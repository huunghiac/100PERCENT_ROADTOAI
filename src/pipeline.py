import json
import zipfile
import os
import re
import sys
import time
from retriever import TableRetriever
from agent import PandasAgent


# ---------------------------------------------------------------------------
# Helpers: build relevant_docs & relevant_tables từ manifest entry
# ---------------------------------------------------------------------------

def _doc_id_from_source_txt(source_txt: str) -> str:
    """
    source_txt = ".../AAA/2015/AAA_financial_statements_2015_consolidated/AAA_..._extracted.txt"
    → "AAA_financial_statements_2015_consolidated"
    Quy tắc BTC: tên thư mục chứa file txt (không phải tên file _extracted.txt).
    """
    if not source_txt:
        return ""
    parts = source_txt.replace("\\", "/").split("/")
    for part in reversed(parts):
        if "_financial_statements_" in part and not part.endswith(".txt"):
            return part
    fname = parts[-1]
    return re.sub(r"_extracted\.txt$", "", fname)


def _build_submission_fields(csv_paths: list, manifest: dict):
    """
    Sinh relevant_docs, relevant_tables, evidence đúng schema BTC.
    evidence[i].variable = "df1", "df2", ...  (hợp lệ Python, không trùng)
    """
    relevant_docs = []
    relevant_tables = []
    evidence = []
    seen_docs = set()

    for i, csv_path in enumerate(csv_paths):
        var_name = f"df{i + 1}"
        # csv_path trong submission phải bắt đầu bằng data/
        arc_path = f"data/{os.path.basename(csv_path)}"

        evidence.append({"variable": var_name, "csv_path": arc_path})

        entry = manifest.get(csv_path, manifest.get(arc_path, {}))
        source_txt = entry.get("source_txt", "")
        table_index = entry.get("source_table_index", None)

        doc_id = _doc_id_from_source_txt(source_txt)
        if doc_id and doc_id not in seen_docs:
            seen_docs.add(doc_id)
            relevant_docs.append(doc_id)
        if doc_id and table_index is not None:
            relevant_tables.append(f"{doc_id}|{table_index}")

    return relevant_docs, relevant_tables, evidence


def run_full_pipeline(questions_file="data/raw_vifinqa/questions.jsonl",
                      output_json="submission.json",
                      output_zip="submission.zip",
                      max_questions=None,
                      checkpoint_interval=10):
    """
    Pipeline chính:
    1. Load checkpoint cũ nếu có
    2. Đọc câu hỏi từ questions.jsonl
    3. TableRetriever tìm CSV
    4. PandasAgent sinh code + thực thi + self-correct
    5. Checkpoint mỗi N câu
    6. Ghi submission.json + submission.zip
    """
    print("=== Khởi động ViFinQA Pipeline ===")

    # Checkpoint loading
    results_map = {}
    used_csv_paths = set()

    if os.path.exists(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
                for item in existing_data:
                    results_map[item["id"]] = item
                    for ev in item.get("evidence", []):
                        if "csv_path" in ev:
                            used_csv_paths.add(ev["csv_path"])
            print(f"[Checkpoint] Loaded {len(results_map)} existing results.")
        except Exception as e:
            print(f"[Checkpoint] Error loading: {e}. Starting fresh.")

    # Init modules
    retriever = TableRetriever(csv_dir="data/processed_csv")
    # backend="auto": dùng transformers (Kaggle GPU) nếu có, fallback ollama (local)
    agent = PandasAgent()

    if not os.path.exists(questions_file):
        print(f"ERROR: {questions_file} not found")
        return

    with open(questions_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if max_questions:
        lines = lines[:max_questions]
        print(f"[Config] Running on {max_questions} questions.")
    else:
        print(f"[Config] Running on all {len(lines)} questions.")

    processed_count = 0
    skipped_no_csv = 0
    t_start = time.time()

    for idx, line in enumerate(lines, 1):
        if not line.strip():
            continue

        q_data = json.loads(line)
        q_id = q_data["id"]
        question = q_data["question"]

        if q_id in results_map:
            continue

        print(f"\n--- [{idx}/{len(lines)}] ID={q_id}: {question[:70]}... ---")

        # 1. Retrieval
        csv_paths = retriever.retrieve(question, top_k=3)
        ticker, year = retriever.extract_entities(question)
        print(f"  Ticker={ticker} Year={year} CSVs={len(csv_paths)}")

        if not csv_paths:
            skipped_no_csv += 1

        for p in csv_paths:
            used_csv_paths.add(p)

        # 2. Agent
        ans, pandas_code, err = agent.run_agent(question, csv_paths, max_retries=2)
        print(f"  Answer: {ans}")
        if err:
            print(f"  Error: {err[:200]}")

        # 3. Format result
        relevant_docs, relevant_tables, evidence = _build_submission_fields(
            csv_paths, retriever.manifest
        )

        # Parse answer to numeric
        formatted_ans = ans
        try:
            formatted_ans = float(ans)
            if formatted_ans == int(formatted_ans) and abs(formatted_ans) < 1e15:
                formatted_ans = int(formatted_ans)
        except (ValueError, TypeError, OverflowError):
            formatted_ans = str(ans)

        results_map[q_id] = {
            "id": q_id,
            "question": question,
            "answer": formatted_ans,
            "relevant_docs": relevant_docs,
            "relevant_tables": relevant_tables,
            "evidence": evidence,
            "pandas_query": pandas_code if pandas_code else ""
        }
        processed_count += 1

        # Checkpoint
        if processed_count % checkpoint_interval == 0:
            elapsed = time.time() - t_start
            rate = processed_count / elapsed * 60 if elapsed > 0 else 0
            print(f"[Checkpoint] {processed_count} done, {rate:.1f} q/min. Saving...")
            _save_json(results_map, output_json)

    # Final save
    print(f"\n[Save] {len(results_map)} results -> {output_json}")
    _save_json(results_map, output_json)

    # Stats
    elapsed = time.time() - t_start
    print(f"[Stats] Processed={processed_count}, Skipped(no CSV)={skipped_no_csv}, Time={elapsed:.0f}s")

    # Zip
    print(f"[Zip] Creating {output_zip}...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(output_json, arcname=os.path.basename(output_json))
        for csv_path in used_csv_paths:
            real_path = csv_path if os.path.exists(csv_path) else csv_path.replace("data/", "", 1)
            if os.path.exists(real_path):
                arc_name = csv_path if csv_path.startswith("data/") else f"data/{os.path.basename(csv_path)}"
                zf.write(real_path, arcname=arc_name)
            else:
                print(f"  [Warn] {real_path} not found, skipping.")

    print(f"=== Done! {output_zip} ready ===")


def _save_json(results_map, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(results_map.values()), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    max_q = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_full_pipeline(max_questions=max_q)

