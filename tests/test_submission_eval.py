"""Mô phỏng bộ chấm điểm của Ban Tổ Chức để tự kiểm tra Submission."""
import json
import os
import sys
import re
import pandas as pd
import numpy as np


def simulate_btc_eval(submission_path="submission.json", csv_dir="data/processed_csv"):
    if not os.path.exists(submission_path):
        print(f"[FAIL] {submission_path} không tồn tại.")
        return

    with open(submission_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"=== Đang mô phỏng chấm điểm {len(data)} câu hỏi ===")

    total = len(data)
    exec_success = 0
    match_answer = 0
    table_format_ok = 0
    errors = []

    for i, item in enumerate(data):
        q_id = item.get("id")
        ans = item.get("answer")
        query = item.get("pandas_query", "").strip()
        evidence = item.get("evidence", [])
        rel_tables = item.get("relevant_tables", [])

        # 1. Kiểm tra format relevant_tables: <doc_id>|<line_number>
        tbl_ok = True
        if not rel_tables and evidence:
            tbl_ok = False  # Lỗi nếu có evidence mà relevant_tables bị rỗng
        for t in rel_tables:
            parts = t.split("|")
            if len(parts) != 2 or not parts[1].isdigit():
                tbl_ok = False
                break
        if tbl_ok and rel_tables:
            table_format_ok += 1

        # 1b. Kiểm tra quy chuẩn AST / Cú pháp query
        query_has_lambda = "lambda" in query
        query_has_df = bool(re.search(r'\bdf\d+\b', query))
        if query_has_lambda:
            errors.append((q_id, "Query contains forbidden 'lambda'", query[:60]))
        elif not query_has_df and query != "0.0":
            errors.append((q_id, "Query does not reference any DataFrame variable", query[:60]))

        # 2. Chuẩn bị môi trường nạp DataFrames
        exec_scope = {"pd": pd, "np": np}
        for ev in evidence:
            var_name = ev.get("variable", "df1")
            csv_path = ev.get("csv_path", "")
            # Tìm file CSV thực tế
            real_path = csv_path if os.path.exists(csv_path) else os.path.join("data", os.path.basename(csv_path))
            if not os.path.exists(real_path):
                bn = os.path.basename(csv_path)
                ticker = bn.split("_")[0] if "_" in bn else ""
                cand = os.path.join(csv_dir, ticker, bn)
                if os.path.exists(cand):
                    real_path = cand

            if os.path.exists(real_path):
                try:
                    exec_scope[var_name] = pd.read_csv(real_path)
                except Exception:
                    pass

        # 3. Thực thi query
        calc_val = None
        if query and not query.startswith("# GENERATION_FAILED"):
            try:
                # Thử eval trước
                calc_val = eval(query, exec_scope)
                exec_success += 1
            except Exception:
                # Thử exec
                try:
                    exec(query, exec_scope)
                    if "result" in exec_scope:
                        calc_val = exec_scope["result"]
                    elif "answer" in exec_scope:
                        calc_val = exec_scope["answer"]
                    exec_success += 1
                except Exception as e:
                    errors.append((q_id, f"Exec error: {e}", query[:60]))

        # 4. So khớp kết quả
        if calc_val is not None and ans is not None:
            try:
                val_num = float(calc_val)
                ans_num = float(ans)
                # Ngưỡng sai số 1% hoặc tuyệt đối 0.01
                if abs(val_num - ans_num) <= max(1e-2, 0.01 * abs(ans_num)):
                    match_answer += 1
            except Exception:
                pass

    print(f"\n--- KẾT QUẢ MÔ PHỎNG ĐÁNH GIÁ ---")
    print(f"Tổng số câu: {total}")
    print(f"Relevant Tables Format Hợp Lệ: {table_format_ok}/{total} ({table_format_ok/total*100:.2f}%)")
    print(f"Execution Accuracy (Query chạy thành công): {exec_success}/{total} ({exec_success/total*100:.2f}%)")
    print(f"Query Result Match Answer: {match_answer}/{total} ({match_answer/total*100:.2f}%)")
    if errors:
        print(f"\nTop 5 lỗi thực thi mẫu:")
        for err in errors[:5]:
            print(f"  ID {err[0]}: {err[1]} | Query: {err[2]}")


if __name__ == "__main__":
    sub_file = sys.argv[1] if len(sys.argv) > 1 else "submission.json"
    simulate_btc_eval(sub_file)
