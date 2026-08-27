import json
import os
import sys
import re
import pandas as pd
import numpy as np

sys.path.insert(0, "src")
from retriever import TableRetriever
from pipeline import _build_submission_fields
from query_formatter import convert_script_to_expression

def fix_submission(in_path="submission500.json", out_path="submission500_fixed.json"):
    r = TableRetriever(csv_dir="data/processed_csv")
    with open(in_path, "r", encoding="utf-8") as f:
        sub = json.load(f)

    fixed = []
    for item in sub:
        qid = item["id"]
        q = item["question"]
        ans = item["answer"]
        ev = item.get("evidence", [])
        old_query = item.get("pandas_query", "")

        # CSV paths from evidence
        csv_paths = []
        for e in ev:
            bn = os.path.basename(e.get("csv_path", ""))
            ticker = bn.split("_")[0] if "_" in bn else ""
            real_path = os.path.join("data", "processed_csv", ticker, bn).replace("\\", "/")
            csv_paths.append(real_path)

        if not csv_paths:
            csv_paths = r.retrieve(q)

        table_items = _build_submission_fields(csv_paths, r.manifest, retriever=r)

        eval_dfs = {}
        for t_item in table_items:
            vn = t_item["var_name"]
            cp = t_item["evidence"]["csv_path"]
            bn = os.path.basename(cp)
            ticker = bn.split("_")[0] if "_" in bn else ""
            real = os.path.join("data", "processed_csv", ticker, bn)
            if os.path.exists(real):
                try:
                    eval_dfs[vn] = pd.read_csv(real)
                except Exception:
                    pass

        try:
            exp_ans = float(ans) if ans is not None else 0.0
        except Exception:
            exp_ans = 0.0

        clean_query = convert_script_to_expression(old_query, eval_dfs, exp_ans)

        used_vars = set(re.findall(r'\bdf(\d+)\b', clean_query))
        if not used_vars and table_items:
            used_vars = {"1"}

        pruned_evidence = [it["evidence"] for it in table_items if it["var_num"] in used_vars]
        pruned_tables = [it["table_entry"] for it in table_items if it["var_num"] in used_vars and it["table_entry"]]
        pruned_docs = list(dict.fromkeys(it["doc_id"] for it in table_items if it["var_num"] in used_vars and it["doc_id"]))

        if not pruned_evidence and table_items:
            pruned_evidence = [table_items[0]["evidence"]]
            if table_items[0]["table_entry"]:
                pruned_tables = [table_items[0]["table_entry"]]
            if table_items[0]["doc_id"]:
                pruned_docs = [table_items[0]["doc_id"]]

        fixed.append({
            "id": qid,
            "question": q,
            "answer": ans,
            "relevant_docs": pruned_docs,
            "relevant_tables": pruned_tables,
            "evidence": pruned_evidence,
            "pandas_query": clean_query
        })

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(fixed, f, ensure_ascii=False, indent=2)

    print(f"Successfully converted {len(fixed)} questions -> {out_path}")

if __name__ == "__main__":
    fix_submission()
