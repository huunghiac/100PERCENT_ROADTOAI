"""Validate submission.json against BTC schema from README."""
import json
import os
import sys


def validate(path="submission.json"):
    if not os.path.exists(path):
        print(f"[FAIL] {path} does not exist yet.")
        print("       Run pipeline.py to generate it.")
        return False

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print("[FAIL] Root must be a JSON array.")
        return False

    required = ["id", "question", "answer", "relevant_docs",
                 "relevant_tables", "evidence", "pandas_query"]
    errors = []
    for i, item in enumerate(data):
        for field in required:
            if field not in item:
                errors.append(f"Item {i} (id={item.get('id','?')}): missing '{field}'")

        # Type checks
        if not isinstance(item.get("id"), int):
            errors.append(f"Item {i}: 'id' should be int, got {type(item.get('id')).__name__}")
        if not isinstance(item.get("question"), str):
            errors.append(f"Item {i}: 'question' should be str")
        if not isinstance(item.get("answer"), (int, float)):
            errors.append(f"Item {i}: 'answer' should be numeric, got {type(item.get('answer')).__name__}: {repr(item.get('answer'))[:60]}")
        if not isinstance(item.get("relevant_docs"), list):
            errors.append(f"Item {i}: 'relevant_docs' should be list")
        if not isinstance(item.get("relevant_tables"), list):
            errors.append(f"Item {i}: 'relevant_tables' should be list")
        if not isinstance(item.get("pandas_query"), str):
            errors.append(f"Item {i}: 'pandas_query' should be str")

        # Evidence structure
        ev = item.get("evidence", [])
        if not isinstance(ev, list):
            errors.append(f"Item {i}: 'evidence' should be list")
        else:
            for j, e in enumerate(ev):
                if "variable" not in e:
                    errors.append(f"Item {i}, evidence[{j}]: missing 'variable'")
                if "csv_path" not in e:
                    errors.append(f"Item {i}, evidence[{j}]: missing 'csv_path'")
                elif not e["csv_path"].startswith("data/"):
                    errors.append(f"Item {i}, evidence[{j}]: csv_path must start with 'data/', got '{e['csv_path']}'")
                elif e["csv_path"].count("/") > 1 or "\\" in e["csv_path"]:
                    errors.append(f"Item {i}, evidence[{j}]: csv_path must be flat 'data/<file>.csv', got '{e['csv_path']}'")

    if errors:
        print(f"[FAIL] {len(errors)} validation errors:")
        for e in errors[:20]:
            print(f"  - {e}")
        if len(errors) > 20:
            print(f"  ... and {len(errors) - 20} more")
        return False

    print(f"[PASS] {path} valid: {len(data)} items, all 7 fields present, types correct.")
    return True


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "submission.json"
    validate(path)
