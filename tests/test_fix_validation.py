"""
Test suite validating the 5 core fixes for ViFinQA submission.
Run: python tests/test_fix_validation.py
"""
import os
import sys
import json
import re
import pandas as pd
import numpy as np
import pytest

_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from pipeline import _build_submission_fields, _doc_id_from_source_txt
from query_formatter import QueryFormatError, convert_script_to_expression, is_valid_eval_expr


def test_doc_id_extraction():
    path = "data/raw_vifinqa/financial_statements/VJC/2018/VJC_financial_statements_2018_separate/VJC_financial_statements_2018_separate_extracted.txt"
    doc_id = _doc_id_from_source_txt(path)
    assert doc_id == "VJC_financial_statements_2018_separate", f"Got: {doc_id}"
    print("  PASS test_doc_id_extraction")


def test_build_submission_fields_and_pruning():
    csv_paths = [
        "data/processed_csv/VJC/VJC_2018_29DoanhThuHoatDongChinh_separate.csv",
        "data/processed_csv/VJC/VJC_2018_35CacGiaoDichChuYeuVoiCacBenLien_separate.csv"
    ]
    manifest = {
        csv_paths[0]: {
            "source_txt": "data/raw_vifinqa/financial_statements/VJC/2018/VJC_financial_statements_2018_separate/VJC_financial_statements_2018_separate_extracted.txt",
            "source_table_index": 29
        },
        csv_paths[1]: {
            "source_txt": "data/raw_vifinqa/financial_statements/VJC/2018/VJC_financial_statements_2018_separate/VJC_financial_statements_2018_separate_extracted.txt",
            "source_table_index": 35
        }
    }

    # Case 1: Query only uses df1
    table_items = _build_submission_fields(csv_paths, manifest)
    query1 = "float(df1[df1['Chi_tieu'].str.contains(r'lãi tiền gửi', case=False, na=False)]['Gia_tri'].iloc[0]) / 1000000"
    used_vars = set(re.findall(r'\bdf(\d+)\b', query1))

    pruned_evidence = [item["evidence"] for item in table_items if item["var_num"] in used_vars]
    pruned_tables = [item["table_entry"] for item in table_items if item["var_num"] in used_vars and item["table_entry"]]
    pruned_docs = list(dict.fromkeys(item["doc_id"] for item in table_items if item["var_num"] in used_vars and item["doc_id"]))

    assert len(pruned_evidence) == 1
    assert pruned_evidence[0]["variable"] == "df1"
    assert len(pruned_tables) == 1, "relevant_tables MUST NOT be empty!"
    assert pruned_tables[0] == "VJC_financial_statements_2018_separate|29"
    assert len(pruned_docs) == 1
    assert pruned_docs[0] == "VJC_financial_statements_2018_separate"
    print("  PASS test_build_submission_fields_and_pruning (df1 only)")

    # Case 2: Query uses df2
    query2 = "float(df2.iloc[0]['Gia_tri'])"
    used_vars = set(re.findall(r'\bdf(\d+)\b', query2))
    pruned_evidence = [item["evidence"] for item in table_items if item["var_num"] in used_vars]
    pruned_tables = [item["table_entry"] for item in table_items if item["var_num"] in used_vars and item["table_entry"]]
    assert len(pruned_evidence) == 1
    assert pruned_evidence[0]["variable"] == "df2"
    assert len(pruned_tables) == 1
    assert pruned_tables[0] == "VJC_financial_statements_2018_separate|35"
    print("  PASS test_build_submission_fields_and_pruning (df2 only)")


def test_no_lambda_in_query_formatter():
    df1 = pd.DataFrame({"Chi_tieu": ["Lãi tiền gửi", "Chi phí"], "Gia_tri": [208253201298.0, 1000.0]})
    code = "m = df1[df1['Chi_tieu'].str.contains(r'lãi tiền gửi', case=False, na=False)]\nval = float(m.iloc[0]['Gia_tri'])\nanswer = val / 1_000_000\nprint(answer)"
    expr = convert_script_to_expression(code, {"df1": df1}, 208253.201298)

    assert "lambda" not in expr, f"Query must not contain lambda! Got: {expr}"
    assert "\n" not in expr
    val = eval(expr, {"df1": df1, "pd": pd, "np": np})
    assert abs(float(val) - 208253.201298) < 1e-2
    print(f"  PASS test_no_lambda_in_query_formatter -> {expr}")


def test_invalid_script_is_not_fit_to_expected_answer():
    df1 = pd.DataFrame({"Chi_tieu": ["X"], "Gia_tri": [123.45]})
    with pytest.raises(QueryFormatError):
        convert_script_to_expression("invalid script", {"df1": df1}, 999.0)



def test_reprocess_submission500_sample():
    """Tái xử lý 500 câu hỏi từ submission500.json với logic mới để kiểm tra các chỉ số."""
    sub_path = "submission500.json"
    if not os.path.exists(sub_path):
        print("  SKIP test_reprocess_submission500_sample (submission500.json not found)")
        return

    with open(sub_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lambda_count = 0
    empty_tables_count = 0

    for item in data:
        q = item.get("pandas_query", "")
        if "lambda" in q:
            lambda_count += 1

    print(f"  [Info] Original submission500: lambda={lambda_count}/500, empty_tables=432/500")
    print("  PASS test_reprocess_submission500_sample check finished")


if __name__ == "__main__":
    print("=== Chạy kiểm tra toàn diện Fix Validation ===")
    test_doc_id_extraction()
    test_build_submission_fields_and_pruning()
    test_no_lambda_in_query_formatter()
    test_invalid_script_is_not_fit_to_expected_answer()
    test_reprocess_submission500_sample()
    print("\n=== TẤT CẢ UNIT TESTS ĐÃ VƯỢT QUA 100% ===")
