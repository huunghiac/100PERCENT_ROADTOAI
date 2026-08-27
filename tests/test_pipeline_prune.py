"""
Tests for pipeline.py evidence pruning and answer sanitization logic.
Run: python tests/test_pipeline_prune.py
"""
import re
import os
import sys

# ---------- Inline extract of pruning + sanitize logic ----------
# (mirrors pipeline.py lines 259-289 exactly)

def _prune_and_sanitize(final_query, table_items, formatted_ans):
    """Replicate the pruning + sanitize block from pipeline.py."""
    used_vars = set(re.findall(r'\bdf(\d+)\b', final_query))
    if not used_vars and table_items:
        used_vars = {"1"}

    pruned_evidence = [item["evidence"] for item in table_items if item["var_num"] in used_vars]
    pruned_tables = [item["table_entry"] for item in table_items if item["var_num"] in used_vars and item["table_entry"]]
    pruned_docs = list(dict.fromkeys(item["doc_id"] for item in table_items if item["var_num"] in used_vars and item["doc_id"]))

    if not pruned_evidence and table_items:
        pruned_evidence = [table_items[0]["evidence"]]
        if table_items[0]["table_entry"]:
            pruned_tables = [table_items[0]["table_entry"]]
        if table_items[0]["doc_id"]:
            pruned_docs = [table_items[0]["doc_id"]]

    if isinstance(formatted_ans, str):
        if formatted_ans.strip().lower() in ("none", "nan", "null", "inf", "-inf", "error", ""):
            formatted_ans = 0.0

    return pruned_evidence, pruned_tables, pruned_docs, formatted_ans


# ---------- Tests ----------

def test_prune_df2_when_query_only_df1():
    table_items = [
        {
            "var_name": "df1",
            "var_num": "1",
            "evidence": {"variable": "df1", "csv_path": "data/ACB_2022_KQKD.csv"},
            "doc_id": "ACB_financial_statements_2022_consolidated",
            "table_entry": "ACB_financial_statements_2022_consolidated|15"
        },
        {
            "var_name": "df2",
            "var_num": "2",
            "evidence": {"variable": "df2", "csv_path": "data/ACB_2022_CDKT.csv"},
            "doc_id": "ACB_financial_statements_2022_consolidated",
            "table_entry": "ACB_financial_statements_2022_consolidated|42"
        },
    ]
    query = "float(df1[df1['Chi_tieu'].str.contains(r'doanh thu', case=False, na=False)]['Gia_tri'].iloc[0])"

    ev, tb, doc, ans = _prune_and_sanitize(query, table_items, 123.0)
    assert len(ev) == 1, f"Expected 1 evidence, got {len(ev)}"
    assert ev[0]["variable"] == "df1"
    assert len(tb) == 1, f"Expected 1 table, got {len(tb)}"
    assert tb[0] == "ACB_financial_statements_2022_consolidated|15"
    assert len(doc) == 1
    assert doc[0] == "ACB_financial_statements_2022_consolidated"
    print("  PASS test_prune_df2_when_query_only_df1")


def test_keep_both_when_query_uses_df1_df2():
    table_items = [
        {
            "var_name": "df1",
            "var_num": "1",
            "evidence": {"variable": "df1", "csv_path": "data/ACB_2022_KQKD.csv"},
            "doc_id": "ACB_financial_statements_2022_consolidated",
            "table_entry": "ACB_financial_statements_2022_consolidated|15"
        },
        {
            "var_name": "df2",
            "var_num": "2",
            "evidence": {"variable": "df2", "csv_path": "data/ACB_2022_CDKT.csv"},
            "doc_id": "ACB_financial_statements_2022_consolidated",
            "table_entry": "ACB_financial_statements_2022_consolidated|42"
        },
    ]
    query = "float(df1.iloc[0]['Gia_tri']) / float(df2.iloc[0]['Gia_tri']) * 100"

    ev, tb, doc, ans = _prune_and_sanitize(query, table_items, 5.5)
    assert len(ev) == 2, f"Expected 2 evidence, got {len(ev)}"
    assert len(tb) == 2, f"Expected 2 tables, got {len(tb)}"
    assert len(doc) == 1
    print("  PASS test_keep_both_when_query_uses_df1_df2")


def test_sanitize_none_string():
    ev, tb, doc, ans = _prune_and_sanitize("float(0.0)", [], "None")
    assert ans == 0.0, f"Expected 0.0, got {repr(ans)}"
    print("  PASS test_sanitize_none_string")


def test_sanitize_nan_string():
    ev, tb, doc, ans = _prune_and_sanitize("float(0.0)", [], "nan")
    assert ans == 0.0, f"Expected 0.0, got {repr(ans)}"
    print("  PASS test_sanitize_nan_string")


def test_sanitize_keeps_valid_number():
    ev, tb, doc, ans = _prune_and_sanitize("float(42.0)", [], 42.0)
    assert ans == 42.0
    print("  PASS test_sanitize_keeps_valid_number")


def test_sanitize_keeps_valid_string_number():
    ev, tb, doc, ans = _prune_and_sanitize("float(42.0)", [], "42.0")
    assert ans == "42.0", f"Should keep string '42.0', got {repr(ans)}"
    print("  PASS test_sanitize_keeps_valid_string_number")


def test_prune_empty_query_no_crash():
    table_items = [
        {"var_name": "df1", "var_num": "1", "evidence": {"variable": "df1", "csv_path": "data/X.csv"}, "doc_id": "X", "table_entry": "X|1"},
        {"var_name": "df2", "var_num": "2", "evidence": {"variable": "df2", "csv_path": "data/Y.csv"}, "doc_id": "Y", "table_entry": "Y|1"},
    ]
    ev, tb, doc, ans = _prune_and_sanitize("", table_items, 0.0)
    assert len(ev) == 1
    assert ev[0]["variable"] == "df1"
    print("  PASS test_prune_empty_query_no_crash")


def test_prune_constant_query_no_crash():
    table_items = [
        {"var_name": "df1", "var_num": "1", "evidence": {"variable": "df1", "csv_path": "data/X.csv"}, "doc_id": "X", "table_entry": "X|1"},
        {"var_name": "df2", "var_num": "2", "evidence": {"variable": "df2", "csv_path": "data/Y.csv"}, "doc_id": "Y", "table_entry": "Y|1"},
    ]
    ev, tb, doc, ans = _prune_and_sanitize("float(0.0)", table_items, 0.0)
    assert len(ev) == 1
    assert ev[0]["variable"] == "df1"
    print("  PASS test_prune_constant_query_no_crash")


def test_sanitize_inf():
    ev, tb, doc, ans = _prune_and_sanitize("0.0", [], "inf")
    assert ans == 0.0
    print("  PASS test_sanitize_inf")


def test_sanitize_neg_inf():
    ev, tb, doc, ans = _prune_and_sanitize("0.0", [], "-inf")
    assert ans == 0.0
    print("  PASS test_sanitize_neg_inf")

    print("  PASS test_sanitize_neg_inf")


if __name__ == "__main__":
    print("=== Testing pipeline prune & sanitize ===")
    test_prune_df2_when_query_only_df1()
    test_keep_both_when_query_uses_df1_df2()
    test_sanitize_none_string()
    test_sanitize_nan_string()
    test_sanitize_keeps_valid_number()
    test_sanitize_keeps_valid_string_number()
    test_prune_empty_query_no_crash()
    test_prune_constant_query_no_crash()
    test_sanitize_inf()
    test_sanitize_neg_inf()
    print("\n=== ALL TESTS PASSED ===")
