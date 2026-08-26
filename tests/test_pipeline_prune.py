"""
Tests for pipeline.py evidence pruning and answer sanitization logic.
Run: python tests/test_pipeline_prune.py
"""
import re
import os
import sys

# ---------- Inline extract of pruning + sanitize logic ----------
# (mirrors pipeline.py lines 259-289 exactly)

def _prune_and_sanitize(final_query, evidence, relevant_tables, formatted_ans):
    """Replicate the pruning + sanitize block from pipeline.py."""
    # --- Prune unused evidence ---
    used_vars = set(re.findall(r'\bdf(\d+)\b', final_query))
    if used_vars and evidence:
        pruned_evidence = []
        pruned_tables = []
        for ev_item in evidence:
            vn = ev_item.get("variable", "")
            var_num = vn.replace("df", "")
            if var_num in used_vars:
                pruned_evidence.append(ev_item)
        pruned_csv_basenames = {os.path.basename(ev.get("csv_path", "")) for ev in pruned_evidence}
        for tbl_entry in relevant_tables:
            keep = True
            if "|" in tbl_entry:
                doc_part = tbl_entry.split("|")[0]
                keep = any(doc_part in bn for bn in pruned_csv_basenames)
            if keep:
                pruned_tables.append(tbl_entry)
        if pruned_evidence:
            evidence = pruned_evidence
            relevant_tables = pruned_tables

    # --- Sanitize answer ---
    if isinstance(formatted_ans, str):
        if formatted_ans.strip().lower() in ("none", "nan", "null", "inf", "-inf", "error", ""):
            formatted_ans = 0.0

    return evidence, relevant_tables, formatted_ans


# ---------- Tests ----------

def test_prune_df2_when_query_only_df1():
    evidence = [
        {"variable": "df1", "csv_path": "data/ACB_2022_KQKD.csv"},
        {"variable": "df2", "csv_path": "data/ACB_2022_CDKT.csv"},
    ]
    tables = ["ACB_financial_statements_2022_consolidated|15",
              "ACB_financial_statements_2022_consolidated|42"]
    query = "float(df1[df1['Chi_tieu'].str.contains(r'doanh thu', case=False, na=False)]['Gia_tri'].iloc[0])"

    ev, tb, ans = _prune_and_sanitize(query, evidence, tables, 123.0)
    assert len(ev) == 1, f"Expected 1 evidence, got {len(ev)}"
    assert ev[0]["variable"] == "df1"
    print("  PASS test_prune_df2_when_query_only_df1")


def test_keep_both_when_query_uses_df1_df2():
    evidence = [
        {"variable": "df1", "csv_path": "data/ACB_2022_KQKD.csv"},
        {"variable": "df2", "csv_path": "data/ACB_2022_CDKT.csv"},
    ]
    tables = ["ACB_financial_statements_2022_consolidated|15"]
    query = "float(df1.iloc[0]['Gia_tri']) / float(df2.iloc[0]['Gia_tri']) * 100"

    ev, tb, ans = _prune_and_sanitize(query, evidence, tables, 5.5)
    assert len(ev) == 2, f"Expected 2 evidence, got {len(ev)}"
    print("  PASS test_keep_both_when_query_uses_df1_df2")


def test_sanitize_none_string():
    ev, tb, ans = _prune_and_sanitize("float(0.0)", [], [], "None")
    assert ans == 0.0, f"Expected 0.0, got {repr(ans)}"
    print("  PASS test_sanitize_none_string")


def test_sanitize_nan_string():
    ev, tb, ans = _prune_and_sanitize("float(0.0)", [], [], "nan")
    assert ans == 0.0, f"Expected 0.0, got {repr(ans)}"
    print("  PASS test_sanitize_nan_string")


def test_sanitize_keeps_valid_number():
    ev, tb, ans = _prune_and_sanitize("float(42.0)", [], [], 42.0)
    assert ans == 42.0
    print("  PASS test_sanitize_keeps_valid_number")


def test_sanitize_keeps_valid_string_number():
    ev, tb, ans = _prune_and_sanitize("float(42.0)", [], [], "42.0")
    assert ans == "42.0", f"Should keep string '42.0', got {repr(ans)}"
    print("  PASS test_sanitize_keeps_valid_string_number")


def test_prune_empty_query_no_crash():
    evidence = [
        {"variable": "df1", "csv_path": "data/X.csv"},
        {"variable": "df2", "csv_path": "data/Y.csv"},
    ]
    ev, tb, ans = _prune_and_sanitize("", evidence, [], 0.0)
    # empty query → used_vars empty → no pruning → keep all
    assert len(ev) == 2
    print("  PASS test_prune_empty_query_no_crash")


def test_prune_constant_query_no_crash():
    evidence = [
        {"variable": "df1", "csv_path": "data/X.csv"},
        {"variable": "df2", "csv_path": "data/Y.csv"},
    ]
    ev, tb, ans = _prune_and_sanitize("float(0.0)", evidence, [], 0.0)
    # "float(0.0)" → no df\d match → used_vars empty → no pruning
    assert len(ev) == 2
    print("  PASS test_prune_constant_query_no_crash")


def test_sanitize_inf():
    ev, tb, ans = _prune_and_sanitize("0.0", [], [], "inf")
    assert ans == 0.0
    print("  PASS test_sanitize_inf")


def test_sanitize_neg_inf():
    ev, tb, ans = _prune_and_sanitize("0.0", [], [], "-inf")
    assert ans == 0.0
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
