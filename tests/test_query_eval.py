"""Unit test cho query_formatter: chuyển multi-line script -> one-line eval expression."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import pytest
from src.query_formatter import (
    QueryExecutionError,
    QueryFormatError,
    convert_script_to_expression,
    is_valid_eval_expr,
)


def test_already_valid():
    df1 = pd.DataFrame({"Chi_tieu": ["Lãi tiền gửi"], "Gia_tri": [208253201298.0]})
    dfs = {"df1": df1}
    expr = "float(df1.iloc[0]['Gia_tri']) / 1000000"
    result = convert_script_to_expression(expr, dfs, 208253.201298)
    assert result == expr, f"Expected passthrough, got: {result}"
    assert is_valid_eval_expr(result, dfs, 208253.201298)
    print("  PASS test_already_valid")


def test_multiline_contains():
    df1 = pd.DataFrame({"Chi_tieu": ["Lãi tiền gửi", "Doanh thu"], "Gia_tri": [208253201298.0, 500000.0]})
    dfs = {"df1": df1}
    script = (
        "m = df1[df1['Chi_tieu'].str.contains(r'lãi tiền gửi', case=False, na=False)]\n"
        "val = float(m.iloc[0]['Gia_tri'])\n"
        "answer = val / 1_000_000\n"
        "print(answer)"
    )
    result = convert_script_to_expression(script, dfs, 208253.201298)
    assert "\n" not in result, f"Contains newline: {result}"
    assert "print(" not in result
    val = eval(result, {"pd": pd, **dfs})
    assert abs(float(val) - 208253.201298) < 1.0, f"Value mismatch: {val}"
    print(f"  PASS test_multiline_contains -> {result}")


def test_signed_value_is_preserved_without_invented_abs():
    df1 = pd.DataFrame({"Chi_tieu": ["Chi phí"], "Gia_tri": [-5947205.0]})
    dfs = {"df1": df1}
    code = "val = float(df1.iloc[0]['Gia_tri'])\nanswer = val\nprint(answer)"
    result = convert_script_to_expression(code, dfs, 5947205.0)
    assert "abs(" not in result
    val = eval(result, {"pd": pd, **dfs})
    assert float(val) == -5947205.0


def test_two_table_sum():
    df1 = pd.DataFrame({"Chi_tieu": ["Nợ ngắn hạn"], "Gia_tri": [100000000000.0]})
    df2 = pd.DataFrame({"Chi_tieu": ["Nợ dài hạn"], "Gia_tri": [50000000000.0]})
    dfs = {"df1": df1, "df2": df2}
    code = (
        "x = float(df1.iloc[0]['Gia_tri'])\n"
        "y = float(df2.iloc[0]['Gia_tri'])\n"
        "answer = (x + y) / 1_000_000_000\nprint(answer)"
    )
    result = convert_script_to_expression(code, dfs, -999.0)
    assert "\n" not in result
    val = eval(result, {"pd": pd, **dfs})
    assert abs(float(val) - 150.0) < 1.0, f"Value mismatch: {val}"
    print(f"  PASS test_two_table_sum -> {result}")


def test_ratio_percent():
    df1 = pd.DataFrame({"Chi_tieu": ["LNST"], "Gia_tri": [300.0]})
    df2 = pd.DataFrame({"Chi_tieu": ["Doanh thu"], "Gia_tri": [1000.0]})
    dfs = {"df1": df1, "df2": df2}
    code = (
        "profit = float(df1.iloc[0]['Gia_tri'])\n"
        "revenue = float(df2.iloc[0]['Gia_tri'])\n"
        "pct = profit / revenue * 100\nprint(pct)"
    )
    result = convert_script_to_expression(code, dfs, -999.0)
    assert "\n" not in result
    val = eval(result, {"pd": pd, **dfs})
    assert abs(float(val) - 30.0) < 1.0, f"Value mismatch: {val}"
    print(f"  PASS test_ratio_percent -> {result}")


def test_invalid_code_never_falls_back_to_a_row_or_constant():
    dfs = {"df1": pd.DataFrame({"Chi_tieu": ["X"], "Gia_tri": [999.0]})}
    with pytest.raises(QueryFormatError):
        convert_script_to_expression("garbage code", dfs, 42.5)


def test_unicode_escape_in_script():
    """Kiểm tra script chứa escape sequence \\u không làm sập converter do lỗi regex template."""
    df1 = pd.DataFrame({"Chi_tieu": ["Thù lao HĐQT"], "Gia_tri": [5000000.0]})
    code = (
        "m = df1[df1['Chi_tieu'].str.contains(r'thu lao h\\u1eafu d\\u1ea1ng', case=False, na=False)]\n"
        "val = float(m.iloc[0]['Gia_tri'])\n"
        "answer = val / 1_000_000\n"
        "print(answer)"
    )
    with pytest.raises(QueryExecutionError):
        convert_script_to_expression(code, {"df1": df1}, 5.0)



if __name__ == "__main__":
    print("=== Testing query_formatter ===")
    test_already_valid()
    test_multiline_contains()
    test_signed_value_is_preserved_without_invented_abs()
    test_two_table_sum()
    test_ratio_percent()
    test_invalid_code_never_falls_back_to_a_row_or_constant()
    print("\n=== ALL TESTS PASSED ===")
