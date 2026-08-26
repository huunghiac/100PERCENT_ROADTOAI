"""Unit test cho query_formatter: chuyển multi-line script -> one-line eval expression."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.query_formatter import convert_script_to_expression, is_valid_eval_expr


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


def test_abs_value():
    df1 = pd.DataFrame({"Chi_tieu": ["Chi phí"], "Gia_tri": [-5947205.0]})
    dfs = {"df1": df1}
    result = convert_script_to_expression("abs(val)", dfs, 5947205.0)
    assert "\n" not in result
    val = eval(result, {"pd": pd, **dfs})
    assert abs(float(val) - 5947205.0) < 1.0, f"Value mismatch: {val}"
    print(f"  PASS test_abs_value -> {result}")


def test_two_table_sum():
    df1 = pd.DataFrame({"Chi_tieu": ["Nợ ngắn hạn"], "Gia_tri": [100000000000.0]})
    df2 = pd.DataFrame({"Chi_tieu": ["Nợ dài hạn"], "Gia_tri": [50000000000.0]})
    dfs = {"df1": df1, "df2": df2}
    result = convert_script_to_expression("x+y", dfs, 150.0)  # 150 tỷ
    assert "\n" not in result
    val = eval(result, {"pd": pd, **dfs})
    assert abs(float(val) - 150.0) < 1.0, f"Value mismatch: {val}"
    print(f"  PASS test_two_table_sum -> {result}")


def test_ratio_percent():
    df1 = pd.DataFrame({"Chi_tieu": ["LNST"], "Gia_tri": [300.0]})
    df2 = pd.DataFrame({"Chi_tieu": ["Doanh thu"], "Gia_tri": [1000.0]})
    dfs = {"df1": df1, "df2": df2}
    result = convert_script_to_expression("pct", dfs, 30.0)  # 30%
    assert "\n" not in result
    val = eval(result, {"pd": pd, **dfs})
    assert abs(float(val) - 30.0) < 1.0, f"Value mismatch: {val}"
    print(f"  PASS test_ratio_percent -> {result}")


def test_fallback_constant():
    dfs = {"df1": pd.DataFrame({"Chi_tieu": ["X"], "Gia_tri": [999.0]})}
    result = convert_script_to_expression("garbage code", dfs, 42.5)
    assert result == "float(42.5)", f"Expected constant fallback, got: {result}"
    val = eval(result, {})
    assert abs(float(val) - 42.5) < 0.01
    print(f"  PASS test_fallback_constant -> {result}")


if __name__ == "__main__":
    print("=== Testing query_formatter ===")
    test_already_valid()
    test_multiline_contains()
    test_abs_value()
    test_two_table_sum()
    test_ratio_percent()
    test_fallback_constant()
    print("\n=== ALL TESTS PASSED ===")
