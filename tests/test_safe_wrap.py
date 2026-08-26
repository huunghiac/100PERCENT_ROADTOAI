"""
Tests for query_formatter._safe_wrap_expr: IndexError guard on BTC evaluator.
Run: python tests/test_safe_wrap.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import pandas as pd
from query_formatter import _safe_wrap_expr


def test_simple_contains_wrapped():
    expr = "float(df1[df1['Chi_tieu'].str.contains(r'doanh thu', case=False, na=False)]['Gia_tri'].iloc[0])"
    wrapped = _safe_wrap_expr(expr)
    assert "lambda _m=" in wrapped, f"Not wrapped: {wrapped}"
    assert "if len(_m) > 0 else 0.0" in wrapped
    # Eval with match
    df1 = pd.DataFrame({"Chi_tieu": ["Doanh thu thuần"], "Gia_tri": [500.0]})
    val = eval(wrapped, {"df1": df1, "pd": pd})
    assert val == 500.0, f"Expected 500.0, got {val}"
    print("  PASS test_simple_contains_wrapped")


def test_contains_no_match_returns_zero():
    expr = "float(df1[df1['Chi_tieu'].str.contains(r'xyz_not_exist', case=False, na=False)]['Gia_tri'].iloc[0])"
    wrapped = _safe_wrap_expr(expr)
    df1 = pd.DataFrame({"Chi_tieu": ["abc"], "Gia_tri": [100.0]})
    val = eval(wrapped, {"df1": df1, "pd": pd})
    assert val == 0.0, f"Expected 0.0, got {val}"
    print("  PASS test_contains_no_match_returns_zero")


def test_contains_with_division():
    expr = "float(df1[df1['Chi_tieu'].str.contains(r'doanh thu', case=False, na=False)]['Gia_tri'].iloc[0]) / 1000"
    wrapped = _safe_wrap_expr(expr)
    assert "lambda _m=" in wrapped
    df1 = pd.DataFrame({"Chi_tieu": ["Doanh thu"], "Gia_tri": [5000.0]})
    val = eval(wrapped, {"df1": df1, "pd": pd})
    assert abs(val - 5.0) < 1e-6, f"Expected 5.0, got {val}"
    print("  PASS test_contains_with_division")


def test_contains_with_abs():
    expr = "abs(float(df1[df1['Chi_tieu'].str.contains(r'chi phí', case=False, na=False)]['Gia_tri'].iloc[0]))"
    wrapped = _safe_wrap_expr(expr)
    df1 = pd.DataFrame({"Chi_tieu": ["Chi phí bán hàng"], "Gia_tri": [-300.0]})
    val = eval(wrapped, {"df1": df1, "pd": pd})
    assert val == 300.0, f"Expected 300.0, got {val}"
    print("  PASS test_contains_with_abs")


def test_ratio_two_contains():
    expr = ("float(df1[df1['Chi_tieu'].str.contains(r'lnst', case=False, na=False)]['Gia_tri'].iloc[0])"
            " / float(df1[df1['Chi_tieu'].str.contains(r'doanh thu', case=False, na=False)]['Gia_tri'].iloc[0]) * 100")
    wrapped = _safe_wrap_expr(expr)
    # Both should be wrapped
    assert wrapped.count("lambda _m=") == 2, f"Expected 2 lambdas, got {wrapped.count('lambda _m=')}"
    df1 = pd.DataFrame({"Chi_tieu": ["LNST", "Doanh thu"], "Gia_tri": [30.0, 100.0]})
    val = eval(wrapped, {"df1": df1, "pd": pd})
    assert abs(val - 30.0) < 1e-6, f"Expected 30.0, got {val}"
    print("  PASS test_ratio_two_contains")


def test_no_contains_passthrough():
    expr = "float(df1.iloc[0]['Gia_tri']) / 1000"
    wrapped = _safe_wrap_expr(expr)
    assert wrapped == expr, f"Should passthrough, got: {wrapped}"
    print("  PASS test_no_contains_passthrough")


def test_constant_passthrough():
    expr = "float(42.5)"
    wrapped = _safe_wrap_expr(expr)
    assert wrapped == expr
    print("  PASS test_constant_passthrough")


def test_df2_variable():
    expr = "float(df2[df2['Chi_tieu'].str.contains(r'vốn', case=False, na=False)]['Gia_tri'].iloc[0])"
    wrapped = _safe_wrap_expr(expr)
    assert "lambda _m=" in wrapped
    df2 = pd.DataFrame({"Chi_tieu": ["Vốn chủ"], "Gia_tri": [999.0]})
    val = eval(wrapped, {"df2": df2, "pd": pd})
    assert val == 999.0
    print("  PASS test_df2_variable")


if __name__ == "__main__":
    print("=== Testing _safe_wrap_expr ===")
    test_simple_contains_wrapped()
    test_contains_no_match_returns_zero()
    test_contains_with_division()
    test_contains_with_abs()
    test_ratio_two_contains()
    test_no_contains_passthrough()
    test_constant_passthrough()
    test_df2_variable()
    print("\n=== ALL TESTS PASSED ===")
