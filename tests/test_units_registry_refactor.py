"""Regression tests for canonical units and deterministic financial formulas.

These tests intentionally use synthetic inputs: formulas must be reusable and
must not depend on a ViFinQA question id or a known answer from the test set.
"""

from __future__ import annotations

import math
import os
import sys

import pytest


_SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src"))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from metric_registry import DEFAULT_REGISTRY, FormulaError  # noqa: E402
from units import (  # noqa: E402
    UnitConversionError,
    UnitDimension,
    conversion_factor,
    convert_value,
    detect_target_unit,
    detect_unit,
)


@pytest.mark.parametrize(
    ("text", "canonical_name", "dimension", "scale"),
    [
        ("Quy đổi sang trăm tỷ đồng", "trăm tỷ đồng", UnitDimension.VND, 1e11),
        ("Đơn vị: nghìn tỷ đồng", "nghìn tỷ đồng", UnitDimension.VND, 1e12),
        ("Kết quả theo triệu đồng", "triệu đồng", UnitDimension.VND, 1e6),
        ("Kết quả theo nghìn đồng", "nghìn đồng", UnitDimension.VND, 1e3),
        ("Kết quả theo triệu USD", "triệu USD", UnitDimension.USD, 1e6),
        ("Tỷ suất là bao nhiêu phần trăm?", "%", UnitDimension.RATIO, 0.01),
        (
            "Chênh lệch bao nhiêu điểm phần trăm?",
            "điểm phần trăm",
            UnitDimension.PERCENTAGE_POINT,
            1.0,
        ),
        ("Hệ số là bao nhiêu lần?", "lần", UnitDimension.RATIO, 1.0),
    ],
)
def test_unit_parser_uses_longest_semantic_match(text, canonical_name, dimension, scale):
    unit = detect_unit(text)
    assert unit is not None
    assert unit.name == canonical_name
    assert unit.dimension == dimension
    assert unit.scale == scale


def test_tram_ty_is_never_parsed_as_ty():
    unit = detect_unit("58 câu hỏi yêu cầu đơn vị trăm tỷ đồng")
    assert unit is not None
    assert unit.name == "trăm tỷ đồng"
    assert unit.scale == 100 * detect_unit("tỷ đồng").scale
    assert detect_target_unit("Giá trị là bao nhiêu trăm tỷ đồng?") == "trăm tỷ đồng"


@pytest.mark.parametrize(
    ("value", "source", "target", "expected"),
    [
        (1.0, "nghìn tỷ đồng", "tỷ đồng", 1_000.0),
        (2.5, "trăm tỷ đồng", "tỷ đồng", 250.0),
        (3.0, "triệu đồng", "nghìn đồng", 3_000.0),
        (4_500.0, "nghìn đồng", "triệu đồng", 4.5),
        (2.0, "triệu USD", "nghìn USD", 2_000.0),
        (15.0, "%", "lần", 0.15),
        (0.15, "lần", "%", 15.0),
        (-2.5, "triệu đồng", "nghìn đồng", -2_500.0),
    ],
)
def test_unit_conversion_is_explicit_and_sign_preserving(value, source, target, expected):
    assert convert_value(value, source, target) == pytest.approx(expected)


def test_unit_conversion_rejects_incompatible_dimensions():
    with pytest.raises(UnitConversionError):
        conversion_factor("triệu USD", "triệu đồng")
    with pytest.raises(UnitConversionError):
        convert_value(5, "điểm phần trăm", "%")
    with pytest.raises(UnitConversionError):
        convert_value(5, "lần", "tỷ đồng")


@pytest.mark.parametrize(
    ("metric", "values", "expected"),
    [
        ("current_ratio", {"current_assets": 200, "current_liabilities": 100}, 2.0),
        (
            "quick_ratio",
            {"current_assets": 200, "inventory": 50, "current_liabilities": 100},
            1.5,
        ),
        ("debt_to_equity", {"total_liabilities": 120, "equity": 80}, 1.5),
        ("debt_to_assets", {"total_liabilities": 120, "total_assets": 300}, 0.4),
        ("gross_margin", {"gross_profit": 40, "net_revenue": 200}, 20.0),
        ("net_margin", {"net_profit": 20, "net_revenue": 200}, 10.0),
        ("operating_margin", {"operating_profit": 30, "net_revenue": 200}, 15.0),
        (
            "roe",
            {"net_profit": 20, "equity": 200, "equity_previous": 100},
            20 / 150 * 100,
        ),
        (
            "roa",
            {"net_profit": 20, "total_assets": 400, "total_assets_previous": 200},
            20 / 300 * 100,
        ),
        (
            "interest_coverage",
            {"profit_before_tax": 30, "interest_expense": 10},
            4.0,
        ),
        ("cfo_to_revenue", {"cfo": 50, "net_revenue": 200}, 25.0),
        ("cfo_to_net_income", {"cfo": 50, "net_profit": 20}, 2.5),
        (
            "cfo_to_current_liabilities",
            {"cfo": 50, "current_liabilities": 100},
            0.5,
        ),
        (
            "working_capital",
            {"current_assets": 200, "current_liabilities": 100},
            100.0,
        ),
        (
            "inventory_days",
            {"inventory": 40, "inventory_previous": 20, "cogs": 120},
            91.25,
        ),
        (
            "fixed_asset_turnover",
            {"net_revenue": 300, "fixed_assets": 100, "fixed_assets_previous": 200},
            2.0,
        ),
        (
            "sga_intensity",
            {"selling_expense": -10, "admin_expense": -20, "net_revenue": 200},
            -15.0,
        ),
        (
            "revenue_growth",
            {"net_revenue": 150, "net_revenue_previous": 100},
            50.0,
        ),
        ("cagr", {"beginning_value": 100, "ending_value": 121, "periods": 2}, 10.0),
        (
            "percentage_change",
            {"current_value": 80, "previous_value": 100},
            -20.0,
        ),
        (
            "percentage_point_change",
            {"current_percentage": 15, "previous_percentage": 10},
            5.0,
        ),
        (
            "margin_spread",
            {"gross_profit": 40, "net_profit": 20, "net_revenue": 200},
            10.0,
        ),
        (
            "inventory_share_change",
            {
                "inventory": 30,
                "total_assets": 150,
                "inventory_previous": 20,
                "total_assets_previous": 200,
            },
            10.0,
        ),
        (
            "operating_leverage",
            {
                "operating_profit": 24,
                "operating_profit_previous": 20,
                "net_revenue": 110,
                "net_revenue_previous": 100,
            },
            2.0,
        ),
    ],
)
def test_metric_registry_formulas(metric, values, expected):
    assert DEFAULT_REGISTRY.evaluate(metric, values) == pytest.approx(expected)


def test_metric_registry_financial_formulas_preserve_negative_signs():
    assert DEFAULT_REGISTRY.evaluate(
        "gross_margin", {"gross_profit": -25, "net_revenue": 100}
    ) == pytest.approx(-25.0)
    assert DEFAULT_REGISTRY.evaluate(
        "working_capital", {"current_assets": 80, "current_liabilities": 100}
    ) == pytest.approx(-20.0)


def test_metric_registry_zero_denominator_fails_instead_of_guessing():
    with pytest.raises(FormulaError):
        DEFAULT_REGISTRY.evaluate(
            "quick_ratio",
            {"current_assets": 100, "inventory": 20, "current_liabilities": 0},
        )


def test_registry_selection_filtering_and_aggregation_are_deterministic():
    values = {"A": 1.0, "B": 7.0, "C": 3.0, "D": 5.0}
    assert DEFAULT_REGISTRY.median_filter(values, above=True) == ["B", "D"]
    assert DEFAULT_REGISTRY.median_filter(values, above=False) == ["A", "C"]
    assert DEFAULT_REGISTRY.select(values, "max") == "B"
    assert DEFAULT_REGISTRY.select(values, "min") == "A"
    assert DEFAULT_REGISTRY.aggregate([1, 2, 3], "count") == 3
    assert DEFAULT_REGISTRY.aggregate([1, 2, 3], "sum") == 6
    assert DEFAULT_REGISTRY.aggregate([1, 2, 3], "average") == 2


def test_registry_build_expression_matches_formula_result():
    expression = DEFAULT_REGISTRY.build_expression(
        "quick_ratio",
        {"current_assets": "200", "inventory": "50", "current_liabilities": "100"},
    )
    assert math.isclose(float(eval(expression, {"__builtins__": {}})), 1.5)
