"""Real-question planner and metric-aware retrieval regressions.

Question ids are used only to locate scenario wording in ``questions.jsonl``.
No expected answer or test-set answer lookup appears in this module.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = str(_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from question_planner import (  # noqa: E402
    QuestionPlan,
    QuestionPlanner,
    QuestionType,
    Scope,
)
from retriever import TableRetriever  # noqa: E402


REGRESSION_IDS = (362, 363, 364, 367, 368, 370, 376, 377, 384, 390, 414, 433, 441, 442, 455)


@pytest.fixture(scope="module")
def questions() -> dict[int, str]:
    wanted = set(REGRESSION_IDS) | {1}
    result: dict[int, str] = {}
    with (_ROOT / "data" / "raw_vifinqa" / "questions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if int(item["id"]) in wanted:
                result[int(item["id"])] = str(item["question"])
    assert set(result) == wanted
    return result


# Each expectation describes semantics only: entities, periods, formulas and
# operations.  It deliberately contains no answer values.
PLAN_EXPECTATIONS = {
    362: {
        "tickers": ("CEO", "HPX", "KBC", "SNZ", "VIC", "VPI", "VRE"),
        "years": ("2022",),
        "target": "current_liabilities",
        "required": {"inventory", "current_liabilities"},
        "filters": {("inventory_to_current_liabilities", ">median")},
        "selection": None,
        "aggregation": "share",
    },
    363: {
        "tickers": ("KBC",),
        "years": ("2016", "2017", "2018", "2019", "2020"),
        "target": "interest_coverage",
        "required": {"profit_before_tax", "interest_expense", "total_liabilities", "equity"},
        "filters": set(),
        "selection": ("debt_to_equity", "max"),
        "aggregation": None,
    },
    364: {
        "tickers": ("GVR", "DPM", "DCM", "PRT"),
        "years": ("2020", "2021"),
        "target": "accrual_ratio",
        "required": {"net_profit", "cfo", "total_assets", "net_revenue"},
        "filters": {("cfo", ">0")},
        "selection": ("revenue_growth", "max_change"),
        "aggregation": None,
    },
    367: {
        "tickers": ("MSN", "MCH", "DBC", "ASM", "OGC"),
        "years": ("2024", "2025"),
        "target": "margin_spread",
        "required": {"gross_profit", "net_profit", "net_revenue", "cfo"},
        "filters": {("cfo", ">0"), ("revenue_growth", "<")},
        "selection": None,
        "aggregation": "average",
    },
    368: {
        "tickers": ("HPG", "HSG", "MSR", "NKG"),
        "years": ("2022",),
        "target": "net_margin",
        "required": {"net_profit", "net_revenue", "current_assets", "inventory", "current_liabilities"},
        "filters": {("quick_ratio", "<median")},
        "selection": None,
        "aggregation": "average",
    },
    370: {
        "tickers": ("GEE", "GEX", "SAM"),
        "years": ("2022", "2023", "2024"),
        "target": "net_margin",
        "required": {"net_profit", "net_revenue", "cfo"},
        "filters": {("cfo", ">0")},
        "selection": ("cagr", "max"),
        "aggregation": None,
    },
    376: {
        "tickers": ("HPX", "KBC", "NVL", "SCR", "VIC", "VPI", "VRE"),
        "years": ("2024",),
        "target": "inventory_to_assets",
        "required": {"inventory", "total_assets", "current_assets", "current_liabilities"},
        "filters": {("current_ratio", ">")},
        "selection": ("quick_ratio", "min"),
        "aggregation": None,
        "question_type": QuestionType.FILTER_THEN_SELECT,
    },
    377: {
        "tickers": ("ASM", "DBC", "MPC", "MSN", "OGC", "QNS"),
        "years": ("2024", "2023"),
        "target": "cfo_to_revenue",
        "required": {"cfo", "net_revenue", "gross_profit"},
        "filters": {("revenue_growth", ">0")},
        "selection": ("gross_margin_change", "min_change"),
        "aggregation": None,
    },
    384: {
        "tickers": ("HPG", "HSG", "NKG"),
        "years": ("2024",),
        "target": "gross_margin",
        "required": {"gross_profit", "net_revenue", "inventory", "total_assets"},
        "filters": set(),
        "selection": ("inventory_share_change", "max_change"),
        "aggregation": None,
    },
    390: {
        "tickers": ("HPG", "HSG", "NKG"),
        "years": ("2024",),
        "target": "inventory",
        "required": {"inventory", "current_assets", "current_liabilities"},
        "filters": set(),
        "selection": ("quick_ratio", "min"),
        "aggregation": None,
    },
    414: {
        "tickers": ("HPG", "HSG", "NKG"),
        "years": ("2024",),
        "target": "operating_margin",
        "required": {"operating_profit", "net_revenue"},
        "filters": {("revenue_growth", ">"), ("operating_profit", ">0")},
        "selection": ("operating_leverage", "max"),
        "aggregation": None,
        "question_type": QuestionType.FILTER_THEN_SELECT,
    },
    433: {
        "tickers": ("CRE", "HPX", "KBC", "KHG", "NVL", "SNZ", "SSH", "VIC", "VPI", "VRE"),
        "years": ("2023",),
        "target": "total_liabilities",
        "required": {
            "total_liabilities",
            "total_assets",
            "short_term_receivables",
            "long_term_receivables",
            "inventory",
        },
        "filters": {("debt_to_assets", ">median"), ("stressed_net_assets", "<0")},
        "selection": None,
        "aggregation": "share",
        "question_type": QuestionType.SCENARIO,
    },
    441: {
        "tickers": ("HPG", "HSG", "MSR", "NKG"),
        "years": ("2024", "2025"),
        "target": "gross_margin",
        "required": {"gross_profit", "net_revenue", "total_liabilities", "equity"},
        "filters": {("debt_to_equity", "<median")},
        "selection": ("revenue_growth", "max_change"),
        "aggregation": None,
    },
    442: {
        "tickers": ("CEO", "DIG", "HPX", "KBC", "NVL", "SCR", "VIC", "VPI", "VRE"),
        "years": ("2024", "2025"),
        "target": "gross_margin",
        "required": {"gross_profit", "net_revenue", "total_liabilities", "equity"},
        "filters": {("debt_to_equity", "<median")},
        "selection": ("revenue_growth", "max_change"),
        "aggregation": None,
    },
    455: {
        "tickers": ("HPG", "HSG", "MSR", "NKG"),
        "years": ("2021", "2022", "2024"),
        "target": "gross_margin",
        "required": {"gross_profit", "net_revenue", "inventory", "cogs"},
        "filters": {("inventory_days", ">median")},
        "selection": ("inventory_days_change", "min_change"),
        "aggregation": None,
    },
}


@pytest.mark.parametrize("question_id", REGRESSION_IDS)
def test_real_analytical_scenario_is_planned_semantically(questions, question_id):
    expected = PLAN_EXPECTATIONS[question_id]
    plan = QuestionPlanner().analyze(questions[question_id])

    assert plan.is_complex
    assert plan.question_type == expected.get("question_type", QuestionType.MULTI_STAGE_ANALYTICAL)
    assert tuple(plan.tickers) == expected["tickers"]
    assert tuple(plan.years) == expected["years"]
    assert plan.target_metric == expected["target"]
    assert expected["required"].issubset(plan.required_metrics)
    assert {(item.metric, item.operator) for item in plan.filters} == expected["filters"]
    if expected["selection"] is None:
        assert plan.selection_operation is None
    else:
        assert plan.selection_operation is not None
        assert (plan.selection_operation.metric, plan.selection_operation.operation) == expected["selection"]
    assert plan.aggregation == expected["aggregation"]


@pytest.mark.parametrize(
    ("question_id", "metric", "expected_years"),
    [
        (364, "cfo", {"2020", "2021"}),
        (364, "total_assets", {"2020", "2021"}),
        (367, "cfo", {"2024", "2025"}),
        (367, "net_revenue", {"2024", "2025"}),
        (370, "cfo", {"2022", "2023", "2024"}),
        (370, "net_revenue", {"2022", "2024"}),
        (377, "gross_profit", {"2023", "2024"}),
        (384, "inventory", {"2023", "2024"}),
        (384, "total_assets", {"2023", "2024"}),
        (414, "operating_profit", {"2023", "2024"}),
        (414, "net_revenue", {"2023", "2024"}),
        (441, "total_liabilities", {"2024"}),
        (441, "gross_profit", {"2025"}),
        (441, "net_revenue", {"2024", "2025"}),
        (442, "equity", {"2024"}),
        (455, "inventory", {"2021", "2022", "2023", "2024"}),
        (455, "cogs", {"2022", "2024"}),
        (455, "gross_profit", {"2024"}),
    ],
)
def test_planner_maps_each_metric_to_the_periods_it_needs(questions, question_id, metric, expected_years):
    plan = QuestionPlanner().analyze(questions[question_id])
    assert set(plan.metric_years[metric]) == expected_years


def test_simple_lookup_regression_stays_on_simple_path(questions):
    plan = QuestionPlanner().analyze(questions[1])
    assert plan.question_type == QuestionType.SIMPLE_LOOKUP
    assert plan.tickers == ["VJC"]
    assert plan.years == ["2018"]
    assert plan.scope == Scope.SEPARATE


@pytest.mark.parametrize("spelling", ["Hoà Phát", "Hòa Phát"])
def test_company_aliases_resolve_both_hoa_spellings(tmp_path, spelling):
    retriever = TableRetriever(
        csv_dir=str(tmp_path),
        manifest_path=str(tmp_path / "missing_manifest.jsonl"),
        line_map_path=str(tmp_path / "missing_line_map.json"),
    )
    _, _, tickers, years = retriever.extract_all_entities(
        f"Trong nhóm {spelling}, Hoa Sen và Nam Kim năm 2024"
    )
    assert tickers == ["HPG", "HSG", "NKG"]
    assert years == ["2024"]


def _write_statement(path: Path, label: str, value: float = 100.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"Chi_tieu": [label], "Gia_tri": [value], "Don_vi": ["triệu đồng"]}
    ).to_csv(path, index=False)


def test_metric_aware_retrieval_has_no_global_cap_and_keeps_stable_mapping(tmp_path):
    # Nine companies x three statement families is intentionally much larger
    # than the historical global max_k=10.
    tickers = [f"C{index:02d}" for index in range(9)]
    for index, ticker in enumerate(tickers):
        folder = tmp_path / ticker
        _write_statement(
            folder / f"{ticker}_2024_01BangCanDoiKeToan_consolidated.csv",
            "Tổng tài sản",
            100 + index,
        )
        _write_statement(
            folder / f"{ticker}_2024_02BaoCaoKetQuaKinhDoanh_consolidated.csv",
            "Doanh thu thuần",
            200 + index,
        )
        _write_statement(
            folder / f"{ticker}_2024_03BaoCaoLuuChuyenTienTe_consolidated.csv",
            "Lưu chuyển tiền thuần từ hoạt động kinh doanh",
            30 + index,
        )

    metrics = ["total_assets", "net_revenue", "cfo"]
    plan = QuestionPlan(
        question="So sánh tài sản, doanh thu và CFO của chín doanh nghiệp năm 2024",
        question_type=QuestionType.MULTI_STAGE_ANALYTICAL,
        tickers=tickers,
        years=["2024"],
        scope=Scope.CONSOLIDATED,
        target_metric="cfo",
        required_metrics=metrics,
        metric_years={metric: ["2024"] for metric in metrics},
    )
    retriever = TableRetriever(
        csv_dir=str(tmp_path),
        manifest_path=str(tmp_path / "missing_manifest.jsonl"),
        line_map_path=str(tmp_path / "missing_line_map.json"),
    )
    bundle = retriever.retrieve_plan(plan)

    assert bundle.complete, bundle.missing_requirements
    assert len(bundle.metric_paths) == 9 * 3
    assert len(bundle.paths) == 9 * 3
    assert len(bundle.paths) > 10
    assert set(bundle.structured) == set(tickers)
    for ticker in tickers:
        statements = bundle.structured[ticker]["2024"]
        assert set(statements) == {"balance_sheet", "income_statement", "cashflow"}
        for metric in metrics:
            key = bundle.requirement_key(ticker, "2024", metric)
            assert len(bundle.metric_paths[key]) == 1

    variables = list(bundle.path_to_variable.values())
    assert variables == [f"df{index}" for index in range(1, 28)]
    assert len(set(variables)) == len(bundle.paths)
    assert {
        variable: path for path, variable in bundle.path_to_variable.items()
    } == bundle.variable_to_path
