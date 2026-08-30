"""Integration tests for deterministic complex solving and the strict save gate.

All financial values and CSVs in this module are synthetic.  The tests exercise
general planner/solver contracts and contain no lookup keyed by a dataset ID.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd
import pytest


_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = str(_ROOT / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from complex_solver import ComplexSolver, SemanticExtractor, StructuredSolveFailure  # noqa: E402
from pipeline import (  # noqa: E402
    _bundle_from_paths,
    _finalize_item,
    _prune_bundle,
)
from query_formatter import execute_expression, referenced_variables  # noqa: E402
from question_planner import (  # noqa: E402
    FilterSpec,
    QuestionPlan,
    QuestionType,
    Scope,
)
from retriever import EvidenceBundle, TableRetriever  # noqa: E402


_BASE_METRICS = (
    "current_assets",
    "inventory",
    "current_liabilities",
    "net_profit",
    "net_revenue",
)


def _normal_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def _write_company_statement(
    root: Path,
    ticker: str,
    *,
    current_assets: float,
    inventory: float,
    current_liabilities: float,
    net_profit: float,
    net_revenue: float,
    omit_metric: str | None = None,
) -> str:
    rows = {
        "current_assets": ("Tài sản ngắn hạn", current_assets, "VND"),
        "inventory": ("Hàng tồn kho", inventory, "VND"),
        "current_liabilities": ("Nợ ngắn hạn", current_liabilities, "VND"),
        "net_profit": ("Lợi nhuận sau thuế", net_profit, "VND"),
        "net_revenue": ("Doanh thu thuần", net_revenue, "VND"),
    }
    records = [
        {"Chi_tieu": label, "Gia_tri": value, "Don_vi": unit}
        for metric, (label, value, unit) in rows.items()
        if metric != omit_metric
    ]
    # A tempting derived row must not substitute for a missing base metric.
    records.append({"Chi_tieu": "Biên lợi nhuận ròng", "Gia_tri": 999, "Don_vi": "%"})
    path = root / f"{ticker}_2024_SyntheticStatements_consolidated.csv"
    pd.DataFrame(records).to_csv(path, index=False, encoding="utf-8")
    return _normal_path(path)


def _analytical_plan(tickers: list[str]) -> QuestionPlan:
    return QuestionPlan(
        question=(
            "Năm 2024, trong nhóm " + ", ".join(tickers)
            + ", các công ty có hệ số thanh toán nhanh thấp hơn trung vị "
            "có biên lợi nhuận ròng bình quân là bao nhiêu phần trăm?"
        ),
        question_type=QuestionType.MULTI_STAGE_ANALYTICAL,
        tickers=list(tickers),
        years=["2024"],
        scope=Scope.CONSOLIDATED,
        target_metric="net_margin",
        required_metrics=list(_BASE_METRICS),
        filters=[FilterSpec("quick_ratio", "<median", "median", ("2024",))],
        grouping="company",
        aggregation="average",
        comparison="<median",
        formula="net_margin",
        target_unit="%",
        mentioned_metrics=["quick_ratio", "net_margin"],
        metric_years={metric: ["2024"] for metric in _BASE_METRICS},
        operations=["filter", "average", "compute:net_margin"],
        complexity_reasons=["multiple_companies", "formula_metrics", "filtering", "aggregation"],
    )


def _simple_plan(*, question: str = "Giá trị chỉ tiêu của AAA năm 2024 là bao nhiêu?") -> QuestionPlan:
    return QuestionPlan(
        question=question,
        question_type=QuestionType.SIMPLE_LOOKUP,
        tickers=["AAA"],
        years=["2024"],
        scope=Scope.CONSOLIDATED,
        target_metric="net_profit",
        required_metrics=["net_profit"],
        target_unit="",
        mentioned_metrics=["net_profit"],
        metric_years={"net_profit": ["2024"]},
    )


def _analytical_bundle(root: Path, *, missing: tuple[str, str] | None = None) -> tuple[QuestionPlan, EvidenceBundle]:
    values = {
        # quick ratios: 2.0, 1.0, 0.6, 0.2; median=0.8.
        # The filtered companies therefore have net margins 30% and 40%.
        "AAA": (100.0, 20.0, 40.0, 10.0, 100.0),
        "BBB": (120.0, 60.0, 60.0, 20.0, 100.0),
        "CCC": (80.0, 50.0, 50.0, 30.0, 100.0),
        "DDD": (100.0, 90.0, 50.0, 40.0, 100.0),
    }
    plan = _analytical_plan(list(values))
    bundle = EvidenceBundle()
    for ticker, numbers in values.items():
        omit = missing[1] if missing and missing[0] == ticker else None
        path = _write_company_statement(
            root,
            ticker,
            current_assets=numbers[0],
            inventory=numbers[1],
            current_liabilities=numbers[2],
            net_profit=numbers[3],
            net_revenue=numbers[4],
            omit_metric=omit,
        )
        for metric in _BASE_METRICS:
            statement = "income_statement" if metric in {"net_profit", "net_revenue"} else "balance_sheet"
            bundle.add(ticker, "2024", metric, statement, path)
    return plan, bundle


def _retriever_for(root: Path) -> TableRetriever:
    return TableRetriever(
        csv_dir=str(root),
        manifest_path=str(root / "missing_manifest.jsonl"),
        line_map_path=str(root / "missing_line_map.json"),
    )


def _frames_for_query(query: str, bundle: EvidenceBundle) -> dict[str, pd.DataFrame]:
    return {
        variable: pd.read_csv(bundle.variable_to_path[variable])
        for variable in referenced_variables(query)
    }


def test_multi_company_metric_aware_solver_and_strict_save_contract(tmp_path):
    plan, bundle = _analytical_bundle(tmp_path)
    result = ComplexSolver().solve(plan, bundle)

    assert result.filtered_candidates == ["CCC", "DDD"]
    assert math.isclose(float(result.answer), (30.0 + 40.0) / 2.0, abs_tol=1e-12)
    assert result.validation["single_row_fallback_used"] is False
    assert set(result.validation["covered_entities"]) == set(plan.tickers)

    observed = {(fact.ticker, fact.year, fact.metric) for fact in result.used_facts}
    expected = {
        (ticker, "2024", metric)
        for ticker in plan.tickers
        for metric in _BASE_METRICS
    }
    assert expected.issubset(observed)
    assert set(referenced_variables(result.pandas_query)) == set(bundle.variable_to_path)

    # The expression is independently executable before it reaches pipeline.
    direct_result = execute_expression(result.pandas_query, _frames_for_query(result.pandas_query, bundle))
    assert math.isclose(float(direct_result), float(result.answer), rel_tol=1e-12, abs_tol=1e-9)

    item, saved_paths = _finalize_item(
        question_id=900001,
        question=plan.question,
        plan=plan,
        bundle=bundle,
        query_or_script=result.pandas_query,
        retriever=_retriever_for(tmp_path),
        facts=result.used_facts,
        solver_validation=result.validation,
    )

    final_mapping = {
        record["variable"]: bundle.variable_to_path[record["variable"]]
        for record in item["evidence"]
    }
    replayed = execute_expression(
        item["pandas_query"],
        {variable: pd.read_csv(path) for variable, path in final_mapping.items()},
    )
    assert math.isclose(float(replayed), float(item["answer"]), rel_tol=1e-12, abs_tol=1e-9)
    assert item["validation"]["valid"] is True
    assert item["validation"]["query_is_source_of_truth"] is True
    assert item["validation"]["single_row_fallback_used"] is False
    assert set(saved_paths) == set(bundle.paths)


def test_pruning_preserves_original_df7_identity_and_final_query_mapping(tmp_path):
    paths: list[str] = []
    for number in range(1, 8):
        path = tmp_path / f"AAA_2024_Table{number}_consolidated.csv"
        pd.DataFrame(
            [{"Chi_tieu": "Lợi nhuận sau thuế", "Gia_tri": float(number), "Don_vi": "VND"}]
        ).to_csv(path, index=False, encoding="utf-8")
        paths.append(_normal_path(path))
    bundle = _bundle_from_paths(paths)

    pruned = _prune_bundle(bundle, {"df7"})
    assert pruned.paths == [paths[6]]
    assert pruned.path_to_variable == {paths[6]: "df7"}
    assert pruned.variable_to_path == {"df7": paths[6]}
    assert "df1" not in pruned.variable_to_path

    query = "float(df7.iloc[0]['Gia_tri'])"
    item, saved_paths = _finalize_item(
        question_id=900002,
        question=_simple_plan().question,
        plan=_simple_plan(),
        bundle=bundle,
        query_or_script=query,
        retriever=_retriever_for(tmp_path),
        solver_validation={"single_row_fallback_used": False},
    )

    assert item["pandas_query"] == query
    assert item["answer"] == 7
    assert item["evidence"] == [{"variable": "df7", "csv_path": f"data/{Path(paths[6]).name}"}]
    assert saved_paths == [paths[6]]
    replayed = execute_expression(query, {"df7": pd.read_csv(paths[6])})
    assert replayed == item["answer"]


def test_zero_is_a_valid_saved_answer_not_a_fallback_trigger(tmp_path):
    path = tmp_path / "AAA_2024_ZeroMetric_consolidated.csv"
    pd.DataFrame(
        [{"Chi_tieu": "Lợi nhuận sau thuế", "Gia_tri": 0.0, "Don_vi": "VND"}]
    ).to_csv(path, index=False, encoding="utf-8")
    bundle = _bundle_from_paths([_normal_path(path)])
    query = "float(df1.iloc[0]['Gia_tri'])"

    item, _ = _finalize_item(
        question_id=900003,
        question=_simple_plan().question,
        plan=_simple_plan(),
        bundle=bundle,
        query_or_script=query,
        retriever=_retriever_for(tmp_path),
        solver_validation={"single_row_fallback_used": False},
    )

    assert item["answer"] == 0
    assert execute_expression(item["pandas_query"], {"df1": pd.read_csv(path)}) == 0
    assert item["validation"]["valid"] is True
    assert item["validation"]["single_row_fallback_used"] is False


def test_missing_complex_metric_is_structured_failure_without_row_fallback(tmp_path):
    plan, bundle = _analytical_bundle(tmp_path, missing=("DDD", "net_revenue"))

    with pytest.raises(StructuredSolveFailure) as captured:
        ComplexSolver().solve(plan, bundle)

    failure = captured.value
    report = failure.to_dict()
    assert failure.code == "missing_metric_facts"
    assert any(
        item.ticker == "DDD" and item.year == "2024" and item.metric == "net_revenue"
        for item in failure.missing
    )
    assert report["retry_layer"] == "metric_retrieval"
    assert report["single_row_fallback_allowed"] is False
    # The synthetic file contains a highly salient derived margin row.  The
    # solver must still fail rather than treating that one row as net revenue.
    assert any("Biên lợi nhuận ròng" in value for value in pd.read_csv(bundle.paths[-1])["Chi_tieu"])


def test_unknown_scope_uses_row_level_metric_coverage(tmp_path):
    consolidated = tmp_path / "AAA_2024_Balance_consolidated.csv"
    separate = tmp_path / "AAA_2024_Statements_separate.csv"
    pd.DataFrame([
        {"Chi_tieu": "Tài sản ngắn hạn", "Gia_tri": 100, "Don_vi": "VND"},
    ]).to_csv(consolidated, index=False, encoding="utf-8")
    pd.DataFrame([
        {"Chi_tieu": "Tài sản ngắn hạn", "Gia_tri": 90, "Don_vi": "VND"},
        {"Chi_tieu": "Lợi nhuận sau thuế TNDN", "Gia_tri": 9, "Don_vi": "VND"},
    ]).to_csv(separate, index=False, encoding="utf-8")
    bundle = EvidenceBundle()
    for metric in ("current_assets", "net_profit"):
        bundle.add("AAA", "2024", metric, "balance_sheet", _normal_path(consolidated))
        bundle.add("AAA", "2024", metric, "notes", _normal_path(separate))
    plan = QuestionPlan(
        question="So sánh tài sản ngắn hạn và lợi nhuận sau thuế AAA năm 2024.",
        question_type=QuestionType.MULTI_STAGE_ANALYTICAL,
        tickers=["AAA"],
        years=["2024"],
        scope=Scope.UNKNOWN,
        target_metric="net_profit",
        required_metrics=["current_assets", "net_profit"],
        metric_years={"current_assets": ["2024"], "net_profit": ["2024"]},
    )
    table = SemanticExtractor().extract(plan, bundle)
    assert table.get("AAA", "2024", "current_assets").path.endswith("_separate.csv")
    assert table.get("AAA", "2024", "net_profit").path.endswith("_separate.csv")
