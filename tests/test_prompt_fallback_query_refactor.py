"""P0 regressions for prompt integrity, fallback safety and query truthfulness."""

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

from agent import PandasAgent, PromptBudgetError  # noqa: E402
from fallback import try_rule_based_answer  # noqa: E402
from query_formatter import (  # noqa: E402
    QueryExecutionError,
    QueryFormatError,
    _safe_df_fallback,
    convert_script_to_expression,
    execute_expression,
)
from question_planner import QuestionPlanner  # noqa: E402
from semantic_validation import validate_answer  # noqa: E402


COMPLEX_IDS = (362, 363, 364, 367, 368, 370, 376, 377, 384, 390, 414, 433, 441, 442, 455)


def _questions(ids) -> dict[int, str]:
    wanted = set(ids)
    result: dict[int, str] = {}
    with (_ROOT / "data" / "raw_vifinqa" / "questions.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            item = json.loads(line)
            if int(item["id"]) in wanted:
                result[int(item["id"])] = str(item["question"])
    assert set(result) == wanted
    return result


def test_largest_complex_question_is_never_truncated_from_prompt():
    questions = _questions(COMPLEX_IDS)
    question = max(questions.values(), key=len) + " [QUESTION_END_47d931]"
    semantic_context = (
        "EVIDENCE_BEGIN\n"
        + "HPG | 2024 | total_assets | 123 | triệu đồng | df1 | 0 | source.csv\n" * 10_000
        + "EVIDENCE_END_MUST_BE_TRUNCATED"
    )
    # Ollama construction is intentionally used so no local transformer model
    # is loaded and no network inference takes place in this unit test.
    agent = PandasAgent(backend="ollama", prompt_token_budget=4096)
    messages = agent._build_messages(question, [], semantic_context=semantic_context)

    user_prompt = messages[1]["content"]
    assert question in user_prompt
    assert "[QUESTION_END_47d931]" in user_prompt
    assert agent._ONE_EXAMPLE in user_prompt
    assert "EVIDENCE_BEGIN" in user_prompt
    assert "EVIDENCE_END_MUST_BE_TRUNCATED" not in user_prompt
    assert "[EVIDENCE_TRUNCATED]" in user_prompt
    assert messages[0]["content"] == agent._SYSTEM_PROMPT

    report = agent.last_prompt_report
    assert report is not None
    assert report.question_preserved
    assert report.evidence_truncated
    assert report.semantic_context
    assert report.raw_csv_tables == 0
    assert report.estimated_tokens <= report.token_budget


def test_prompt_guard_raises_instead_of_truncating_an_oversized_question():
    agent = PandasAgent(backend="ollama", prompt_token_budget=100)
    question = "CÂU HỎI KHÔNG ĐƯỢC CẮT " + "rất dài " * 5_000 + "QUESTION_END"
    with pytest.raises(PromptBudgetError):
        agent._build_messages(question, [], semantic_context="one compact fact")


@pytest.mark.parametrize("question_id", COMPLEX_IDS)
def test_complex_question_can_never_use_single_row_fallback(tmp_path, question_id):
    question = _questions(COMPLEX_IDS)[question_id]
    tempting_row = tmp_path / "HPG_2024_01BangCanDoiKeToan_consolidated.csv"
    pd.DataFrame(
        {
            "Chi_tieu": [
                "Tổng tài sản",
                "Hàng tồn kho",
                "Nợ ngắn hạn",
                "Doanh thu thuần",
                "Lợi nhuận gộp",
            ],
            "Gia_tri": [999_999_999_999, 100, 50, 200, 40],
            "Don_vi": ["triệu đồng"] * 5,
        }
    ).to_csv(tempting_row, index=False)
    plan = QuestionPlanner().analyze(question)
    assert plan.is_complex
    assert try_rule_based_answer(question, [str(tempting_row)], plan=plan) is None


def test_simple_fallback_does_not_expand_to_an_unauthorized_sibling_csv(tmp_path):
    authorized = tmp_path / "HPG_2024_99Other_consolidated.csv"
    unauthorized = tmp_path / "HPG_2024_01BangCanDoiKeToan_consolidated.csv"
    pd.DataFrame(
        {"Chi_tieu": ["Chi phí khác"], "Gia_tri": [1], "Don_vi": ["triệu đồng"]}
    ).to_csv(authorized, index=False)
    pd.DataFrame(
        {"Chi_tieu": ["Tổng tài sản"], "Gia_tri": [999], "Don_vi": ["triệu đồng"]}
    ).to_csv(unauthorized, index=False)

    result = try_rule_based_answer(
        "Tổng tài sản của HPG năm 2024 là bao nhiêu triệu đồng?",
        [str(authorized)],
    )
    assert result is None


def test_query_formatter_ignores_expected_answer_and_preserves_generated_semantics():
    frame = pd.DataFrame({"Chi_tieu": ["A", "B"], "Gia_tri": [10.0, 999.0]})
    code = "value = float(df1.iloc[0]['Gia_tri'])\nanswer = value\nprint(answer)"

    expression = convert_script_to_expression(code, {"df1": frame}, expected_ans=999.0)

    assert execute_expression(expression, {"df1": frame}) == 10
    assert "iloc[0]" in expression
    assert "iloc[1]" not in expression


def test_query_formatter_has_no_answer_fitting_dataframe_fallback():
    frame = pd.DataFrame({"Chi_tieu": ["A"], "Gia_tri": [123.0]})
    with pytest.raises(QueryFormatError):
        convert_script_to_expression("this is not valid Python @@@", {"df1": frame}, expected_ans=123.0)
    with pytest.raises(QueryFormatError):
        _safe_df_fallback({"df1": frame}, expected_ans=123.0)


def test_query_execution_requires_the_exact_final_evidence_mapping():
    frame = pd.DataFrame({"Chi_tieu": ["A"], "Gia_tri": [10.0]})
    with pytest.raises(QueryExecutionError, match="missing evidence"):
        execute_expression("float(df7.iloc[0]['Gia_tri'])", {"df1": frame})
    assert execute_expression("float(df7.iloc[0]['Gia_tri'])", {"df7": frame}) == 10


def test_subset_share_may_legitimately_be_derived_from_currency_facts():
    frame = pd.DataFrame({"Chi_tieu": ["Nợ ngắn hạn", "Nợ ngắn hạn"], "Gia_tri": [60.0, 40.0]})
    plan = {
        "is_complex": False,
        "question": "Tỷ trọng nợ ngắn hạn chiếm bao nhiêu phần trăm?",
        "aggregation": "share",
        "target_metric": "current_liabilities",
        "target_unit": "%",
    }
    query = "float(df1.iloc[0]['Gia_tri']) / (float(df1.iloc[0]['Gia_tri']) + float(df1.iloc[1]['Gia_tri'])) * 100"
    facts = [
        {
            "ticker": "AAA",
            "year": "2024",
            "metric": "current_liabilities",
            "value": 60.0,
            "unit": "VND",
            "variable": "df1",
            "row_index": 0,
        }
    ]
    report = validate_answer(
        60.0,
        query,
        {"df1": frame},
        plan=plan,
        facts=facts,
        dataframes={"df1": frame},
    )
    assert report.valid, report.error_messages
    assert not any(issue.code == "ratio_uses_currency_fact" for issue in report.errors)
