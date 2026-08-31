from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from submission_contract import (  # noqa: E402
    account_ids,
    package_submission_atomic,
    validate_items,
    validate_submission_zip,
)
from units import detect_target_unit  # noqa: E402


def _item(question_id=1, question="Q1?"):
    return {
        "id": question_id, "question": question, "answer": 1.0,
        "relevant_docs": ["doc"], "relevant_tables": ["doc|1"],
        "evidence": [{"variable": "df1", "csv_path": "data/a.csv"}],
        "pandas_query": "float(df1.iloc[0]['value'])",
    }


def test_schema_requires_exact_seven_fields_finite_answer_and_question_mapping():
    assert validate_items([_item()], {1: "Q1?"}) == []
    invalid = _item()
    invalid["diagnostics"] = {}
    invalid["answer"] = float("nan")
    errors = validate_items([invalid], {1: "Different?"})
    assert any("fields must equal BTC_FIELDS" in error for error in errors)
    assert any("finite numeric" in error for error in errors)
    assert any("question does not match" in error for error in errors)


def test_accounting_detects_duplicate_missing_extra_and_overlap_ids():
    report = account_ids({1, 2, 3}, [_item(1), _item(1), _item(4)], [{"id": 1}, {"id": 2}])
    assert report.duplicate_saved_ids == {1}
    assert report.missing_ids == {3}
    assert report.extra_ids == {4}
    assert report.overlap_ids == {1}
    assert not report.run_accounted
    assert not report.submission_ready


def test_submission_ready_requires_every_expected_id_saved_without_failures():
    complete = account_ids({1, 2}, [_item(1), _item(2)], [])
    partial = account_ids({1, 2}, [_item(1)], [{"id": 2}])
    assert complete.run_accounted and complete.submission_ready
    assert partial.run_accounted and not partial.submission_ready
    assert partial.to_dict()["selected_equals_saved_plus_failed"] is True


def test_atomic_package_has_exact_csv_closure_and_canonical_submission_name(tmp_path):
    submission = tmp_path / "checkpoint.json"
    submission.write_text(json.dumps([_item()]), encoding="utf-8")
    csv = tmp_path / "a.csv"
    csv.write_text("value\n1\n", encoding="utf-8")
    destination = tmp_path / "submission.zip"
    package_submission_atomic(str(destination), str(submission), [str(csv)])
    with zipfile.ZipFile(destination) as archive:
        assert archive.namelist() == ["submission.json", "data/a.csv"]


def test_zip_replay_validates_exact_closure_query_and_provenance(tmp_path):
    submission = tmp_path / "submission.json"
    submission.write_text(json.dumps([_item()]), encoding="utf-8")
    csv = tmp_path / "a.csv"
    csv.write_text("value\n1\n", encoding="utf-8")
    destination = tmp_path / "submission.zip"
    package_submission_atomic(str(destination), str(submission), [str(csv)])
    report = validate_submission_zip(destination)
    assert report.valid
    assert report.replayed_count == 1
    assert report.errors == ()


def test_zip_replay_rejects_extra_csv_unused_variable_and_answer_mismatch(tmp_path):
    item = _item()
    item["answer"] = 2.0
    item["evidence"].append({"variable": "df2", "csv_path": "data/b.csv"})
    destination = tmp_path / "submission.zip"
    with zipfile.ZipFile(destination, "w") as archive:
        archive.writestr("submission.json", json.dumps([item]))
        archive.writestr("data/a.csv", "value\n1\n")
        archive.writestr("data/b.csv", "value\n2\n")
        archive.writestr("data/extra.csv", "value\n3\n")
    report = validate_submission_zip(destination)
    assert not report.valid
    assert any("Undeclared CSV files" in error for error in report.errors)
    assert any("evidence variables unused" in error for error in report.errors)


def test_zip_replay_rejects_missing_evidence_and_unexpected_entry(tmp_path):
    destination = tmp_path / "submission.zip"
    with zipfile.ZipFile(destination, "w") as archive:
        archive.writestr("submission.json", json.dumps([_item()]))
        archive.writestr("notes.txt", "not allowed")
    report = validate_submission_zip(destination)
    assert not report.valid
    assert any("Unexpected archive entries" in error for error in report.errors)
    assert any("Missing declared evidence files" in error for error in report.errors)


def test_atomic_package_rejects_basename_collision_without_overwriting_destination(tmp_path):
    submission = tmp_path / "submission.json"
    submission.write_text("[]", encoding="utf-8")
    destination = tmp_path / "submission.zip"
    destination.write_bytes(b"existing")
    first, second = tmp_path / "x" / "a.csv", tmp_path / "y" / "a.csv"
    first.parent.mkdir(); second.parent.mkdir()
    first.write_text("x", encoding="utf-8"); second.write_text("y", encoding="utf-8")
    with pytest.raises(ValueError, match="basename collision"):
        package_submission_atomic(str(destination), str(submission), [str(first), str(second)])
    assert destination.read_bytes() == b"existing"


@pytest.mark.parametrize("question", [
    "Tổng tài sản của Công ty Cổ phần ABC năm 2024 là bao nhiêu?",
    "Công ty cổ phần XYZ có doanh thu năm 2023 bằng bao nhiêu?",
])
def test_company_legal_name_does_not_imply_share_unit(question):
    assert detect_target_unit(question) == ""


def test_explicit_share_answer_clause_is_still_detected():
    assert detect_target_unit("Công ty Cổ phần ABC phát hành bao nhiêu cổ phần?") == "cổ phần"
