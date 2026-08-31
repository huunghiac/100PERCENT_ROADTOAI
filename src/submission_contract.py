"""BTC submission schema, ID accounting, and packaging gates."""
from __future__ import annotations

import json
import math
import os
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

BTC_FIELDS = frozenset({
    "id", "question", "answer", "relevant_docs",
    "relevant_tables", "evidence", "pandas_query",
})
EVIDENCE_FIELDS = frozenset({"variable", "csv_path"})


@dataclass(frozen=True)
class AccountingReport:
    expected_ids: frozenset[int]
    saved_ids: frozenset[int]
    failure_ids: frozenset[int]
    duplicate_saved_ids: frozenset[int]
    missing_ids: frozenset[int]
    extra_ids: frozenset[int]
    overlap_ids: frozenset[int]

    @property
    def run_accounted(self) -> bool:
        return not (self.duplicate_saved_ids or self.missing_ids or self.extra_ids or self.overlap_ids)

    @property
    def submission_ready(self) -> bool:
        return self.run_accounted and self.saved_ids == self.expected_ids and not self.failure_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "expected": len(self.expected_ids), "saved": len(self.saved_ids),
            "failed": len(self.failure_ids), "selected_equals_saved_plus_failed": self.run_accounted,
            "submission_ready": self.submission_ready,
            "duplicate_saved_ids": sorted(self.duplicate_saved_ids),
            "missing_ids": sorted(self.missing_ids), "extra_ids": sorted(self.extra_ids),
            "overlap_ids": sorted(self.overlap_ids),
        }


def load_questions(path: str | os.PathLike[str]) -> dict[int, str]:
    questions: dict[int, str] = {}
    with open(path, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            question_id = int(item["id"])
            if question_id in questions:
                raise ValueError(f"Duplicate question id {question_id} at line {line_number}")
            questions[question_id] = str(item["question"])
    return questions


def validate_items(items: object, questions: Mapping[int, str] | None = None) -> list[str]:
    if not isinstance(items, list):
        return ["Root must be a JSON array"]
    errors: list[str] = []
    seen: set[int] = set()
    for index, item in enumerate(items):
        label = f"Item {index}"
        if not isinstance(item, dict):
            errors.append(f"{label}: must be an object")
            continue
        if set(item) != BTC_FIELDS:
            errors.append(f"{label}: fields must equal BTC_FIELDS; missing={sorted(BTC_FIELDS-set(item))}, extra={sorted(set(item)-BTC_FIELDS)}")
        question_id = item.get("id")
        if isinstance(question_id, bool) or not isinstance(question_id, int):
            errors.append(f"{label}: id must be int")
        elif question_id in seen:
            errors.append(f"{label}: duplicate id {question_id}")
        else:
            seen.add(question_id)
            if questions is not None:
                if question_id not in questions:
                    errors.append(f"{label}: id {question_id} not in expected question universe")
                elif item.get("question") != questions[question_id]:
                    errors.append(f"{label}: question does not match source for id {question_id}")
        if not isinstance(item.get("question"), str) or not item.get("question", "").strip():
            errors.append(f"{label}: question must be non-empty str")
        answer = item.get("answer")
        if isinstance(answer, bool) or not isinstance(answer, (int, float)) or not math.isfinite(float(answer)):
            errors.append(f"{label}: answer must be finite numeric")
        for field in ("relevant_docs", "relevant_tables"):
            if not isinstance(item.get(field), list):
                errors.append(f"{label}: {field} must be list")
        evidence = item.get("evidence")
        variables: set[str] = set()
        if not isinstance(evidence, list) or not evidence:
            errors.append(f"{label}: evidence must be non-empty list")
        else:
            for ev_index, record in enumerate(evidence):
                ev_label = f"{label} evidence[{ev_index}]"
                if not isinstance(record, dict) or set(record) != EVIDENCE_FIELDS:
                    errors.append(f"{ev_label}: fields must equal {sorted(EVIDENCE_FIELDS)}")
                    continue
                variable, csv_path = record.get("variable"), record.get("csv_path")
                if not isinstance(variable, str) or not variable:
                    errors.append(f"{ev_label}: variable must be non-empty str")
                elif variable in variables:
                    errors.append(f"{ev_label}: duplicate variable {variable}")
                else:
                    variables.add(variable)
                if not isinstance(csv_path, str) or not csv_path.startswith("data/") or csv_path.count("/") != 1 or "\\" in csv_path or not csv_path.endswith(".csv"):
                    errors.append(f"{ev_label}: csv_path must be flat data/<file>.csv")
        if not isinstance(item.get("pandas_query"), str) or not item.get("pandas_query", "").strip():
            errors.append(f"{label}: pandas_query must be non-empty str")
    return errors


def account_ids(expected_ids: Iterable[int], items: Sequence[Mapping[str, Any]], failures: Sequence[Mapping[str, Any]] = ()) -> AccountingReport:
    expected = frozenset(int(value) for value in expected_ids)
    saved_list = [int(item["id"]) for item in items if isinstance(item.get("id"), int)]
    saved = frozenset(saved_list)
    failed = frozenset(int(item["id"]) for item in failures if isinstance(item.get("id"), int))
    duplicates = frozenset(value for value in saved if saved_list.count(value) > 1)
    present = saved | failed
    return AccountingReport(expected, saved, failed, duplicates, expected-present, present-expected, saved & failed)


def package_submission_atomic(output_zip: str, submission_json: str, csv_paths: Iterable[str]) -> None:
    """Write exact CSV closure atomically; reject missing files and basename collisions."""
    by_name: dict[str, str] = {}
    for raw_path in csv_paths:
        resolved = os.path.abspath(raw_path)
        if not os.path.isfile(resolved):
            raise FileNotFoundError(f"Evidence CSV not found: {raw_path}")
        name = os.path.basename(resolved)
        previous = by_name.get(name)
        if previous and os.path.normcase(previous) != os.path.normcase(resolved):
            raise ValueError(f"Evidence basename collision: {name}: {previous} vs {resolved}")
        by_name[name] = resolved
    destination = Path(output_zip).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp", dir=destination.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(submission_json, arcname="submission.json")
            for name, path in sorted(by_name.items()):
                archive.write(path, arcname=f"data/{name}")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


__all__ = ["AccountingReport", "BTC_FIELDS", "account_ids", "load_questions", "package_submission_atomic", "validate_items"]
