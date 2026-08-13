"""Deterministic, source-grounded quality evaluation for the ViFinQA extractor.

The evaluator deliberately separates structural validity from content accuracy.
Each gold record addresses one raw table in ``tests/gold`` and all metrics are
computed from extractor APIs rather than from generated production CSV files.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median
from types import SimpleNamespace
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import data_extractor as extractor  # noqa: E402


DEFAULT_GOLD_DIR = Path(__file__).resolve().parent / "gold"
DEFAULT_CASES_PATH = Path(__file__).resolve().parent / "gold_cases.jsonl"
CORE_SLUGS = {
    "BangCanDoiKeToan",
    "BaoCaoTinhHinhTaiChinh",
    "BaoCaoKetQuaKinhDoanh",
    "BaoCaoLuuChuyenTienTe",
}


def load_gold_cases(path: str | Path = DEFAULT_CASES_PATH) -> list[dict[str, Any]]:
    """Load and minimally validate the line-oriented gold contract."""

    cases: list[dict[str, Any]] = []
    required = {
        "case_id",
        "source",
        "report_year",
        "table_index",
        "expected_accept",
        "expected_slug",
        "expected_column",
        "expected_header_contains",
        "expected_period_contains",
        "expected_unit",
        "logical_table_id",
        "key_rows",
    }
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            case = json.loads(line)
            missing = required - set(case)
            if missing:
                raise ValueError(f"Gold line {line_number} lacks fields: {sorted(missing)}")
            cases.append(case)
    identifiers = [str(case["case_id"]) for case in cases]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Gold case_id values must be unique")
    return cases


def _percent(numerator: int, denominator: int) -> float:
    return round(100.0 * numerator / denominator, 4) if denominator else 100.0


def _contains_folded(value: object, expected_fragment: object) -> bool:
    fragment = extractor.fold_text(expected_fragment)
    return not fragment or fragment in extractor.fold_text(value)


def _legacy_canonical_table_slug(title: str) -> str:
    """Reproduce the broad phase-1 substring mapper (including its false core slug)."""

    folded = extractor.fold_text(title)
    if "bang can doi ke toan" in folded:
        return "BangCanDoiKeToan"
    if "bao cao tinh hinh tai chinh" in folded or "tinh hinh tai chinh" in folded:
        return "BaoCaoTinhHinhTaiChinh"
    if "ket qua hoat dong kinh doanh" in folded or "bao cao ket qua kinh doanh" in folded:
        return "BaoCaoKetQuaKinhDoanh"
    if ("luu chuyen tien" in folded or "lu chuyen tien" in folded) and "te" in folded:
        return "BaoCaoLuuChuyenTienTe"
    if "thay doi von chu so huu" in folded:
        return "BaoCaoThayDoiVonChuSoHuu"

    cleaned = re.sub(
        r"\b(tiep theo|tiep|continued|hop nhat|tong hop|rieng le)\b", " ", folded
    )
    words = re.findall(r"[a-z0-9]+", cleaned)
    ignored = {"don", "vi", "tinh", "nam", "tai", "ngay", "bang"}
    words = [word for word in words if word not in ignored]
    if not words:
        return "BangTaiChinh"
    return ("".join(word.capitalize() for word in words[:10])[:64] or "BangTaiChinh")


def _legacy_is_continuation_title(title: str) -> bool:
    return bool(
        re.search(r"\b(tiep theo|tiep|continued)\b", extractor.fold_text(title))
    )


def _legacy_base_table_title(title: str) -> str:
    cleaned = re.sub(
        r"(?i)\s*[\[(]?\s*(tiếp theo|tiếp|continued)\s*[\])]?[\s]*", " ", title
    )
    return extractor.normalize_space(cleaned).strip("-:;,. ") or "Bảng tài chính"


def _legacy_candidate_shape(rows: Sequence[Sequence[str]]) -> tuple[int, int, int]:
    numeric_cells = numeric_rows = total_cells = 0
    for row in rows:
        row_numeric = 0
        for cell in row:
            if extractor.normalize_space(cell):
                total_cells += 1
            if extractor.parse_number(cell) is not None:
                numeric_cells += 1
                row_numeric += 1
        numeric_rows += int(bool(row_numeric))
    return numeric_cells, numeric_rows, total_cells


def _legacy_is_financial_candidate(candidate: Any) -> tuple[bool, str]:
    """Phase-1 classification coupled to the broad legacy slug function."""

    if len(candidate.rows) < 2:
        return False, "fewer_than_two_rows"
    numeric_cells, numeric_rows, total_cells = _legacy_candidate_shape(candidate.rows)
    if numeric_cells < 2 or numeric_rows < 2:
        return False, "insufficient_numeric_rows"

    first_rows = extractor.fold_text(" ".join(" ".join(row) for row in candidate.rows[:3]))
    all_text = extractor.fold_text(
        candidate.title + " " + " ".join(" ".join(row) for row in candidate.rows[:30])
    )
    if "trang" in first_rows:
        return False, "table_of_contents"

    known_statement = _legacy_canonical_table_slug(candidate.title) in set(
        extractor.KNOWN_TABLE_SLUGS
    )
    financial_hits = sum(
        keyword in all_text for keyword in extractor.FINANCIAL_KEYWORDS
    )
    administrative_hits = sum(
        keyword in all_text for keyword in extractor.ADMINISTRATIVE_KEYWORDS
    )
    if administrative_hits >= 2 and not known_statement:
        return False, "administrative_or_staff_table"
    if not known_statement and financial_hits == 0:
        return False, "no_financial_context"
    if numeric_cells / max(total_cells, 1) < 0.08 and not known_statement:
        return False, "numeric_density_too_low"
    return True, ""


def _legacy_column_values(
    rows: Sequence[Sequence[str]], column: int
) -> list[int | float]:
    return [
        parsed
        for row in rows
        if column < len(row)
        for parsed in [extractor.parse_number(row[column])]
        if parsed is not None
    ]


def _legacy_header_text(rows: Sequence[Sequence[str]], column: int) -> str:
    parts: list[str] = []
    for row in rows[:6]:
        if column >= len(row):
            continue
        cell = extractor.normalize_space(row[column])
        if not cell:
            continue
        folded = extractor.fold_text(cell)
        if extractor.parse_number(cell) is None or re.search(r"\b(19|20)\d{2}\b", folded):
            parts.append(cell)
    return extractor.normalize_space(" ".join(dict.fromkeys(parts)))


def _legacy_decision_for(
    rows: Sequence[Sequence[str]], report_year: int
) -> dict[str, Any]:
    """Exact phase-1 len/magnitude/current-period column scoring."""

    width = max((len(row) for row in rows), default=0)
    candidates: list[tuple[float, int, str, int]] = []
    year_text = str(report_year)
    for column in range(width):
        values = _legacy_column_values(rows, column)
        if len(values) < 2:
            continue
        header = _legacy_header_text(rows, column)
        folded_header = extractor.fold_text(header)
        if any(keyword in folded_header for keyword in extractor.METADATA_COLUMN_KEYWORDS):
            continue
        text_count = sum(
            column < len(row)
            and bool(extractor.normalize_space(row[column]))
            and extractor.parse_number(row[column]) is None
            for row in rows
        )
        if column == 0 and text_count > len(values):
            continue
        absolute_values = [abs(float(value)) for value in values if float(value) != 0]
        magnitude = median(absolute_values) if absolute_values else 0.0
        magnitude_score = min(math.log10(magnitude + 1.0), 12.0)
        small_integer_ratio = sum(
            float(value).is_integer() and abs(float(value)) <= 999 for value in values
        ) / len(values)
        score = len(values) * 5.0 + magnitude_score - column * 0.05
        if re.search(rf"(?<!\d){re.escape(year_text)}(?!\d)", folded_header):
            score += 120.0
        if re.search(rf"31[./-]12[./-]{re.escape(year_text)}", folded_header):
            score += 45.0
        if re.search(rf"0?1[./-]0?1[./-]{re.escape(year_text)}", folded_header):
            score -= 15.0
        if any(
            token in folded_header for token in ("nam nay", "ky nay", "cuoi nam", "tai ngay")
        ):
            score += 25.0
        other_years = [int(item) for item in re.findall(r"\b(20\d{2})\b", folded_header)]
        if other_years and report_year not in other_years:
            score -= 60.0
        if not header and small_integer_ratio > 0.8 and magnitude < 1000:
            score -= 25.0
        candidates.append((score, column, header, len(values)))

    if not candidates:
        return {
            "column": None,
            "method": "legacy_none",
            "header": "",
            "value_period": "",
            "confidence": "legacy",
            "candidates": [],
            "warnings": ["no_reliable_value_column"],
        }
    candidates.sort(key=lambda item: (-item[0], item[1]))
    score, column, header, _ = candidates[0]
    exact_year = bool(
        re.search(rf"(?<!\d){re.escape(year_text)}(?!\d)", extractor.fold_text(header))
    )
    warnings: list[str] = []
    if exact_year:
        value_period = header or year_text
    else:
        value_period = header or f"heuristic_current_period_{year_text}"
        alternatives = ",".join(str(item[1]) for item in candidates[:3])
        warnings.append(
            f"value_column_heuristic:selected={column};candidates={alternatives};score={score:.2f}"
        )
    return {
        "column": column,
        "method": "legacy_scored_heuristic",
        "header": header,
        "value_period": value_period,
        "confidence": "legacy",
        "candidates": [
            {"column": item[1], "header": item[2], "score": item[0]}
            for item in candidates
        ],
        "warnings": warnings,
    }


def _legacy_converted_table(
    candidate: Any, report_year: int, decision: dict[str, Any] | None = None
) -> tuple[Any | None, str]:
    accepted, reason = _legacy_is_financial_candidate(candidate)
    if not accepted:
        return None, reason
    column_decision = decision or _legacy_decision_for(candidate.rows, report_year)
    value_column = column_decision["column"]
    if value_column is None:
        return None, "no_reliable_value_column"

    rows, merge_warnings = extractor.merge_wrapped_label_rows(
        candidate.rows, value_column
    )
    records: list[dict[str, Any]] = []
    for row in rows:
        if value_column >= len(row):
            continue
        raw_value = extractor.normalize_space(row[value_column])
        value = extractor.parse_number(raw_value)
        if value is None:
            continue
        label_index = extractor._label_index(row, value_column)
        if label_index is None:
            continue
        label = extractor.normalize_space(row[label_index])
        if not label or extractor._is_header_label(label):
            continue
        records.append(
            {
                "Chi_tieu": label,
                "Gia_tri": value,
                "Don_vi": extractor.infer_row_unit(label, raw_value, candidate.unit),
            }
        )
    if len(records) < 2:
        return None, "fewer_than_two_normalized_rows"

    units = {str(record["Don_vi"]) for record in records if record["Don_vi"]}
    manifest_unit = next(iter(units)) if len(units) == 1 else ("mixed" if units else "")
    title = _legacy_base_table_title(candidate.title)
    return (
        SimpleNamespace(
            table_title=title,
            table_slug=_legacy_canonical_table_slug(title),
            unit=manifest_unit,
            value_period=column_decision["value_period"],
            parser=candidate.parser,
            records=records,
            source_table_indices=[candidate.source_table_index],
            continued=_legacy_is_continuation_title(candidate.title),
            warnings=list(
                dict.fromkeys(
                    list(column_decision["warnings"])
                    + list(candidate.warnings)
                    + list(merge_warnings)
                )
            ),
        ),
        "",
    )


def _legacy_merge_continuations(tables: Sequence[Any]) -> list[Any]:
    """Phase-1 merge: explicit marker plus any previous table with the same slug."""

    merged: list[Any] = []
    last_by_slug: dict[str, int] = {}
    for table in tables:
        target_index = last_by_slug.get(table.table_slug)
        if table.continued and target_index is not None:
            target = merged[target_index]
            existing = {
                (str(row["Chi_tieu"]), row["Gia_tri"], str(row["Don_vi"]))
                for row in target.records
            }
            for record in table.records:
                key = (str(record["Chi_tieu"]), record["Gia_tri"], str(record["Don_vi"]))
                if key not in existing:
                    target.records.append(record)
                    existing.add(key)
            target.source_table_indices.extend(table.source_table_indices)
            target.warnings.extend(table.warnings)
            target.warnings.append("merged_continuation_table")
            target.warnings = list(dict.fromkeys(target.warnings))
            continue
        if table.continued:
            table.warnings.append("continuation_without_preceding_table")
        merged.append(table)
        last_by_slug[table.table_slug] = len(merged) - 1
    return merged


def _legacy_extract_outputs(
    candidates: Sequence[Any], report_year: int
) -> list[Any]:
    extracted: list[Any] = []
    for candidate in candidates:
        converted, _ = _legacy_converted_table(candidate, report_year)
        if converted is not None:
            extracted.append(converted)
    return _legacy_merge_continuations(extracted)


def _decision_for(rows: Sequence[Sequence[str]], report_year: int) -> dict[str, Any]:
    """Use the auditable phase-2 decision API, with a phase-1 compatibility path."""

    decide = getattr(extractor, "decide_value_column", None)
    if callable(decide):
        decision = decide(rows, report_year)
        return {
            "column": decision.column,
            "method": decision.method,
            "header": decision.header,
            "value_period": decision.value_period,
            "confidence": decision.confidence,
            "candidates": decision.candidates,
            "warnings": decision.warnings,
        }

    column, value_period, warnings = extractor.select_value_column(rows, report_year)
    header = ""
    header_reader = getattr(extractor, "_header_text_for_column", None)
    if column is not None and callable(header_reader):
        header = header_reader(rows, column)
    return {
        "column": column,
        "method": "legacy_select_value_column",
        "header": header,
        "value_period": value_period,
        "confidence": "legacy",
        "candidates": [],
        "warnings": warnings,
    }


def _converted_table(candidate: Any, report_year: int) -> tuple[Any | None, str]:
    result = extractor.convert_candidate(candidate, report_year)
    if isinstance(result, tuple):
        return result[0], str(result[1])
    return result, ""


def _record_matches(record: dict[str, Any], expected: dict[str, Any]) -> bool:
    if extractor.fold_text(record.get("Chi_tieu", "")) != extractor.fold_text(
        expected["label"]
    ):
        return False
    try:
        actual_value = float(record["Gia_tri"])
        expected_value = float(expected["value"])
    except (KeyError, TypeError, ValueError):
        return False
    return math.isclose(actual_value, expected_value, rel_tol=1e-12, abs_tol=1e-9)


def _matching_key_count(records: Iterable[dict[str, Any]], key_rows: Sequence[dict[str, Any]]) -> int:
    actual = list(records)
    return sum(any(_record_matches(record, expected) for record in actual) for expected in key_rows)


def _candidate_tables(content: str) -> list[Any]:
    candidates = extractor.extract_html_tables(content)
    candidates.extend(extractor.extract_plain_text_tables(content))
    return candidates


def evaluate_quality(
    gold_dir: str | Path = DEFAULT_GOLD_DIR,
    cases_path: str | Path = DEFAULT_CASES_PATH,
    *,
    mode: str = "current",
) -> dict[str, Any]:
    """Evaluate classification, extraction, units, and logical-table grouping."""

    if mode not in {"current", "legacy"}:
        raise ValueError(f"Unsupported evaluation mode: {mode!r}")
    gold_root = Path(gold_dir)
    cases = load_gold_cases(cases_path)
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_source[str(case["source"])].append(case)

    classification_tp = classification_fp = classification_fn = classification_tn = 0
    value_total = value_correct = header_correct = period_correct = 0
    slug_total = slug_correct = 0
    numeric_total = numeric_correct = 0
    unit_tp = unit_fp = unit_fn = 0
    unknown_unit_total = unknown_unit_correct = 0
    mapped_case_outputs: dict[str, set[str]] = defaultdict(set)
    output_mappings: list[dict[str, Any]] = []
    case_results: list[dict[str, Any]] = []

    for source_name in sorted(by_source):
        source_path = gold_root / source_name
        if not source_path.exists():
            raise FileNotFoundError(f"Gold source does not exist: {source_path}")
        source_cases = sorted(by_source[source_name], key=lambda item: int(item["table_index"]))
        content = source_path.read_text(encoding="utf-8")
        candidates = _candidate_tables(content)
        expected_indices = {int(case["table_index"]) for case in source_cases}
        actual_indices = set(range(len(candidates)))
        if expected_indices != actual_indices:
            raise ValueError(
                f"Gold/raw table mismatch for {source_name}: "
                f"expected={sorted(expected_indices)}, actual={sorted(actual_indices)}"
            )

        source_case_by_index = {int(case["table_index"]): case for case in source_cases}
        for table_index, candidate in enumerate(candidates):
            case = source_case_by_index[table_index]
            expected_accept = bool(case["expected_accept"])
            classifier = (
                _legacy_is_financial_candidate
                if mode == "legacy"
                else extractor.is_financial_candidate
            )
            actual_accept, classification_reason = classifier(candidate)
            if expected_accept and actual_accept:
                classification_tp += 1
            elif expected_accept:
                classification_fn += 1
            elif actual_accept:
                classification_fp += 1
            else:
                classification_tn += 1

            if mode == "legacy":
                decision = _legacy_decision_for(candidate.rows, int(case["report_year"]))
                converted, conversion_reason = _legacy_converted_table(
                    candidate, int(case["report_year"]), decision
                )
                actual_slug = _legacy_canonical_table_slug(candidate.title)
            else:
                decision = _decision_for(candidate.rows, int(case["report_year"]))
                converted, conversion_reason = _converted_table(
                    candidate, int(case["report_year"])
                )
                actual_slug = extractor.canonical_table_slug(candidate.title)
            actual_unit = (
                str(getattr(converted, "unit", "") or "")
                if actual_accept and converted is not None
                else ""
            )
            keys_correct = 0

            if expected_accept:
                slug_total += 1
                slug_correct += int(actual_slug == case["expected_slug"])
                value_total += 1
                column_matches = actual_accept and decision["column"] == case["expected_column"]
                value_correct += int(column_matches)
                header_correct += int(
                    column_matches
                    and _contains_folded(decision["header"], case["expected_header_contains"])
                )
                period_correct += int(
                    column_matches
                    and _contains_folded(
                        decision["value_period"], case["expected_period_contains"]
                    )
                )

                records = list(getattr(converted, "records", [])) if converted is not None else []
                numeric_total += len(case["key_rows"])
                keys_correct = _matching_key_count(records, case["key_rows"])
                numeric_correct += keys_correct

                expected_unit = str(case["expected_unit"] or "")
                if expected_unit:
                    if actual_unit == expected_unit:
                        unit_tp += 1
                    else:
                        unit_fn += 1
                        if actual_unit:
                            unit_fp += 1
                else:
                    unknown_unit_total += 1
                    if not actual_unit:
                        unknown_unit_correct += 1
                    else:
                        unit_fp += 1

            case_results.append(
                {
                    "case_id": case["case_id"],
                    "source": source_name,
                    "table_index": table_index,
                    "expected_accept": expected_accept,
                    "actual_accept": actual_accept,
                    "classification_reason": classification_reason,
                    "expected_slug": case["expected_slug"],
                    "actual_slug": actual_slug,
                    "expected_column": case["expected_column"],
                    "actual_column": decision["column"],
                    "value_column_method": decision["method"],
                    "value_column_header": decision["header"],
                    "value_column_period": decision["value_period"],
                    "value_column_confidence": decision["confidence"],
                    "expected_unit": case["expected_unit"],
                    "actual_unit": actual_unit,
                    "key_rows_correct": keys_correct,
                    "key_rows_total": len(case["key_rows"]),
                    "converted": converted is not None,
                    "conversion_reason": conversion_reason,
                }
            )

        report_years = {int(case["report_year"]) for case in source_cases}
        if len(report_years) != 1:
            raise ValueError(f"Gold source mixes report years: {source_name}")
        report_year = next(iter(report_years))
        if mode == "legacy":
            outputs = _legacy_extract_outputs(candidates, report_year)
        else:
            outputs, _, _ = extractor.extract_tables_from_text(content, report_year)
        for output_index, output in enumerate(outputs):
            output_key = f"{source_name}#output-{output_index}"
            matching_cases: set[str] = set()
            source_indices = {
                int(index) for index in getattr(output, "source_table_indices", [])
            }
            for case in source_cases:
                if not case["expected_accept"]:
                    continue
                table_index = int(case["table_index"])
                source_index_match = table_index in source_indices
                record_match = bool(case["key_rows"]) and _matching_key_count(
                    getattr(output, "records", []), case["key_rows"]
                ) > 0
                if source_index_match or record_match:
                    case_id = str(case["case_id"])
                    matching_cases.add(case_id)
                    mapped_case_outputs[case_id].add(output_key)
            logical_ids = sorted(
                {
                    str(case["logical_table_id"])
                    for case in source_cases
                    if case["case_id"] in matching_cases
                }
            )
            output_mappings.append(
                {
                    "output": output_key,
                    "slug": output.table_slug,
                    "source_table_indices": sorted(source_indices),
                    "matched_cases": sorted(matching_cases),
                    "logical_table_ids": logical_ids,
                }
            )

    accepted_cases = [case for case in cases if case["expected_accept"]]
    logical_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in accepted_cases:
        logical_groups[str(case["logical_table_id"])].append(case)

    fragmented: dict[str, list[str]] = {}
    covered_logical: set[str] = set()
    for logical_id, group in sorted(logical_groups.items()):
        outputs_for_group = set().union(
            *(mapped_case_outputs.get(str(case["case_id"]), set()) for case in group)
        )
        if all(mapped_case_outputs.get(str(case["case_id"])) for case in group):
            covered_logical.add(logical_id)
        if len(outputs_for_group) > 1:
            fragmented[logical_id] = sorted(outputs_for_group)

    wrong_merges = [
        mapping for mapping in output_mappings if len(mapping["logical_table_ids"]) > 1
    ]
    core_logical = {
        logical_id
        for logical_id, group in logical_groups.items()
        if any(case["expected_slug"] in CORE_SLUGS for case in group)
    }
    covered_core = core_logical & covered_logical

    metrics = {
        "table_classification_precision": _percent(
            classification_tp, classification_tp + classification_fp
        ),
        "table_classification_recall": _percent(
            classification_tp, classification_tp + classification_fn
        ),
        "table_slug_accuracy": _percent(slug_correct, slug_total),
        "value_column_accuracy": _percent(value_correct, value_total),
        "value_column_header_accuracy": _percent(header_correct, value_total),
        "value_period_accuracy": _percent(period_correct, value_total),
        "numeric_value_accuracy": _percent(numeric_correct, numeric_total),
        "unit_extraction_precision": _percent(unit_tp, unit_tp + unit_fp),
        "unit_extraction_recall": _percent(unit_tp, unit_tp + unit_fn),
        "unknown_unit_accuracy": _percent(unknown_unit_correct, unknown_unit_total),
        "core_table_coverage": _percent(len(covered_core), len(core_logical)),
    }
    counts = {
        "gold_sources": len(by_source),
        "raw_tables": len(cases),
        "expected_accepted_tables": len(accepted_cases),
        "expected_rejected_tables": len(cases) - len(accepted_cases),
        "classification_tp": classification_tp,
        "classification_fp": classification_fp,
        "classification_fn": classification_fn,
        "classification_tn": classification_tn,
        "value_columns_correct": value_correct,
        "value_columns_total": value_total,
        "numeric_values_correct": numeric_correct,
        "numeric_values_total": numeric_total,
        "known_units_correct": unit_tp,
        "known_units_total": unit_tp + unit_fn,
        "unknown_units_correct": unknown_unit_correct,
        "unknown_units_total": unknown_unit_total,
        "expected_logical_tables": len(logical_groups),
        "expected_core_logical_tables": len(core_logical),
        "covered_core_logical_tables": len(covered_core),
        "fragmented_logical_tables": len(fragmented),
        "fragmentation_excess_files": sum(len(items) - 1 for items in fragmented.values()),
        "wrong_merge_outputs": len(wrong_merges),
    }
    failures = {
        "classification": [
            result["case_id"]
            for result in case_results
            if result["expected_accept"] != result["actual_accept"]
        ],
        "slug": [
            result["case_id"]
            for result in case_results
            if result["expected_accept"]
            and result["expected_slug"] != result["actual_slug"]
        ],
        "value_column": [
            result["case_id"]
            for result in case_results
            if result["expected_accept"]
            and result["expected_column"] != result["actual_column"]
        ],
        "numeric": [
            result["case_id"]
            for result in case_results
            if result["key_rows_correct"] != result["key_rows_total"]
        ],
        "unit": [
            result["case_id"]
            for result in case_results
            if result["expected_accept"]
            and str(result["expected_unit"] or "") != result["actual_unit"]
        ],
        "fragmented": fragmented,
        "wrong_merges": wrong_merges,
    }
    return {
        "mode": mode,
        "gold_cases_path": Path(cases_path).as_posix(),
        "gold_dir": gold_root.as_posix(),
        "metrics": metrics,
        "counts": counts,
        "failures": failures,
        "case_results": case_results,
        "output_mappings": output_mappings,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate ViFinQA extractor content accuracy.")
    parser.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD_DIR)
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument(
        "--mode",
        choices=("current", "legacy"),
        default="current",
        help="Evaluate current production behavior or the preserved phase-1 baseline adapter.",
    )
    parser.add_argument("--output", type=Path, help="Optional JSON report destination.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report = evaluate_quality(args.gold_dir, args.cases, mode=args.mode)
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
