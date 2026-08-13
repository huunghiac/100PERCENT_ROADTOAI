"""Streaming structural audit for all generated ViFinQA extraction artifacts.

The audit intentionally checks structure and traceability only.  Numeric accuracy is
measured by the separate gold evaluator; a parseable CSV is not evidence that a
financial value is correct.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable


EXPECTED_COLUMNS = ["Chi_tieu", "Gia_tri", "Don_vi"]
FILENAME_PATTERN = re.compile(
    r"^[A-Z0-9]+_\d{4}_[A-Za-z0-9]+_"
    r"(?:consolidated|separate|aggregated|unknown)(?:_\d{2})?\.csv$"
)
CORE_TABLES = {
    "BangCanDoiKeToan",
    "BaoCaoTinhHinhTaiChinh",
    "BaoCaoKetQuaKinhDoanh",
    "BaoCaoLuuChuyenTienTe",
}
REPORT_TYPES = {"consolidated", "separate", "aggregated", "unknown"}
CONFIDENCE_LEVELS = {"high", "medium", "low"}

BASE_MANIFEST_FIELDS = {
    "csv_path",
    "source_txt",
    "ticker",
    "company_name",
    "report_year",
    "report_type",
    "table_title",
    "table_slug",
    "unit",
    "value_period",
    "parser",
    "row_count",
    "warnings",
}
PHASE2_MANIFEST_FIELDS = {
    "value_column_method",
    "value_column_header",
    "value_column_confidence",
    "candidate_columns",
    "logical_table_id",
    "unit_source",
    "unit_confidence",
    "source_table_index",
}
REQUIRED_MANIFEST_FIELDS = BASE_MANIFEST_FIELDS | PHASE2_MANIFEST_FIELDS

REJECTED_CELL_FIELDS = {
    "source_txt",
    "ticker",
    "report_year",
    "table_title",
    "source_table_index",
    "source_row",
    "source_column",
    "raw_cell",
    "reason",
    "candidate_split",
    "confidence",
}
QUARANTINE_FIELDS = {
    "source_txt",
    "ticker",
    "report_year",
    "report_type",
    "table_title",
    "table_slug",
    "source_table_index",
    "reason",
    "value_column_method",
    "value_column_header",
    "value_column_confidence",
    "candidate_columns",
    "warnings",
}

Failure = tuple[str, str]


def _resolved(root: Path, value: object) -> Path:
    path = Path(str(value))
    return (root / path).resolve() if not path.is_absolute() else path.resolve()


def _path_key(path: Path) -> str:
    """Return a Windows-safe identity key, also deterministic on POSIX."""

    return os.path.normcase(str(path.resolve())).casefold()


def _is_nonempty_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_nonnegative_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _valid_candidate_columns(value: object) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, dict) for item in value)


def _validate_manifest_entry(
    entry: dict[str, object], label: str, root: Path, output_dir: Path
) -> list[Failure]:
    failures: list[Failure] = []
    missing = sorted(REQUIRED_MANIFEST_FIELDS - entry.keys())
    for field in missing:
        failures.append((label, f"missing_manifest_field:{field}"))
    if missing:
        return failures

    required_strings = (
        "csv_path",
        "source_txt",
        "ticker",
        "company_name",
        "table_title",
        "table_slug",
        "parser",
        "value_column_method",
        "logical_table_id",
    )
    for field in required_strings:
        if not _is_nonempty_string(entry[field]):
            failures.append((label, f"invalid_manifest_field:{field}"))
    for field in ("unit", "value_period", "value_column_header", "unit_source"):
        if not isinstance(entry[field], str):
            failures.append((label, f"invalid_manifest_field:{field}"))

    year = entry["report_year"]
    if not _is_nonnegative_int(year) or not 1900 <= int(year) <= 2100:
        failures.append((label, "invalid_manifest_field:report_year"))
    if not isinstance(entry["report_type"], str) or entry["report_type"] not in REPORT_TYPES:
        failures.append((label, "invalid_manifest_field:report_type"))
    if not _is_nonnegative_int(entry["row_count"]) or int(entry["row_count"]) == 0:
        failures.append((label, "invalid_manifest_field:row_count"))
    if not _is_nonnegative_int(entry["source_table_index"]):
        failures.append((label, "invalid_manifest_field:source_table_index"))
    if (
        not isinstance(entry["value_column_confidence"], str)
        or entry["value_column_confidence"] not in CONFIDENCE_LEVELS
    ):
        failures.append((label, "invalid_manifest_field:value_column_confidence"))
    elif entry["value_column_confidence"] == "low":
        failures.append((label, "low_confidence_table_exported"))
    if not isinstance(entry["unit_confidence"], str) or entry["unit_confidence"] not in CONFIDENCE_LEVELS:
        failures.append((label, "invalid_manifest_field:unit_confidence"))
    if not _valid_candidate_columns(entry["candidate_columns"]):
        failures.append((label, "invalid_manifest_field:candidate_columns"))
    warnings = entry["warnings"]
    if not isinstance(warnings, list) or not all(isinstance(item, str) for item in warnings):
        failures.append((label, "invalid_manifest_field:warnings"))

    target = _resolved(root, entry["csv_path"])
    if target.parent != output_dir.resolve():
        failures.append((label, "manifest_target_outside_output_dir"))
    source = _resolved(root, entry["source_txt"])
    if not source.is_file():
        failures.append((label, "source_target_missing"))
    return failures


def _audit_csv(path: Path, display_path: str, expected_row_count: int) -> list[Failure]:
    failures: list[Failure] = []
    if not path.is_file():
        return [(display_path, "manifest_target_missing")]
    if not FILENAME_PATTERN.fullmatch(path.name):
        failures.append((display_path, "filename"))

    row_count = 0
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = csv.reader(handle)
            header = next(rows, [])
            if header != EXPECTED_COLUMNS:
                failures.append((display_path, "schema"))
            for line_number, row in enumerate(rows, 2):
                row_count += 1
                if len(row) != len(EXPECTED_COLUMNS):
                    failures.append((display_path, f"column_count_at_line_{line_number}"))
                    continue
                raw_value = row[1].strip()
                try:
                    number = Decimal(raw_value)
                    if not raw_value or not number.is_finite():
                        raise InvalidOperation
                except (InvalidOperation, ValueError):
                    failures.append((f"{display_path}:{line_number}", "numeric"))
    except (OSError, UnicodeError, csv.Error) as exc:
        failures.append((display_path, f"read_error:{type(exc).__name__}"))
        return failures

    if row_count == 0:
        failures.append((display_path, "empty"))
    if row_count != expected_row_count:
        failures.append((display_path, "row_count"))
    return failures


def _validate_sidecar_entry(
    entry: dict[str, object], kind: str, label: str, root: Path
) -> list[Failure]:
    required = REJECTED_CELL_FIELDS if kind == "rejected_cell" else QUARANTINE_FIELDS
    failures: list[Failure] = []
    for field in sorted(required - entry.keys()):
        failures.append((label, f"missing_{kind}_field:{field}"))
    if required - entry.keys():
        return failures

    string_fields = {"source_txt", "ticker", "table_title", "reason"}
    if kind == "rejected_cell":
        string_fields |= {"raw_cell"}
    else:
        string_fields |= {"table_slug", "value_column_method"}
    for field in string_fields:
        if not _is_nonempty_string(entry[field]):
            failures.append((label, f"invalid_{kind}_field:{field}"))

    if not _is_nonnegative_int(entry["report_year"]) or not 1900 <= int(entry["report_year"]) <= 2100:
        failures.append((label, f"invalid_{kind}_field:report_year"))
    if not _is_nonnegative_int(entry["source_table_index"]):
        failures.append((label, f"invalid_{kind}_field:source_table_index"))
    if not _resolved(root, entry["source_txt"]).is_file():
        failures.append((label, "source_target_missing"))

    if kind == "rejected_cell":
        for field in ("source_row", "source_column"):
            if not _is_nonnegative_int(entry[field]):
                failures.append((label, f"invalid_rejected_cell_field:{field}"))
        if not isinstance(entry["candidate_split"], list):
            failures.append((label, "invalid_rejected_cell_field:candidate_split"))
        if not isinstance(entry["confidence"], str) or entry["confidence"] not in CONFIDENCE_LEVELS:
            failures.append((label, "invalid_rejected_cell_field:confidence"))
    else:
        if not isinstance(entry["report_type"], str) or entry["report_type"] not in REPORT_TYPES:
            failures.append((label, "invalid_quarantine_field:report_type"))
        if not isinstance(entry["value_column_header"], str):
            failures.append((label, "invalid_quarantine_field:value_column_header"))
        if (
            not isinstance(entry["value_column_confidence"], str)
            or entry["value_column_confidence"] not in CONFIDENCE_LEVELS
        ):
            failures.append((label, "invalid_quarantine_field:value_column_confidence"))
        if not isinstance(entry["candidate_columns"], list) or not all(
            isinstance(item, dict) for item in entry["candidate_columns"]
        ):
            failures.append((label, "invalid_quarantine_field:candidate_columns"))
        if not isinstance(entry["warnings"], list) or not all(
            isinstance(item, str) for item in entry["warnings"]
        ):
            failures.append((label, "invalid_quarantine_field:warnings"))
    return failures


def _audit_jsonl_sidecar(path: Path, kind: str, root: Path) -> tuple[int, list[Failure]]:
    failures: list[Failure] = []
    count = 0
    if not path.is_file():
        return 0, [(path.as_posix(), f"missing_{kind}_sidecar")]
    try:
        with path.open("r", encoding="utf-8", errors="strict") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                label = f"{path.as_posix()}:{line_number}"
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    failures.append((label, f"invalid_{kind}_json"))
                    continue
                if not isinstance(entry, dict):
                    failures.append((label, f"invalid_{kind}_record"))
                    continue
                count += 1
                failures.extend(_validate_sidecar_entry(entry, kind, label, root))
    except (OSError, UnicodeError) as exc:
        failures.append((path.as_posix(), f"read_error:{type(exc).__name__}"))
    return count, failures


def _drain_futures(
    pending: set[Future[list[Failure]]], failures: list[Failure], *, all_pending: bool = False
) -> None:
    if not pending:
        return
    done, still_pending = wait(
        pending, return_when=ALL_COMPLETED if all_pending else FIRST_COMPLETED
    )
    for future in done:
        failures.extend(future.result())
    pending.clear()
    pending.update(still_pending)


def _iter_output_csvs(output_dir: Path) -> Iterable[Path]:
    with os.scandir(output_dir) as entries:
        for entry in entries:
            if entry.is_file() and entry.name.casefold().endswith(".csv"):
                yield Path(entry.path).resolve()


def audit(
    output_dir: Path,
    root: Path,
    sample_files: int | None = None,
    *,
    max_workers: int | None = None,
) -> tuple[dict[str, object], list[Failure]]:
    """Audit generated files while retaining only indexes and bounded futures.

    ``sample_files`` is retained for compatibility with the phase-1 helper.  It
    limits CSV content reads only; manifest, target, collision, and sidecar checks
    always cover 100% of entries.  Omitting it audits every CSV.
    """

    root = root.resolve()
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "_manifest.jsonl"
    if sample_files is not None and sample_files <= 0:
        raise ValueError("sample_files must be positive")
    worker_count = max_workers or min(32, (os.cpu_count() or 1) + 4)
    if worker_count <= 0:
        raise ValueError("max_workers must be positive")

    failures: list[Failure] = []
    actual_by_key: dict[str, Path] = {}
    actual_csv_count = 0
    output_case_collisions = 0
    for csv_path in _iter_output_csvs(output_dir):
        actual_csv_count += 1
        key = _path_key(csv_path)
        previous = actual_by_key.get(key)
        if previous is not None and str(previous) != str(csv_path):
            output_case_collisions += 1
            failures.append((csv_path.as_posix(), "output_case_insensitive_collision"))
        else:
            actual_by_key[key] = csv_path

    seen_csv_paths: set[str] = set()
    seen_csv_casefold: dict[str, str] = {}
    seen_target_keys: set[str] = set()
    seen_logical_ids: dict[str, str] = {}
    manifest_rows = 0
    manifest_duplicates = 0
    manifest_case_collisions = 0
    audited_csv_count = 0
    tickers: set[str] = set()
    years: set[int] = set()
    report_types: Counter[str] = Counter()
    parsers: Counter[str] = Counter()
    units: Counter[str] = Counter()
    warnings: Counter[str] = Counter()
    confidences: Counter[str] = Counter()
    core_tables: Counter[str] = Counter()
    missing_company_names = 0
    pending: set[Future[list[Failure]]] = set()
    pending_limit = max(worker_count * 4, 1)

    if not manifest_path.is_file():
        failures.append((manifest_path.as_posix(), "manifest_missing"))
    else:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            try:
                with manifest_path.open("r", encoding="utf-8", errors="strict") as handle:
                    for line_number, line in enumerate(handle, 1):
                        if not line.strip():
                            continue
                        manifest_rows += 1
                        line_label = f"{manifest_path.as_posix()}:{line_number}"
                        try:
                            entry = json.loads(line)
                        except json.JSONDecodeError:
                            failures.append((line_label, "invalid_manifest_json"))
                            continue
                        if not isinstance(entry, dict):
                            failures.append((line_label, "invalid_manifest_record"))
                            continue

                        raw_csv_path = entry.get("csv_path")
                        label = str(raw_csv_path) if raw_csv_path else line_label
                        failures.extend(_validate_manifest_entry(entry, label, root, output_dir))
                        if not _is_nonempty_string(raw_csv_path):
                            continue

                        csv_text = str(raw_csv_path)
                        if csv_text in seen_csv_paths:
                            manifest_duplicates += 1
                            failures.append((label, "manifest_duplicate_exact"))
                        else:
                            seen_csv_paths.add(csv_text)
                            folded_path = csv_text.casefold()
                            previous = seen_csv_casefold.get(folded_path)
                            if previous is not None and previous != csv_text:
                                manifest_case_collisions += 1
                                failures.append((label, "manifest_case_insensitive_collision"))
                            else:
                                seen_csv_casefold[folded_path] = csv_text

                        target = _resolved(root, csv_text)
                        seen_target_keys.add(_path_key(target))
                        logical_id = entry.get("logical_table_id")
                        if _is_nonempty_string(logical_id):
                            logical_text = str(logical_id)
                            previous_path = seen_logical_ids.get(logical_text)
                            if previous_path is not None and previous_path != csv_text:
                                failures.append((label, "duplicate_logical_table_id"))
                            else:
                                seen_logical_ids[logical_text] = csv_text

                        ticker_value = entry.get("ticker")
                        if _is_nonempty_string(ticker_value):
                            tickers.add(str(ticker_value))
                        year_value = entry.get("report_year")
                        if _is_nonnegative_int(year_value):
                            years.add(int(year_value))
                        report_types[str(entry.get("report_type", ""))] += 1
                        parsers[str(entry.get("parser", ""))] += 1
                        units[str(entry.get("unit", ""))] += 1
                        confidences[str(entry.get("value_column_confidence", ""))] += 1
                        slug = str(entry.get("table_slug", ""))
                        if slug in CORE_TABLES:
                            core_tables[slug] += 1
                        if not entry.get("company_name"):
                            missing_company_names += 1
                        warning_items = entry.get("warnings", [])
                        if isinstance(warning_items, list):
                            for warning in warning_items:
                                if isinstance(warning, str):
                                    warnings[warning.split(":", 1)[0].split("=", 1)[0]] += 1

                        expected_count = entry.get("row_count")
                        should_audit = sample_files is None or audited_csv_count < sample_files
                        if should_audit and _is_nonnegative_int(expected_count):
                            pending.add(executor.submit(_audit_csv, target, csv_text, int(expected_count)))
                            audited_csv_count += 1
                            if len(pending) >= pending_limit:
                                _drain_futures(pending, failures)
            except (OSError, UnicodeError) as exc:
                failures.append((manifest_path.as_posix(), f"read_error:{type(exc).__name__}"))
            _drain_futures(pending, failures, all_pending=True)

    for key, path in actual_by_key.items():
        if key not in seen_target_keys:
            failures.append((path.as_posix(), "csv_missing_from_manifest"))

    rejected_count, rejected_failures = _audit_jsonl_sidecar(
        output_dir / "_rejected_cells.jsonl", "rejected_cell", root
    )
    quarantine_count, quarantine_failures = _audit_jsonl_sidecar(
        output_dir / "_quarantine.jsonl", "quarantine", root
    )
    failures.extend(rejected_failures)
    failures.extend(quarantine_failures)
    failures.sort()

    result = {
        "csv_count": actual_csv_count,
        "audited_csv_count": audited_csv_count,
        "manifest_rows": manifest_rows,
        "manifest_duplicates": manifest_duplicates,
        "manifest_case_collisions": manifest_case_collisions,
        "output_case_collisions": output_case_collisions,
        "failure_count": len(failures),
        "tickers": len(tickers),
        "years": sorted(years),
        "report_types": dict(report_types),
        "parsers": dict(parsers),
        "core_tables": dict(core_tables),
        "missing_company_names": missing_company_names,
        "top_units": units.most_common(8),
        "top_warnings": warnings.most_common(12),
        "value_column_confidences": dict(confidences),
        "rejected_cells": rejected_count,
        "quarantined_tables": quarantine_count,
        "max_workers": worker_count,
    }
    return result, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/processed_csv")
    parser.add_argument(
        "--sample-files",
        type=int,
        help="Optional compatibility smoke mode; omit for the required 100%% audit.",
    )
    parser.add_argument("--max-workers", type=int, help="Bounded CSV validation worker count.")
    parser.add_argument("--failure-limit", type=int, default=50)
    args = parser.parse_args()
    if args.failure_limit <= 0:
        parser.error("--failure-limit must be positive")
    root = Path.cwd()
    result, failures = audit(
        Path(args.output_dir), root, args.sample_files, max_workers=args.max_workers
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if failures:
        print(
            json.dumps(
                {"failure_examples": failures[: args.failure_limit]},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
