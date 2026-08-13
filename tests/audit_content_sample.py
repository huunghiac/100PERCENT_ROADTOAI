"""Deterministic, stratified source-to-output content audit.

This audit is deliberately separate from ``audit_processed_csv.py``.  It does
not treat a valid schema or a parseable number as evidence of financial
accuracy.  Instead it selects a deterministic sample across business strata,
re-reads each source TXT, reconstructs the referenced logical table, and checks
that the selected raw column can reproduce the exact CSV labels, values, units,
and provenance metadata.  Independent correctness is measured by the gold-set
evaluator in ``evaluate_extractor_quality.py``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src import data_extractor as extractor  # noqa: E402


CORE_TABLES = {
    "BangCanDoiKeToan",
    "BaoCaoTinhHinhTaiChinh",
    "BaoCaoKetQuaKinhDoanh",
    "BaoCaoLuuChuyenTienTe",
}
STRATUM_DIMENSIONS = (
    "report_type",
    "report_year",
    "ticker",
    "value_column_confidence",
    "table_class",
)


@dataclass(frozen=True)
class ManifestCandidate:
    line_number: int
    csv_path: str
    source_txt: str
    ticker: str
    report_year: int
    report_type: str
    value_column_confidence: str
    table_slug: str

    @property
    def table_class(self) -> str:
        return "core" if self.table_slug in CORE_TABLES else "general"

    def strata(self) -> tuple[str, ...]:
        values = {
            "report_type": self.report_type,
            "report_year": str(self.report_year),
            "ticker": self.ticker,
            "value_column_confidence": self.value_column_confidence,
            "table_class": self.table_class,
        }
        return tuple(f"{dimension}={values[dimension]}" for dimension in STRATUM_DIMENSIONS)


def _resolved(root: Path, value: object) -> Path:
    path = Path(str(value))
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _stable_rank(candidate: ManifestCandidate, seed: str) -> str:
    payload = f"{seed}\0{candidate.csv_path}\0{candidate.line_number}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_manifest_candidates(manifest_path: Path) -> list[ManifestCandidate]:
    candidates: list[ManifestCandidate] = []
    with manifest_path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            entry = json.loads(line)
            candidates.append(
                ManifestCandidate(
                    line_number=line_number,
                    csv_path=str(entry["csv_path"]),
                    source_txt=str(entry["source_txt"]),
                    ticker=str(entry["ticker"]),
                    report_year=int(entry["report_year"]),
                    report_type=str(entry["report_type"]),
                    value_column_confidence=str(entry["value_column_confidence"]),
                    table_slug=str(entry["table_slug"]),
                )
            )
    return candidates


def select_stratified(
    candidates: Sequence[ManifestCandidate], sample_size: int, seed: str
) -> list[ManifestCandidate]:
    """Select a deterministic set-cover sample, then balance every dimension."""

    if sample_size <= 0:
        raise ValueError("sample_size must be positive")
    if not candidates:
        return []
    target = min(sample_size, len(candidates))
    ranks = {candidate.line_number: _stable_rank(candidate, seed) for candidate in candidates}
    all_strata = {stratum for candidate in candidates for stratum in candidate.strata()}
    uncovered = set(all_strata)
    selected: list[ManifestCandidate] = []
    selected_lines: set[int] = set()

    # First cover every observed category (including rare aggregated reports).
    while uncovered and len(selected) < target:
        remaining = (item for item in candidates if item.line_number not in selected_lines)
        best = min(
            remaining,
            key=lambda item: (
                -len(uncovered.intersection(item.strata())),
                ranks[item.line_number],
            ),
        )
        gain = uncovered.intersection(best.strata())
        if not gain:
            break
        selected.append(best)
        selected_lines.add(best.line_number)
        uncovered.difference_update(gain)

    # Fill the remaining quota by favouring currently under-represented strata.
    counts: Counter[str] = Counter(
        stratum for candidate in selected for stratum in candidate.strata()
    )
    while len(selected) < target:
        remaining = [item for item in candidates if item.line_number not in selected_lines]
        best = min(
            remaining,
            key=lambda item: (
                -sum(1.0 / (1 + counts[stratum]) for stratum in item.strata()),
                ranks[item.line_number],
            ),
        )
        selected.append(best)
        selected_lines.add(best.line_number)
        counts.update(best.strata())

    return sorted(selected, key=lambda item: (item.source_txt, item.csv_path))


def _load_selected_entries(
    manifest_path: Path, selected: Sequence[ManifestCandidate]
) -> list[dict[str, object]]:
    selected_lines = {candidate.line_number for candidate in selected}
    entries: list[dict[str, object]] = []
    with manifest_path.open("r", encoding="utf-8", errors="strict") as handle:
        for line_number, line in enumerate(handle, 1):
            if line_number in selected_lines and line.strip():
                entries.append(json.loads(line))
    return sorted(entries, key=lambda entry: (str(entry["source_txt"]), str(entry["csv_path"])))


def _decimal(value: object) -> Decimal:
    try:
        result = Decimal(str(value).strip())
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"not a decimal: {value!r}") from exc
    if not result.is_finite():
        raise ValueError(f"not a finite decimal: {value!r}")
    return result


def _read_csv_records(path: Path) -> list[tuple[str, Decimal, str]]:
    records: list[tuple[str, Decimal, str]] = []
    with path.open("r", encoding="utf-8-sig", errors="strict", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(extractor.CSV_COLUMNS):
            raise ValueError(f"unexpected CSV fields: {reader.fieldnames!r}")
        for row in reader:
            records.append(
                (
                    extractor.normalize_space(row["Chi_tieu"]),
                    _decimal(row["Gia_tri"]),
                    extractor.normalize_space(row["Don_vi"]),
                )
            )
    return records


def _normalized_records(table: extractor.ExtractedTable) -> list[tuple[str, Decimal, str]]:
    return [
        (
            extractor.normalize_space(str(record["Chi_tieu"])),
            _decimal(record["Gia_tri"]),
            extractor.normalize_space(str(record["Don_vi"])),
        )
        for record in table.records
    ]


def _raw_value_counter(parts: Sequence[extractor.RawTable], report_year: int) -> Counter[Decimal]:
    values: Counter[Decimal] = Counter()
    for part in parts:
        decision = extractor.decide_value_column(part.rows, report_year)
        if decision.column is None:
            continue
        for row in part.rows:
            if decision.column >= len(row):
                continue
            raw_value = extractor.normalize_space(row[decision.column])
            parsed = extractor.parse_number(raw_value)
            if parsed is not None:
                values[_decimal(parsed)] += 1
                continue
            for split in extractor._split_unique_thousands_concatenation(raw_value):
                split_value = extractor.parse_number(split)
                if split_value is not None:
                    values[_decimal(split_value)] += 1
    return values


def _raw_text_tokens(parts: Sequence[extractor.RawTable]) -> set[str]:
    tokens: set[str] = set()
    for part in parts:
        for row in part.rows:
            for cell in row:
                tokens.update(extractor.fold_text(cell).split())
    return tokens


def _record_failure(
    failures: list[dict[str, object]], csv_path: str, check: str, detail: str
) -> None:
    failures.append({"csv_path": csv_path, "check": check, "detail": detail})


def _dimension_counts(
    candidates: Iterable[ManifestCandidate], dimension: str
) -> dict[str, int]:
    values: Counter[str] = Counter()
    for candidate in candidates:
        value = candidate.table_class if dimension == "table_class" else str(getattr(candidate, dimension))
        values[value] += 1
    return dict(sorted(values.items()))


def audit_content_sample(
    output_dir: Path,
    root: Path,
    *,
    sample_size: int = 160,
    seed: str = "vifinqa-phase2-content-audit-v1",
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Audit a deterministic sample against its source TXT and current rules."""

    root = root.resolve()
    output_dir = output_dir.resolve()
    manifest_path = output_dir / "_manifest.jsonl"
    candidates = _load_manifest_candidates(manifest_path)
    selected = select_stratified(candidates, sample_size, seed)
    entries = _load_selected_entries(manifest_path, selected)

    failures: list[dict[str, object]] = []
    source_cache: dict[str, dict[int, extractor.RawTable]] = {}
    source_parse_failures: dict[str, str] = {}
    tables_exact = 0
    rows_compared = 0
    labels_exact = 0
    numeric_exact = 0
    units_exact = 0
    raw_numeric_provenance_hits = 0
    raw_label_token_hits = 0
    metadata_fields_compared = 0
    metadata_fields_exact = 0

    for entry in entries:
        csv_text = str(entry["csv_path"])
        source_text = str(entry["source_txt"])
        source_path = _resolved(root, source_text)
        if source_text not in source_cache and source_text not in source_parse_failures:
            try:
                content = source_path.read_text(encoding="utf-8", errors="replace")
                raw_tables = extractor.extract_html_tables(content)
                raw_tables.extend(extractor.extract_plain_text_tables(content))
                by_index: dict[int, extractor.RawTable] = {}
                duplicate_indexes: set[int] = set()
                for raw_table in raw_tables:
                    if raw_table.source_table_index in by_index:
                        duplicate_indexes.add(raw_table.source_table_index)
                    by_index[raw_table.source_table_index] = raw_table
                if duplicate_indexes:
                    raise ValueError(f"duplicate source indexes: {sorted(duplicate_indexes)}")
                source_cache[source_text] = by_index
            except (OSError, UnicodeError, ValueError) as exc:
                source_parse_failures[source_text] = f"{type(exc).__name__}: {exc}"

        if source_text in source_parse_failures:
            _record_failure(
                failures, csv_text, "source_read_or_parse", source_parse_failures[source_text]
            )
            continue

        raw_by_index = source_cache[source_text]
        source_indexes = [int(value) for value in entry.get("source_table_indices", [])]
        if not source_indexes:
            source_indexes = [int(entry["source_table_index"])]
        missing_indexes = [index for index in source_indexes if index not in raw_by_index]
        if missing_indexes:
            _record_failure(
                failures,
                csv_text,
                "source_table_provenance",
                f"indexes missing from source: {missing_indexes}",
            )
            continue
        parts = [raw_by_index[index] for index in source_indexes]
        metadata = extractor.ReportMetadata(
            ticker=str(entry["ticker"]),
            company_name=str(entry["company_name"]),
            report_year=int(entry["report_year"]),
            report_type=str(entry["report_type"]),
            source_txt=source_path,
        )
        rebuilt_parts: list[extractor.ExtractedTable] = []
        part_failure = ""
        for part in parts:
            rebuilt, reason = extractor.convert_candidate(
                part, metadata.report_year, metadata=metadata
            )
            if rebuilt is None:
                part_failure = f"source index {part.source_table_index} rejected: {reason}"
                break
            rebuilt_parts.append(rebuilt)
        if part_failure:
            _record_failure(failures, csv_text, "reconstruction", part_failure)
            continue

        extractor._inherit_continuation_units(rebuilt_parts)
        logical_tables = extractor.merge_continuation_tables(rebuilt_parts)
        if len(logical_tables) != 1:
            _record_failure(
                failures,
                csv_text,
                "logical_table_reconstruction",
                f"manifest references one table but current rules yield {len(logical_tables)}",
            )
            continue
        rebuilt = logical_tables[0]
        rebuilt.logical_table_id = extractor._logical_table_id(source_text, rebuilt)

        metadata_checks = {
            "table_title": rebuilt.table_title,
            "table_slug": rebuilt.table_slug,
            "unit": rebuilt.unit,
            "unit_source": rebuilt.unit_source,
            "unit_confidence": rebuilt.unit_confidence,
            "value_period": rebuilt.value_period,
            "value_column_method": rebuilt.value_column_method,
            "value_column_header": rebuilt.value_column_header,
            "value_column_confidence": rebuilt.value_column_confidence,
            "candidate_columns": rebuilt.candidate_columns,
            "source_table_indices": rebuilt.source_table_indices,
            "logical_table_id": rebuilt.logical_table_id,
            "parser": rebuilt.parser,
            "row_count": len(rebuilt.records),
        }
        for field, actual in metadata_checks.items():
            metadata_fields_compared += 1
            if entry.get(field) == actual:
                metadata_fields_exact += 1
            else:
                _record_failure(
                    failures,
                    csv_text,
                    f"manifest_{field}",
                    f"manifest={entry.get(field)!r}; rebuilt={actual!r}",
                )

        try:
            csv_records = _read_csv_records(_resolved(root, csv_text))
        except (OSError, UnicodeError, csv.Error, ValueError) as exc:
            _record_failure(failures, csv_text, "csv_read", f"{type(exc).__name__}: {exc}")
            continue
        rebuilt_records = _normalized_records(rebuilt)
        if csv_records == rebuilt_records:
            tables_exact += 1
        else:
            _record_failure(
                failures,
                csv_text,
                "record_sequence",
                f"CSV rows={len(csv_records)}; rebuilt rows={len(rebuilt_records)}",
            )

        row_pairs = list(zip(csv_records, rebuilt_records))
        rows_compared += max(len(csv_records), len(rebuilt_records))
        labels_exact += sum(left[0] == right[0] for left, right in row_pairs)
        numeric_exact += sum(left[1] == right[1] for left, right in row_pairs)
        units_exact += sum(left[2] == right[2] for left, right in row_pairs)

        raw_values = _raw_value_counter(parts, metadata.report_year)
        raw_tokens = _raw_text_tokens(parts)
        for label, value, _unit in csv_records:
            if raw_values[value] > 0:
                raw_numeric_provenance_hits += 1
                raw_values[value] -= 1
            label_tokens = extractor.fold_text(label).split()
            if label_tokens and all(token in raw_tokens for token in label_tokens):
                raw_label_token_hits += 1

    sampled_rows = sum(
        int(entry.get("row_count", 0)) for entry in entries
    )
    population_strata = {
        dimension: _dimension_counts(candidates, dimension) for dimension in STRATUM_DIMENSIONS
    }
    sample_strata = {
        dimension: _dimension_counts(selected, dimension) for dimension in STRATUM_DIMENSIONS
    }
    uncovered_strata = {
        dimension: sorted(set(population_strata[dimension]) - set(sample_strata[dimension]))
        for dimension in STRATUM_DIMENSIONS
    }
    uncovered_strata = {
        dimension: values for dimension, values in uncovered_strata.items() if values
    }

    def rate(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 6) if denominator else 0.0

    report: dict[str, object] = {
        "audit_kind": "stratified_source_to_output_reproducibility",
        "accuracy_scope": (
            "This is a provenance/reproducibility audit, not an independent estimate of "
            "financial numeric accuracy; use the gold evaluator for correctness."
        ),
        "seed": seed,
        "manifest_entries": len(candidates),
        "requested_sample_size": sample_size,
        "sampled_tables": len(entries),
        "sampled_sources": len({str(entry["source_txt"]) for entry in entries}),
        "sampled_manifest_rows": sampled_rows,
        "population_strata": population_strata,
        "sample_strata": sample_strata,
        "uncovered_strata": uncovered_strata,
        "metrics": {
            "exact_table_reproduction_rate": rate(tables_exact, len(entries)),
            "exact_tables": tables_exact,
            "row_positions_compared": rows_compared,
            "label_reproduction_rate": rate(labels_exact, rows_compared),
            "numeric_reproduction_rate": rate(numeric_exact, rows_compared),
            "unit_reproduction_rate": rate(units_exact, rows_compared),
            "raw_numeric_provenance_rate": rate(
                raw_numeric_provenance_hits, sampled_rows
            ),
            "raw_label_token_provenance_rate": rate(raw_label_token_hits, sampled_rows),
            "manifest_reproduction_rate": rate(
                metadata_fields_exact, metadata_fields_compared
            ),
        },
        "failure_count": len(failures),
        "failures": failures,
    }
    return report, failures


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit a deterministic stratified sample against source TXT content."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed_csv"))
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--sample-size", type=int, default=160)
    parser.add_argument("--seed", default="vifinqa-phase2-content-audit-v1")
    parser.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    report, failures = audit_content_sample(
        args.output_dir,
        args.root,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
