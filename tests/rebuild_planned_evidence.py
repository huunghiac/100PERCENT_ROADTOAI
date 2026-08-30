"""Rebuild a question subset's evidence from raw reports.

This utility is intentionally driven by ``QuestionPlanner`` output.  IDs only
select scenarios to run; there are no ID-specific metrics, answers, paths, or
formulas in this script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data_extractor import process_all_reports  # noqa: E402
from src.question_planner import QuestionPlanner  # noqa: E402
from src.retriever import TableRetriever  # noqa: E402


def _parse_ids(value: str) -> set[int]:
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def planned_pairs(questions_path: Path, ids: set[int]) -> list[tuple[str, int]]:
    resolver = TableRetriever()
    planner = QuestionPlanner(entity_resolver=resolver)
    pairs: set[tuple[str, int]] = set()
    seen: set[int] = set()
    with questions_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            question_id = int(record["id"])
            if question_id not in ids:
                continue
            seen.add(question_id)
            plan = planner.analyze(str(record["question"]))
            years = sorted(set(sum(plan.metric_years.values(), [])) or set(plan.years))
            pairs.update((ticker, int(year)) for ticker in plan.tickers for year in years)
    missing = sorted(ids - seen)
    if missing:
        raise ValueError(f"Question IDs not present in source: {missing}")
    return sorted(pairs)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Re-extract all ticker/year evidence required by planned questions."
    )
    parser.add_argument("--ids", required=True, type=_parse_ids, help="Comma-separated question IDs")
    parser.add_argument(
        "--questions", type=Path, default=ROOT / "data/raw_vifinqa/questions.jsonl"
    )
    parser.add_argument(
        "--raw-dir", type=Path, default=ROOT / "data/raw_vifinqa/financial_statements"
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    pairs = planned_pairs(args.questions, args.ids)
    totals = {"pairs": len(pairs), "txt_scanned": 0, "csv_written": 0, "errors": 0}
    for index, (ticker, year) in enumerate(pairs, 1):
        stats = process_all_reports(
            args.raw_dir,
            args.output_dir,
            ticker=ticker,
            year=year,
        )
        totals["txt_scanned"] += stats.txt_scanned
        totals["csv_written"] += stats.csv_written
        totals["errors"] += stats.errors
        print(
            f"[{index}/{len(pairs)}] {ticker} {year}: "
            f"txt={stats.txt_scanned} csv={stats.csv_written} errors={stats.errors}",
            flush=True,
        )
    print(json.dumps(totals, ensure_ascii=False))
    return 1 if totals["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
