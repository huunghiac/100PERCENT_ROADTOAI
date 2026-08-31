"""Validate BTC schema and expected-ID completeness."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = str(ROOT / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from submission_contract import (  # noqa: E402
    account_ids,
    load_questions,
    validate_items,
    validate_submission_zip,
)


def _load_json_list(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, list):
        raise ValueError(f"{path}: root must be a JSON array")
    return value


def validate(
    path: str = "submission.json",
    *,
    questions_path: str | None = None,
    failures_path: str | None = None,
    require_complete: bool = False,
) -> bool:
    if not os.path.exists(path):
        print(f"[SCHEMA FAIL] {path} does not exist")
        return False
    try:
        items = _load_json_list(path)
        questions = load_questions(questions_path) if questions_path else None
        failures = _load_json_list(failures_path) if failures_path and os.path.exists(failures_path) else []
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"[SCHEMA FAIL] {exc}")
        return False

    errors = validate_items(items, questions)
    if errors:
        print(f"[SCHEMA FAIL] {len(errors)} errors")
        for error in errors[:20]:
            print(f"  - {error}")
    else:
        print(f"[SCHEMA PASS] {len(items)} items; exact 7-field BTC schema")

    completeness_pass = True
    if questions is not None:
        accounting = account_ids(questions, items, failures)
        completeness_pass = accounting.submission_ready
        label = "PASS" if completeness_pass else "FAIL"
        print(f"[COMPLETENESS {label}] {json.dumps(accounting.to_dict(), ensure_ascii=False)}")
    elif require_complete:
        completeness_pass = False
        print("[COMPLETENESS FAIL] --questions is required with --require-complete")
    else:
        print("[COMPLETENESS SKIP] expected question universe not provided")

    package_pass = not errors and (completeness_pass or not require_complete)
    print(f"[PACKAGE {'PASS' if package_pass else 'FAIL'}]")
    return not errors and (completeness_pass if require_complete else True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", default="submission.json")
    parser.add_argument("--questions")
    parser.add_argument("--failures")
    parser.add_argument("--require-complete", action="store_true")
    parser.add_argument("--zip", dest="zip_path", help="Replay and validate a ZIP in isolation")
    args = parser.parse_args(argv)
    json_valid = validate(
        args.path,
        questions_path=args.questions,
        failures_path=args.failures,
        require_complete=args.require_complete,
    )
    zip_valid = True
    if args.zip_path:
        report = validate_submission_zip(args.zip_path)
        zip_valid = report.valid
        print(f"[ZIP REPLAY {'PASS' if zip_valid else 'FAIL'}] {json.dumps(report.to_dict(), ensure_ascii=False)}")
    return 0 if json_valid and zip_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
