import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data_extractor import CSV_COLUMNS, process_all_reports


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "e2e"
RAW_STATEMENTS = FIXTURE_ROOT / "raw" / "financial_statements"
EXPECTED_DIR = FIXTURE_ROOT / "expected"


def load_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class ExtractorEndToEndTests(unittest.TestCase):
    maxDiff = None

    def test_process_all_reports_matches_paired_csv_and_manifest_fixtures(self):
        source_files = sorted(RAW_STATEMENTS.rglob("*.txt"))
        expected_csvs = sorted(EXPECTED_DIR.glob("*.csv"))
        expected_manifest = load_jsonl(EXPECTED_DIR / "manifest_subset.jsonl")

        with tempfile.TemporaryDirectory(prefix="vifinqa-e2e-") as temporary:
            output_dir = Path(temporary) / "processed"
            stats = process_all_reports(RAW_STATEMENTS, output_dir)

            self.assertEqual(stats.txt_scanned, len(source_files))
            self.assertEqual(stats.tables_detected, 3)
            self.assertEqual(stats.csv_written, len(expected_csvs))
            self.assertEqual(stats.tables_skipped, 0)
            self.assertEqual(stats.errors, 0)

            actual_csvs = sorted(output_dir.glob("*.csv"))
            self.assertEqual(
                [path.name for path in actual_csvs],
                [path.name for path in expected_csvs],
            )
            for expected_path in expected_csvs:
                actual_path = output_dir / expected_path.name
                actual = pd.read_csv(actual_path)
                expected = pd.read_csv(expected_path)
                self.assertEqual(tuple(actual.columns), CSV_COLUMNS)
                self.assertTrue(pd.api.types.is_numeric_dtype(actual["Gia_tri"]))
                pd.testing.assert_frame_equal(actual, expected, check_dtype=False)

            actual_manifest = load_jsonl(output_dir / "_manifest.jsonl")
            actual_by_name = {
                Path(str(entry["csv_path"])).name: entry for entry in actual_manifest
            }
            self.assertEqual(set(actual_by_name), {entry["csv_name"] for entry in expected_manifest})

            exact_fields = (
                "ticker",
                "company_name",
                "report_year",
                "report_type",
                "table_title",
                "table_slug",
                "unit",
                "parser",
                "row_count",
            )
            for expected in expected_manifest:
                actual = actual_by_name[str(expected["csv_name"])]
                self.assertEqual(
                    Path(str(actual["source_txt"])).name,
                    expected["source_name"],
                )
                for field in exact_fields:
                    self.assertEqual(actual[field], expected[field], field)
                self.assertIn(
                    str(expected["value_period_contains"]),
                    str(actual["value_period"]),
                )
                for warning in expected["warnings_contains"]:
                    self.assertIn(warning, actual["warnings"])


if __name__ == "__main__":
    unittest.main()
