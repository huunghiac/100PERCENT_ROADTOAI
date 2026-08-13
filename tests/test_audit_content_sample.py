import csv
import tempfile
import unittest
from decimal import Decimal
from pathlib import Path

from src.data_extractor import process_all_reports
from tests.audit_content_sample import (
    ManifestCandidate,
    audit_content_sample,
    select_stratified,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
RAW_FIXTURE = REPO_ROOT / "tests" / "fixtures" / "e2e" / "raw" / "financial_statements"


class ContentSampleAuditTests(unittest.TestCase):
    def test_stratified_selection_is_deterministic_and_covers_dimensions(self):
        candidates = [
            ManifestCandidate(
                line_number=index,
                csv_path=f"out/{ticker}_{year}_{report_type}_{table_slug}.csv",
                source_txt=f"raw/{ticker}_{year}.txt",
                ticker=ticker,
                report_year=year,
                report_type=report_type,
                value_column_confidence=confidence,
                table_slug=table_slug,
            )
            for index, (ticker, year, report_type, confidence, table_slug) in enumerate(
                (
                    ("AAA", 2021, "consolidated", "high", "BangCanDoiKeToan"),
                    ("AAA", 2022, "separate", "medium", "GeneralNote"),
                    ("ACB", 2023, "aggregated", "high", "BaoCaoKetQuaKinhDoanh"),
                    ("VNM", 2022, "consolidated", "medium", "GeneralNote"),
                    ("VJC", 2021, "separate", "high", "BaoCaoLuuChuyenTienTe"),
                ),
                1,
            )
        ]
        first = select_stratified(candidates, len(candidates), "fixed-seed")
        second = select_stratified(candidates, len(candidates), "fixed-seed")
        self.assertEqual(first, second)
        self.assertEqual({item.ticker for item in first}, {"AAA", "ACB", "VNM", "VJC"})
        self.assertEqual({item.report_type for item in first}, {"consolidated", "separate", "aggregated"})
        self.assertEqual({item.table_class for item in first}, {"core", "general"})

    def test_audit_reconstructs_fixture_csv_from_source(self):
        with tempfile.TemporaryDirectory(prefix="vifinqa-content-audit-") as temporary:
            output_dir = Path(temporary) / "processed"
            stats = process_all_reports(RAW_FIXTURE, output_dir)
            self.assertEqual(stats.errors, 0)
            report, failures = audit_content_sample(
                output_dir, REPO_ROOT, sample_size=10, seed="fixture"
            )

        self.assertEqual(failures, [])
        self.assertEqual(report["sampled_tables"], stats.csv_written)
        metrics = report["metrics"]
        self.assertEqual(metrics["exact_table_reproduction_rate"], 1.0)
        self.assertEqual(metrics["numeric_reproduction_rate"], 1.0)
        self.assertEqual(metrics["raw_numeric_provenance_rate"], 1.0)
        self.assertEqual(metrics["manifest_reproduction_rate"], 1.0)

    def test_audit_detects_numeric_content_tampering(self):
        with tempfile.TemporaryDirectory(prefix="vifinqa-content-tamper-") as temporary:
            output_dir = Path(temporary) / "processed"
            process_all_reports(RAW_FIXTURE, output_dir)
            csv_path = sorted(output_dir.glob("*.csv"))[0]
            with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.reader(handle))
            rows[1][1] = str(Decimal(rows[1][1]) + Decimal("1"))
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                csv.writer(handle).writerows(rows)

            report, failures = audit_content_sample(
                output_dir, REPO_ROOT, sample_size=10, seed="fixture"
            )

        self.assertTrue(any(item["check"] == "record_sequence" for item in failures))
        self.assertLess(report["metrics"]["numeric_reproduction_rate"], 1.0)
        self.assertLess(report["metrics"]["raw_numeric_provenance_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
