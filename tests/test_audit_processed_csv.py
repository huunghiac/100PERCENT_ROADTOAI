import csv
import json
import tempfile
import unittest
from pathlib import Path

from tests.audit_processed_csv import EXPECTED_COLUMNS, audit


class ProcessedCsvAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.output = self.root / "data" / "processed_csv"
        self.output.mkdir(parents=True)
        self.source = self.root / "data" / "raw_vifinqa" / "financial_statements" / "AAA.txt"
        self.source.parent.mkdir(parents=True)
        self.source.write_text("source", encoding="utf-8")
        (self.output / "_rejected_cells.jsonl").write_text("", encoding="utf-8")
        (self.output / "_quarantine.jsonl").write_text("", encoding="utf-8")

    def tearDown(self):
        self.temporary.cleanup()

    def _relative(self, path):
        return path.relative_to(self.root).as_posix()

    def _write_csv(self, name="AAA_2023_BangCanDoiKeToan_consolidated.csv", rows=None):
        path = self.output / name
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(EXPECTED_COLUMNS)
            writer.writerows(rows or [["Tiền", "12345678901234567890.25", "VND"], ["Lỗ", "-1.5", "VND"]])
        return path

    def _entry(self, csv_path, **updates):
        entry = {
            "csv_path": self._relative(csv_path),
            "source_txt": self._relative(self.source),
            "ticker": "AAA",
            "company_name": "Công ty AAA",
            "report_year": 2023,
            "report_type": "consolidated",
            "table_title": "Bảng cân đối kế toán",
            "table_slug": "BangCanDoiKeToan",
            "unit": "VND",
            "value_period": "31/12/2023",
            "parser": "html",
            "row_count": 2,
            "warnings": [],
            "value_column_method": "exact_report_year",
            "value_column_header": "Tại ngày | 31/12/2023 | VND",
            "value_column_confidence": "high",
            "candidate_columns": [{"index": 1, "score": 140.0, "header": "31/12/2023"}],
            "logical_table_id": "logical-aaa-2023-balance",
            "unit_source": "header",
            "unit_confidence": "high",
            "source_table_index": 7,
        }
        entry.update(updates)
        return entry

    def _write_manifest(self, entries):
        with (self.output / "_manifest.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for entry in entries:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _reasons(self, failures):
        return {reason for _, reason in failures}

    def test_full_streaming_audit_accepts_valid_artifacts(self):
        csv_path = self._write_csv()
        self._write_manifest([self._entry(csv_path)])
        rejected = {
            "source_txt": self._relative(self.source),
            "ticker": "AAA",
            "report_year": 2023,
            "table_title": "Bảng cân đối kế toán",
            "source_table_index": 7,
            "source_row": 12,
            "source_column": 3,
            "raw_cell": "1.000 2.000",
            "reason": "ambiguous_numeric_concatenation",
            "candidate_split": [["1.000", "2.000"]],
            "confidence": "low",
        }
        quarantine = {
            "source_txt": self._relative(self.source),
            "ticker": "AAA",
            "report_year": 2023,
            "report_type": "consolidated",
            "table_title": "Bảng tài chính",
            "table_slug": "BangTaiChinh",
            "source_table_index": 8,
            "reason": "low_value_column_confidence",
            "value_column_method": "score_margin_too_small",
            "value_column_header": "",
            "value_column_confidence": "low",
            "candidate_columns": [],
            "warnings": ["value_column_low_confidence"],
        }
        (self.output / "_rejected_cells.jsonl").write_text(
            json.dumps(rejected, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        (self.output / "_quarantine.jsonl").write_text(
            json.dumps(quarantine, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        result, failures = audit(self.output, self.root, max_workers=2)
        self.assertEqual(failures, [])
        self.assertEqual(result["csv_count"], 1)
        self.assertEqual(result["audited_csv_count"], 1)
        self.assertEqual(result["manifest_rows"], 1)
        self.assertEqual(result["rejected_cells"], 1)
        self.assertEqual(result["quarantined_tables"], 1)

    def test_csv_schema_numeric_row_count_and_filename_are_checked(self):
        csv_path = self.output / "bad-name.csv"
        csv_path.write_text("Chi_tieu,Gia_tri,Wrong\nA,NaN,VND\n", encoding="utf-8")
        self._write_manifest([self._entry(csv_path, row_count=2)])

        _, failures = audit(self.output, self.root, max_workers=1)
        reasons = self._reasons(failures)
        self.assertIn("filename", reasons)
        self.assertIn("schema", reasons)
        self.assertIn("numeric", reasons)
        self.assertIn("row_count", reasons)

    def test_manifest_metadata_exact_duplicate_and_case_collision_are_checked(self):
        csv_path = self._write_csv()
        first = self._entry(csv_path)
        duplicate = dict(first)
        case_variant = dict(first)
        case_variant["csv_path"] = first["csv_path"].swapcase()
        case_variant["logical_table_id"] = "logical-case-variant"
        del first["unit_source"]
        self._write_manifest([first, duplicate, case_variant])

        result, failures = audit(self.output, self.root, max_workers=2)
        reasons = self._reasons(failures)
        self.assertIn("missing_manifest_field:unit_source", reasons)
        self.assertIn("manifest_duplicate_exact", reasons)
        self.assertIn("manifest_case_insensitive_collision", reasons)
        self.assertEqual(result["manifest_duplicates"], 1)
        self.assertEqual(result["manifest_case_collisions"], 1)

    def test_missing_manifest_target_source_and_unmanifested_csv_are_checked(self):
        orphan = self._write_csv("AAA_2023_BaoCaoKetQuaKinhDoanh_consolidated.csv")
        missing = self.output / "AAA_2023_BangCanDoiKeToan_consolidated.csv"
        entry = self._entry(missing, source_txt="data/raw_vifinqa/missing.txt")
        self._write_manifest([entry])

        _, failures = audit(self.output, self.root, max_workers=1)
        reasons = self._reasons(failures)
        self.assertIn("manifest_target_missing", reasons)
        self.assertIn("source_target_missing", reasons)
        self.assertIn("csv_missing_from_manifest", reasons)
        self.assertTrue(orphan.exists())

    def test_low_confidence_export_and_invalid_sidecar_trace_are_rejected(self):
        csv_path = self._write_csv()
        self._write_manifest([self._entry(csv_path, value_column_confidence="low")])
        rejected = {
            "source_txt": self._relative(self.source),
            "ticker": "AAA",
            "report_year": 2023,
            "table_title": "Bảng cân đối kế toán",
            "source_table_index": 0,
            "source_row": -1,
            "source_column": 1,
            "raw_cell": "1.0002.000",
            "reason": "ambiguous",
            "candidate_split": "1.000 | 2.000",
            "confidence": "unknown",
        }
        (self.output / "_rejected_cells.jsonl").write_text(
            json.dumps(rejected, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        quarantine = {
            "source_txt": self._relative(self.source),
            "ticker": "AAA",
            "report_year": 2023,
            "report_type": "bad",
            "table_title": "Bảng",
            "table_slug": "Bang",
            "source_table_index": 1,
            "reason": "ambiguous",
            "value_column_method": "score",
            "value_column_header": "",
            "value_column_confidence": "unknown",
            "candidate_columns": "bad",
            "warnings": "bad",
        }
        (self.output / "_quarantine.jsonl").write_text(
            json.dumps(quarantine, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        _, failures = audit(self.output, self.root, max_workers=1)
        reasons = self._reasons(failures)
        self.assertIn("low_confidence_table_exported", reasons)
        self.assertIn("invalid_rejected_cell_field:source_row", reasons)
        self.assertIn("invalid_rejected_cell_field:candidate_split", reasons)
        self.assertIn("invalid_rejected_cell_field:confidence", reasons)
        self.assertIn("invalid_quarantine_field:report_type", reasons)
        self.assertIn("invalid_quarantine_field:value_column_confidence", reasons)
        self.assertIn("invalid_quarantine_field:candidate_columns", reasons)
        self.assertIn("invalid_quarantine_field:warnings", reasons)

    def test_missing_sidecars_are_reported(self):
        csv_path = self._write_csv()
        self._write_manifest([self._entry(csv_path)])
        (self.output / "_rejected_cells.jsonl").unlink()
        (self.output / "_quarantine.jsonl").unlink()

        _, failures = audit(self.output, self.root, max_workers=1)
        reasons = self._reasons(failures)
        self.assertIn("missing_rejected_cell_sidecar", reasons)
        self.assertIn("missing_quarantine_sidecar", reasons)


if __name__ == "__main__":
    unittest.main()
