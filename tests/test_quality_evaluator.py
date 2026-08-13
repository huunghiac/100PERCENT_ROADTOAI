import unittest
from pathlib import Path

from tests.evaluate_extractor_quality import (
    DEFAULT_CASES_PATH,
    DEFAULT_GOLD_DIR,
    evaluate_quality,
    load_gold_cases,
)


class ExtractorQualityEvaluatorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases = load_gold_cases(DEFAULT_CASES_PATH)
        cls.report = evaluate_quality(DEFAULT_GOLD_DIR, DEFAULT_CASES_PATH)

    def test_gold_contract_covers_every_raw_table_and_required_scenarios(self):
        self.assertEqual(len(self.cases), 24)
        sources = {str(case["source"]) for case in self.cases}
        self.assertEqual(sources, {path.name for path in Path(DEFAULT_GOLD_DIR).glob("*.txt")})
        self.assertEqual(self.report["counts"]["raw_tables"], 24)
        self.assertEqual(self.report["counts"]["expected_accepted_tables"], 20)
        self.assertEqual(self.report["counts"]["expected_rejected_tables"], 4)

        scenarios = {
            scenario for case in self.cases for scenario in case.get("scenarios", [])
        }
        required = {
            "ordinary_company",
            "bank",
            "consolidated",
            "separate",
            "aggregated",
            "multi_level_header",
            "continuation",
            "continuation_marker_middle",
            "two_periods",
            "missing_unit",
            "ocr_concatenation",
            "non_financial",
        }
        self.assertTrue(required <= scenarios, required - scenarios)

    def test_quality_thresholds_and_logical_table_integrity(self):
        metrics = self.report["metrics"]
        thresholds = {
            "table_classification_precision": 95.0,
            "table_classification_recall": 95.0,
            "value_column_accuracy": 98.0,
            "numeric_value_accuracy": 99.0,
            "unit_extraction_precision": 95.0,
            "unit_extraction_recall": 95.0,
            "unknown_unit_accuracy": 100.0,
            "core_table_coverage": 95.0,
        }
        for metric, threshold in thresholds.items():
            self.assertGreaterEqual(
                metrics[metric],
                threshold,
                f"{metric}: {metrics[metric]} < {threshold}; failures={self.report['failures']}",
            )
        self.assertEqual(self.report["counts"]["fragmented_logical_tables"], 0)
        self.assertEqual(self.report["counts"]["wrong_merge_outputs"], 0)


if __name__ == "__main__":
    unittest.main()
