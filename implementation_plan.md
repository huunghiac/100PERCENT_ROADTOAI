# Implementation Plan - Evaluator & Full Pipeline Batch Processing

## Overview
The goal is to build a local evaluation script (src/evaluator.py) strictly matching the BTC evaluation specification:
- 3.1 Table Retrieval: Precision, Recall, and Macro F2 score (5 * P * R) / (4 * P + R).
- 3.2 Answer Accuracy: Exact/Tolerance numerical matching.
- 3.3 Execution Accuracy: Code execution success rate.

And optimize src/pipeline.py for batch processing all 1012 questions with incremental checkpointing.

## Types
- GroundTruthItem: Dict with id, question, expected_answer, expected_evidence.
- EvaluationMetrics: Dict with macro_precision, macro_recall, macro_f2, execution_accuracy, answer_accuracy.

## Files

### New Files
1. src/evaluator.py - Evaluation engine implementing BTC formulas 3.1, 3.2, 3.3.
2. data/mock_ground_truth.jsonl - Ground truth benchmark file for testing.

### Modified Files
1. src/pipeline.py - Batch processing with checkpointing every 10 items.
2. TODO.md - Checklist updates.

## Functions

### src/evaluator.py`n- calculate_retrieval_metrics(predicted_paths: list, expected_paths: list) -> tuple`n- compare_answers(predicted, expected, tolerance: float = 1e-3) -> bool`n- evaluate_submission(submission_path: str, ground_truth_path: str, tolerance: float = 1e-3) -> dict`n
### src/pipeline.py`n- 
un_full_pipeline(questions_file, output_json, output_zip, max_questions, checkpoint_interval=10)`n
## Implementation Order
1. Create benchmark dataset data/mock_ground_truth.jsonl.
2. Build src/evaluator.py implementing metrics 3.1, 3.2, 3.3.
3. Upgrade src/pipeline.py with checkpoint saving.
4. Validate pipeline output against mock ground truth using src/evaluator.py.