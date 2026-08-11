import json
import math
import argparse
from typing import List, Dict, Any, Tuple

def calculate_retrieval_metrics(predicted_paths: List[str], expected_paths: List[str]) -> Tuple[float, float, float]:
    """
    Calculate Precision, Recall, and F2 score for table retrieval.
    F2 score formula: (5 * Precision * Recall) / (4 * Precision + Recall)
    """
    # Normalize paths (strip leading/trailing whitespace, unify slashes)
    pred_set = set(p.replace("\\", "/").strip() for p in predicted_paths if p)
    exp_set = set(p.replace("\\", "/").strip() for p in expected_paths if p)

    if not exp_set:
        recall = 1.0 if not pred_set else 0.0
        precision = 1.0 if not pred_set else 0.0
    else:
        intersection = pred_set.intersection(exp_set)
        precision = len(intersection) / len(pred_set) if pred_set else 0.0
        recall = len(intersection) / len(exp_set)

    denom = 4 * precision + recall
    f2 = (5 * precision * recall) / denom if denom > 0 else 0.0

    return precision, recall, f2

def compare_answers(predicted: Any, expected: Any, tolerance: float = 1e-3) -> bool:
    """
    Compare predicted and expected answers with absolute or relative numerical tolerance.
    """
    if predicted is None:
        return False

    # Try numeric comparison
    try:
        pred_num = float(predicted)
        exp_num = float(expected)
        
        # Absolute difference check
        if abs(pred_num - exp_num) <= tolerance:
            return True
            
        # Relative difference check for large numbers
        if abs(exp_num) > 1e-9 and (abs(pred_num - exp_num) / abs(exp_num)) <= tolerance:
            return True
            
        return False
    except (ValueError, TypeError):
        # String fallback after normalization
        pred_str = str(predicted).strip().lower()
        exp_str = str(expected).strip().lower()
        return pred_str == exp_str

def evaluate_submission(submission_path: str, ground_truth_path: str, tolerance: float = 1e-3) -> Dict[str, float]:
    """
    Evaluate submission JSON against ground truth JSONL file.
    Returns metrics matching BTC specifications (Precision, Recall, F2, Answer Accuracy, Execution Accuracy).
    """
    # Load submission
    with open(submission_path, "r", encoding="utf-8") as f:
        sub_data = json.load(f)

    sub_dict = {item["id"]: item for item in sub_data}

    # Load ground truth
    gt_items = []
    with open(ground_truth_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                gt_items.append(json.loads(line))

    if not gt_items:
        return {
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "macro_f2": 0.0,
            "answer_accuracy": 0.0,
            "execution_accuracy": 0.0,
            "total_questions": 0
        }

    total_precision = 0.0
    total_recall = 0.0
    total_f2 = 0.0
    correct_answers = 0
    successful_executions = 0

    for gt in gt_items:
        q_id = gt["id"]
        exp_answer = gt.get("expected_answer")
        exp_evidence = [ev.get("csv_path", "") for ev in gt.get("expected_evidence", [])]

        sub_item = sub_dict.get(q_id, {})
        pred_answer = sub_item.get("answer")
        pred_evidence = [ev.get("csv_path", "") for ev in sub_item.get("evidence", [])]

        # 1. Retrieval Metrics
        p, r, f2 = calculate_retrieval_metrics(pred_evidence, exp_evidence)
        total_precision += p
        total_recall += r
        total_f2 += f2

        # 2. Execution Accuracy (answer is present and not None / Error indicator)
        if pred_answer is not None and not str(pred_answer).startswith("Lỗi:"):
            successful_executions += 1

        # 3. Answer Accuracy
        if compare_answers(pred_answer, exp_answer, tolerance=tolerance):
            correct_answers += 1

    total_count = len(gt_items)
    metrics = {
        "macro_precision": round(total_precision / total_count, 4),
        "macro_recall": round(total_recall / total_count, 4),
        "macro_f2": round(total_f2 / total_count, 4),
        "answer_accuracy": round(correct_answers / total_count, 4),
        "execution_accuracy": round(successful_executions / total_count, 4),
        "total_questions": total_count
    }

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ViFinQA pipeline submissions.")
    parser.add_argument("--submission", type=str, default="submission.json", help="Path to submission.json")
    parser.add_argument("--ground_truth", type=str, default="data/mock_ground_truth.jsonl", help="Path to ground truth jsonl")
    parser.add_argument("--tolerance", type=float, default=1e-3, help="Numerical tolerance for accuracy")
    
    args = parser.parse_args()
    results = evaluate_submission(args.submission, args.ground_truth, tolerance=args.tolerance)
    
    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"{k}: {v}")
