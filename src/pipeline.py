"""Validated ViFinQA pipeline.

The save contract is intentionally strict:

    final evidence -> final pandas expression -> saved answer

No answer is used to search for a query, zero is a valid result, and complex
questions never fall back to a single statement row.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

_SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from agent import PandasAgent
from complex_solver import ComplexSolver, SolveResult, StructuredSolveFailure
from fallback import try_rule_based_answer
from query_formatter import (
    QueryExecutionError,
    QueryFormatError,
    convert_script_to_expression,
    execute_expression,
    referenced_variables,
)
from question_planner import QuestionPlan, QuestionPlanner, QuestionType
from retriever import EvidenceBundle, TableRetriever
from semantic_validation import SemanticValidationError, validate_answer


class PipelineItemError(RuntimeError):
    """A question failed safely and must not be written to the submission."""

    def __init__(self, stage: str, message: str, *, details: Mapping[str, Any] | None = None):
        super().__init__(message)
        self.stage = stage
        self.message = message
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {"stage": self.stage, "message": self.message, "details": self.details}


@dataclass
class PipelineStats:
    attempted: int = 0
    saved: int = 0
    failed: int = 0
    simple_questions: int = 0
    complex_questions: int = 0
    fallback_simple: int = 0
    fallback_complex: int = 0
    generation_failures: int = 0
    entity_extraction_failures: int = 0
    prompt_truncations: int = 0
    evidence_context_truncations: int = 0
    evidence_simple: list[int] = field(default_factory=list)
    evidence_complex: list[int] = field(default_factory=list)

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 6) if denominator else 0.0

    def report(self) -> dict[str, Any]:
        avg_simple = sum(self.evidence_simple) / len(self.evidence_simple) if self.evidence_simple else 0.0
        avg_complex = sum(self.evidence_complex) / len(self.evidence_complex) if self.evidence_complex else 0.0
        return {
            "attempted": self.attempted,
            "saved": self.saved,
            "failed": self.failed,
            "query_execution_rate_saved": 1.0 if self.saved else 0.0,
            "answer_query_consistency_saved": 1.0 if self.saved else 0.0,
            "empty_evidence_saved": 0,
            "fallback_rate_simple": self._rate(self.fallback_simple, self.simple_questions),
            "fallback_rate_complex": self._rate(self.fallback_complex, self.complex_questions),
            "generation_failure": self.generation_failures,
            "entity_extraction_failure": self.entity_extraction_failures,
            "prompt_truncation_count": self.prompt_truncations,
            "evidence_context_truncation_count": self.evidence_context_truncations,
            "average_evidence_count_simple": round(avg_simple, 6),
            "average_evidence_count_complex": round(avg_complex, 6),
        }


def _doc_id_from_source_txt(source_txt: str) -> str:
    if not source_txt:
        return ""
    parts = source_txt.replace("\\", "/").split("/")
    for part in reversed(parts):
        if "_financial_statements_" in part and not part.endswith(".txt"):
            return part
    return re.sub(r"_extracted\.txt$", "", parts[-1])


def _mapped_variable(path: str, index: int, mapping: Mapping[str, str] | None) -> str:
    if mapping:
        normalized = path.replace("\\", "/")
        variable = mapping.get(path) or mapping.get(normalized)
        if variable:
            return variable
        raise PipelineItemError(
            "evidence_mapping", f"Evidence path has no official dataframe variable: {path}"
        )
    return f"df{index + 1}"


def _build_submission_fields(
    csv_paths: Sequence[str],
    manifest: Mapping[str, dict] | None,
    retriever: TableRetriever | None = None,
    path_to_variable: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Build synchronized evidence/docs/tables without renumbering variables."""

    items: list[dict[str, Any]] = []
    manifest = manifest or {}
    for index, csv_path in enumerate(csv_paths):
        variable = _mapped_variable(csv_path, index, path_to_variable)
        match = re.fullmatch(r"df([1-9]\d*)", variable)
        if not match:
            raise PipelineItemError("evidence_mapping", f"Invalid dataframe variable {variable!r}")
        flat_path = f"data/{os.path.basename(csv_path)}"
        normalized = csv_path.replace("\\", "/")
        if retriever is not None:
            entry = retriever.get_manifest_entry(normalized)
        else:
            entry = manifest.get(normalized, manifest.get(csv_path, manifest.get(flat_path, {})))
        source_txt = entry.get("source_txt", "")
        table_index = entry.get("source_table_index")
        doc_id = _doc_id_from_source_txt(source_txt)
        table_entry = ""
        if doc_id and table_index is not None:
            line_number = (
                retriever.get_source_line_number(doc_id, table_index)
                if retriever is not None
                else table_index
            )
            table_entry = f"{doc_id}|{line_number}"
        items.append({
            "var_name": variable,
            "var_num": match.group(1),
            "evidence": {"variable": variable, "csv_path": flat_path},
            "doc_id": doc_id,
            "table_entry": table_entry,
            "source_path": normalized,
            "metadata_status": entry.get("metadata_status", "manifest" if entry else "missing"),
        })
    return items


def _resolve_csv_path(path: str, csv_dir: str = "data/processed_csv") -> str | None:
    candidates = [path]
    if path.startswith("data/"):
        candidates.append(path.replace("data/", "", 1))
    basename = os.path.basename(path)
    ticker = basename.split("_")[0] if "_" in basename else ""
    candidates.extend((
        os.path.join(csv_dir, ticker, basename),
        os.path.join("data", basename),
    ))
    return next((candidate for candidate in candidates if os.path.isfile(candidate)), None)


def _load_dataframes(
    variable_to_path: Mapping[str, str], csv_dir: str = "data/processed_csv"
) -> dict[str, pd.DataFrame]:
    frames: dict[str, pd.DataFrame] = {}
    for variable, path in variable_to_path.items():
        resolved = _resolve_csv_path(path, csv_dir)
        if resolved is None:
            raise PipelineItemError(
                "evidence_load", f"Evidence file for {variable} does not exist", details={"path": path}
            )
        try:
            frames[variable] = pd.read_csv(resolved)
        except Exception as exc:
            raise PipelineItemError(
                "evidence_load",
                f"Cannot load {variable}: {type(exc).__name__}: {exc}",
                details={"path": path},
            ) from exc
    return frames


def _bundle_from_paths(paths: Sequence[str]) -> EvidenceBundle:
    bundle = EvidenceBundle()
    for path in paths:
        normalized = path.replace("\\", "/")
        if normalized in bundle.path_to_variable:
            continue
        variable = f"df{len(bundle.paths) + 1}"
        bundle.paths.append(normalized)
        bundle.path_to_variable[normalized] = variable
        bundle.variable_to_path[variable] = normalized
    return bundle


def _prune_bundle(bundle: EvidenceBundle, variables: set[str]) -> EvidenceBundle:
    """Prune paths only after query variables are final; never renumber dfN."""

    final = EvidenceBundle()
    for path in bundle.paths:
        variable = bundle.path_to_variable.get(path)
        if variable not in variables:
            continue
        final.paths.append(path)
        final.path_to_variable[path] = variable
        final.variable_to_path[variable] = path
    for key, paths in bundle.metric_paths.items():
        kept = [path for path in paths if path in final.path_to_variable]
        if kept:
            final.metric_paths[key] = kept
    for ticker, year_map in bundle.structured.items():
        for year, statement_map in year_map.items():
            for statement, paths in statement_map.items():
                kept = [path for path in paths if path in final.path_to_variable]
                if kept:
                    final.structured.setdefault(ticker, {}).setdefault(year, {})[statement] = kept
    # Retrieval misses that did not prevent deterministic execution are logged
    # in the solve/failure report, not attached to final evidence.
    return final


def _numeric(value: Any) -> float | int:
    number = float(value)
    if not math.isfinite(number):
        raise PipelineItemError("answer", "Answer is NaN or infinity")
    if number.is_integer() and abs(number) < 1e15:
        return int(number)
    return number


def _checkpoint_item_valid(
    item: Mapping[str, Any], csv_dir: str = "data/processed_csv"
) -> bool:
    """Trust only items previously emitted by this validated save gate."""

    if not bool((item.get("validation") or {}).get("valid")):
        return False
    evidence = item.get("evidence") or []
    query = str(item.get("pandas_query") or "")
    mapping = {
        str(record.get("variable") or ""): str(record.get("csv_path") or "")
        for record in evidence
    }
    if not mapping or referenced_variables(query) != set(mapping):
        return False
    try:
        result = execute_expression(query, _load_dataframes(mapping, csv_dir))
        return math.isclose(float(result), float(item.get("answer")), rel_tol=1e-12, abs_tol=1e-9)
    except Exception:
        return False


def _finalize_item(
    *,
    question_id: int,
    question: str,
    plan: QuestionPlan,
    bundle: EvidenceBundle,
    query_or_script: str,
    retriever: TableRetriever,
    facts: Sequence[Any] = (),
    solver_validation: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], list[str]]:
    if not bundle.paths:
        raise PipelineItemError("finalize", "Final evidence is empty")
    full_frames = _load_dataframes(bundle.variable_to_path, retriever.csv_dir)
    try:
        expression = convert_script_to_expression(query_or_script, full_frames)
    except (QueryFormatError, QueryExecutionError) as exc:
        raise PipelineItemError("query_format", str(exc)) from exc
    variables = referenced_variables(expression)
    if not variables:
        raise PipelineItemError("query_mapping", "Final query does not reference evidence")
    final_bundle = _prune_bundle(bundle, variables)
    if set(final_bundle.variable_to_path) != variables:
        raise PipelineItemError(
            "query_mapping",
            "Final query and evidence variables differ",
            details={
                "query_variables": sorted(variables),
                "evidence_variables": sorted(final_bundle.variable_to_path),
            },
        )
    final_frames = _load_dataframes(final_bundle.variable_to_path, retriever.csv_dir)
    answer = _numeric(execute_expression(expression, final_frames))
    final_facts = [fact for fact in facts if getattr(fact, "variable", "") in variables]
    report = validate_answer(
        answer,
        expression,
        final_bundle,
        plan=plan,
        facts=final_facts,
        dataframes=final_frames,
    )
    try:
        report.require_valid()
    except SemanticValidationError as exc:
        raise PipelineItemError(
            "semantic_validation", str(exc), details=report.to_dict()
        ) from exc

    table_items = _build_submission_fields(
        final_bundle.paths,
        retriever.manifest,
        retriever=retriever,
        path_to_variable=final_bundle.path_to_variable,
    )
    evidence = [item["evidence"] for item in table_items]
    relevant_tables = [item["table_entry"] for item in table_items if item["table_entry"]]
    relevant_docs = list(dict.fromkeys(item["doc_id"] for item in table_items if item["doc_id"]))
    validation = report.to_dict()
    if solver_validation:
        validation["solver"] = dict(solver_validation)
    validation["query_is_source_of_truth"] = True
    validation["single_row_fallback_used"] = bool(
        solver_validation and solver_validation.get("single_row_fallback_used")
    )
    item = {
        "id": int(question_id),
        "question": question,
        "answer": answer,
        "relevant_docs": relevant_docs,
        "relevant_tables": relevant_tables,
        "evidence": evidence,
        "pandas_query": expression,
        "validation": validation,
        "question_plan": plan.to_dict(),
    }
    return item, final_bundle.paths


def _solve_complex_with_retry(
    plan: QuestionPlan,
    retriever: TableRetriever,
    solver: ComplexSolver,
) -> tuple[SolveResult, EvidenceBundle, int]:
    last_failure: StructuredSolveFailure | None = None
    for attempt, per_metric_k in enumerate((1, 3), 1):
        bundle = retriever.retrieve_plan(plan, per_metric_k=per_metric_k)
        try:
            return solver.solve(plan, bundle), bundle, attempt
        except StructuredSolveFailure as exc:
            last_failure = exc
            # Retry belongs at metric retrieval.  A formula/planner-domain
            # failure without missing facts will not improve with more tables.
            if attempt == 1 and exc.missing:
                continue
            raise
    assert last_failure is not None
    raise last_failure


def run_full_pipeline(
    questions_file: str = "data/raw_vifinqa/questions.jsonl",
    output_json: str = "submission.json",
    output_zip: str = "submission.zip",
    max_questions: int | None = None,
    checkpoint_interval: int = 10,
    agent: PandasAgent | None = None,
    *,
    question_ids: Iterable[int] | None = None,
    deterministic_only: bool = False,
    csv_dir: str = "data/processed_csv",
    manifest_path: str | None = None,
) -> dict[str, Any]:
    """Run the planner-routed pipeline and return its measured quality report."""

    # Windows runners may expose a legacy CP1252 console even though all
    # corpus files are UTF-8.  Logging must never abort financial execution.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError):
            pass
    print("=== Khởi động ViFinQA Pipeline (validated) ===")
    selected_ids = {int(item) for item in question_ids} if question_ids is not None else None
    retriever = TableRetriever(
        csv_dir=csv_dir,
        manifest_path=manifest_path or os.path.join(csv_dir, "_manifest.jsonl"),
    )
    planner = QuestionPlanner(entity_resolver=retriever)
    solver = ComplexSolver()
    stats = PipelineStats()
    results_map: dict[int, dict[str, Any]] = {}
    failures: dict[int, dict[str, Any]] = {}
    used_csv_paths: set[str] = set()

    if os.path.exists(output_json):
        try:
            with open(output_json, "r", encoding="utf-8") as handle:
                for item in json.load(handle):
                    question_id = int(item.get("id"))
                    if _checkpoint_item_valid(item, retriever.csv_dir):
                        results_map[question_id] = item
                        for record in item.get("evidence", []):
                            resolved = _resolve_csv_path(record.get("csv_path", ""), retriever.csv_dir)
                            if resolved:
                                used_csv_paths.add(resolved.replace("\\", "/"))
            print(f"[Checkpoint] retained {len(results_map)} validated items.")
        except Exception as exc:
            print(f"[Checkpoint] ignored unreadable checkpoint: {exc}")

    if not os.path.exists(questions_file):
        alternatives = (
            "data/raw_vifinqa/questions.jsonl", "questions.jsonl",
            "data/questions.jsonl", "raw_vifinqa/questions.jsonl",
        )
        questions_file = next((path for path in alternatives if os.path.exists(path)), "")
    if not questions_file:
        raise FileNotFoundError("questions.jsonl was not found")
    with open(questions_file, "r", encoding="utf-8") as handle:
        questions = [json.loads(line) for line in handle if line.strip()]
    if selected_ids is not None:
        questions = [item for item in questions if int(item["id"]) in selected_ids]
    if max_questions is not None:
        questions = questions[:max_questions]

    started = time.time()
    for position, record in enumerate(questions, 1):
        question_id, question = int(record["id"]), str(record["question"])
        if question_id in results_map:
            continue
        stats.attempted += 1
        print(f"\n--- [{position}/{len(questions)}] ID={question_id}: {question[:90]} ---")
        plan: QuestionPlan | None = None
        try:
            plan = planner.analyze(question)
            if not plan.tickers or not plan.years:
                stats.entity_extraction_failures += 1
                raise PipelineItemError(
                    "planner", "Planner could not resolve all primary entities/years",
                    details={"tickers": plan.tickers, "years": plan.years},
                )

            if plan.is_complex:
                stats.complex_questions += 1
                result, bundle, retrieval_attempts = _solve_complex_with_retry(plan, retriever, solver)
                item, saved_paths = _finalize_item(
                    question_id=question_id,
                    question=question,
                    plan=plan,
                    bundle=bundle,
                    query_or_script=result.pandas_query,
                    retriever=retriever,
                    facts=result.used_facts,
                    solver_validation={
                        **result.validation,
                        "confidence": result.confidence,
                        "retrieval_attempts": retrieval_attempts,
                        "solver_mode": "complex_deterministic",
                        "single_row_fallback_used": False,
                    },
                )
                stats.evidence_complex.append(len(item["evidence"]))
            else:
                stats.simple_questions += 1
                paths = retriever.retrieve(question)
                bundle = _bundle_from_paths(paths)
                code = ""
                generation_error = None
                used_single_row_fallback = False
                if not deterministic_only:
                    if agent is None:
                        agent = PandasAgent()
                    _, code, generation_error = agent.run_agent(
                        question,
                        bundle.paths,
                        max_retries=2,
                        path_to_variable=bundle.path_to_variable,
                    )
                    if agent.last_prompt_report:
                        if not agent.last_prompt_report.question_preserved:
                            stats.prompt_truncations += 1
                        if agent.last_prompt_report.evidence_truncated:
                            stats.evidence_context_truncations += 1
                if generation_error or "GENERATION_FAILED" in code or not code.strip():
                    if not deterministic_only:
                        stats.generation_failures += 1
                    fallback = try_rule_based_answer(question, bundle.paths, plan=plan)
                    if fallback is None:
                        raise PipelineItemError(
                            "simple_solver",
                            generation_error or "No high-confidence simple fallback",
                        )
                    stats.fallback_simple += 1
                    used_single_row_fallback = True
                    code = fallback.pandas_query
                try:
                    item, saved_paths = _finalize_item(
                        question_id=question_id,
                        question=question,
                        plan=plan,
                        bundle=bundle,
                        query_or_script=code,
                        retriever=retriever,
                        solver_validation={
                            "single_row_fallback_used": used_single_row_fallback,
                            "solver_mode": (
                                "simple_rule_fallback"
                                if used_single_row_fallback
                                else "simple_llm"
                            ),
                        },
                    )
                except PipelineItemError as first_error:
                    # Conversion/semantic failure is a legitimate reason to try
                    # the conservative fallback; answer==0 is deliberately not.
                    fallback = try_rule_based_answer(question, bundle.paths, plan=plan)
                    if fallback is None or fallback.pandas_query == code:
                        raise first_error
                    stats.fallback_simple += 1
                    used_single_row_fallback = True
                    item, saved_paths = _finalize_item(
                        question_id=question_id,
                        question=question,
                        plan=plan,
                        bundle=bundle,
                        query_or_script=fallback.pandas_query,
                        retriever=retriever,
                        solver_validation={
                            "single_row_fallback_used": True,
                            "solver_mode": "simple_rule_fallback",
                        },
                    )
                stats.evidence_simple.append(len(item["evidence"]))

            results_map[question_id] = item
            used_csv_paths.update(saved_paths)
            stats.saved += 1
            print(f"  saved answer={item['answer']} evidence={len(item['evidence'])}")
        except StructuredSolveFailure as exc:
            stats.failed += 1
            failures[question_id] = {
                "id": question_id,
                "question": question,
                "plan": plan.to_dict() if plan is not None else None,
                **exc.to_dict(),
            }
            print(f"  structured failure [{exc.code}]: {exc.message}")
        except (PipelineItemError, QueryFormatError, QueryExecutionError) as exc:
            stats.failed += 1
            detail = exc.to_dict() if isinstance(exc, PipelineItemError) else {
                "stage": "query", "message": str(exc), "details": {}
            }
            failures[question_id] = {
                "id": question_id,
                "question": question,
                "plan": plan.to_dict() if plan is not None else None,
                "single_row_fallback_allowed": not getattr(plan, "is_complex", True),
                **detail,
            }
            print(f"  failed safely [{detail['stage']}]: {detail['message']}")
        except Exception as exc:  # Preserve progress while surfacing unexpected defects.
            stats.failed += 1
            failures[question_id] = {
                "id": question_id,
                "question": question,
                "plan": plan.to_dict() if plan is not None else None,
                "stage": "unexpected",
                "message": f"{type(exc).__name__}: {exc}",
                "single_row_fallback_allowed": False if plan and plan.is_complex else True,
            }
            print(f"  unexpected failure: {type(exc).__name__}: {exc}")

        if stats.attempted % max(1, checkpoint_interval) == 0:
            _save_json(results_map, output_json)
            _save_failures(failures, output_json)

    _save_json(results_map, output_json)
    failure_path = _save_failures(failures, output_json)
    quality = stats.report()
    quality["elapsed_seconds"] = round(time.time() - started, 3)
    quality["failure_report"] = failure_path
    quality_path = str(Path(output_json).with_suffix(".quality.json"))
    with open(quality_path, "w", encoding="utf-8") as handle:
        json.dump(quality, handle, ensure_ascii=False, indent=2)

    if output_zip:
        with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(output_json, arcname=os.path.basename(output_json))
            for csv_path in sorted(used_csv_paths):
                resolved = _resolve_csv_path(csv_path, retriever.csv_dir)
                if resolved:
                    archive.write(resolved, arcname=f"data/{os.path.basename(resolved)}")
    print(f"\n[Quality] {json.dumps(quality, ensure_ascii=False)}")
    return quality


def _save_json(results_map: Mapping[int, dict[str, Any]], path: str) -> None:
    ordered = [results_map[key] for key in sorted(results_map)]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(ordered, handle, ensure_ascii=False, indent=2)


def _save_failures(failures: Mapping[int, dict[str, Any]], output_json: str) -> str:
    path = str(Path(output_json).with_suffix(".failures.json"))
    ordered = [failures[key] for key in sorted(failures)]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(ordered, handle, ensure_ascii=False, indent=2)
    return path


if __name__ == "__main__":
    maximum = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run_full_pipeline(max_questions=maximum)


__all__ = [
    "PipelineItemError",
    "PipelineStats",
    "_build_submission_fields",
    "_bundle_from_paths",
    "_finalize_item",
    "_prune_bundle",
    "run_full_pipeline",
]
