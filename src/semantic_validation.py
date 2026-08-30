"""Semantic validation gate for final ViFinQA answers.

Executing a pandas expression proves only that the expression is syntactically
valid.  This module validates the stronger contract used by the pipeline:

``final evidence -> final pandas query -> saved answer``

The validator deliberately accepts the planner, retriever, and semantic fact
objects by protocol (attribute access) as well as plain mappings.  This keeps
the save gate independent from the concrete complex-solver implementation and
makes the validation report easy to serialize alongside a submission.
"""

from __future__ import annotations

import ast
import math
import os
import re
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

try:  # Support both ``src.semantic_validation`` and legacy top-level imports.
    from .metric_registry import DEFAULT_REGISTRY, MetricRegistry, normalize_metric_text
    from .query_formatter import (
        QueryExecutionError,
        QueryFormatError,
        execute_expression,
        referenced_variables,
    )
    from .units import UnitDimension, resolve_unit
except ImportError:  # pragma: no cover - exercised by legacy entry points
    from metric_registry import DEFAULT_REGISTRY, MetricRegistry, normalize_metric_text
    from query_formatter import (
        QueryExecutionError,
        QueryFormatError,
        execute_expression,
        referenced_variables,
    )
    from units import UnitDimension, resolve_unit


class ValidationSeverity(str, Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    """A machine-readable validation finding."""

    code: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.ERROR
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["severity"] = self.severity.value
        return result

    def __str__(self) -> str:
        return f"{self.code}: {self.message}"


@dataclass
class ValidationReport:
    """Result of the final semantic save gate."""

    valid: bool = False
    confidence: float = 0.0
    answer: float | int | None = None
    query_result: float | int | None = None
    errors: list[ValidationIssue] = field(default_factory=list)
    warnings: list[ValidationIssue] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    coverage: dict[str, Any] = field(default_factory=dict)

    @property
    def error_messages(self) -> list[str]:
        return [str(issue) for issue in self.errors]

    @property
    def warning_messages(self) -> list[str]:
        return [str(issue) for issue in self.warnings]

    def add_error(self, code: str, message: str, **details: Any) -> None:
        self.errors.append(
            ValidationIssue(code, message, ValidationSeverity.ERROR, details)
        )

    def add_warning(self, code: str, message: str, **details: Any) -> None:
        self.warnings.append(
            ValidationIssue(code, message, ValidationSeverity.WARNING, details)
        )

    def require_valid(self) -> "ValidationReport":
        """Raise a concise error instead of allowing an invalid item to save."""

        if not self.valid:
            summary = "; ".join(self.error_messages) or "unknown validation failure"
            raise SemanticValidationError(summary, report=self)
        return self

    def to_dict(self) -> dict[str, Any]:
        return {
            "valid": self.valid,
            "confidence": self.confidence,
            "answer": self.answer,
            "query_result": self.query_result,
            "errors": [issue.to_dict() for issue in self.errors],
            "warnings": [issue.to_dict() for issue in self.warnings],
            "checks": dict(self.checks),
            "coverage": dict(self.coverage),
        }


class SemanticValidationError(ValueError):
    """Raised when callers explicitly require a valid report."""

    def __init__(self, message: str, *, report: ValidationReport | None = None) -> None:
        super().__init__(message)
        self.report = report


@dataclass(frozen=True)
class _FactView:
    ticker: str
    year: str
    metric: str
    value: float | None
    unit: str
    path: str
    row_index: int | None
    variable: str
    label: str
    confidence: float
    provenance: str


@dataclass
class _EvidenceView:
    variable_to_path: dict[str, str] = field(default_factory=dict)
    paths: list[str] = field(default_factory=list)
    mapping_errors: list[tuple[str, str, dict[str, Any]]] = field(default_factory=list)
    missing_requirements: list[Any] = field(default_factory=list)

    @property
    def variables(self) -> set[str]:
        return set(self.variable_to_path)


_DF_VARIABLE = re.compile(r"df[1-9]\d*\Z")
_PSEUDO_METRICS = {
    "beginning_value",
    "ending_value",
    "periods",
    "current_value",
    "previous_value",
    "current_percentage",
    "previous_percentage",
    "beginning_period",
    "ending_period",
}


def _value(obj: Any, *names: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        for name in names:
            if name in obj:
                return obj[name]
        return default
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _norm_path(value: Any) -> str:
    if value is None:
        return ""
    return os.path.normcase(os.path.normpath(str(value)))


def _display_path(value: Any) -> str:
    return str(value or "").replace("\\", "/")


def _as_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _flatten_facts(facts: Any) -> list[Any]:
    """Return semantic facts without assuming a concrete table container."""

    if facts is None:
        return []
    contained = _value(facts, "facts", default=None)
    if contained is not None and contained is not facts:
        return _flatten_facts(contained)
    if isinstance(facts, Mapping):
        # A mapping with fact fields is one fact; other mappings are containers.
        if any(name in facts for name in ("metric", "metric_name", "ticker", "year")):
            return [facts]
        result: list[Any] = []
        for item in facts.values():
            result.extend(_flatten_facts(item))
        return result
    if isinstance(facts, Sequence) and not isinstance(facts, (str, bytes, bytearray)):
        result = []
        for item in facts:
            result.extend(_flatten_facts(item))
        return result
    return [facts]


def _fact_view(fact: Any) -> _FactView:
    row = _value(fact, "row_index", "row", "index", default=None)
    try:
        row_index = int(row) if row is not None else None
    except (TypeError, ValueError, OverflowError):
        row_index = None
    confidence = _as_float(_value(fact, "confidence", "score", "match_score", default=1.0))
    # ``SemanticExtractor`` exposes a deterministic row score on a 0..120
    # scale, whereas external fact providers normally expose confidence on
    # 0..1.  Normalize only the known score-shaped field.
    if confidence is not None and confidence > 1.0 and hasattr(fact, "match_score"):
        confidence /= 120.0
    provenance_parts = [
        _value(fact, "statement_type", "statement", "report_type", default=""),
        _value(fact, "source", "provenance", default=""),
    ]
    return _FactView(
        ticker=str(_value(fact, "ticker", "entity", "company", default="") or "").upper(),
        year=str(_value(fact, "year", "period", default="") or ""),
        metric=str(_value(fact, "metric", "metric_name", default="") or "")
        .removesuffix("_previous"),
        value=_as_float(_value(fact, "value", "normalized_value", default=None)),
        unit=str(_value(fact, "unit", "source_unit", default="") or ""),
        path=_display_path(_value(fact, "path", "csv_path", "source_path", default="")),
        row_index=row_index,
        variable=str(_value(fact, "variable", "df_variable", "var_name", default="") or ""),
        label=str(_value(fact, "label", "row_label", "chi_tieu", default="") or ""),
        confidence=max(0.0, min(1.0, confidence if confidence is not None else 0.0)),
        provenance=" ".join(str(part or "") for part in provenance_parts).strip(),
    )


def _missing_to_dict(item: Any) -> dict[str, Any]:
    if is_dataclass(item):
        return asdict(item)
    if isinstance(item, Mapping):
        return dict(item)
    return {
        "ticker": _value(item, "ticker", default=""),
        "year": _value(item, "year", default=""),
        "metric": _value(item, "metric", default=""),
        "reason": _value(item, "reason", default=str(item)),
    }


def _evidence_view(evidence: Any) -> _EvidenceView:
    view = _EvidenceView()
    if evidence is None:
        return view

    missing = _value(evidence, "missing_requirements", default=None)
    if missing:
        view.missing_requirements = list(missing)

    paths_attr = _value(evidence, "paths", default=None)
    if paths_attr is not None and not isinstance(paths_attr, (str, bytes)):
        view.paths = [_display_path(path) for path in paths_attr]

    path_to_variable = _value(evidence, "path_to_variable", default=None)
    variable_to_path = _value(evidence, "variable_to_path", default=None)
    if isinstance(path_to_variable, Mapping):
        for raw_path, raw_variable in path_to_variable.items():
            _add_evidence_mapping(view, str(raw_variable), str(raw_path))
    if isinstance(variable_to_path, Mapping):
        for raw_variable, raw_path in variable_to_path.items():
            variable, path = str(raw_variable), str(raw_path)
            existing = view.variable_to_path.get(variable)
            if existing is not None and _norm_path(existing) != _norm_path(path):
                view.mapping_errors.append((
                    "inconsistent_inverse_mapping",
                    f"{variable} maps to two evidence paths",
                    {"variable": variable, "first": existing, "second": path},
                ))
            _add_evidence_mapping(view, variable, path)

    # Submission-style evidence is a list of {variable, csv_path} records.
    if isinstance(evidence, Sequence) and not isinstance(evidence, (str, bytes, bytearray)):
        seen_records: set[tuple[str, str]] = set()
        for item in evidence:
            variable = str(_value(item, "variable", "var_name", default="") or "")
            path = str(_value(item, "csv_path", "path", default="") or "")
            record = (variable, _norm_path(path))
            if record in seen_records:
                view.mapping_errors.append((
                    "duplicate_evidence_record",
                    "Final evidence repeats the same variable/path record",
                    {"variable": variable, "path": path},
                ))
                continue
            seen_records.add(record)
            _add_evidence_mapping(view, variable, path)

    # A direct {dfN: frame-or-path} mapping is convenient for tests and tools.
    if isinstance(evidence, Mapping) and not path_to_variable and not variable_to_path:
        for key, item in evidence.items():
            key_text = str(key)
            if _DF_VARIABLE.fullmatch(key_text):
                if isinstance(item, pd.DataFrame):
                    path = str(_value(item, "attrs", default={}).get("source_path", ""))
                else:
                    path = str(_value(item, "csv_path", "path", default=item) or "")
                _add_evidence_mapping(view, key_text, path)
            elif _DF_VARIABLE.fullmatch(str(item)):
                _add_evidence_mapping(view, str(item), key_text)

    # ``paths`` and both mapping directions must describe the same final set.
    if view.paths:
        mapped_paths = {_norm_path(path) for path in view.variable_to_path.values()}
        declared_paths = {_norm_path(path) for path in view.paths}
        if mapped_paths != declared_paths:
            view.mapping_errors.append((
                "evidence_paths_mapping_mismatch",
                "Evidence paths and path-to-variable mapping differ",
                {
                    "unmapped_paths": sorted(declared_paths - mapped_paths),
                    "undeclared_paths": sorted(mapped_paths - declared_paths),
                },
            ))
    return view


def _add_evidence_mapping(view: _EvidenceView, variable: str, path: str) -> None:
    if not _DF_VARIABLE.fullmatch(variable):
        view.mapping_errors.append((
            "invalid_evidence_variable",
            f"Invalid evidence variable {variable!r}",
            {"variable": variable, "path": path},
        ))
        return
    if not path:
        # A DataFrame supplied directly has no filesystem path.  The variable
        # still participates in exact query mapping.
        path = f"<in-memory:{variable}>"
    existing = view.variable_to_path.get(variable)
    if existing is not None and _norm_path(existing) != _norm_path(path):
        view.mapping_errors.append((
            "duplicate_evidence_variable",
            f"{variable} is assigned to more than one path",
            {"variable": variable, "first": existing, "second": path},
        ))
        return
    for other_variable, other_path in view.variable_to_path.items():
        if other_variable != variable and _norm_path(other_path) == _norm_path(path):
            view.mapping_errors.append((
                "duplicate_evidence_path",
                "One evidence path is assigned to multiple variables",
                {"path": path, "variables": [other_variable, variable]},
            ))
            return
    view.variable_to_path[variable] = _display_path(path)


def _plan_is_complex(plan: Any) -> bool:
    explicit = _value(plan, "is_complex", default=None)
    if explicit is not None:
        return bool(explicit)
    question_type = _value(plan, "question_type", default="")
    kind = str(getattr(question_type, "value", question_type))
    return bool(kind and kind != "SIMPLE_LOOKUP")


def _previous_year(year: str) -> str:
    try:
        return str(int(year) - 1)
    except (TypeError, ValueError):
        return year


class SemanticValidator:
    """Validate the final answer before it is persisted."""

    def __init__(
        self,
        *,
        registry: MetricRegistry | None = None,
        absolute_tolerance: float = 1e-9,
        relative_tolerance: float = 1e-12,
    ) -> None:
        self.registry = registry or DEFAULT_REGISTRY
        self.absolute_tolerance = float(absolute_tolerance)
        self.relative_tolerance = float(relative_tolerance)

    def validate(
        self,
        *,
        answer: Any,
        pandas_query: str,
        evidence: Any,
        plan: Any | None = None,
        facts: Any = None,
        dataframes: Mapping[str, pd.DataFrame] | None = None,
    ) -> ValidationReport:
        report = ValidationReport()
        evidence_view = _evidence_view(evidence)
        fact_views = [_fact_view(fact) for fact in _flatten_facts(facts)]

        numeric_answer = _as_float(answer)
        if numeric_answer is None:
            report.add_error("non_numeric_answer", "Answer is not a finite numeric scalar", answer=repr(answer))
            report.checks["answer_is_finite_numeric"] = False
        else:
            report.answer = int(numeric_answer) if numeric_answer.is_integer() and abs(numeric_answer) < 1e15 else numeric_answer
            report.checks["answer_is_finite_numeric"] = True

        for code, message, details in evidence_view.mapping_errors:
            report.add_error(code, message, **details)
        report.checks["evidence_mapping_is_bijective"] = not evidence_view.mapping_errors

        query_variables = referenced_variables(pandas_query)
        evidence_variables = evidence_view.variables
        if evidence_variables:
            report.checks["final_evidence_non_empty"] = True
            missing_variables = sorted(query_variables - evidence_variables)
            unused_variables = sorted(evidence_variables - query_variables)
            if missing_variables:
                report.add_error(
                    "query_uses_unknown_evidence",
                    "Query references variables absent from final evidence",
                    variables=missing_variables,
                )
            if unused_variables:
                report.add_error(
                    "final_evidence_not_pruned",
                    "Final evidence contains variables not referenced by the final query",
                    variables=unused_variables,
                )
            if not query_variables:
                report.add_error(
                    "constant_query_with_evidence",
                    "A query with final evidence must reference that evidence",
                )
            report.checks["query_evidence_variables_exact"] = (
                query_variables == evidence_variables
            )
        else:
            report.add_error(
                "empty_final_evidence",
                "A saved answer must be grounded in non-empty final evidence",
            )
            report.checks["final_evidence_non_empty"] = False
            report.checks["query_evidence_variables_exact"] = not query_variables
            if query_variables:
                report.add_error(
                    "query_uses_unknown_evidence",
                    "Query references dataframe variables but final evidence is empty",
                    variables=sorted(query_variables),
                )

        frames = self._resolve_frames(dataframes, evidence_view, report)
        self._execute_and_compare(
            pandas_query, frames, numeric_answer, report
        )

        if evidence_view.missing_requirements and _plan_is_complex(plan):
            report.add_error(
                "retrieval_incomplete",
                "Metric-aware retrieval still has missing requirements",
                missing=[_missing_to_dict(item) for item in evidence_view.missing_requirements],
            )
            report.checks["retrieval_complete"] = False
        else:
            report.checks["retrieval_complete"] = True

        self._validate_fact_evidence_mapping(fact_views, evidence_view, report)
        self._validate_complex_coverage(plan, fact_views, report)
        self._validate_output_semantics(plan, numeric_answer, fact_views, report)
        self._validate_cfo_provenance(plan, fact_views, report)
        self._validate_statement_sums(pandas_query, plan, report)

        report.valid = not report.errors and all(report.checks.values())
        report.confidence = self._confidence(report, fact_views)
        return report

    def _resolve_frames(
        self,
        supplied: Mapping[str, pd.DataFrame] | None,
        evidence: _EvidenceView,
        report: ValidationReport,
    ) -> dict[str, pd.DataFrame]:
        frames: dict[str, pd.DataFrame] = {}
        if supplied is not None:
            for variable, frame in supplied.items():
                if not _DF_VARIABLE.fullmatch(str(variable)):
                    report.add_error(
                        "invalid_dataframe_variable",
                        f"Loaded dataframe uses invalid variable {variable!r}",
                    )
                    continue
                if not isinstance(frame, pd.DataFrame):
                    report.add_error(
                        "invalid_dataframe_object",
                        f"{variable} is not a pandas DataFrame",
                        actual_type=type(frame).__name__,
                    )
                    continue
                frames[str(variable)] = frame
        else:
            for variable, raw_path in evidence.variable_to_path.items():
                if raw_path.startswith("<in-memory:"):
                    continue
                path = Path(raw_path)
                if not path.is_file():
                    continue
                try:
                    frames[variable] = pd.read_csv(path)
                except Exception as exc:
                    report.add_error(
                        "evidence_read_failure",
                        f"Cannot read exact evidence for {variable}",
                        variable=variable,
                        path=_display_path(path),
                        error=f"{type(exc).__name__}: {exc}",
                    )

        expected = evidence.variables
        actual = set(frames)
        if expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            if missing:
                report.add_error(
                    "missing_loaded_evidence",
                    "Not every final evidence variable has a loaded DataFrame",
                    variables=missing,
                )
            if extra:
                report.add_error(
                    "extra_loaded_evidence",
                    "Loaded DataFrames include variables outside final evidence",
                    variables=extra,
                )
            report.checks["loaded_evidence_is_exact"] = actual == expected
        else:
            report.checks["loaded_evidence_is_exact"] = not actual
            if actual:
                report.add_error(
                    "extra_loaded_evidence",
                    "DataFrames were supplied without corresponding final evidence",
                    variables=sorted(actual),
                )
        return frames

    def _execute_and_compare(
        self,
        query: str,
        frames: Mapping[str, pd.DataFrame],
        answer: float | None,
        report: ValidationReport,
    ) -> None:
        try:
            query_result = execute_expression(query, frames)
        except (QueryFormatError, QueryExecutionError) as exc:
            report.add_error(
                "query_execution_failure",
                "Final pandas query cannot be executed on final evidence",
                error=str(exc),
            )
            report.checks["query_executes"] = False
            report.checks["answer_matches_query"] = False
            return
        report.query_result = query_result
        report.checks["query_executes"] = True
        if answer is None:
            report.checks["answer_matches_query"] = False
            return
        matches = math.isclose(
            float(query_result),
            answer,
            rel_tol=self.relative_tolerance,
            abs_tol=self.absolute_tolerance,
        )
        report.checks["answer_matches_query"] = matches
        if not matches:
            report.add_error(
                "answer_query_mismatch",
                "Saved answer is not the exact result of the final pandas query",
                answer=answer,
                query_result=query_result,
                absolute_difference=abs(answer - float(query_result)),
                absolute_tolerance=self.absolute_tolerance,
                relative_tolerance=self.relative_tolerance,
            )

    def _validate_fact_evidence_mapping(
        self,
        facts: Sequence[_FactView],
        evidence: _EvidenceView,
        report: ValidationReport,
    ) -> None:
        errors_before = len(report.errors)
        for index, fact in enumerate(facts):
            if fact.variable and fact.variable not in evidence.variables:
                report.add_error(
                    "fact_uses_unknown_evidence",
                    "Semantic fact references a variable outside final evidence",
                    fact_index=index,
                    variable=fact.variable,
                    metric=fact.metric,
                )
                continue
            if fact.variable and fact.path:
                mapped = evidence.variable_to_path.get(fact.variable, "")
                if mapped and _norm_path(mapped) != _norm_path(fact.path):
                    report.add_error(
                        "fact_evidence_path_mismatch",
                        "Semantic fact variable does not map to its source path",
                        fact_index=index,
                        variable=fact.variable,
                        fact_path=fact.path,
                        evidence_path=mapped,
                    )
            if fact.path:
                path_variable = next(
                    (
                        variable
                        for variable, path in evidence.variable_to_path.items()
                        if _norm_path(path) == _norm_path(fact.path)
                    ),
                    None,
                )
                if path_variable is None:
                    report.add_error(
                        "fact_source_not_in_evidence",
                        "Semantic fact source path is absent from final evidence",
                        fact_index=index,
                        path=fact.path,
                        metric=fact.metric,
                    )
                elif fact.variable and path_variable != fact.variable:
                    report.add_error(
                        "fact_variable_path_mismatch",
                        "Semantic fact path belongs to a different evidence variable",
                        fact_index=index,
                        expected_variable=path_variable,
                        actual_variable=fact.variable,
                    )
        report.checks["facts_map_to_exact_evidence"] = len(report.errors) == errors_before

    def _metric_dependencies(
        self,
        metric: str,
        year: str,
        result: set[tuple[str, str]],
        seen: set[tuple[str, str]],
    ) -> None:
        metric = metric.removesuffix("_previous")
        key = (metric, year)
        if key in seen:
            return
        seen.add(key)
        if metric in _PSEUDO_METRICS:
            return
        try:
            definition = self.registry.get(metric)
        except KeyError:
            result.add(key)
            return
        if not definition.required_metrics:
            result.add(key)
            return
        for dependency in definition.required_metrics:
            if dependency in _PSEUDO_METRICS:
                continue
            previous = dependency.endswith("_previous")
            dependency_name = dependency.removesuffix("_previous")
            dependency_year = _previous_year(year) if previous else year
            self._metric_dependencies(dependency_name, dependency_year, result, seen)

    def _expected_coverage(self, plan: Any) -> set[tuple[str, str, str]]:
        tickers = [str(item).upper() for item in (_value(plan, "tickers", default=[]) or [])]
        years = [str(item) for item in (_value(plan, "years", default=[]) or [])]
        metric_years = _value(plan, "metric_years", default={}) or {}
        seeds: list[tuple[str, str]] = []
        if isinstance(metric_years, Mapping):
            for metric, assigned_years in metric_years.items():
                for year in assigned_years or years:
                    seeds.append((str(metric), str(year)))

        # Modern planners expose the exact base-metric/year matrix after
        # expanding target, filter and selection formulas.  Treat it as the
        # source of truth.  Expanding every derived operation again for every
        # question year would invent requirements (for example gross profit in
        # 2024 when only the 2025 output margin is requested).
        if seeds:
            expanded: set[tuple[str, str]] = set()
            for metric, year in seeds:
                self._metric_dependencies(metric, year, expanded, set())
            return {
                (ticker, year, metric)
                for ticker in tickers
                for metric, year in expanded
                if ticker and year and metric
            }

        additional_metrics: list[str] = []
        target = _value(plan, "target_metric", default=None)
        if target:
            additional_metrics.append(str(target))
        for filter_spec in _value(plan, "filters", default=[]) or []:
            metric = _value(filter_spec, "metric", default=None)
            filter_years = [str(y) for y in (_value(filter_spec, "years", default=[]) or years)]
            if metric:
                for year in filter_years:
                    seeds.append((str(metric), year))
                additional_metrics.append(str(metric))
        selection = _value(plan, "selection_operation", "selection", default=None)
        selection_metric = _value(selection, "metric", default=None)
        if selection_metric:
            additional_metrics.append(str(selection_metric))

        seeded_names = {metric for metric, _ in seeds}
        for metric in additional_metrics:
            if metric in seeded_names:
                continue
            # The current planner publishes role-aware base metric/year
            # requirements.  Do not re-expand a derived operation for every
            # explicit year when all of its bases are already mapped: doing so
            # invents periods (for example 2021 and 2023 for a 2022->2024
            # endpoint growth calculation).
            dependency_probe: set[tuple[str, str]] = set()
            self._metric_dependencies(metric, "2001", dependency_probe, set())
            dependency_names = {name for name, _ in dependency_probe}
            if dependency_names and dependency_names.issubset(seeded_names):
                continue
            for year in years:
                seeds.append((metric, year))

        expanded: set[tuple[str, str]] = set()
        for metric, year in seeds:
            self._metric_dependencies(metric, year, expanded, set())

        # Some planner versions expose only base requirements.  Add only bases
        # not already reached from the semantic operations, avoiding an overly
        # broad all-years matrix when dependency/year information is available.
        reached_metrics = {metric for metric, _ in expanded}
        for required in _value(plan, "required_metrics", default=[]) or []:
            name = str(required).removesuffix("_previous")
            if name in _PSEUDO_METRICS or name in reached_metrics:
                continue
            for year in years:
                self._metric_dependencies(name, year, expanded, set())

        return {
            (ticker, year, metric)
            for ticker in tickers
            for metric, year in expanded
            if ticker and year and metric
        }

    def _validate_complex_coverage(
        self,
        plan: Any,
        facts: Sequence[_FactView],
        report: ValidationReport,
    ) -> None:
        if plan is None or not _plan_is_complex(plan):
            report.checks["complex_semantic_coverage"] = True
            report.coverage = {
                "required": 0,
                "observed": len(facts),
                "missing": [],
            }
            return

        expected = self._expected_coverage(plan)
        observed = {
            (fact.ticker, fact.year, fact.metric)
            for fact in facts
            if fact.ticker and fact.year and fact.metric and fact.value is not None
        }
        missing = sorted(expected - observed)
        report.coverage = {
            "required": len(expected),
            "observed": len(expected & observed),
            "fact_count": len(facts),
            "expected": [
                {"ticker": ticker, "year": year, "metric": metric}
                for ticker, year, metric in sorted(expected)
            ],
            "missing": [
                {"ticker": ticker, "year": year, "metric": metric}
                for ticker, year, metric in missing
            ],
        }
        if not expected:
            report.add_error(
                "complex_plan_has_no_requirements",
                "Complex question plan has no concrete entity/year/metric requirements",
            )
        if missing:
            report.add_error(
                "incomplete_complex_coverage",
                "Complex answer lacks facts for required companies, years, or metrics",
                missing=report.coverage["missing"],
            )
        report.checks["complex_semantic_coverage"] = bool(expected) and not missing

    def _target_output_kind(self, plan: Any) -> str:
        aggregation = str(_value(plan, "aggregation", default="") or "").lower()
        if aggregation == "share":
            return "percent"
        if aggregation == "count":
            return "count"
        target_metric = str(_value(plan, "target_metric", default="") or "")
        if target_metric:
            try:
                definition = self.registry.get(target_metric)
                # Formula semantics are authoritative.  For a base row, an
                # explicit requested unit (not the registry's default
                # ``currency`` kind) must still be honoured.
                if definition.derived:
                    return definition.output_kind
            except KeyError:
                pass
        target_unit = _value(plan, "target_unit", default="")
        unit = resolve_unit(target_unit)
        if unit is not None:
            if unit.dimension == UnitDimension.PERCENTAGE_POINT:
                return "percentage_point"
            if unit.dimension == UnitDimension.RATIO:
                return "percent" if unit.name == "%" else "times"
            if unit.dimension in {UnitDimension.VND, UnitDimension.USD}:
                return "currency"
            if unit.dimension == UnitDimension.SHARES:
                return "shares"
            return unit.dimension.value
        if target_metric:
            try:
                return self.registry.get(target_metric).output_kind
            except KeyError:
                pass
        return ""

    def _validate_output_semantics(
        self,
        plan: Any,
        answer: float | None,
        facts: Sequence[_FactView],
        report: ValidationReport,
    ) -> None:
        if answer is None:
            report.checks["output_semantics"] = False
            return
        errors_before = len(report.errors)
        aggregation = str(_value(plan, "aggregation", default="") or "").lower()
        question = normalize_metric_text(_value(plan, "question", default="") or "")
        target_metric = str(_value(plan, "target_metric", default="") or "")
        output_kind = self._target_output_kind(plan)

        count_question = aggregation == "count" or target_metric == "count"
        if count_question and (answer < 0 or not answer.is_integer()):
            report.add_error(
                "invalid_count",
                "A count must be a non-negative integer",
                answer=answer,
            )

        ownership_question = bool(
            re.search(r"\b(?:ty le|phan tram|muc) so huu\b|\bquyen so huu\b", question)
            or target_metric in {"ownership", "ownership_percentage", "ownership_ratio"}
        )
        if ownership_question and not 0.0 <= answer <= 100.0:
            report.add_error(
                "ownership_out_of_range",
                "Ownership percentage must lie between 0 and 100",
                answer=answer,
            )

        # These are broad semantic guards rather than arbitrary answer caps.
        # They catch the observed failure mode where raw VND (billions/trillions)
        # is returned for a dimensionless question while retaining unusual but
        # possible financial percentages and ratios.
        if output_kind == "percent":
            if abs(answer) > 1_000_000:
                report.add_error(
                    "percentage_has_currency_scale",
                    "Percentage answer has a raw-currency-sized magnitude",
                    answer=answer,
                )
            elif abs(answer) > 10_000:
                report.add_warning(
                    "extreme_percentage",
                    "Percentage magnitude is extreme and should be reviewed",
                    answer=answer,
                )
        elif output_kind == "percentage_point":
            if abs(answer) > 100_000:
                report.add_error(
                    "percentage_point_has_currency_scale",
                    "Percentage-point answer has a raw-currency-sized magnitude",
                    answer=answer,
                )
            elif abs(answer) > 1_000:
                report.add_warning(
                    "extreme_percentage_point_change",
                    "Percentage-point change is extreme and should be reviewed",
                    answer=answer,
                )
        elif output_kind in {"times", "ratio"}:
            if abs(answer) > 1_000_000:
                report.add_error(
                    "ratio_has_currency_scale",
                    "Dimensionless ratio has a raw-currency-sized magnitude",
                    answer=answer,
                )
            elif abs(answer) > 10_000:
                report.add_warning(
                    "extreme_ratio",
                    "Dimensionless ratio is extreme and should be reviewed",
                    answer=answer,
                )

        target_unit = resolve_unit(_value(plan, "target_unit", default=""))
        if target_unit is not None and target_unit.dimension == UnitDimension.RATIO:
            # A direct currency row cannot truthfully produce a ``lần`` or
            # percent answer.  A subset-share aggregation is different: its
            # numerator and denominator are intentionally currency facts and
            # their quotient is dimensionless.  Rejecting those source facts
            # made every valid ``tỷ trọng ... chiếm bao nhiêu phần trăm`` plan
            # fail after its query had already reproduced the answer exactly.
            if aggregation != "share" and target_metric and facts:
                target_facts = [fact for fact in facts if fact.metric == target_metric]
                if target_facts:
                    currency_facts = [
                        fact
                        for fact in target_facts
                        if (resolve_unit(fact.unit) is not None and resolve_unit(fact.unit).is_currency)
                    ]
                    if currency_facts:
                        report.add_error(
                            "ratio_uses_currency_fact",
                            "A dimensionless result was sourced directly from a currency row",
                            metric=target_metric,
                        )

        report.checks["output_semantics"] = len(report.errors) == errors_before

    def _plan_requires_metric(self, plan: Any, metric: str) -> bool:
        candidates: list[str] = []
        for field_name in ("target_metric",):
            value = _value(plan, field_name, default=None)
            if value:
                candidates.append(str(value))
        candidates.extend(str(item) for item in (_value(plan, "mentioned_metrics", default=[]) or []))
        candidates.extend(str(item) for item in (_value(plan, "required_metrics", default=[]) or []))
        for filter_spec in _value(plan, "filters", default=[]) or []:
            value = _value(filter_spec, "metric", default=None)
            if value:
                candidates.append(str(value))
        selection = _value(plan, "selection_operation", "selection", default=None)
        selection_metric = _value(selection, "metric", default=None)
        if selection_metric:
            candidates.append(str(selection_metric))
        for candidate in candidates:
            dependencies: set[tuple[str, str]] = set()
            self._metric_dependencies(candidate, "2001", dependencies, set())
            if metric in {name for name, _ in dependencies}:
                return True
        return False

    def _validate_cfo_provenance(
        self,
        plan: Any,
        facts: Sequence[_FactView],
        report: ValidationReport,
    ) -> None:
        requires_cfo = self._plan_requires_metric(plan, "cfo") if plan is not None else False
        if not requires_cfo:
            report.checks["cfo_provenance"] = True
            return
        cfo_facts = [fact for fact in facts if fact.metric == "cfo" and fact.value is not None]
        if not cfo_facts:
            report.add_error(
                "missing_cfo_fact",
                "Question requires CFO but no raw CFO fact is present",
            )
            report.checks["cfo_provenance"] = False
            return

        invalid: list[dict[str, Any]] = []
        for fact in cfo_facts:
            source = normalize_metric_text(" ".join((fact.path, fact.provenance)))
            label = normalize_metric_text(fact.label)
            compact_source = source.replace(" ", "")
            source_is_cashflow = bool(re.search(
                r"cash ?flow|luu chuyen tien|dong tien|hoat dong kinh doanh",
                source,
            )) or any(
                marker in compact_source
                for marker in ("cashflow", "luuchuyentien", "dongtien", "hoatdongkinhdoanh")
            )
            label_is_raw_cfo = bool(re.search(
                r"(?:^cfo$|luu chuyen tien (?:thuan )?tu hoat dong kinh doanh|"
                r"tien (?:thuan )?(?:thu )?tu hoat dong kinh doanh|"
                r"net cash flows? from operating activities)",
                label,
            ))
            label_is_derived = bool(re.search(r"margin|bien|ty le|tren doanh thu", label))
            if label_is_derived or not (source_is_cashflow or label_is_raw_cfo):
                invalid.append({
                    "ticker": fact.ticker,
                    "year": fact.year,
                    "label": fact.label,
                    "path": fact.path,
                })
        if invalid:
            report.add_error(
                "invalid_cfo_provenance",
                "CFO facts must come from a raw operating cash-flow row",
                facts=invalid,
            )
        report.checks["cfo_provenance"] = not invalid

    @staticmethod
    def _is_pd_series_constructor(node: ast.AST) -> bool:
        if not isinstance(node, ast.Call):
            return False
        func = node.func
        return (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "pd"
            and func.attr == "Series"
        )

    @staticmethod
    def _contains_dataframe(node: ast.AST) -> bool:
        return any(
            isinstance(child, ast.Name) and _DF_VARIABLE.fullmatch(child.id)
            for child in ast.walk(node)
        )

    @staticmethod
    def _has_explicit_row_index(node: ast.AST) -> bool:
        return any(
            isinstance(child, ast.Attribute) and child.attr in {"iloc", "iat", "at"}
            for child in ast.walk(node)
        )

    @staticmethod
    def _has_broad_label_match(node: ast.AST) -> bool:
        broad_methods = {"contains", "match", "startswith", "endswith", "find", "isin"}
        if any(
            isinstance(child, ast.Attribute) and child.attr in broad_methods
            for child in ast.walk(node)
        ):
            return True
        for child in ast.walk(node):
            if isinstance(child, ast.Constant) and isinstance(child.value, str):
                folded = normalize_metric_text(child.value)
                if folded in {"chi tieu", "chitieu", "label", "metric", "ten chi tieu"}:
                    return True
        return False

    def _validate_statement_sums(
        self,
        query: str,
        plan: Any,
        report: ValidationReport,
    ) -> None:
        try:
            tree = ast.parse(query, mode="eval")
        except SyntaxError:
            # Query execution already reports the syntax error.
            report.checks["no_unsafe_statement_sum"] = False
            return
        unsafe: list[str] = []
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "sum"
            ):
                continue
            receiver = node.func.value
            if self._is_pd_series_constructor(receiver):
                # Aggregating a compact Series of already resolved scalar facts
                # is safe; it cannot double-count statement hierarchy rows.
                continue
            if not self._contains_dataframe(receiver):
                continue
            broad = self._has_broad_label_match(receiver)
            explicit_row = self._has_explicit_row_index(receiver)
            if broad or not explicit_row:
                try:
                    rendered = ast.unparse(node)
                except Exception:  # pragma: no cover - ast.unparse is standard
                    rendered = ".sum(...)"
                unsafe.append(rendered)
        if unsafe:
            report.add_error(
                "unsafe_statement_sum",
                "Query sums raw statement rows selected broadly and may double-count parent/child totals",
                expressions=unsafe,
                planner_aggregation=_value(plan, "aggregation", default=None),
            )
        report.checks["no_unsafe_statement_sum"] = not unsafe

    @staticmethod
    def _confidence(report: ValidationReport, facts: Sequence[_FactView]) -> float:
        if report.errors:
            # Invalid answers are never represented as high-confidence merely
            # because their underlying row matcher emitted a high score.
            return 0.0
        check_ratio = (
            sum(bool(value) for value in report.checks.values()) / len(report.checks)
            if report.checks
            else 0.0
        )
        fact_confidence = (
            sum(fact.confidence for fact in facts) / len(facts)
            if facts
            else 0.85
        )
        warning_penalty = min(0.25, 0.04 * len(report.warnings))
        return round(max(0.0, min(1.0, 0.55 * check_ratio + 0.45 * fact_confidence - warning_penalty)), 6)


def validate_answer(
    answer: Any,
    pandas_query: str,
    evidence: Any,
    plan: Any | None = None,
    facts: Any = None,
    dataframes: Mapping[str, pd.DataFrame] | None = None,
    *,
    registry: MetricRegistry | None = None,
    absolute_tolerance: float = 1e-9,
    relative_tolerance: float = 1e-12,
) -> ValidationReport:
    """Convenience wrapper used by the pipeline save gate."""

    return SemanticValidator(
        registry=registry,
        absolute_tolerance=absolute_tolerance,
        relative_tolerance=relative_tolerance,
    ).validate(
        answer=answer,
        pandas_query=pandas_query,
        evidence=evidence,
        plan=plan,
        facts=facts,
        dataframes=dataframes,
    )


__all__ = [
    "SemanticValidationError",
    "SemanticValidator",
    "ValidationIssue",
    "ValidationReport",
    "ValidationSeverity",
    "validate_answer",
]
