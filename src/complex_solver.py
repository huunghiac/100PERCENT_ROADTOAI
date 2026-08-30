"""Deterministic execution for analytical ViFinQA questions.

The complex path never asks a language model to reason over raw CSV previews.
It first reduces the retrieved evidence to row-level semantic facts and then
constructs one auditable pandas expression from those facts.  Missing or
ambiguous inputs are reported as structured failures; a complex question is
never degraded to a lexical single-row answer.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

try:  # Support package and legacy top-level imports.
    from .metric_registry import (
        DEFAULT_REGISTRY,
        FormulaError,
        MetricDefinition,
        MetricRegistry,
        normalize_metric_text,
    )
    from .query_formatter import QueryExecutionError, QueryFormatError, execute_expression, referenced_variables
    from .question_planner import FilterSpec, QuestionPlan
    from .retriever import EvidenceBundle
    from .units import UnitDimension, detect_unit, resolve_unit
except ImportError:  # pragma: no cover - legacy entry points import src directly
    from metric_registry import (  # type: ignore
        DEFAULT_REGISTRY,
        FormulaError,
        MetricDefinition,
        MetricRegistry,
        normalize_metric_text,
    )
    from query_formatter import QueryExecutionError, QueryFormatError, execute_expression, referenced_variables  # type: ignore
    from question_planner import FilterSpec, QuestionPlan  # type: ignore
    from retriever import EvidenceBundle  # type: ignore
    from units import UnitDimension, detect_unit, resolve_unit  # type: ignore


_STRUCTURAL_TOKENS = {
    "a", "b", "c", "d", "i", "ii", "iii", "iv", "v", "vi", "vii",
    "viii", "ix", "x", "ma", "so", "chi", "tieu", "tndn", "thu", "nhap",
    "doanh", "nghiep",
}


def _stable_unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _as_number(value: object) -> float:
    """Parse a CSV scalar without altering its accounting sign."""

    if value is None or (isinstance(value, float) and math.isnan(value)):
        raise ValueError("empty numeric value")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        numeric = float(value)
    else:
        text = str(value).strip().replace("\u00a0", "").replace(" ", "")
        negative_parentheses = text.startswith("(") and text.endswith(")")
        if negative_parentheses:
            text = text[1:-1]
        if text in {"", "-", "--", "nan", "None", "null"}:
            raise ValueError("empty numeric value")
        # Extracted CSVs normally contain plain numerics.  These branches only
        # normalise common presentation formats; they never call abs().
        if re.fullmatch(r"[-+]?\d{1,3}(?:[.,]\d{3})+", text):
            text = text.replace(".", "").replace(",", "")
        elif "," in text and "." not in text:
            left, right = text.rsplit(",", 1)
            text = left.replace(",", "") + ("." + right if len(right) != 3 else right)
        else:
            text = text.replace(",", "")
        numeric = float(text)
        if negative_parentheses:
            numeric = -numeric
    if not math.isfinite(numeric):
        raise ValueError("numeric value is not finite")
    return numeric


def _column(frame: pd.DataFrame, candidates: Sequence[str]) -> str | None:
    normalized = {normalize_metric_text(column): str(column) for column in frame.columns}
    for candidate in candidates:
        found = normalized.get(normalize_metric_text(candidate))
        if found is not None:
            return found
    return None


def _structural_remainder(text: str, alias: str) -> bool:
    """Whether text surrounding an alias is only a statement code/formula."""

    if text == alias:
        return True
    if alias in text:
        remainder = (text[: text.index(alias)] + " " + text[text.index(alias) + len(alias):]).strip()
        tokens = remainder.split()
        return bool(tokens) and all(token.isdigit() or token in _STRUCTURAL_TOKENS for token in tokens)
    return False


@dataclass(frozen=True)
class MissingFact:
    ticker: str
    year: str
    metric: str
    reason: str
    searched_paths: tuple[str, ...] = ()
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["searched_paths"] = list(self.searched_paths)
        return result


@dataclass(frozen=True)
class SemanticFact:
    ticker: str
    year: str
    metric: str
    raw_value: float
    value: float
    source_unit: str
    base_unit: str
    dimension: str
    path: str
    variable: str
    row_position: int
    row_index: object
    label: str
    value_column: str
    match_score: float
    ambiguous_matches: int = 0

    @property
    def expression(self) -> str:
        raw = f"float({self.variable}.iloc[{self.row_position}][{self.value_column!r}])"
        if self.raw_value == 0:
            # Unit scale is not recoverable by division when the actual value is
            # zero, so derive it directly from the parsed unit.
            spec = resolve_unit(self.source_unit)
            factor = spec.scale if spec is not None else 1.0
        else:
            factor = self.value / self.raw_value
        return raw if factor == 1.0 else f"({raw}) * {factor!r}"

    def compact_record(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "year": self.year,
            "metric": self.metric,
            "value": self.raw_value,
            "unit": self.source_unit,
            "df": self.variable,
            "row": self.row_position,
            "source": self.path,
        }


@dataclass
class SemanticTable:
    facts: list[SemanticFact] = field(default_factory=list)
    missing: list[MissingFact] = field(default_factory=list)
    frames: dict[str, pd.DataFrame] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._index: dict[tuple[str, str, str], SemanticFact] = {
            (fact.ticker, fact.year, fact.metric): fact for fact in self.facts
        }

    def add_fact(self, fact: SemanticFact) -> None:
        key = (fact.ticker, fact.year, fact.metric)
        if key not in self._index:
            self.facts.append(fact)
            self._index[key] = fact

    def add_missing(self, missing: MissingFact) -> None:
        key = (missing.ticker, missing.year, missing.metric, missing.reason)
        if key not in {
            (item.ticker, item.year, item.metric, item.reason) for item in self.missing
        }:
            self.missing.append(missing)

    def get(self, ticker: str, year: str, metric: str) -> SemanticFact | None:
        return self._index.get((ticker, str(year), metric))

    def records(self) -> list[dict[str, Any]]:
        return [fact.compact_record() for fact in self.facts]

    def to_compact_table(self) -> str:
        """Return the only evidence shape suitable for a complex LLM prompt."""

        header = "ticker | year | metric | value | unit | df | row | source"
        rows = [header]
        for record in self.records():
            source = str(record["source"]).replace("|", "/")
            rows.append(
                f"{record['ticker']} | {record['year']} | {record['metric']} | "
                f"{record['value']:.15g} | {record['unit']} | {record['df']} | "
                f"{record['row']} | {source}"
            )
        return "\n".join(rows)

    # Friendly aliases for prompt builders and diagnostics.
    compact = to_compact_table
    to_markdown = to_compact_table


@dataclass(frozen=True)
class MetricValue:
    metric: str
    value: float
    expression: str
    unit: str
    dimension: str
    fact_keys: frozenset[tuple[str, str, str]]


class StructuredSolveFailure(RuntimeError):
    """Expected analytical failure with machine-readable retry information."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        missing: Sequence[MissingFact] = (),
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.missing = list(missing)
        self.details = dict(details or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "code": self.code,
            "message": self.message,
            "missing": [item.to_dict() for item in self.missing],
            "details": self.details,
            "retry_layer": "metric_retrieval" if self.missing else "planner_or_solver",
            "single_row_fallback_allowed": False,
        }


@dataclass
class SolveResult:
    answer: float | int
    pandas_query: str
    unit: str
    semantic_table: SemanticTable
    used_facts: list[SemanticFact]
    used_paths: list[str]
    used_variables: list[str]
    selected_candidates: list[str]
    filtered_candidates: list[str]
    confidence: float
    validation: dict[str, Any]

    @property
    def compact_evidence(self) -> str:
        return self.semantic_table.to_compact_table()

    @property
    def query(self) -> str:
        return self.pandas_query

    @property
    def final_evidence(self) -> list[str]:
        return list(self.used_paths)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": True,
            "answer": self.answer,
            "pandas_query": self.pandas_query,
            "unit": self.unit,
            "evidence": list(self.used_paths),
            "variables": list(self.used_variables),
            "selected_candidates": list(self.selected_candidates),
            "filtered_candidates": list(self.filtered_candidates),
            "confidence": self.confidence,
            "validation": dict(self.validation),
            "semantic_facts": [fact.compact_record() for fact in self.used_facts],
        }


class SemanticExtractor:
    """Extract exact metric rows only from paths officially present in a bundle."""

    def __init__(self, registry: MetricRegistry | None = None) -> None:
        self.registry = registry or DEFAULT_REGISTRY
        self._frame_cache: dict[str, pd.DataFrame] = {}
        self._preferred_scope: dict[tuple[str, str], str] = {}

    @staticmethod
    def _path_variable(bundle: EvidenceBundle, path: str) -> str | None:
        normalized = path.replace("\\", "/")
        return bundle.path_to_variable.get(normalized) or bundle.path_to_variable.get(path)

    def _read_frame(self, path: str) -> pd.DataFrame:
        normalized = path.replace("\\", "/")
        if normalized not in self._frame_cache:
            last_error: Exception | None = None
            for encoding in ("utf-8-sig", "utf-8"):
                try:
                    self._frame_cache[normalized] = pd.read_csv(path, encoding=encoding)
                    break
                except (UnicodeDecodeError, OSError, pd.errors.ParserError) as exc:
                    last_error = exc
            else:
                assert last_error is not None
                raise last_error
        return self._frame_cache[normalized]

    def _candidate_paths(
        self, bundle: EvidenceBundle, ticker: str, year: str, metric: str
    ) -> list[str]:
        key = bundle.requirement_key(ticker, str(year), metric)
        direct = list(bundle.metric_paths.get(key, ()))
        # A table retrieved for another metric may legitimately contain this
        # metric too.  It is still official evidence and already has a stable
        # variable.  No filesystem/global candidate expansion is permitted.
        structured: list[str] = []
        for paths in bundle.structured.get(ticker, {}).get(str(year), {}).values():
            structured.extend(paths)
        candidates = _stable_unique((*direct, *structured))
        candidates = [path for path in candidates if self._path_variable(bundle, path) is not None]
        preferred = self._preferred_scope.get((ticker, str(year)))
        if preferred:
            scoped_or_neutral = [
                path for path in candidates
                if f"_{preferred}" in path.casefold()
                or not any(f"_{scope}" in path.casefold() for scope in ("consolidated", "separate"))
            ]
            # Mixing report scopes changes the economic entity and is more
            # dangerous than returning a structured missing-data failure.
            candidates = scoped_or_neutral
        return candidates

    def _scope_bonus(self, ticker: str, year: str, path: str) -> float:
        preferred = self._preferred_scope.get((ticker, str(year)))
        if not preferred:
            return 0.0
        normalized = path.casefold().replace("\\", "/")
        return 3.0 if f"_{preferred}" in normalized else 0.0

    def _path_has_metric_row(self, path: str, metric: str) -> bool:
        """Check semantic row coverage, not merely retrieval-path coverage."""

        try:
            frame = self._read_frame(path)
        except Exception:
            return False
        label_column = _column(frame, ("Chi_tieu", "Chỉ tiêu", "indicator", "metric"))
        value_column = _column(frame, ("Gia_tri", "Giá trị", "value"))
        if label_column is None or value_column is None:
            return False
        definition = self.registry.get(metric)
        for position in range(len(frame)):
            if self._row_score(frame.iloc[position][label_column], definition) is None:
                continue
            try:
                _as_number(frame.iloc[position][value_column])
            except ValueError:
                continue
            return True
        return False

    def _row_score(self, label: object, definition: MetricDefinition) -> float | None:
        raw_label = "" if label is None else str(label)
        text = normalize_metric_text(raw_label)
        if not text:
            return None
        without_parentheticals = normalize_metric_text(re.sub(r"\([^)]*\)", " ", raw_label))
        polarity_folded = re.sub(r"^(?:lo|lai)\s+(?=loi nhuan\b)", "", text)
        variants = _stable_unique((text, without_parentheticals, polarity_folded))
        aliases = _stable_unique(
            normalize_metric_text(alias)
            for alias in (definition.name.replace("_", " "), *definition.aliases)
        )
        best: float | None = None
        for alias in aliases:
            for variant in variants:
                if variant == alias:
                    score = 120.0 + len(alias) / 1000.0
                elif _structural_remainder(variant, alias):
                    score = 115.0 + len(alias) / 1000.0
                elif definition.exact_total:
                    continue
                elif re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", variant):
                    score = 80.0 + len(alias) / 1000.0
                else:
                    continue
                # Rows describing changes, proportions, or components must not
                # outrank an exact raw statement line.
                if any(token in text for token in ("tang giam", "chenh lech", "ty le", "phan tram")):
                    score -= 25.0
                best = score if best is None else max(best, score)
        return best

    @staticmethod
    def _base_value(raw_value: float, unit_text: object) -> tuple[float, str, str]:
        spec = detect_unit(unit_text)
        if spec is None:
            # Some source CSVs genuinely have an empty Don_vi cell.  Preserve
            # the raw scalar and mark its dimension unknown; do not infer a
            # dominant unit from neighbouring rows.  Dimensionless formulas
            # can still be evaluated, while target currency conversion below
            # will reject this fact explicitly.
            return raw_value, "", "unknown"
        value = raw_value * spec.scale
        if spec.dimension == UnitDimension.VND:
            return value, "VND", spec.dimension.value
        if spec.dimension == UnitDimension.USD:
            return value, "USD", spec.dimension.value
        if spec.dimension == UnitDimension.RATIO:
            return value, "lần", spec.dimension.value
        if spec.dimension == UnitDimension.PERCENTAGE_POINT:
            return value, "điểm phần trăm", spec.dimension.value
        if spec.dimension == UnitDimension.SHARES:
            return value, "cổ phần", spec.dimension.value
        return value, spec.name, spec.dimension.value

    def extract_fact(
        self,
        table: SemanticTable,
        bundle: EvidenceBundle,
        ticker: str,
        year: str,
        metric: str,
    ) -> SemanticFact | None:
        existing = table.get(ticker, str(year), metric)
        if existing is not None:
            return existing
        definition = self.registry.get(metric)
        paths = self._candidate_paths(bundle, ticker, str(year), metric)
        if not paths:
            table.add_missing(MissingFact(ticker, str(year), metric, "no_mapped_evidence", ()))
            return None

        candidates: list[tuple[float, int, int, SemanticFact]] = []
        read_errors: list[str] = []
        for path_order, path in enumerate(paths):
            variable = self._path_variable(bundle, path)
            if variable is None:  # Defensive; _candidate_paths already excludes it.
                continue
            if not os.path.exists(path):
                read_errors.append(f"{path}: file_not_found")
                continue
            try:
                frame = self._read_frame(path)
            except Exception as exc:
                read_errors.append(f"{path}: {type(exc).__name__}: {exc}")
                continue
            table.frames[variable] = frame
            label_column = _column(frame, ("Chi_tieu", "Chỉ tiêu", "indicator", "metric"))
            value_column = _column(frame, ("Gia_tri", "Giá trị", "value"))
            unit_column = _column(frame, ("Don_vi", "Đơn vị", "unit"))
            if label_column is None or value_column is None or unit_column is None:
                read_errors.append(f"{path}: required_columns_missing")
                continue
            for position in range(len(frame)):
                row = frame.iloc[position]
                score = self._row_score(row[label_column], definition)
                if score is None:
                    continue
                try:
                    raw_value = _as_number(row[value_column])
                    raw_unit = row[unit_column]
                    source_unit = "" if pd.isna(raw_unit) else str(raw_unit).strip()
                    value, base_unit, dimension = self._base_value(raw_value, source_unit)
                except ValueError as exc:
                    read_errors.append(f"{path} row {position}: {exc}")
                    continue
                fact = SemanticFact(
                    ticker=ticker,
                    year=str(year),
                    metric=metric,
                    raw_value=raw_value,
                    value=value,
                    source_unit=source_unit,
                    base_unit=base_unit,
                    dimension=dimension,
                    path=path.replace("\\", "/"),
                    variable=variable,
                    row_position=position,
                    row_index=frame.index[position],
                    label=str(row[label_column]),
                    value_column=value_column,
                    match_score=score,
                )
                if not fact.source_unit or fact.dimension == "unknown":
                    table.warnings.append(
                        f"{ticker}|{year}|{metric}: selected row has no parseable unit; "
                        "raw signed value retained without scale inference"
                    )
                unit_bonus = 0.25 if fact.dimension != "unknown" else 0.0
                candidates.append((score + self._scope_bonus(ticker, str(year), path) + unit_bonus, -path_order, -position, fact))

        if not candidates:
            reason = "row_or_unit_not_found" if not read_errors else "unusable_metric_row"
            table.add_missing(
                MissingFact(
                    ticker,
                    str(year),
                    metric,
                    reason,
                    tuple(paths),
                    "; ".join(read_errors[:5]),
                )
            )
            return None
        candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        best_score = candidates[0][0]
        tied = [item for item in candidates if abs(item[0] - best_score) < 1e-12]
        chosen = candidates[0][3]
        if len(tied) > 1:
            chosen = SemanticFact(**{**asdict(chosen), "ambiguous_matches": len(tied)})
            table.warnings.append(
                f"{ticker}|{year}|{metric}: {len(tied)} equally-ranked exact rows; "
                f"used {chosen.variable}.iloc[{chosen.row_position}] by stable evidence order"
            )
        table.add_fact(chosen)
        return chosen

    def extract(self, plan: QuestionPlan, bundle: EvidenceBundle) -> SemanticTable:
        table = SemanticTable()
        self._preferred_scope = {}
        explicit_scope = getattr(plan.scope, "value", str(plan.scope))
        if explicit_scope in {"consolidated", "separate"}:
            for ticker in plan.tickers:
                for year in plan.years:
                    self._preferred_scope[(ticker, str(year))] = explicit_scope
        else:
            # Select one coherent report scope per entity/period by *row-level*
            # metric coverage.  A path retrieved for a metric does not prove
            # that the metric exists in that report scope (for example, a
            # consolidated balance sheet can rank behind a missing consolidated
            # income statement).  Counting paths caused correct separate rows
            # to be discarded before extraction.
            coverage: dict[tuple[str, str], dict[str, set[str]]] = {}
            for key, paths in bundle.metric_paths.items():
                parts = key.split("|", 2)
                if len(parts) != 3:
                    continue
                ticker, year, metric = parts
                bucket = coverage.setdefault((ticker, year), {"consolidated": set(), "separate": set()})
                for path in paths:
                    lowered = path.casefold()
                    for scope in ("consolidated", "separate"):
                        if f"_{scope}" in lowered and self._path_has_metric_row(path, metric):
                            bucket[scope].add(metric)
            for key, scopes in coverage.items():
                consolidated = len(scopes["consolidated"])
                separate = len(scopes["separate"])
                self._preferred_scope[key] = "consolidated" if consolidated >= separate else "separate"
        for missing in bundle.missing_requirements:
            table.add_missing(
                MissingFact(
                    missing.ticker,
                    str(missing.year),
                    missing.metric,
                    missing.reason,
                    (),
                    f"required statement types: {', '.join(missing.statement_types)}",
                )
            )

        # The bundle keys are the authoritative retrieval requirements.
        requirements: list[tuple[str, str, str]] = []
        for key in bundle.metric_paths:
            parts = key.split("|", 2)
            if len(parts) == 3 and parts[2] in self.registry.names():
                requirements.append((parts[0], parts[1], parts[2]))

        # Also cover planner requirements so a missing retrieval key becomes an
        # explicit retry target instead of disappearing from diagnostics.
        default_years = plan.years or [""]
        for ticker in plan.tickers:
            for metric in plan.required_metrics:
                base = metric.removesuffix("_previous")
                if base not in self.registry.names() or self.registry.get(base).derived:
                    continue
                years = plan.metric_years.get(base, default_years)
                for year in years:
                    requirements.append((ticker, str(year), base))
        for ticker, year, metric in dict.fromkeys(requirements):
            self.extract_fact(table, bundle, ticker, year, metric)
        return table


class ComplexSolver:
    def __init__(
        self,
        registry: MetricRegistry | None = None,
        extractor: SemanticExtractor | None = None,
    ) -> None:
        self.registry = registry or DEFAULT_REGISTRY
        self.extractor = extractor or SemanticExtractor(self.registry)
        self._plan: QuestionPlan | None = None
        self._bundle: EvidenceBundle | None = None
        self._table: SemanticTable | None = None
        self._cache: dict[tuple[str, str, str], MetricValue] = {}

    @staticmethod
    def _previous_year(plan: QuestionPlan, current_year: str) -> str:
        numeric = sorted(int(year) for year in plan.years if str(year).isdigit())
        if current_year.isdigit():
            prior = [year for year in numeric if year < int(current_year)]
            if prior:
                return str(prior[-1])
            return str(int(current_year) - 1)
        raise StructuredSolveFailure(
            "invalid_period", f"Cannot derive previous year from {current_year!r}"
        )

    @staticmethod
    def _immediately_previous_year(current_year: str) -> str:
        if current_year.isdigit():
            return str(int(current_year) - 1)
        raise StructuredSolveFailure(
            "invalid_period", f"Cannot derive previous year from {current_year!r}"
        )

    def _requirement_year(self, metric: str, required: str, current_year: str) -> str:
        if not required.endswith("_previous"):
            return current_year
        assert self._plan is not None
        # Change metrics compare the explicit endpoints named in the question.
        # Average-balance formulas (ROE/ROA/inventory days/turnover) require the
        # immediately preceding annual value even when the question names two
        # non-consecutive endpoint years.
        if metric in {
            "revenue_growth", "gross_margin_change", "inventory_share_change",
            "operating_leverage",
        }:
            return self._previous_year(self._plan, current_year)
        return self._immediately_previous_year(current_year)

    def _subject_metric(self, generic_metric: str) -> str:
        assert self._plan is not None
        ignored = {"cagr", "percentage_change", "percentage_point_change", generic_metric}
        candidates = [
            metric for metric in self._plan.mentioned_metrics
            if metric not in ignored and metric in self.registry.names()
        ]
        if generic_metric == "percentage_point_change":
            derived = [metric for metric in candidates if self.registry.get(metric).derived]
            if derived:
                return derived[-1]
        bases = [metric for metric in candidates if not self.registry.get(metric).derived]
        if bases:
            return bases[-1]
        if candidates:
            return candidates[-1]
        raise StructuredSolveFailure(
            "ambiguous_formula_subject",
            f"Planner did not identify the subject metric for {generic_metric}",
            details={"mentioned_metrics": self._plan.mentioned_metrics},
        )

    def _missing_failure(self, ticker: str, year: str, metric: str) -> StructuredSolveFailure:
        assert self._table is not None
        matching = [
            item for item in self._table.missing
            if item.ticker == ticker and item.year == str(year) and item.metric == metric
        ]
        if not matching:
            matching = [MissingFact(ticker, str(year), metric, "fact_not_extracted")]
        return StructuredSolveFailure(
            "missing_metric_facts",
            f"Missing {metric} for {ticker} {year}",
            missing=matching,
        )

    @staticmethod
    def _output_semantics(definition: MetricDefinition, inputs: Sequence[MetricValue]) -> tuple[str, str]:
        kind = definition.output_kind
        if kind == "percent":
            return "%", UnitDimension.RATIO.value
        if kind == "percentage_point":
            return "điểm phần trăm", UnitDimension.PERCENTAGE_POINT.value
        if kind == "times":
            return "lần", UnitDimension.RATIO.value
        if kind == "days":
            return "ngày", "days"
        currency_dimensions = _stable_unique(
            item.dimension
            for item in inputs
            if item.dimension in {UnitDimension.VND.value, UnitDimension.USD.value}
        )
        if len(currency_dimensions) > 1:
            raise StructuredSolveFailure(
                "incompatible_units",
                "Formula combines VND and USD facts without an exchange rate",
                details={"dimensions": currency_dimensions},
            )
        if currency_dimensions:
            dimension = currency_dimensions[0]
            return ("VND" if dimension == UnitDimension.VND.value else "USD"), dimension
        return (inputs[0].unit, inputs[0].dimension) if inputs else ("", "unknown")

    def _fact_value(self, ticker: str, year: str, metric: str) -> MetricValue:
        assert self._table is not None
        fact = self._table.get(ticker, year, metric)
        if fact is None:
            assert self._bundle is not None
            fact = self.extractor.extract_fact(self._table, self._bundle, ticker, year, metric)
        if fact is None:
            raise self._missing_failure(ticker, year, metric)
        return MetricValue(
            metric,
            fact.value,
            fact.expression,
            fact.base_unit,
            fact.dimension,
            frozenset({(ticker, year, metric)}),
        )

    def _inventory_days_change(self, ticker: str, year: str) -> MetricValue:
        assert self._plan is not None
        previous_endpoint = self._previous_year(self._plan, year)
        current = self._metric_value(ticker, year, "inventory_days")
        previous = self._metric_value(ticker, previous_endpoint, "inventory_days")
        return MetricValue(
            "inventory_days_change",
            current.value - previous.value,
            f"({current.expression}) - ({previous.expression})",
            "ngày",
            "days",
            current.fact_keys | previous.fact_keys,
        )

    def _generic_value(self, ticker: str, year: str, metric: str) -> MetricValue:
        assert self._plan is not None
        subject = self._subject_metric(metric)
        years = sorted((str(item) for item in self._plan.years), key=lambda item: int(item) if item.isdigit() else item)
        current_year = years[-1] if years else year
        previous_year = years[0] if len(years) > 1 else self._previous_year(self._plan, current_year)
        current = self._metric_value(ticker, current_year, subject)
        previous = self._metric_value(ticker, previous_year, subject)
        if metric == "cagr":
            periods = int(current_year) - int(previous_year)
            if previous.value == 0 or periods <= 0 or current.value / previous.value < 0:
                raise StructuredSolveFailure("invalid_formula_domain", "CAGR endpoints/period are invalid")
            value = ((current.value / previous.value) ** (1.0 / periods) - 1.0) * 100.0
            expression = f"((({current.expression}) / ({previous.expression})) ** (1.0 / {periods}) - 1.0) * 100"
            unit, dimension = "%", UnitDimension.RATIO.value
        elif metric == "percentage_change":
            if previous.value == 0:
                raise StructuredSolveFailure("zero_denominator", "Percentage-change denominator is zero")
            value = (current.value - previous.value) / previous.value * 100.0
            expression = f"(({current.expression}) - ({previous.expression})) / ({previous.expression}) * 100"
            unit, dimension = "%", UnitDimension.RATIO.value
        else:
            value = current.value - previous.value
            expression = f"({current.expression}) - ({previous.expression})"
            unit, dimension = "điểm phần trăm", UnitDimension.PERCENTAGE_POINT.value
        return MetricValue(metric, value, expression, unit, dimension, current.fact_keys | previous.fact_keys)

    def _metric_value(self, ticker: str, year: str, metric: str) -> MetricValue:
        key = (ticker, str(year), metric)
        if key in self._cache:
            return self._cache[key]
        if metric == "stressed_net_assets":
            assert self._plan is not None
            q = normalize_metric_text(self._plan.question)
            haircut_mentions = list(re.finditer(r"giam\s+(\d+(?:[.,]\d+)?)\s*(?:%|phan tram)", q))

            def closest_haircut(phrase: str) -> float | None:
                position = q.find(phrase)
                preceding = [match for match in haircut_mentions if 0 <= match.end() <= position]
                if not preceding:
                    return None
                return float(preceding[-1].group(1).replace(",", ".")) / 100.0

            receivable_haircut = closest_haircut("phai thu")
            inventory_haircut = closest_haircut("hang ton kho")
            if receivable_haircut is None or inventory_haircut is None:
                raise StructuredSolveFailure(
                    "unsupported_scenario",
                    "Stressed-net-assets filter requires explicit receivables and inventory haircuts",
                )
            assets = self._metric_value(ticker, str(year), "total_assets")
            liabilities = self._metric_value(ticker, str(year), "total_liabilities")
            short_receivables = self._metric_value(ticker, str(year), "short_term_receivables")
            long_receivables = self._metric_value(ticker, str(year), "long_term_receivables")
            inventory = self._metric_value(ticker, str(year), "inventory")
            inputs = (assets, liabilities, short_receivables, long_receivables, inventory)
            currency_dimensions = {
                item.dimension for item in inputs
                if item.dimension in {UnitDimension.VND.value, UnitDimension.USD.value}
            }
            if len(currency_dimensions) > 1:
                raise StructuredSolveFailure(
                    "incompatible_units", "Stressed-net-assets scenario mixes VND and USD facts"
                )
            numeric = (
                assets.value
                - receivable_haircut * (short_receivables.value + long_receivables.value)
                - inventory_haircut * inventory.value
                - liabilities.value
            )
            expression = (
                f"({assets.expression}) - {receivable_haircut!r} * "
                f"(({short_receivables.expression}) + ({long_receivables.expression})) - "
                f"{inventory_haircut!r} * ({inventory.expression}) - ({liabilities.expression})"
            )
            fact_keys: set[tuple[str, str, str]] = set()
            for item in inputs:
                fact_keys.update(item.fact_keys)
            unit = assets.unit or liabilities.unit
            dimension = assets.dimension if assets.dimension != "unknown" else liabilities.dimension
            value = MetricValue(metric, numeric, expression, unit, dimension, frozenset(fact_keys))
            self._cache[key] = value
            return value
        if metric not in self.registry.names():
            raise StructuredSolveFailure("unknown_metric", f"Unknown metric {metric!r}")
        definition = self.registry.get(metric)
        if not definition.derived:
            value = self._fact_value(ticker, str(year), metric)
            self._cache[key] = value
            return value
        if metric == "inventory_days_change":
            value = self._inventory_days_change(ticker, str(year))
            self._cache[key] = value
            return value
        if metric in {"cagr", "percentage_change", "percentage_point_change"}:
            value = self._generic_value(ticker, str(year), metric)
            self._cache[key] = value
            return value

        input_values: dict[str, float] = {}
        input_expressions: dict[str, str] = {}
        inputs: list[MetricValue] = []
        fact_keys: set[tuple[str, str, str]] = set()
        assert self._plan is not None
        for required in definition.required_metrics:
            base = required.removesuffix("_previous")
            required_year = self._requirement_year(metric, required, str(year))
            item = self._metric_value(ticker, required_year, base)
            input_values[required] = item.value
            input_expressions[required] = item.expression
            fact_keys.update(item.fact_keys)
            inputs.append(item)
        try:
            numeric = self.registry.evaluate(metric, input_values)
            expression = self.registry.build_expression(metric, input_expressions)
        except FormulaError as exc:
            raise StructuredSolveFailure(
                "formula_error", f"Cannot evaluate {metric}: {exc}",
                details={"ticker": ticker, "year": year, "metric": metric},
            ) from exc
        unit, dimension = self._output_semantics(definition, inputs)
        value = MetricValue(metric, numeric, expression, unit, dimension, frozenset(fact_keys))
        self._cache[key] = value
        return value

    @staticmethod
    def _series_expression(values: Mapping[str, MetricValue]) -> str:
        items = ", ".join(f"{candidate!r}: ({item.expression})" for candidate, item in values.items())
        return f"pd.Series({{{items}}}, dtype='float64')"

    @staticmethod
    def _compare(value: float, operator: str, threshold: float) -> bool:
        return {
            ">": value > threshold,
            "<": value < threshold,
            ">=": value >= threshold,
            "<=": value <= threshold,
            "==": value == threshold,
            "!=": value != threshold,
            ">0": value > 0,
            "<0": value < 0,
        }.get(operator, False)

    @staticmethod
    def _comparison_expression(series: str, operator: str, threshold: float | str) -> str:
        if operator in {">0", "<0"}:
            return f"({series}) {operator[0]} 0"
        if operator in {">", "<", ">=", "<=", "==", "!="}:
            return f"({series}) {operator} {float(threshold)!r}"
        if operator in {">median", "<median"}:
            symbol = ">" if operator.startswith(">") else "<"
            return f"({series}) {symbol} ({series}).median()"
        raise StructuredSolveFailure("unsupported_filter", f"Unsupported filter operator {operator!r}")

    def _candidates(self, plan: QuestionPlan) -> tuple[list[str], str]:
        if len(plan.tickers) > 1:
            return list(plan.tickers), "company"
        if plan.grouping == "year" and (plan.selection_operation or plan.aggregation):
            return list(plan.years), "year"
        if plan.tickers:
            return [plan.tickers[0]], "company"
        raise StructuredSolveFailure("missing_entities", "Planner found no company ticker")

    @staticmethod
    def _context(candidate: str, axis: str, plan: QuestionPlan) -> tuple[str, str]:
        if axis == "company":
            years = [str(year) for year in plan.years]
            if not years:
                raise StructuredSolveFailure("missing_periods", "Planner found no year")
            year = max(years, key=lambda item: int(item) if item.isdigit() else item)
            return candidate, year
        return plan.tickers[0], candidate

    def _filter_values(
        self,
        spec: FilterSpec,
        candidates: Sequence[str],
        axis: str,
        plan: QuestionPlan,
    ) -> tuple[dict[str, bool], str, set[tuple[str, str, str]]]:
        if not spec.metric:
            raise StructuredSolveFailure(
                "ambiguous_filter_metric", "Planner produced a filter without a metric",
                details={"filter": asdict(spec)},
            )
        years = list(spec.years)
        masks: dict[str, bool] = {candidate: True for candidate in candidates}
        per_year_expressions: list[str] = []
        fact_keys: set[tuple[str, str, str]] = set()
        evaluation_years = years if axis == "company" and years else [""]
        for filter_year in evaluation_years:
            values: dict[str, MetricValue] = {}
            for candidate in candidates:
                ticker, default_year = self._context(candidate, axis, plan)
                year = str(filter_year or default_year)
                item = self._metric_value(ticker, year, spec.metric)
                values[candidate] = item
                fact_keys.update(item.fact_keys)
            series = self._series_expression(values)
            per_year_expressions.append(self._comparison_expression(series, spec.operator, spec.threshold or 0.0))
            if spec.operator in {">median", "<median"}:
                ordered = [item.value for item in values.values()]
                threshold = float(pd.Series(ordered, dtype="float64").median())
                operator = ">" if spec.operator.startswith(">") else "<"
            else:
                threshold = float(spec.threshold or 0.0)
                operator = spec.operator
            for candidate, item in values.items():
                masks[candidate] = masks[candidate] and self._compare(item.value, operator, threshold)
        expression = " & ".join(f"({item})" for item in per_year_expressions)
        return masks, expression, fact_keys

    @staticmethod
    def _target_unit(value: MetricValue, target_unit: str) -> tuple[float, str, str]:
        if not target_unit:
            return value.value, value.expression, value.unit
        source = resolve_unit(value.unit)
        target = resolve_unit(target_unit)
        if source is None or target is None or source.dimension != target.dimension:
            # Ratio/dimensionless metric (lần, %) — bỏ qua yêu cầu convert sang đơn vị khác.
            # Planner hay detect_target_unit đôi khi bắt nhầm đơn vị từ tên công ty / văn cảnh.
            if source is not None and source.dimension == UnitDimension.RATIO:
                return value.value, value.expression, value.unit
            # Currency metric, target là ratio — bỏ qua tương tự.
            if (source is not None and source.is_currency
                    and target is not None and target.dimension == UnitDimension.RATIO):
                return value.value, value.expression, value.unit
            raise StructuredSolveFailure(
                "incompatible_target_unit",
                f"Cannot convert deterministic result from {value.unit!r} to {target_unit!r}",
            )
        factor = source.scale / target.scale
        expression = value.expression if factor == 1.0 else f"({value.expression}) * {factor!r}"
        return value.value * factor, expression, target.name

    def solve(self, plan: QuestionPlan, bundle: EvidenceBundle) -> SolveResult:
        if not plan.is_complex:
            raise StructuredSolveFailure(
                "not_complex", "ComplexSolver only handles analytical plans",
                details={"question_type": plan.question_type.value},
            )
        if not plan.target_metric:
            raise StructuredSolveFailure("missing_target_metric", "Planner found no target metric")

        self._plan, self._bundle = plan, bundle
        self._table = self.extractor.extract(plan, bundle)
        self._cache = {}
        candidates, axis = self._candidates(plan)

        filter_mask = {candidate: True for candidate in candidates}
        mask_expressions: list[str] = []
        prefix_masks: list[dict[str, bool]] = []
        used_keys: set[tuple[str, str, str]] = set()
        for spec in plan.filters:
            mask, expression, facts = self._filter_values(spec, candidates, axis, plan)
            for candidate in candidates:
                filter_mask[candidate] = filter_mask[candidate] and mask[candidate]
            mask_expressions.append(expression)
            used_keys.update(facts)
            prefix_masks.append(dict(filter_mask))
        filtered = [candidate for candidate in candidates if filter_mask[candidate]]
        if not filtered:
            raise StructuredSolveFailure(
                "empty_filtered_subset",
                "No candidate satisfies all deterministic filters",
                details={"candidates": candidates, "filters": [asdict(item) for item in plan.filters]},
            )
        combined_mask = " & ".join(f"({item})" for item in mask_expressions) if mask_expressions else ""

        selection_values: dict[str, MetricValue] = {}
        selected = list(filtered)
        selection_series = ""
        selection_method = ""
        if plan.selection_operation is not None:
            metric = plan.selection_operation.metric
            if not metric:
                raise StructuredSolveFailure(
                    "ambiguous_selection_metric", "Planner produced a selection without a metric"
                )
            for candidate in candidates:
                ticker, year = self._context(candidate, axis, plan)
                item = self._metric_value(ticker, year, metric)
                selection_values[candidate] = item
                used_keys.update(item.fact_keys)
            eligible_values = {candidate: selection_values[candidate].value for candidate in filtered}
            operation = plan.selection_operation.operation
            if operation in {"max", "max_change", "argmax", "highest"}:
                chosen = max(eligible_values, key=eligible_values.get)
                selection_method = "idxmax"
            elif operation in {"min", "min_change", "argmin", "lowest"}:
                chosen = min(eligible_values, key=eligible_values.get)
                selection_method = "idxmin"
            else:
                raise StructuredSolveFailure("unsupported_selection", f"Unsupported selection {operation!r}")
            selected = [chosen]
            selection_series = self._series_expression(selection_values)

        target_values: dict[str, MetricValue] = {}
        converted_values: dict[str, float] = {}
        converted_expressions: dict[str, str] = {}
        result_unit = ""
        for candidate in candidates:
            if plan.aggregation == "count":
                target_values[candidate] = MetricValue(
                    plan.target_metric, 1.0, "1.0", "count", "count", frozenset()
                )
                converted_values[candidate] = 1.0
                converted_expressions[candidate] = "1.0"
                result_unit = "count"
                continue
            ticker, year = self._context(candidate, axis, plan)
            item = self._metric_value(ticker, year, plan.target_metric)
            target_values[candidate] = item
            used_keys.update(item.fact_keys)
            # Count does not consume a target measure, and subset share first
            # aggregates the raw target measure before producing a percentage.
            # A '%' phrase in either question must therefore not be interpreted
            # as a request to convert every currency input into percent.
            requested_unit = "" if plan.aggregation in {"count", "share"} else plan.target_unit
            try:
                numeric, expression, unit = self._target_unit(item, requested_unit)
            except StructuredSolveFailure as exc:
                if requested_unit and (not item.unit or item.dimension == "unknown"):
                    assert self._table is not None
                    unknown_facts = [
                        fact for fact in self._table.facts
                        if (fact.ticker, fact.year, fact.metric) in item.fact_keys
                        and (not fact.source_unit or fact.dimension == "unknown")
                    ]
                    missing = [
                        MissingFact(
                            fact.ticker,
                            fact.year,
                            fact.metric,
                            "unknown_row_unit",
                            (fact.path,),
                            f"selected row {fact.variable}.iloc[{fact.row_position}] has empty/unparseable Don_vi",
                        )
                        for fact in unknown_facts
                    ]
                    raise StructuredSolveFailure(
                        "missing_metric_units",
                        "Target-unit conversion requires row-level units that are absent",
                        missing=missing,
                    ) from exc
                raise
            converted_values[candidate] = numeric
            converted_expressions[candidate] = expression
            result_unit = unit
        target_series = self._series_expression({
            candidate: MetricValue(
                plan.target_metric,
                converted_values[candidate],
                converted_expressions[candidate],
                result_unit,
                target_values[candidate].dimension,
                target_values[candidate].fact_keys,
            )
            for candidate in candidates
        })

        eligible_expression = target_series
        if combined_mask:
            eligible_expression = f"({target_series})[{combined_mask}]"
        if selection_series:
            selector = selection_series
            if combined_mask:
                selector = f"({selection_series})[{combined_mask}]"
            selected_index = f"({selector}).{selection_method}()"
            if plan.aggregation:
                eligible_expression = f"({target_series}).loc[[{selected_index}]]"
            else:
                eligible_expression = f"({target_series}).loc[{selected_index}]"

        aggregation = plan.aggregation
        values_for_result = [converted_values[candidate] for candidate in selected]
        if aggregation == "count":
            expected = len(values_for_result)
            if combined_mask:
                query = f"({combined_mask}).sum()"
            else:
                ones = ", ".join(f"{candidate!r}: 1.0" for candidate in candidates)
                query = f"pd.Series({{{ones}}}, dtype='float64').count()"
            result_unit = "count"
        elif aggregation in {"average", "mean"}:
            expected = float(sum(values_for_result) / len(values_for_result))
            query = f"({eligible_expression}).mean()"
        elif aggregation == "sum":
            expected = float(sum(values_for_result))
            query = f"({eligible_expression}).sum()"
        elif aggregation == "share":
            # Filters are sequential stages.  The numerator uses every stage;
            # for a nested scenario the denominator is the population entering
            # the final filter (all preceding filters).  With a single filter,
            # the denominator remains the original candidate set.
            if len(prefix_masks) >= 2:
                denominator_candidates = [
                    candidate for candidate in candidates if prefix_masks[-2][candidate]
                ]
                denominator_mask = " & ".join(
                    f"({item})" for item in mask_expressions[:-1]
                )
                denominator_expression = f"({target_series})[{denominator_mask}]"
            else:
                denominator_candidates = list(candidates)
                denominator_expression = target_series
            denominator = float(sum(converted_values[item] for item in denominator_candidates))
            if denominator == 0:
                raise StructuredSolveFailure("zero_denominator", "Subset-share denominator is zero")
            expected = float(sum(values_for_result) / denominator * 100.0)
            query = f"({eligible_expression}).sum() / ({denominator_expression}).sum() * 100"
            result_unit = "%"
        elif aggregation in {"max", "min"}:
            expected = max(values_for_result) if aggregation == "max" else min(values_for_result)
            query = f"({eligible_expression}).{aggregation}()"
        elif aggregation is None:
            if len(selected) != 1:
                raise StructuredSolveFailure(
                    "ambiguous_multiple_results",
                    "Analytical plan yields multiple target values without selection or aggregation",
                    details={"candidates": selected},
                )
            expected = converted_values[selected[0]]
            if selection_series:
                query = eligible_expression
            elif combined_mask:
                query = f"({eligible_expression}).iloc[0]"
            else:
                query = converted_expressions[selected[0]]
        else:
            raise StructuredSolveFailure("unsupported_aggregation", f"Unsupported aggregation {aggregation!r}")

        variables = referenced_variables(query)
        dfs = {variable: self._table.frames[variable] for variable in variables if variable in self._table.frames}
        missing_variables = sorted(variables - set(dfs))
        if missing_variables:
            raise StructuredSolveFailure(
                "evidence_variable_mismatch",
                "Query references dataframes absent from final evidence",
                details={"missing_variables": missing_variables},
            )
        try:
            answer = execute_expression(query, dfs)
        except (QueryExecutionError, QueryFormatError) as exc:
            raise StructuredSolveFailure("query_execution_failed", str(exc), details={"query": query}) from exc
        if not math.isclose(float(answer), float(expected), rel_tol=1e-12, abs_tol=1e-9):
            raise StructuredSolveFailure(
                "answer_query_mismatch",
                "Deterministic calculation disagrees with its final pandas expression",
                details={"expected": expected, "query_result": answer, "query": query},
            )

        used_facts = [
            fact for fact in self._table.facts
            if (fact.ticker, fact.year, fact.metric) in used_keys and fact.variable in variables
        ]
        used_paths = _stable_unique(fact.path for fact in used_facts)
        used_variables = _stable_unique(fact.variable for fact in used_facts)
        confidence = min((min(fact.match_score / 120.0, 1.0) for fact in used_facts), default=0.0)
        if any(fact.ambiguous_matches for fact in used_facts):
            confidence *= 0.85
        if any(fact.dimension == "unknown" for fact in used_facts):
            confidence *= 0.80
        validation = {
            "query_executed": True,
            "answer_query_match": True,
            "all_query_variables_mapped": True,
            "single_row_fallback_used": False,
            "required_entity_count": len(plan.tickers),
            "covered_entities": _stable_unique(fact.ticker for fact in used_facts),
            "covered_years": _stable_unique(fact.year for fact in used_facts),
            "warnings": list(self._table.warnings),
        }
        return SolveResult(
            answer=answer,
            pandas_query=query,
            unit=result_unit,
            semantic_table=self._table,
            used_facts=used_facts,
            used_paths=used_paths,
            used_variables=used_variables,
            selected_candidates=selected,
            filtered_candidates=filtered,
            confidence=confidence,
            validation=validation,
        )

    def solve_safe(self, plan: QuestionPlan, bundle: EvidenceBundle) -> SolveResult | dict[str, Any]:
        try:
            return self.solve(plan, bundle)
        except StructuredSolveFailure as exc:
            return exc.to_dict()


def solve_complex_question(
    plan: QuestionPlan,
    bundle: EvidenceBundle,
    registry: MetricRegistry | None = None,
) -> SolveResult:
    return ComplexSolver(registry=registry).solve(plan, bundle)


__all__ = [
    "ComplexSolver",
    "MetricValue",
    "MissingFact",
    "SemanticExtractor",
    "SemanticFact",
    "SemanticTable",
    "SolveResult",
    "StructuredSolveFailure",
    "solve_complex_question",
]
