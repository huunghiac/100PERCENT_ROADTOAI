"""Deterministic question analysis for simple and analytical ViFinQA tasks."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Callable, Mapping, Sequence

try:  # Support both ``src.question_planner`` and top-level imports.
    from .metric_registry import DEFAULT_REGISTRY, MetricRegistry, normalize_metric_text
    from .units import detect_target_unit
except ImportError:  # pragma: no cover - exercised by legacy entry points
    from metric_registry import DEFAULT_REGISTRY, MetricRegistry, normalize_metric_text
    from units import detect_target_unit


class QuestionType(str, Enum):
    SIMPLE_LOOKUP = "SIMPLE_LOOKUP"
    RATIO = "RATIO"
    MULTI_YEAR = "MULTI_YEAR"
    MULTI_COMPANY = "MULTI_COMPANY"
    FILTER_THEN_SELECT = "FILTER_THEN_SELECT"
    AGGREGATION = "AGGREGATION"
    SCENARIO = "SCENARIO"
    MULTI_STAGE_ANALYTICAL = "MULTI_STAGE_ANALYTICAL"


class Scope(str, Enum):
    SEPARATE = "separate"
    CONSOLIDATED = "consolidated"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FilterSpec:
    metric: str | None
    operator: str
    threshold: float | str | None = None
    years: tuple[str, ...] = ()
    raw_text: str = ""


@dataclass(frozen=True)
class SelectionSpec:
    operation: str
    metric: str | None
    raw_text: str = ""


@dataclass
class QuestionPlan:
    question: str
    question_type: QuestionType
    tickers: list[str]
    years: list[str]
    scope: Scope
    target_metric: str | None
    required_metrics: list[str]
    filters: list[FilterSpec] = field(default_factory=list)
    grouping: str | None = None
    selection_operation: SelectionSpec | None = None
    aggregation: str | None = None
    comparison: str | None = None
    time_operation: str | None = None
    formula: str | None = None
    target_unit: str = ""
    mentioned_metrics: list[str] = field(default_factory=list)
    metric_years: dict[str, list[str]] = field(default_factory=dict)
    period_roles: list[str] = field(default_factory=list)
    operations: list[str] = field(default_factory=list)
    complexity_reasons: list[str] = field(default_factory=list)

    @property
    def is_complex(self) -> bool:
        return self.question_type != QuestionType.SIMPLE_LOOKUP

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["question_type"] = self.question_type.value
        result["scope"] = self.scope.value
        result["is_complex"] = self.is_complex
        return result


_NOISE_TICKERS = {
    "CFO", "ROA", "ROE", "EPS", "EBIT", "EBITDA", "COGS", "CAGR", "DOH",
    "NIM", "CIR", "NPL", "CAR", "LDR", "SGA", "SG", "TNDN",
    "VND", "USD", "CTCP", "TNHH", "TMCP", "TCTD", "NHNN", "HĐQT", "BCTC",
}

# Reusable public-name aliases.  The retriever augments these from code_stock.csv.
_COMMON_ALIASES = {
    "hoa phat": "HPG", "tap doan hoa phat": "HPG", "hoa sen": "HSG",
    "tap doan hoa sen": "HSG", "nam kim": "NKG", "thep nam kim": "NKG",
    "masan": "MSN", "tap doan masan": "MSN", "vinamilk": "VNM",
    "dai duong": "OGC", "minh phu": "MPC", "dabaco": "DBC",
    "dam phu my": "DPM", "dam ca mau": "DCM", "binh son": "BSR",
    "pvtrans": "PVT", "vinhomes": "VHM", "vincom retail": "VRE",
    "vingroup": "VIC", "do thi kinh bac": "KBC", "hai phat": "HPX",
    "van phu invest": "VPI", "the gioi di dong": "MWG", "fpt": "FPT",
    "vietjet": "VJC", "acb": "ACB", "bao viet": "BVH",
}


def _default_entities(question: str) -> tuple[str | None, str | None, list[str], list[str]]:
    folded = normalize_metric_text(question)
    positioned: list[tuple[int, str]] = []
    for alias, ticker in sorted(_COMMON_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        match = re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", folded)
        if match:
            positioned.append((match.start(), ticker))
    for match in re.finditer(r"\b([A-Z][A-Z0-9]{1,4})\b", question):
        ticker = match.group(1)
        if ticker not in _NOISE_TICKERS:
            positioned.append((match.start(), ticker))
    tickers: list[str] = []
    for _, ticker in sorted(positioned):
        if ticker not in tickers:
            tickers.append(ticker)

    # Anchor filter: nếu câu có đúng 1 ticker ngoặc đơn "(GEE)" không phải noise
    # → chỉ giữ anchor, loại các ticker từ partial name match.
    paren_tickers = [
        t for t in re.findall(r'\(([A-Z]{2,5})\)', question)
        if t not in _NOISE_TICKERS
    ]
    if len(paren_tickers) == 1 and len(tickers) > 1:
        tickers = [paren_tickers[0]]

    years: list[str] = []
    for match in re.finditer(r"\b(20\d{2})\s*[-–—]\s*(20\d{2})\b", question):
        start, end = int(match.group(1)), int(match.group(2))
        if start <= end and end - start <= 15:
            years.extend(str(year) for year in range(start, end + 1))
    for year in re.findall(r"\b(20\d{2})\b", question):
        if year not in years:
            years.append(year)
    years = list(dict.fromkeys(years))
    return (tickers[0] if tickers else None, years[0] if years else None, tickers, years)


class QuestionPlanner:
    def __init__(
        self,
        entity_resolver: Any | None = None,
        registry: MetricRegistry | None = None,
    ) -> None:
        self.entity_resolver = entity_resolver
        self.registry = registry or DEFAULT_REGISTRY

    def _entities(self, question: str) -> tuple[list[str], list[str]]:
        resolver = self.entity_resolver
        result: Any
        if resolver is None:
            result = _default_entities(question)
        elif hasattr(resolver, "extract_all_entities"):
            result = resolver.extract_all_entities(question)
        elif callable(resolver):
            result = resolver(question)
        else:
            raise TypeError("entity_resolver must be callable or expose extract_all_entities")
        if isinstance(result, tuple) and len(result) == 4:
            tickers, years = list(result[2]), list(result[3])
        elif isinstance(result, Mapping):
            tickers = list(result.get("tickers", []))
            years = [str(year) for year in result.get("years", [])]
        else:
            raise TypeError("Entity resolver returned an unsupported shape")
        return list(dict.fromkeys(tickers)), list(dict.fromkeys(str(y) for y in years))

    @staticmethod
    def _scope(normalized: str) -> Scope:
        if any(phrase in normalized for phrase in ("cong ty me", "bao cao rieng", "rieng le")):
            return Scope.SEPARATE
        if any(phrase in normalized for phrase in ("hop nhat", "toan tap doan", "bao cao tong hop")):
            return Scope.CONSOLIDATED
        return Scope.UNKNOWN

    def _metric_positions(self, question: str) -> list[tuple[int, str]]:
        """Return semantic metric mentions with longest-span precedence.

        Financial aliases are heavily nested: ``tổng nợ`` is contained in
        ``tổng nợ ngắn hạn`` and ``lợi nhuận gộp`` is contained in ``biên lợi
        nhuận gộp``.  Treating every substring as an independent mention was a
        major source of wrong plans.  We retain non-overlapping longest spans,
        while allowing a metric to occur again in a different clause.
        """

        normalized = normalize_metric_text(question)
        occurrences: list[tuple[int, int, int, str]] = []
        for name in self.registry.names():
            definition = self.registry.get(name)
            for alias in (name.replace("_", " "), *definition.aliases):
                alias_norm = normalize_metric_text(alias)
                for match in re.finditer(rf"(?<![a-z0-9]){re.escape(alias_norm)}(?![a-z0-9])", normalized):
                    occurrences.append((match.start(), match.end(), len(alias_norm), name))

        accepted: list[tuple[int, int, int, str]] = []
        for candidate in sorted(occurrences, key=lambda item: (-item[2], item[0], item[3])):
            start, end, _, _ = candidate
            if any(not (end <= old_start or start >= old_end) for old_start, old_end, _, _ in accepted):
                continue
            accepted.append(candidate)
        return [(start, name) for start, _, _, name in sorted(accepted)]

    def _target_metric(self, question: str, positions: list[tuple[int, str]]) -> str | None:
        if not positions:
            return None
        normalized = normalize_metric_text(question)
        if re.search(r"chenh lech(?: binh quan)? giua bien loi nhuan gop va bien loi nhuan rong", normalized):
            return "margin_spread"
        if re.search(r"loi nhuan gop.+?(?:bang|chiem).+?phan tram.+?doanh thu thuan", normalized):
            return "gross_margin"
        if re.search(r"loi nhuan sau thue.+?(?:bang|chiem).+?phan tram.+?doanh thu thuan", normalized):
            return "net_margin"
        if re.search(r"loi nhuan (?:thuan tu )?hoat dong.+?phan tram.+?doanh thu", normalized):
            return "operating_margin"

        def prefer_derived(candidates: list[tuple[int, str]]) -> str | None:
            derived = [item for item in candidates if self.registry.get(item[1]).derived]
            chosen = derived[-1] if derived else (candidates[-1] if candidates else None)
            return chosen[1] if chosen else None

        # In "giá trị X của doanh nghiệp có Y ..." X is the requested result,
        # while Y is only a selection metric.
        selection_boundary = re.search(
            r"\bcua (?:cac )?(?:doanh nghiep|cong ty|ma|nam)(?: co)?\b",
            normalized,
        )
        if selection_boundary:
            prefix_candidates = [item for item in positions if item[0] < selection_boundary.start()]
            if prefix_candidates:
                # The output is the metric immediately before "của ... có";
                # a derived filter earlier in the sentence must not override it.
                return prefix_candidates[-1][1]
        # Otherwise analytical questions conventionally state the output metric
        # in the final clause. Prefer the last derived/base mention.
        return prefer_derived(positions)

    def _metric_year_mapping(
        self, question: str, mentioned: Sequence[str], explicit_years: Sequence[str]
    ) -> dict[str, list[str]]:
        normalized = normalize_metric_text(question)
        mentioned_years: dict[str, list[str]] = {}
        for name in mentioned:
            definition = self.registry.get(name)
            occurrences: list[tuple[int, int]] = []
            for alias in (name.replace("_", " "), *definition.aliases):
                alias_norm = normalize_metric_text(alias)
                occurrences.extend(
                    (match.start(), match.end())
                    for match in re.finditer(rf"(?<![a-z0-9]){re.escape(alias_norm)}(?![a-z0-9])", normalized)
                )
            local_years: list[str] = []
            for start, end in occurrences:
                window = normalized[max(0, start - 90): min(len(normalized), end + 90)]
                for range_match in re.finditer(r"\b(20\d{2})\s+(?:den|toi)\s+(20\d{2})\b", window):
                    first, last = int(range_match.group(1)), int(range_match.group(2))
                    if first <= last and last - first <= 15:
                        local_years.extend(str(year) for year in range(first, last + 1))
                local_years.extend(re.findall(r"\b(20\d{2})\b", window))
            mentioned_years[name] = list(dict.fromkeys(local_years or explicit_years))
            if (
                len(explicit_years) > 2
                and explicit_years[0] in mentioned_years[name]
                and explicit_years[-1] in mentioned_years[name]
            ):
                mentioned_years[name] = list(explicit_years)

        mapping: dict[str, list[str]] = {}
        for name, metric_years in mentioned_years.items():
            definition = self.registry.get(name)
            requirements = definition.required_metrics or (name,)
            for requirement in requirements:
                base = requirement.removesuffix("_previous")
                if base not in self.registry.names():
                    continue
                years_for_base = list(metric_years)
                if requirement.endswith("_previous") and len(metric_years) <= 1:
                    years_for_base = [str(int(year) - 1) for year in metric_years if str(year).isdigit()]
                mapping.setdefault(base, [])
                for year in years_for_base:
                    if year not in mapping[base]:
                        mapping[base].append(year)
        return mapping

    def _operation_metric_year_mapping(
        self,
        question: str,
        years: Sequence[str],
        target_metric: str | None,
        filters: Sequence[FilterSpec],
        selection: SelectionSpec | None,
        mentioned: Sequence[str],
    ) -> dict[str, list[str]]:
        """Map formula inputs to the periods in which they are actually used.

        This replaces the old ±90-character heuristic for analytical plans.
        It is deliberately role-aware: a 2024 median filter and a 2025 output
        do not force every balance-sheet input to be retrieved for both years.
        """

        explicit = [str(year) for year in years]
        if explicit and all(year.isdigit() for year in explicit):
            explicit = sorted(dict.fromkeys(explicit), key=int)
        latest = [explicit[-1]] if explicit else []
        mapping: dict[str, list[str]] = {}

        def add(base: str, needed_years: Sequence[str]) -> None:
            if base not in self.registry.names():
                return
            mapping.setdefault(base, [])
            for year in needed_years:
                if year and year not in mapping[base]:
                    mapping[base].append(str(year))

        def previous_year(current_years: Sequence[str]) -> list[str]:
            if len(current_years) >= 2:
                return [str(current_years[-2])]
            if current_years and str(current_years[-1]).isdigit():
                return [str(int(str(current_years[-1])) - 1)]
            return []

        def add_metric(metric: str | None, metric_years: Sequence[str], *, role: str) -> None:
            if not metric or metric not in self.registry.names():
                return
            definition = self.registry.get(metric)
            selected_years = list(dict.fromkeys(str(year) for year in metric_years))

            if metric == "cagr":
                # CAGR in this dataset is applied to the explicitly mentioned
                # revenue series unless another base series is named.
                base = "net_revenue" if "net_revenue" in mentioned else None
                if base:
                    endpoints = [selected_years[0], selected_years[-1]] if len(selected_years) > 1 else selected_years
                    add(base, endpoints)
                return
            if metric == "inventory_days_change":
                # Two inventory-days observations each require a two-period
                # average inventory.  Infer only the immediately preceding
                # years; no global top-k truncation is involved downstream.
                endpoints = sorted({int(y) for y in selected_years if str(y).isdigit()})
                if len(endpoints) >= 2:
                    comparison_years = [endpoints[-2], endpoints[-1]]
                else:
                    comparison_years = endpoints
                for endpoint in comparison_years:
                    add("inventory", [str(endpoint - 1), str(endpoint)])
                    add("cogs", [str(endpoint)])
                return

            requirements = definition.required_metrics or (metric,)
            for requirement in requirements:
                base = requirement.removesuffix("_previous")
                if base not in self.registry.names():
                    continue
                if requirement.endswith("_previous"):
                    add(base, previous_year(selected_years))
                else:
                    # Change metrics use their final year as current and the
                    # ``_previous`` dependency for the comparison year.
                    if any(req.endswith("_previous") for req in requirements) and role != "series":
                        add(base, selected_years[-1:] if selected_years else [])
                    else:
                        add(base, selected_years)

        # A target after a cross-company selection normally belongs to the
        # latest stated year.  For one-company/multi-year argmax questions it
        # must be available in every candidate year.
        entity_count = len(self._entities(question)[0])
        target_is_base = bool(
            target_metric
            and target_metric in self.registry.names()
            and not self.registry.get(target_metric).derived
        )
        target_is_year_series = (
            entity_count == 1
            and len(explicit) > 1
            and (selection is not None or target_is_base)
        )
        target_years = explicit if target_is_year_series else latest
        add_metric(target_metric, target_years, role="target")

        for spec in filters:
            filter_years = list(spec.years) or explicit
            if spec.metric in {"revenue_growth", "gross_margin_change", "operating_leverage"}:
                filter_years = explicit[-2:] if len(explicit) >= 2 else filter_years
            add_metric(spec.metric, filter_years, role="filter")

        if selection:
            selection_years = explicit
            if selection.metric == "inventory_share_change" and len(explicit) == 1 and explicit[0].isdigit():
                selection_years = [str(int(explicit[0]) - 1), explicit[0]]
            add_metric(selection.metric, selection_years, role="selection")
        return mapping

    def _filters(
        self, question: str, mentioned: list[str], years: list[str]
    ) -> list[FilterSpec]:
        q = normalize_metric_text(question)
        filters: list[FilterSpec] = []
        median_match = re.search(r"(cao hon|lon hon|vuot|tren|thap hon|nho hon|duoi) (?:muc )?trung vi", q)
        if median_match:
            operator = ">median" if median_match.group(1) in {"cao hon", "lon hon", "vuot", "tren"} else "<median"
            metric = mentioned[0] if mentioned else None
            # Use the metric immediately before the comparison phrase.
            positions = self._metric_positions(question)
            before = [item for item in positions if item[0] < median_match.start()]
            if before:
                metric = before[-1][1]
            if re.search(r"hang ton kho binh quan.+?gia von hang ban.+?(?:nhan|x) 365", q):
                metric = "inventory_days"
            local_years = re.findall(r"\b(20\d{2})\b", q[max(0, median_match.start() - 120):median_match.end()])
            if metric == "inventory_days" and local_years:
                # The earlier year is the average-inventory input; the filter
                # itself is one inventory-days observation at the later year.
                local_years = [max(local_years, key=int)]
            filters.append(FilterSpec(metric, operator, "median", tuple(dict.fromkeys(local_years or years)), median_match.group(0)))

        metric_positions = self._metric_positions(question)
        for sign_word, operator in (("duong", ">0"), ("am", "<0")):
            for match in re.finditer(rf"\b{sign_word}\b", q):
                before = [item for item in metric_positions if item[0] < match.start()]
                if not before:
                    continue
                metric = before[-1][1]
                local = q[max(0, match.start() - 100):min(len(q), match.end() + 100)]
                local_years = list(dict.fromkeys(re.findall(r"\b(20\d{2})\b", local)))
                filter_years = local_years or list(years)
                if any(phrase in local for phrase in ("ca hai nam", "ca ba nam", "trong ca hai nam", "trong ca ba nam")):
                    filter_years = list(years)
                if metric in {"revenue_growth", "gross_margin_change", "operating_leverage"} and len(years) >= 2:
                    filter_years = [max((str(year) for year in years), key=int)]
                if (
                    any(phrase in local for phrase in ("ca hai ky", "hai ky", "ky so sanh"))
                    and len(filter_years) == 1
                ):
                    current = int(filter_years[0])
                    filter_years = [str(current - 1), str(current)]
                filters.append(FilterSpec(metric, operator, 0.0, tuple(filter_years), match.group(0)))

        # Directional revenue language denotes a growth filter even when the
        # question does not literally use the noun "tăng trưởng".
        revenue_direction = re.search(
            r"doanh thu thuan(?: nam 20\d{2})?\s+(tang|giam)(?:\s+(?:tren|hon)\s+(-?\d+(?:[.,]\d+)?)\s*%)?\s+so voi",
            q,
        )
        if revenue_direction:
            direction, threshold = revenue_direction.groups()
            operator = ">" if direction == "tang" else "<"
            value = float((threshold or "0").replace(",", "."))
            local = q[revenue_direction.start(): min(len(q), revenue_direction.end() + 80)]
            local_years = list(dict.fromkeys(re.findall(r"\b(20\d{2})\b", local))) or list(years)
            if local_years:
                local_years = [max((str(year) for year in local_years), key=int)]
            filters.append(FilterSpec("revenue_growth", operator, value, tuple(local_years), revenue_direction.group(0)))

        q_numbers = unicodedata.normalize("NFKC", question).casefold().replace("đ", "d")
        q_numbers = "".join(c for c in unicodedata.normalize("NFD", q_numbers) if unicodedata.category(c) != "Mn")
        q_numbers = re.sub(r"(?<=\d),(?=\d)", ".", q_numbers)
        q_numbers = re.sub(r"[^a-z0-9%.]+", " ", q_numbers)
        threshold_pattern = re.compile(
            r"(lon hon|cao hon|tren|nho hon|thap hon|duoi)\s+(-?\d+(?:[,.]\d+)?)\s*(%|lan)?"
        )
        for match in threshold_pattern.finditer(q_numbers):
            if "trung vi" in q_numbers[match.start():match.end() + 20]:
                continue
            before_metrics = [item for item in self._metric_positions(question) if item[0] < match.start()]
            metric = before_metrics[-1][1] if before_metrics else None
            local_before = q_numbers[max(0, match.start() - 80):match.start()]
            if "doanh thu" in local_before and any(word in local_before for word in ("tang", "tang truong")):
                metric = "revenue_growth"
            number = float(match.group(2).replace(",", "."))
            operator = ">" if match.group(1) in {"lon hon", "cao hon", "tren"} else "<"
            filters.append(FilterSpec(metric, operator, number, tuple(years), match.group(0)))
        # Stable de-duplication matters when "dương ở cả hai năm" matches twice.
        unique: list[FilterSpec] = []
        seen = set()
        for spec in filters:
            key = (spec.metric, spec.operator, spec.threshold, spec.years)
            if key not in seen:
                seen.add(key)
                unique.append(spec)
        return unique

    def _selection(self, question: str) -> SelectionSpec | None:
        q = normalize_metric_text(question)
        semantic_patterns = (
            (r"muc tang ty trong hang ton kho(?: tren tong tai san)?.*?(?:cao nhat|lon nhat)", "max_change", "inventory_share_change"),
            (r"muc thay doi bien loi nhuan gop.*?(?:thap nhat|nho nhat)", "min_change", "gross_margin_change"),
            (r"muc giam (?:lon nhat|manh nhat).*?gia tri nay", "min_change", "inventory_days_change"),
            (r"(?:don bay kinh doanh|he so don bay kinh doanh).*?(?:cao nhat|lon nhat)", "max", "operating_leverage"),
            (r"(?:cagr|toc do tang truong kep).*?(?:cao nhat|lon nhat)", "max", "cagr"),
            (r"(?:tang truong|toc do tang|muc tang) doanh thu thuan.*?(?:cao nhat|lon nhat)", "max_change", "revenue_growth"),
            (r"(?:he so|ty so|ty le) thanh toan nhanh.*?(?:thap nhat|nho nhat)", "min", "quick_ratio"),
            (r"(?:ty so d/e|d/e|no phai tra (?:tren|chia cho) von chu so huu).*?(?:cao nhat|lon nhat)", "max", "debt_to_equity"),
        )
        for pattern, operation, metric in semantic_patterns:
            match = re.search(pattern, q)
            if match:
                return SelectionSpec(operation, metric, match.group(0))
        patterns = (
            (r"muc tang .*? lon nhat", "max_change"),
            (r"muc giam .*? lon nhat|sut giam sau nhat", "min_change"),
            (r"(?:cao nhat|lon nhat|manh nhat)", "max"),
            (r"(?:thap nhat|nho nhat)", "min"),
        )
        metric_positions = self._metric_positions(question)
        for pattern, operation in patterns:
            match = re.search(pattern, q)
            if not match:
                continue
            before = [item for item in metric_positions if item[0] < match.start()]
            metric = before[-1][1] if before else None
            return SelectionSpec(operation, metric, match.group(0))
        return None

    @staticmethod
    def _aggregation(normalized: str) -> str | None:
        if re.search(r"\bco bao nhieu (?:doanh nghiep|cong ty|nam)\b", normalized):
            return "count"
        if re.search(r"(?:binh quan|trung binh).{0,100}(?:la|dat|bang) bao nhieu", normalized):
            return "average"
        if any(phrase in normalized for phrase in ("dong gop bao nhieu phan tram", "ty trong tong", "chiem bao nhieu phan tram tong")):
            return "share"
        if any(phrase in normalized for phrase in ("tong cong", "tong gia tri", "tong loi nhuan")):
            return "sum"
        return None

    @staticmethod
    def _time_operation(normalized: str, years: Sequence[str]) -> str | None:
        if "cagr" in normalized or "tang truong kep" in normalized:
            return "cagr"
        if any(phrase in normalized for phrase in ("so voi nam", "tu nam", "sang nam", "tang truong", "muc thay doi", "muc tang", "muc giam")):
            return "change"
        if any(phrase in normalized for phrase in ("nam ngay sau", "nam ke tiep", "nam lien truoc", "nam truoc do")):
            return "relative_year"
        if len(years) > 1:
            return "multi_period"
        if "dau nam" in normalized and "cuoi nam" in normalized:
            return "begin_end_period"
        return None

    def analyze(self, question: str) -> QuestionPlan:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")
        normalized = normalize_metric_text(question)
        tickers, years = self._entities(question)
        positions = self._metric_positions(question)
        mentioned = []
        for _, name in positions:
            if name not in mentioned:
                mentioned.append(name)
        target_metric = self._target_metric(question, positions)
        filters = self._filters(question, mentioned, years)
        selection = self._selection(question)
        aggregation = self._aggregation(normalized)
        time_operation = self._time_operation(normalized, years)
        target_unit = detect_target_unit(question)

        operational_metrics = [target_metric]
        operational_metrics.extend(spec.metric for spec in filters)
        if selection:
            operational_metrics.append(selection.metric)
        required: list[str] = []
        for metric in operational_metrics:
            if metric and metric in self.registry.names():
                required.extend(self.registry.expand_required([metric]))
        # CAGR's registry inputs are abstract endpoints; retrieval needs the
        # concrete series named in the question.
        if selection and selection.metric == "cagr" and "net_revenue" in mentioned:
            required.append("net_revenue")
        required = [name for name in dict.fromkeys(required) if name in self.registry.names()]

        metric_years = self._operation_metric_year_mapping(
            question, years, target_metric, filters, selection, mentioned
        )

        period_roles: list[str] = []
        for metric in mentioned:
            for requirement in self.registry.get(metric).required_metrics:
                if requirement.endswith("_previous") and requirement not in period_roles:
                    period_roles.append(requirement)
        if "dau nam" in normalized and "beginning_period" not in period_roles:
            period_roles.append("beginning_period")
        if "cuoi nam" in normalized and "ending_period" not in period_roles:
            period_roles.append("ending_period")

        scenario = any(
            phrase in normalized
            for phrase in ("gia su", "neu ", "kich ban", "giu nguyen", "giam 30", "giam 50", "tang 20", "phat hanh them", "uoc tinh")
        )
        has_filter = bool(filters)
        has_selection = selection is not None
        derived_count = sum(self.registry.get(metric).derived for metric in mentioned)
        operation_count = sum(bool(value) for value in (has_filter, has_selection, aggregation, time_operation))

        reasons: list[str] = []
        if len(tickers) > 1:
            reasons.append("multiple_companies")
        if len(years) > 1 or time_operation in {"change", "cagr", "relative_year", "multi_period"}:
            reasons.append("multiple_periods")
        if derived_count:
            reasons.append("formula_metrics")
        if has_filter:
            reasons.append("filtering")
        if has_selection:
            reasons.append("selection")
        if aggregation:
            reasons.append("aggregation")
        if scenario:
            reasons.append("scenario")

        if scenario:
            question_type = QuestionType.SCENARIO
        elif has_filter and has_selection and not aggregation and not time_operation:
            # A pure filter -> select pipeline is useful to distinguish from
            # questions that add aggregation or cross-period arithmetic.
            question_type = QuestionType.FILTER_THEN_SELECT
        elif (
            operation_count >= 2
            or (has_filter and (len(tickers) > 1 or len(years) > 1))
            or (has_selection and (len(tickers) > 1 or len(years) > 1) and derived_count > 0)
        ):
            question_type = QuestionType.MULTI_STAGE_ANALYTICAL
        elif aggregation:
            question_type = QuestionType.AGGREGATION
        elif len(tickers) > 1:
            question_type = QuestionType.MULTI_COMPANY
        elif len(years) > 1 or time_operation:
            question_type = QuestionType.MULTI_YEAR
        elif derived_count or (target_metric and self.registry.get(target_metric).derived):
            question_type = QuestionType.RATIO
        else:
            question_type = QuestionType.SIMPLE_LOOKUP

        operations: list[str] = []
        if has_filter:
            operations.append("filter")
        if selection:
            operations.append(selection.operation)
        if aggregation:
            operations.append(aggregation)
        if time_operation:
            operations.append(time_operation)
        if target_metric:
            operations.append(f"compute:{target_metric}")

        formula = target_metric if target_metric and self.registry.get(target_metric).derived else None
        comparison = filters[0].operator if filters else None
        grouping = "company" if len(tickers) > 1 else ("year" if len(years) > 1 else None)
        return QuestionPlan(
            question=question,
            question_type=question_type,
            tickers=tickers,
            years=years,
            scope=self._scope(normalized),
            target_metric=target_metric,
            required_metrics=required,
            filters=filters,
            grouping=grouping,
            selection_operation=selection,
            aggregation=aggregation,
            comparison=comparison,
            time_operation=time_operation,
            formula=formula,
            target_unit=target_unit,
            mentioned_metrics=mentioned,
            metric_years=metric_years,
            period_roles=period_roles,
            operations=operations,
            complexity_reasons=reasons,
        )


def analyze_question(
    question: str,
    entity_resolver: Any | None = None,
    registry: MetricRegistry | None = None,
) -> QuestionPlan:
    return QuestionPlanner(entity_resolver=entity_resolver, registry=registry).analyze(question)


__all__ = [
    "FilterSpec",
    "QuestionPlan",
    "QuestionPlanner",
    "QuestionType",
    "Scope",
    "SelectionSpec",
    "analyze_question",
]
