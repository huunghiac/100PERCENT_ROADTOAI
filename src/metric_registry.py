"""Reusable financial metric and formula registry for ViFinQA.

All formulas live here so retrieval, deterministic execution, validation, and
tests share one definition.  Formula evaluation preserves source signs; no
metric silently applies ``abs`` to financial-statement values.
"""

from __future__ import annotations

import math
import re
import statistics
import unicodedata
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence


def normalize_metric_text(value: object) -> str:
    text = "" if value is None else unicodedata.normalize("NFKC", str(value)).casefold()
    text = text.replace("đ", "d")
    text = "".join(
        char for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )
    text = re.sub(r"[^a-z0-9%]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    # Vietnamese spelling permits both ``tỉ`` and ``tỷ``.  Accent folding
    # yields ``ti`` and ``ty`` respectively, so canonicalize the standalone
    # word to keep aliases deterministic.
    return re.sub(r"\bti\b", "ty", text)


class FormulaError(ValueError):
    """A deterministic metric cannot be evaluated from the supplied facts."""


Formula = Callable[[Mapping[str, float]], float]


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    aliases: tuple[str, ...]
    required_metrics: tuple[str, ...] = ()
    statement_types: tuple[str, ...] = ()
    formula: Formula | None = None
    expression_template: str | None = None
    output_kind: str = "currency"
    exact_total: bool = False

    @property
    def derived(self) -> bool:
        return self.formula is not None


def _required(values: Mapping[str, float], *names: str) -> list[float]:
    missing = [name for name in names if name not in values or values[name] is None]
    if missing:
        raise FormulaError(f"Missing formula inputs: {missing}")
    result = [float(values[name]) for name in names]
    if not all(math.isfinite(value) for value in result):
        raise FormulaError("Formula inputs must be finite")
    return result


def _divide(numerator: float, denominator: float, name: str) -> float:
    if denominator == 0:
        raise FormulaError(f"{name} denominator is zero")
    return numerator / denominator


def _ratio(a: str, b: str, *, percent: bool = False, name: str = "ratio") -> Formula:
    def calculate(values: Mapping[str, float]) -> float:
        numerator, denominator = _required(values, a, b)
        result = _divide(numerator, denominator, name)
        return result * 100.0 if percent else result
    return calculate


def _average(values: Mapping[str, float], current: str, previous: str) -> float:
    current_value, previous_value = _required(values, current, previous)
    return (current_value + previous_value) / 2.0


BASE_METRICS: tuple[MetricDefinition, ...] = (
    MetricDefinition("current_assets", ("tai san ngan han", "tong tai san ngan han"), statement_types=("balance_sheet",), exact_total=True),
    MetricDefinition("current_liabilities", ("no ngan han", "tong no ngan han", "no phai tra ngan han"), statement_types=("balance_sheet",), exact_total=True),
    MetricDefinition("inventory", ("hang ton kho", "gia tri hang ton kho"), statement_types=("balance_sheet",)),
    MetricDefinition("total_liabilities", ("no phai tra", "tong no phai tra", "tong no"), statement_types=("balance_sheet",), exact_total=True),
    MetricDefinition("equity", ("von chu so huu", "tong von chu so huu", "von chu"), statement_types=("balance_sheet",), exact_total=True),
    MetricDefinition("total_assets", ("tong tai san", "tong cong tai san", "tong nguon von"), statement_types=("balance_sheet",), exact_total=True),
    MetricDefinition("long_term_assets", ("tai san dai han", "tong tai san dai han"), statement_types=("balance_sheet",), exact_total=True),
    MetricDefinition("short_term_receivables", ("phai thu ngan han", "cac khoan phai thu ngan han"), statement_types=("balance_sheet",), exact_total=True),
    MetricDefinition("long_term_receivables", ("phai thu dai han", "cac khoan phai thu dai han"), statement_types=("balance_sheet",), exact_total=True),
    MetricDefinition("fixed_assets", ("tai san co dinh thuan", "tai san co dinh", "gia tri con lai tai san co dinh"), statement_types=("balance_sheet",)),
    MetricDefinition(
        "net_revenue",
        (
            "doanh thu thuan",
            "doanh thu thuan ve ban hang va cung cap dich vu",
            "doanh thu thuan ban hang va cung cap dich vu",
        ),
        statement_types=("income_statement",),
        exact_total=True,
    ),
    MetricDefinition(
        "gross_profit",
        (
            "loi nhuan gop",
            "lai gop",
            "loi nhuan gop ve ban hang va cung cap dich vu",
            "loi nhuan gop ban hang va cung cap dich vu",
        ),
        statement_types=("income_statement",),
        exact_total=True,
    ),
    MetricDefinition(
        "net_profit",
        (
            "loi nhuan sau thue",
            "loi nhuan thuan sau thue",
            "loi nhuan sau thue tndn",
            "loi nhuan sau thue thu nhap doanh nghiep",
            "lo sau thue",
            "lo sau thue tndn",
            "lnst",
        ),
        statement_types=("income_statement",),
        exact_total=True,
    ),
    MetricDefinition("operating_profit", ("loi nhuan thuan tu hoat dong kinh doanh", "loi nhuan hoat dong", "ebit"), statement_types=("income_statement",), exact_total=True),
    MetricDefinition("profit_before_tax", ("loi nhuan truoc thue", "tong loi nhuan ke toan truoc thue", "lntt"), statement_types=("income_statement",), exact_total=True),
    MetricDefinition("interest_expense", ("chi phi lai vay", "lai vay", "chi phi lai"), statement_types=("income_statement", "notes")),
    MetricDefinition("cfo", ("luu chuyen tien thuan tu hoat dong kinh doanh", "luu chuyen tien hoat dong", "dong tien tu hoat dong kinh doanh", "dong tien hoat dong", "cfo"), statement_types=("cashflow",), exact_total=True),
    MetricDefinition(
        "cogs",
        (
            "gia von hang ban",
            "gia von hang hoa",
            "gia von hang ban va dich vu cung cap",
            "gia von hang ban va cung cap dich vu",
            "cogs",
        ),
        statement_types=("income_statement",),
        exact_total=True,
    ),
    MetricDefinition("selling_expense", ("chi phi ban hang",), statement_types=("income_statement",), exact_total=True),
    MetricDefinition("admin_expense", ("chi phi quan ly doanh nghiep",), statement_types=("income_statement",), exact_total=True),
    MetricDefinition(
        "shares",
        (
            "so co phieu dang luu hanh",
            "co phieu dang luu hanh",
            "co phieu dang ky phat hanh",
            "co phieu binh quan",
            "so luong co phieu",
            "so luong co phan",
            "tong so luong co phan",
            "so co phan da dang ky",
        ),
        statement_types=("notes",),
    ),
    MetricDefinition("eps", ("lai co ban tren co phieu", "eps co ban", "eps"), statement_types=("income_statement", "notes")),
)


def _quick_ratio(values: Mapping[str, float]) -> float:
    assets, inventory, liabilities = _required(values, "current_assets", "inventory", "current_liabilities")
    return _divide(assets - inventory, liabilities, "quick_ratio")


def _working_capital(values: Mapping[str, float]) -> float:
    assets, liabilities = _required(values, "current_assets", "current_liabilities")
    return assets - liabilities


def _roe(values: Mapping[str, float]) -> float:
    profit = _required(values, "net_profit")[0]
    equity = _average(values, "equity", "equity_previous")
    return _divide(profit, equity, "roe") * 100.0


def _roa(values: Mapping[str, float]) -> float:
    profit = _required(values, "net_profit")[0]
    assets = _average(values, "total_assets", "total_assets_previous")
    return _divide(profit, assets, "roa") * 100.0


def _interest_coverage(values: Mapping[str, float]) -> float:
    profit, interest = _required(values, "profit_before_tax", "interest_expense")
    return _divide(profit + interest, interest, "interest_coverage")


def _inventory_days(values: Mapping[str, float]) -> float:
    inventory = _average(values, "inventory", "inventory_previous")
    cogs = _required(values, "cogs")[0]
    return _divide(inventory, cogs, "inventory_days") * 365.0


def _fixed_asset_turnover(values: Mapping[str, float]) -> float:
    revenue = _required(values, "net_revenue")[0]
    assets = _average(values, "fixed_assets", "fixed_assets_previous")
    return _divide(revenue, assets, "fixed_asset_turnover")


def _sga_intensity(values: Mapping[str, float]) -> float:
    selling, admin, revenue = _required(values, "selling_expense", "admin_expense", "net_revenue")
    return _divide(selling + admin, revenue, "sga_intensity") * 100.0


def _growth(values: Mapping[str, float]) -> float:
    current, previous = _required(values, "current_value", "previous_value")
    return _divide(current - previous, previous, "percentage_change") * 100.0


def _cagr(values: Mapping[str, float]) -> float:
    ending, beginning, periods = _required(values, "ending_value", "beginning_value", "periods")
    if beginning == 0 or periods <= 0 or ending / beginning < 0:
        raise FormulaError("CAGR requires non-zero same-sign endpoints and positive periods")
    return ((ending / beginning) ** (1.0 / periods) - 1.0) * 100.0


def _percentage_point_change(values: Mapping[str, float]) -> float:
    current, previous = _required(values, "current_percentage", "previous_percentage")
    return current - previous


def _accrual_ratio(values: Mapping[str, float]) -> float:
    profit, cfo = _required(values, "net_profit", "cfo")
    assets = _average(values, "total_assets", "total_assets_previous")
    return _divide(profit - cfo, assets, "accrual_ratio") * 100.0


def _margin_spread(values: Mapping[str, float]) -> float:
    gross, profit, revenue = _required(values, "gross_profit", "net_profit", "net_revenue")
    return (_divide(gross, revenue, "gross_margin") - _divide(profit, revenue, "net_margin")) * 100.0


def _gross_margin_change(values: Mapping[str, float]) -> float:
    gross, revenue, gross_previous, revenue_previous = _required(
        values, "gross_profit", "net_revenue", "gross_profit_previous", "net_revenue_previous"
    )
    return (_divide(gross, revenue, "gross_margin") - _divide(gross_previous, revenue_previous, "gross_margin_previous")) * 100.0


def _inventory_share_change(values: Mapping[str, float]) -> float:
    inventory, assets, inventory_previous, assets_previous = _required(
        values, "inventory", "total_assets", "inventory_previous", "total_assets_previous"
    )
    return (_divide(inventory, assets, "inventory_share") - _divide(inventory_previous, assets_previous, "inventory_share_previous")) * 100.0


def _operating_leverage(values: Mapping[str, float]) -> float:
    operating, revenue, operating_previous, revenue_previous = _required(
        values, "operating_profit", "net_revenue", "operating_profit_previous", "net_revenue_previous"
    )
    operating_growth = _divide(operating - operating_previous, operating_previous, "operating_profit_growth")
    revenue_growth = _divide(revenue - revenue_previous, revenue_previous, "revenue_growth")
    return _divide(operating_growth, revenue_growth, "operating_leverage")


def _stressed_net_assets(values: Mapping[str, float]) -> float:
    """Net assets after reusable liquidation haircuts.

    The 30% receivables / 50% inventory haircut is a recurring analytical
    scenario in ViFinQA.  It belongs in the formula registry (and remains
    parameter-free only for that explicitly described scenario), not in an
    answer or question-id lookup.
    """

    assets, short_receivables, long_receivables, inventory, liabilities = _required(
        values,
        "total_assets",
        "short_term_receivables",
        "long_term_receivables",
        "inventory",
        "total_liabilities",
    )
    stressed_assets = assets - 0.30 * short_receivables - 0.30 * long_receivables - 0.50 * inventory
    return stressed_assets - liabilities


DERIVED_METRICS: tuple[MetricDefinition, ...] = (
    MetricDefinition("inventory_to_current_liabilities", ("he so hang ton kho no ngan han", "hang ton kho/no ngan han", "hang ton kho chia cho no ngan han"), ("inventory", "current_liabilities"), formula=_ratio("inventory", "current_liabilities", name="inventory_to_current_liabilities"), expression_template="({inventory}) / ({current_liabilities})", output_kind="times"),
    MetricDefinition("inventory_to_assets", ("ty trong hang ton kho tren tong tai san", "hang ton kho chiem bao nhieu phan tram tong tai san", "hang ton kho tren tong tai san"), ("inventory", "total_assets"), formula=_ratio("inventory", "total_assets", percent=True, name="inventory_to_assets"), expression_template="({inventory}) / ({total_assets}) * 100", output_kind="percent"),
    MetricDefinition("current_ratio", ("he so thanh toan hien hanh", "ty so thanh toan hien hanh", "tai san ngan han chia cho no ngan han"), ("current_assets", "current_liabilities"), formula=_ratio("current_assets", "current_liabilities", name="current_ratio"), expression_template="({current_assets}) / ({current_liabilities})", output_kind="times"),
    MetricDefinition("quick_ratio", ("he so thanh toan nhanh", "ty so thanh toan nhanh", "ty le thanh toan nhanh", "tai san ngan han tru hang ton kho"), ("current_assets", "inventory", "current_liabilities"), formula=_quick_ratio, expression_template="(({current_assets}) - ({inventory})) / ({current_liabilities})", output_kind="times"),
    MetricDefinition("debt_to_equity", ("ty so d/e", "d/e", "no phai tra tren von chu so huu", "no phai tra chia cho von chu so huu", "he so no phai tra tren von chu so huu"), ("total_liabilities", "equity"), formula=_ratio("total_liabilities", "equity", name="debt_to_equity"), expression_template="({total_liabilities}) / ({equity})", output_kind="times"),
    MetricDefinition("debt_to_assets", ("no phai tra tren tong tai san", "no phai tra chia cho tong tai san", "ty le no tren tai san"), ("total_liabilities", "total_assets"), formula=_ratio("total_liabilities", "total_assets", name="debt_to_assets"), expression_template="({total_liabilities}) / ({total_assets})", output_kind="times"),
    MetricDefinition("gross_margin", ("bien loi nhuan gop", "loi nhuan gop tren doanh thu thuan", "ty le loi nhuan gop tren doanh thu thuan"), ("gross_profit", "net_revenue"), formula=_ratio("gross_profit", "net_revenue", percent=True, name="gross_margin"), expression_template="({gross_profit}) / ({net_revenue}) * 100", output_kind="percent"),
    MetricDefinition("net_margin", ("bien loi nhuan rong", "loi nhuan sau thue tren doanh thu thuan", "ty le loi nhuan sau thue tren doanh thu thuan"), ("net_profit", "net_revenue"), formula=_ratio("net_profit", "net_revenue", percent=True, name="net_margin"), expression_template="({net_profit}) / ({net_revenue}) * 100", output_kind="percent"),
    MetricDefinition("operating_margin", ("bien loi nhuan hoat dong", "loi nhuan hoat dong tren doanh thu thuan", "loi nhuan thuan tu hoat dong kinh doanh tren doanh thu thuan"), ("operating_profit", "net_revenue"), formula=_ratio("operating_profit", "net_revenue", percent=True, name="operating_margin"), expression_template="({operating_profit}) / ({net_revenue}) * 100", output_kind="percent"),
    MetricDefinition("roe", ("roe", "ty suat loi nhuan tren von chu so huu"), ("net_profit", "equity", "equity_previous"), formula=_roe, expression_template="({net_profit}) / ((({equity}) + ({equity_previous})) / 2) * 100", output_kind="percent"),
    MetricDefinition("roa", ("roa", "ty suat loi nhuan tren tong tai san"), ("net_profit", "total_assets", "total_assets_previous"), formula=_roa, expression_template="({net_profit}) / ((({total_assets}) + ({total_assets_previous})) / 2) * 100", output_kind="percent"),
    MetricDefinition("interest_coverage", ("he so kha nang thanh toan lai vay", "ty le thanh toan lai vay", "interest coverage", "ebit chia cho chi phi lai vay"), ("profit_before_tax", "interest_expense"), formula=_interest_coverage, expression_template="(({profit_before_tax}) + ({interest_expense})) / ({interest_expense})", output_kind="times"),
    MetricDefinition("cfo_to_revenue", ("cfo tren doanh thu thuan", "cfo/doanh thu", "cfo margin", "dong tien hoat dong tren doanh thu"), ("cfo", "net_revenue"), formula=_ratio("cfo", "net_revenue", percent=True, name="cfo_to_revenue"), expression_template="({cfo}) / ({net_revenue}) * 100", output_kind="percent"),
    MetricDefinition("cfo_to_net_income", ("cfo tren lnst", "cfo/lnst", "he so chuyen doi loi nhuan", "dong tien hoat dong tren loi nhuan sau thue"), ("cfo", "net_profit"), formula=_ratio("cfo", "net_profit", name="cfo_to_net_income"), expression_template="({cfo}) / ({net_profit})", output_kind="times"),
    MetricDefinition("cfo_to_current_liabilities", ("cfo tren no ngan han", "he so dong tien hoat dong tren no ngan han", "dong tien hoat dong tren no ngan han"), ("cfo", "current_liabilities"), formula=_ratio("cfo", "current_liabilities", name="cfo_to_current_liabilities"), expression_template="({cfo}) / ({current_liabilities})", output_kind="times"),
    MetricDefinition("working_capital", ("von luu dong rong", "von luu dong"), ("current_assets", "current_liabilities"), formula=_working_capital, expression_template="({current_assets}) - ({current_liabilities})"),
    MetricDefinition("inventory_days", ("so ngay ton kho", "vong quay hang ton kho theo ngay", "doh"), ("inventory", "inventory_previous", "cogs"), formula=_inventory_days, expression_template="((({inventory}) + ({inventory_previous})) / 2) / ({cogs}) * 365", output_kind="days"),
    MetricDefinition("fixed_asset_turnover", ("vong quay tai san co dinh",), ("net_revenue", "fixed_assets", "fixed_assets_previous"), formula=_fixed_asset_turnover, expression_template="({net_revenue}) / ((({fixed_assets}) + ({fixed_assets_previous})) / 2)", output_kind="times"),
    MetricDefinition("total_asset_turnover", ("vong quay tong tai san",), ("net_revenue", "total_assets", "total_assets_previous"), formula=lambda v: _divide(_required(v, "net_revenue")[0], _average(v, "total_assets", "total_assets_previous"), "total_asset_turnover"), expression_template="({net_revenue}) / ((({total_assets}) + ({total_assets_previous})) / 2)", output_kind="times"),
    MetricDefinition("sga_intensity", ("sga intensity", "ty le sg&a", "chi phi ban hang va chi phi quan ly doanh nghiep tren doanh thu"), ("selling_expense", "admin_expense", "net_revenue"), formula=_sga_intensity, expression_template="(({selling_expense}) + ({admin_expense})) / ({net_revenue}) * 100", output_kind="percent"),
    MetricDefinition("revenue_growth", ("tang truong doanh thu thuan", "toc do tang doanh thu thuan", "muc tang doanh thu thuan"), ("net_revenue", "net_revenue_previous"), formula=lambda v: _growth({"current_value": v.get("net_revenue"), "previous_value": v.get("net_revenue_previous")}), expression_template="(({net_revenue}) - ({net_revenue_previous})) / ({net_revenue_previous}) * 100", output_kind="percent"),
    MetricDefinition("cagr", ("cagr", "toc do tang truong kep"), ("beginning_value", "ending_value", "periods"), formula=_cagr, output_kind="percent"),
    MetricDefinition("percentage_change", ("phan tram thay doi", "toc do thay doi", "muc thay doi phan tram"), ("current_value", "previous_value"), formula=_growth, expression_template="(({current_value}) - ({previous_value})) / ({previous_value}) * 100", output_kind="percent"),
    MetricDefinition("percentage_point_change", ("diem phan tram", "chenh lech bien loi nhuan", "muc thay doi bien loi nhuan"), ("current_percentage", "previous_percentage"), formula=_percentage_point_change, expression_template="({current_percentage}) - ({previous_percentage})", output_kind="percentage_point"),
    MetricDefinition("accrual_ratio", ("ty so don tich", "ty le don tich"), ("net_profit", "cfo", "total_assets", "total_assets_previous"), formula=_accrual_ratio, expression_template="(({net_profit}) - ({cfo})) / ((({total_assets}) + ({total_assets_previous})) / 2) * 100", output_kind="percent"),
    MetricDefinition("margin_spread", ("chenh lech giua bien loi nhuan gop va bien loi nhuan rong", "hieu giua bien loi nhuan gop va bien loi nhuan rong"), ("gross_profit", "net_profit", "net_revenue"), formula=_margin_spread, expression_template="(({gross_profit}) / ({net_revenue}) - ({net_profit}) / ({net_revenue})) * 100", output_kind="percentage_point"),
    MetricDefinition("gross_margin_change", ("muc thay doi bien loi nhuan gop", "muc tang bien loi nhuan gop", "muc giam bien loi nhuan gop"), ("gross_profit", "net_revenue", "gross_profit_previous", "net_revenue_previous"), formula=_gross_margin_change, expression_template="(({gross_profit}) / ({net_revenue}) - ({gross_profit_previous}) / ({net_revenue_previous})) * 100", output_kind="percentage_point"),
    MetricDefinition("inventory_share_change", ("muc tang ty trong hang ton kho tren tong tai san", "thay doi ty trong hang ton kho tren tong tai san"), ("inventory", "total_assets", "inventory_previous", "total_assets_previous"), formula=_inventory_share_change, expression_template="(({inventory}) / ({total_assets}) - ({inventory_previous}) / ({total_assets_previous})) * 100", output_kind="percentage_point"),
    # ``inventory_days_change`` spans two independently calculated inventory-
    # days periods.  Its cross-period expression is assembled by the complex
    # executor because the two comparison years are supplied by the plan.
    MetricDefinition("inventory_days_change", ("muc giam so ngay ton kho", "muc thay doi so ngay ton kho"), ("inventory", "inventory_previous", "cogs"), formula=_inventory_days, output_kind="days"),
    MetricDefinition("operating_leverage", ("don bay kinh doanh", "he so don bay kinh doanh"), ("operating_profit", "net_revenue", "operating_profit_previous", "net_revenue_previous"), formula=_operating_leverage, expression_template="((({operating_profit}) - ({operating_profit_previous})) / ({operating_profit_previous})) / ((({net_revenue}) - ({net_revenue_previous})) / ({net_revenue_previous}))", output_kind="times"),
    MetricDefinition(
        "stressed_net_assets",
        (
            "gia tri tai san rong chiu ap luc thanh ly",
            "tai san rong sau ap luc thanh ly",
            "tai san rong sau khi dieu chinh thanh ly",
        ),
        (
            "total_assets",
            "short_term_receivables",
            "long_term_receivables",
            "inventory",
            "total_liabilities",
        ),
        formula=_stressed_net_assets,
        expression_template="({total_assets}) - 0.30 * ({short_term_receivables}) - 0.30 * ({long_term_receivables}) - 0.50 * ({inventory}) - ({total_liabilities})",
        output_kind="currency",
    ),
)


class MetricRegistry:
    def __init__(self, definitions: Iterable[MetricDefinition] | None = None) -> None:
        defs = tuple(definitions or (*BASE_METRICS, *DERIVED_METRICS))
        self._definitions = {definition.name: definition for definition in defs}
        if len(self._definitions) != len(defs):
            raise ValueError("Metric names must be unique")
        self._alias_index: list[tuple[str, str]] = []
        for definition in defs:
            for alias in (definition.name.replace("_", " "), *definition.aliases):
                normalized = normalize_metric_text(alias)
                if normalized:
                    self._alias_index.append((normalized, definition.name))
        self._alias_index.sort(key=lambda item: len(item[0]), reverse=True)

    def names(self) -> tuple[str, ...]:
        return tuple(self._definitions)

    def get(self, name: str) -> MetricDefinition:
        try:
            return self._definitions[name]
        except KeyError as exc:
            raise KeyError(f"Unknown financial metric: {name}") from exc

    def detect(self, text: str) -> list[str]:
        normalized = f" {normalize_metric_text(text)} "
        found: list[tuple[int, int, str]] = []
        for alias, name in self._alias_index:
            match = re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", normalized)
            if match:
                found.append((match.start(), -len(alias), name))
        result: list[str] = []
        for _, _, name in sorted(found):
            if name not in result:
                result.append(name)
        return result

    def expand_required(self, metrics: Iterable[str]) -> list[str]:
        result: list[str] = []

        def visit(name: str) -> None:
            definition = self.get(name)
            if not definition.required_metrics:
                if name not in result:
                    result.append(name)
                return
            for required_name in definition.required_metrics:
                # Period-role pseudo inputs are retrieved from the same base
                # metric in another year/period and remain explicit in plans.
                base_name = required_name.removesuffix("_previous")
                if base_name in self._definitions:
                    visit(base_name)
                elif required_name not in result:
                    result.append(required_name)

        for metric in metrics:
            visit(metric)
        return result

    def statement_types_for(self, metric: str) -> tuple[str, ...]:
        name = metric.removesuffix("_previous")
        definition = self.get(name)
        if definition.statement_types:
            return definition.statement_types
        statements: list[str] = []
        for required in definition.required_metrics:
            base = required.removesuffix("_previous")
            if base in self._definitions:
                for statement in self.statement_types_for(base):
                    if statement not in statements:
                        statements.append(statement)
        return tuple(statements)

    def evaluate(self, metric: str, values: Mapping[str, float]) -> float:
        definition = self.get(metric)
        if definition.formula is None:
            return _required(values, metric)[0]
        value = float(definition.formula(values))
        if not math.isfinite(value):
            raise FormulaError(f"Formula {metric} returned a non-finite value")
        return value

    def build_expression(self, metric: str, expressions: Mapping[str, str]) -> str:
        definition = self.get(metric)
        if definition.formula is None:
            try:
                return expressions[metric]
            except KeyError as exc:
                raise FormulaError(f"Missing expression for {metric}") from exc
        if not definition.expression_template:
            raise FormulaError(f"Metric {metric} has no scalar expression template")
        try:
            return definition.expression_template.format(**expressions)
        except KeyError as exc:
            raise FormulaError(f"Missing expression input for {metric}: {exc.args[0]}") from exc

    @staticmethod
    def median_filter(values: Mapping[str, float], *, above: bool) -> list[str]:
        finite = {key: float(value) for key, value in values.items() if math.isfinite(float(value))}
        if not finite:
            return []
        threshold = statistics.median(finite.values())
        return [key for key, value in finite.items() if value > threshold] if above else [key for key, value in finite.items() if value < threshold]

    @staticmethod
    def select(values: Mapping[str, float], operation: str) -> str:
        finite = {key: float(value) for key, value in values.items() if math.isfinite(float(value))}
        if not finite:
            raise FormulaError("Selection has no finite candidates")
        if operation in {"max", "argmax", "highest"}:
            return max(finite, key=finite.get)
        if operation in {"min", "argmin", "lowest"}:
            return min(finite, key=finite.get)
        raise FormulaError(f"Unknown selection operation: {operation}")

    @staticmethod
    def aggregate(values: Sequence[float], operation: str) -> float:
        finite = [float(value) for value in values if math.isfinite(float(value))]
        if operation == "count":
            return float(len(finite))
        if not finite:
            raise FormulaError("Aggregation has no finite values")
        if operation == "sum":
            return sum(finite)
        if operation in {"average", "mean"}:
            return statistics.fmean(finite)
        if operation == "median":
            return statistics.median(finite)
        if operation == "max":
            return max(finite)
        if operation == "min":
            return min(finite)
        raise FormulaError(f"Unknown aggregation operation: {operation}")


DEFAULT_REGISTRY = MetricRegistry()


__all__ = [
    "BASE_METRICS",
    "DEFAULT_REGISTRY",
    "DERIVED_METRICS",
    "FormulaError",
    "MetricDefinition",
    "MetricRegistry",
    "normalize_metric_text",
]
