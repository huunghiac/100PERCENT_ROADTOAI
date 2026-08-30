"""Canonical unit parsing and deterministic conversion for ViFinQA.

The module deliberately keeps unit detection separate from metric semantics.  In
particular, percentage points are not interchangeable with percentages, and a
currency value is never converted to a dimensionless value.  Conversion is
sign preserving; callers that need an absolute magnitude must request that as
part of the question semantics rather than as a unit side effect.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Optional, Union


Number = Union[int, float]


class UnitDimension(str, Enum):
    """Dimensions that may be converted without an exchange rate."""

    VND = "currency_vnd"
    USD = "currency_usd"
    RATIO = "ratio"
    PERCENTAGE_POINT = "percentage_point"
    SHARES = "shares"
    VND_PER_SHARE = "vnd_per_share"


@dataclass(frozen=True)
class UnitSpec:
    """A canonical unit.

    ``scale`` converts a value in this unit to the base unit of its dimension.
    For example, one ``triệu đồng`` is ``1e6`` VND and one percent is ``0.01``
    of a dimensionless ratio.
    """

    name: str
    dimension: UnitDimension
    scale: float
    aliases: tuple[str, ...]

    @property
    def is_currency(self) -> bool:
        return self.dimension in {UnitDimension.VND, UnitDimension.USD}

    @property
    def is_dimensionless(self) -> bool:
        return self.dimension == UnitDimension.RATIO


class UnitConversionError(ValueError):
    """Raised when two units cannot be converted without extra information."""


def normalize_unit_text(value: object) -> str:
    """Fold accents and punctuation while retaining ``%`` and ``/``."""

    if value is None:
        return ""
    text = unicodedata.normalize("NFKC", str(value)).casefold()
    text = text.replace("đ", "d")
    text = "".join(
        char
        for char in unicodedata.normalize("NFD", text)
        if unicodedata.category(char) != "Mn"
    )
    text = re.sub(r"[^a-z0-9%/]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# More specific units intentionally appear before their suffixes.  Detection
# also sorts aliases by normalized length, so this invariant is maintained when
# aliases are extended.
UNIT_SPECS: tuple[UnitSpec, ...] = (
    UnitSpec(
        "nghìn tỷ đồng",
        UnitDimension.VND,
        1_000_000_000_000.0,
        (
            "nghìn tỷ đồng",
            "ngàn tỷ đồng",
            "nghìn tỷ VND",
            "ngan ty dong",
            "nghin ty dong",
            "nghin ty vnd",
            "trillion VND",
        ),
    ),
    UnitSpec(
        "trăm tỷ đồng",
        UnitDimension.VND,
        100_000_000_000.0,
        (
            "trăm tỷ đồng",
            "trăm tỷ VND",
            "tram ty dong",
            "tram ty vnd",
            "hundred billion VND",
        ),
    ),
    UnitSpec(
        "triệu đồng",
        UnitDimension.VND,
        1_000_000.0,
        ("triệu đồng", "triệu VND", "trieu dong", "trieu vnd", "million VND"),
    ),
    UnitSpec(
        "nghìn đồng",
        UnitDimension.VND,
        1_000.0,
        (
            "nghìn đồng",
            "ngàn đồng",
            "nghìn VND",
            "ngàn VND",
            "nghin dong",
            "ngan dong",
            "nghin vnd",
            "thousand VND",
        ),
    ),
    UnitSpec(
        "tỷ đồng",
        UnitDimension.VND,
        1_000_000_000.0,
        ("tỷ đồng", "tỉ đồng", "tỷ VND", "ty dong", "ti dong", "ty vnd", "billion VND"),
    ),
    UnitSpec(
        "triệu USD",
        UnitDimension.USD,
        1_000_000.0,
        ("triệu USD", "trieu usd", "million USD", "USD million"),
    ),
    UnitSpec(
        "nghìn USD",
        UnitDimension.USD,
        1_000.0,
        ("nghìn USD", "ngàn USD", "nghin usd", "ngan usd", "thousand USD", "USD thousand"),
    ),
    UnitSpec(
        "VND/cổ phiếu",
        UnitDimension.VND_PER_SHARE,
        1.0,
        (
            "VND/cổ phiếu",
            "VNĐ/cổ phiếu",
            "đồng/cổ phiếu",
            "vnd/co phieu",
            "dong/co phieu",
            "VND per share",
        ),
    ),
    UnitSpec(
        "điểm phần trăm",
        UnitDimension.PERCENTAGE_POINT,
        1.0,
        (
            "điểm phần trăm",
            "diem phan tram",
            "percentage points",
            "percentage point",
            "percent points",
            "percent point",
            "pp",
        ),
    ),
    UnitSpec(
        "%",
        UnitDimension.RATIO,
        0.01,
        ("phần trăm", "phan tram", "percent", "pct", "%"),
    ),
    UnitSpec(
        "lần",
        UnitDimension.RATIO,
        1.0,
        ("lần", "lan", "times", "time", "multiple"),
    ),
    UnitSpec(
        "cổ phần",
        UnitDimension.SHARES,
        1.0,
        ("cổ phần", "cổ phiếu", "co phan", "co phieu", "shares", "share"),
    ),
    UnitSpec(
        "USD",
        UnitDimension.USD,
        1.0,
        ("US dollars", "US dollar", "dollars", "dollar", "USD"),
    ),
    UnitSpec(
        "VND",
        UnitDimension.VND,
        1.0,
        ("Việt Nam đồng", "Vietnam dong", "VNĐ", "VND", "đồng", "dong"),
    ),
)


def _alias_patterns(specs: Iterable[UnitSpec]) -> list[tuple[str, re.Pattern[str], UnitSpec]]:
    patterns: list[tuple[str, re.Pattern[str], UnitSpec]] = []
    for spec in specs:
        for raw_alias in (spec.name, *spec.aliases):
            alias = normalize_unit_text(raw_alias)
            if not alias:
                continue
            escaped = re.escape(alias).replace(r"\ ", r"\s+")
            if alias == "%":
                pattern = re.compile(re.escape(alias))
            else:
                pattern = re.compile(rf"(?<![a-z0-9]){escaped}(?![a-z0-9])")
            patterns.append((alias, pattern, spec))
    # Longest-match is essential: "trăm tỷ đồng" must never fall through to
    # "tỷ đồng", and "điểm phần trăm" must precede "phần trăm".
    patterns.sort(key=lambda item: len(item[0]), reverse=True)
    return patterns


_PATTERNS = _alias_patterns(UNIT_SPECS)
_CANONICAL = {normalize_unit_text(spec.name): spec for spec in UNIT_SPECS}


def detect_unit(text: object) -> Optional[UnitSpec]:
    """Return the longest canonical unit mentioned in ``text``."""

    normalized = normalize_unit_text(text)
    if not normalized:
        return None
    best: Optional[tuple[int, int, UnitSpec]] = None
    for alias, pattern, spec in _PATTERNS:
        match = pattern.search(normalized)
        if match is None:
            continue
        candidate = (len(alias), -match.start(), spec)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best[2] if best else None


def resolve_unit(unit: object) -> Optional[UnitSpec]:
    """Resolve a canonical name, legacy spelling, or text containing a unit."""

    if isinstance(unit, UnitSpec):
        return unit
    normalized = normalize_unit_text(unit)
    if not normalized:
        return None
    exact = _CANONICAL.get(normalized)
    return exact if exact is not None else detect_unit(unit)


def detect_unit_name(text: object, default: str = "") -> str:
    spec = detect_unit(text)
    return spec.name if spec is not None else default


def detect_target_unit(question: object) -> str:
    """Compatibility helper returning a canonical target-unit string."""

    return detect_unit_name(question)


def is_compatible(source_unit: object, target_unit: object) -> bool:
    source = resolve_unit(source_unit)
    target = resolve_unit(target_unit)
    return bool(source and target and source.dimension == target.dimension)


def conversion_factor(source_unit: object, target_unit: object) -> float:
    """Return the sign-preserving multiplicative conversion factor.

    VND and USD are deliberately distinct dimensions; this function never
    invents an exchange rate.  Percent and ``lần`` share the ratio dimension,
    so ``15 %`` converts to ``0.15 lần`` and vice versa.  Percentage points use
    their own dimension because they describe a difference, not a level.
    """

    source = resolve_unit(source_unit)
    target = resolve_unit(target_unit)
    if source is None:
        raise UnitConversionError(f"Unknown source unit: {source_unit!r}")
    if target is None:
        raise UnitConversionError(f"Unknown target unit: {target_unit!r}")
    if source.dimension != target.dimension:
        raise UnitConversionError(
            f"Cannot convert {source.name!r} ({source.dimension.value}) to "
            f"{target.name!r} ({target.dimension.value})"
        )
    return source.scale / target.scale


def convert_value(
    value: Number,
    source_unit: object,
    target_unit: object,
    *,
    strict: bool = True,
) -> float:
    """Convert ``value`` without changing its sign.

    With ``strict=False`` an unknown or incompatible unit leaves the numeric
    value unchanged.  The pipeline should normally use strict mode so missing
    row-level unit evidence becomes a structured failure rather than a guess.
    """

    numeric = float(value)
    try:
        factor = conversion_factor(source_unit, target_unit)
    except UnitConversionError:
        if strict:
            raise
        return numeric
    return numeric * factor


__all__ = [
    "UNIT_SPECS",
    "UnitConversionError",
    "UnitDimension",
    "UnitSpec",
    "conversion_factor",
    "convert_value",
    "detect_target_unit",
    "detect_unit",
    "detect_unit_name",
    "is_compatible",
    "normalize_unit_text",
    "resolve_unit",
]
