import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

import pandas as pd

try:  # Support both package imports and the legacy ``sys.path += ['src']`` entrypoint.
    from .metric_registry import DEFAULT_REGISTRY, normalize_metric_text
    from .question_planner import QuestionPlan, QuestionPlanner, QuestionType
    from .units import (
        UnitConversionError,
        conversion_factor as canonical_conversion_factor,
        convert_value as canonical_convert_value,
        detect_target_unit as canonical_detect_target_unit,
        resolve_unit,
    )
except ImportError:  # pragma: no cover - exercised by the legacy CLI entrypoint
    from metric_registry import DEFAULT_REGISTRY, normalize_metric_text
    from question_planner import QuestionPlan, QuestionPlanner, QuestionType
    from units import (
        UnitConversionError,
        conversion_factor as canonical_conversion_factor,
        convert_value as canonical_convert_value,
        detect_target_unit as canonical_detect_target_unit,
        resolve_unit,
    )


_QUESTION_PLANNER = QuestionPlanner()


@dataclass
class FallbackResult:
    answer: float
    pandas_query: str
    csv_path: str
    row_index: int
    score: float


_STOPWORDS = {
    "cua", "cho", "va", "vao", "cuoi", "trong", "nam", "la", "bao", "nhieu",
    "dong", "trieu", "ty", "nghin", "ngay", "thang", "den", "tai", "voi",
    "cong", "ty", "ctcp", "tnhh", "tmcp", "ngan", "hang", "tong", "tap", "doan",
    "me", "hop", "nhat", "rieng", "bao", "cao", "don", "vi", "theo", "cac",
    "co", "phan", "phai", "duoc", "hoi", "so", "du",
}

_PHRASES = [
    ("chi phi quan ly doanh nghiep", ["chi", "phi", "quan", "ly", "doanh", "nghiep"]),
    ("chi phi luong", ["chi", "phi", "luong"]),
    ("khac theo luong", ["khac", "luong"]),
    ("chi phi phat", ["chi", "phi", "phat"]),
    ("tien va cac khoan tuong duong tien", ["tien", "tuong", "duong", "tien"]),
    ("von co phan", ["von", "co", "phan"]),
    ("tong tai san", ["tong", "tai", "san"]),
    ("chung khoan kinh doanh", ["chung", "khoan", "kinh", "doanh"]),
    ("thue tndn con phai nop", ["thue", "tndn", "con", "phai", "nop"]),
    ("thue tndn phai nop", ["thue", "tndn", "phai", "nop"]),
    ("vay ngan han", ["vay", "ngan", "han"]),
    ("vay dai han", ["vay", "dai", "han"]),
    ("no ngan han", ["no", "ngan", "han"]),
    ("no dai han", ["no", "dai", "han"]),
    ("luu chuyen tien thuan tu hoat dong kinh doanh", ["luu", "chuyen", "tien", "thuan", "hoat", "dong", "kinh", "doanh"]),
    ("luu chuyen tien thuan", ["luu", "chuyen", "tien", "thuan"]),
    ("nguyen gia", ["nguyen", "gia"]),
    ("chi phi du phong rui ro tin dung", ["chi", "phi", "du", "phong", "rui", "ro", "tin", "dung"]),
    ("chi phi du phong", ["chi", "phi", "du", "phong"]),
    ("loi nhuan sau thue", ["loi", "nhuan", "sau", "thue"]),
    ("loi nhuan gop", ["loi", "nhuan", "gop"]),
    ("doanh thu thuan", ["doanh", "thu", "thuan"]),
    ("tong quy luong", ["tong", "quy", "luong"]),
    ("chi phi dich vu mua ngoai", ["chi", "phi", "dich", "vu", "mua", "ngoai"]),
    ("phai thu ngan han khac", ["phai", "thu", "ngan", "han", "khac"]),
    ("thue thu nhap doanh nghiep", ["thue", "thu", "nhap", "doanh", "nghiep"]),
    ("thue thu nhap doanh nghiep phai tra", ["thue", "thu", "nhap", "doanh", "nghiep", "phai", "tra"]),
    ("thue tndn phai tra", ["thue", "tndn", "phai", "tra"]),
    ("ty le quyen bieu quyet", ["ty", "le", "quyen", "bieu", "quyet"]),
    ("ty le bieu quyet", ["ty", "le", "bieu", "quyet"]),
    ("thu lao", ["thu", "lao"]),
    ("thanh vien hdqt", ["thanh", "vien", "hdqt"]),
    ("cam ket cho thue hoat dong", ["cam", "ket", "cho", "thue", "hoat", "dong"]),
    ("loi the thuong mai", ["loi", "the", "thuong", "mai"]),
    ("cho vay khach hang", ["cho", "vay", "khach", "hang"]),
    ("chi phi khac", ["chi", "phi", "khac"]),
    ("thu nhap khac", ["thu", "nhap", "khac"]),
    ("chung khoan no", ["chung", "khoan", "no"]),
    ("bat dong san dau tu", ["bat", "dong", "san", "dau", "tu"]),
    ("gia tri con lai", ["gia", "tri", "con", "lai"]),
    # --- NEW PHRASES ---
    ("gia von hang hoa", ["gia", "von", "hang", "hoa"]),
    ("gia von hang ban", ["gia", "von", "hang", "ban"]),
    ("gia von cung cap dich vu", ["gia", "von", "cung", "cap", "dich", "vu"]),
    ("tien gui tai cac tctd khac", ["tien", "gui", "tctd", "khac"]),
    ("tien gui tai tctd", ["tien", "gui", "tctd"]),
    ("tien gui tai cac to chuc tin dung", ["tien", "gui", "to", "chuc", "tin", "dung"]),
    ("du no cho vay", ["du", "no", "cho", "vay"]),
    ("thuong mai dich vu", ["thuong", "mai", "dich", "vu"]),
    ("vay va no thue tai chinh", ["vay", "no", "thue", "tai", "chinh"]),
    ("vay va no", ["vay", "va", "no"]),
    ("du phong rui ro cho vay khach hang", ["du", "phong", "rui", "ro", "cho", "vay", "khach", "hang"]),
    ("du phong rui ro cho vay", ["du", "phong", "rui", "ro", "cho", "vay"]),
    ("so du du phong", ["so", "du", "du", "phong"]),
    ("chi phi thue hien hanh", ["chi", "phi", "thue", "hien", "hanh"]),
    ("thue thu nhap doanh nghiep hien hanh", ["thue", "thu", "nhap", "hien", "hanh"]),
    ("ty le so huu", ["ty", "le", "so", "huu"]),
    ("von gop truc tiep", ["von", "gop", "truc", "tiep"]),
    ("lai vay phai tra", ["lai", "vay", "phai", "tra"]),
    ("cam ket giao dich hoi doai", ["cam", "ket", "giao", "dich", "hoi", "doai"]),
    ("cam ket giao dich", ["cam", "ket", "giao", "dich"]),
    ("luu chuyen tien tu hoat dong kinh doanh", ["luu", "chuyen", "hoat", "dong", "kinh", "doanh"]),
    ("nam hien hanh", ["nam", "hien", "hanh"]),
    ("quy dau tu gia tri bao viet", ["quy", "dau", "tu", "gia", "tri", "bao", "viet"]),
    ("dau tu vao cong ty con va bvif", ["dau", "tu", "cong", "ty", "con", "bvif"]),
    ("bvif", ["bvif"]),
]


def normalize_text(text) -> str:
    text = "" if text is None else str(text).lower()
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    text = text.replace("đ", "d")
    return re.sub(r"\s+", " ", text).strip()


def detect_target_unit(question: str) -> str:
    """Return the canonical target unit (legacy public compatibility API).

    Unit matching is delegated to :mod:`units`, whose longest-match parser is
    deliberately ordered so ``trăm tỷ đồng`` cannot be misread as
    ``tỷ đồng``.
    """

    return canonical_detect_target_unit(question)


def convert_unit(value, source_unit, target_unit: str) -> float:
    """Convert using the *selected row's* unit and preserve the source sign.

    An empty target means no conversion was requested.  A requested conversion
    with an empty/unknown source is an error: fallback must not guess a unit
    from the magnitude or from unrelated rows in the CSV.
    """

    numeric = float(value)
    if not target_unit:
        return numeric
    if pd.isna(source_unit) or not str(source_unit).strip():
        raise UnitConversionError("Selected evidence row has no source unit")
    return canonical_convert_value(numeric, source_unit, target_unit, strict=True)



def _question_tokens(question: str) -> list:
    q = re.sub(r"\b20\d{2}\b", " ", normalize_text(question))
    toks = re.findall(r"[a-z0-9]+", q)
    return [t for t in toks if len(t) > 1 and t not in _STOPWORDS]


def _extract_ticker_year(question: str):
    year_match = re.search(r"\b(20\d{2})\b", question)
    tickers = re.findall(r"\b[A-Z]{2,5}\b", question)
    ticker = tickers[-1] if tickers else None
    return ticker, year_match.group(1) if year_match else None


def _candidate_paths(question: str, csv_paths: list) -> list:
    """Return only evidence paths explicitly supplied by the caller.

    Older code searched the ticker directory and silently appended files that
    were not part of ``relevant_docs``.  Those files had no stable dataframe
    variable and were later emitted as ``df1``.  Keeping this helper (and its
    historical signature) is useful to callers, but expansion is forbidden.
    """

    del question  # The question must never broaden the authorized evidence.
    paths: list[str] = []
    for raw_path in csv_paths or ():
        path = os.fspath(raw_path)
        if path not in paths:
            paths.append(path)
    return paths


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = f" {normalize_metric_text(text)} "
    normalized_phrase = normalize_metric_text(phrase)
    return bool(
        normalized_phrase
        and re.search(
            rf"(?<![a-z0-9]){re.escape(normalized_phrase)}(?![a-z0-9])",
            normalized_text,
        )
    )


def _semantic_metric_match(question: str, row_label: object, plan: QuestionPlan) -> float:
    """Return a conservative semantic-match bonus, or zero when ambiguous.

    A lexical overlap alone is insufficient for a fallback answer.  For known
    registry metrics the selected row must contain a registered alias for the
    planned *target* metric.  For note-table questions not represented by the
    registry, a multi-token domain phrase (or nearly exact row label) must be
    present in both question and row.
    """

    row = "" if row_label is None else str(row_label)
    target = plan.target_metric
    if target:
        try:
            definition = DEFAULT_REGISTRY.get(target)
        except KeyError:
            return 0.0
        if definition.derived:
            # A derived metric is not a source row.  The planner should already
            # classify it as complex, but retaining this guard makes the safety
            # rule local and explicit.
            return 0.0
        aliases = (target.replace("_", " "), *definition.aliases)
        matched = [alias for alias in aliases if _contains_phrase(row, alias)]
        if not matched:
            return 0.0
        # Exact-total metrics must resolve to their total row, not a child row
        # that merely happens to share generic vocabulary.
        if definition.exact_total:
            normalized_row = normalize_metric_text(row)
            exact_aliases = {normalize_metric_text(alias) for alias in matched}
            stripped_row = re.sub(r"^(?:[ivxlcdm]+|\d+)\s*", "", normalized_row).strip()
            exact_shape = any(
                re.fullmatch(
                    rf"(?:(?:[a-z]|[ivxlcdm]+|\d+)\s+)*{re.escape(alias)}(?:\s+\d+)*",
                    stripped_row,
                )
                or stripped_row.startswith(f"tong {alias}")
                or stripped_row.startswith(f"{alias} tong cong")
                for alias in exact_aliases
            )
            if not exact_shape:
                # Some statements suffix the exact total with punctuation or a
                # short period qualifier.  A close token-length match remains
                # safe; hierarchical rows are substantially longer.
                alias_tokens = min(len(alias.split()) for alias in exact_aliases)
                if len(stripped_row.split()) > alias_tokens + 2:
                    return 0.0
        return 14.0

    qnorm = normalize_text(question)
    rnorm = normalize_text(row)
    best_phrase_tokens = 0
    for phrase, tokens in _PHRASES:
        if len(tokens) < 2:
            continue
        if phrase in qnorm and phrase in rnorm:
            best_phrase_tokens = max(best_phrase_tokens, len(tokens))
        elif all(token in qnorm for token in tokens) and all(token in rnorm for token in tokens):
            best_phrase_tokens = max(best_phrase_tokens, len(tokens))
    if best_phrase_tokens >= 2:
        return 10.0 + min(best_phrase_tokens, 5)

    # Notes often use a proper-name row not covered by the financial registry.
    # Accept only a near-exact, multi-token label mention in that case.
    row_tokens = [token for token in re.findall(r"[a-z0-9]+", rnorm) if token not in _STOPWORDS]
    question_tokens = set(_question_tokens(question))
    if len(row_tokens) >= 2:
        overlap = sum(token in question_tokens for token in row_tokens)
        if overlap >= 3 and overlap / len(row_tokens) >= 0.75:
            return 12.0
    return 0.0


def _row_score(question: str, path: str, chi_tieu: str) -> float:
    q = normalize_text(question)
    row = normalize_text(chi_tieu)
    p = normalize_text(path)
    qtoks = _question_tokens(question)
    if not row:
        return 0.0
    score = 0.0
    for tok in qtoks:
        if tok in row:
            score += 1.0
    for phrase, toks in _PHRASES:
        phrase_in_q = phrase in q or all(t in q for t in toks)
        phrase_in_row = phrase in row or all(t in row for t in toks)
        if phrase_in_q and phrase_in_row:
            score += 8.0 + min(len(toks), 5)
    if ("cuoi nam" in q or "31/12" in q) and ("cuoi nam" in row or "cuoi ky" in row or "con phai nop" in row or row.startswith("tong") or row.startswith("i.")):
        score += 4.0
    if "dau nam" in row and ("cuoi nam" in q or "31/12" in q):
        score -= 5.0
    if "cong ty me" in q or "rieng" in q:
        if "separate" in p or "rieng" in p:
            score += 1.5
        if "consolidated" in p or "hop nhat" in p:
            score -= 2.0
    if "luu chuyen" not in q and "luuchuyentiente" in p:
        score -= 3.0
    if "luu chuyen" in q and "luuchuyentiente" in p:
        score += 4.0
    if "tong" in q and row.startswith("tong"):
        score += 3.0
    if "phai thu" in q and "phai tra" in row:
        score -= 8.0
    if "phai tra" in q and "phai thu" in row:
        score -= 8.0
    if "vay ngan han" in q and "cho vay ngan han" in row:
        score -= 10.0
    if "loi the thuong mai" in q and "loi the thuong mai" not in row:
        score -= 10.0
    if "chi phi du phong" in q and "truoc chi phi du phong" in row:
        score -= 8.0
    if ("so du" in q or "phai tra" in q or "phai nop" in q) and row.startswith("chi phi"):
        score -= 8.0
    if "quyen bieu quyet" in q and "quyen bieu quyet" not in row and "bieu quyet" not in row:
        score -= 5.0
    if "thuong mai" in q and "thuong mai" not in row:
        score -= 10.0
    if "chu thi binh" in q and "chu thi binh" not in row:
        score -= 10.0
    if "cam ket" in q and "cam ket" not in row:
        score -= 10.0
    if "gia tri con lai" in q and "gia tri con lai" not in row and "con lai" not in row:
        score -= 8.0
    # --- NEW GUARDS ---
    if "gia von hang hoa" in q and "hang hoa" not in row:
        score -= 10.0
    if "du phong rui ro" in q and "du phong" not in row and "rui ro" not in row:
        score -= 8.0
    if "so du" in q and "du phong" in q:
        # Prefer rows with "so du tai ngay 31" for balance-type questions
        if "so du tai ngay 31" in row:
            score += 5.0
        elif "so du" in row:
            score += 3.0
    if "tien gui" in q and "tctd" in q and "tctd" not in row and "to chuc tin dung" not in row and "tien gui" not in row:
        score -= 8.0
    if "von gop" in q and "von gop" not in row:
        score -= 8.0
    if "lai vay" in q and "lai vay" not in row and "lai" not in row:
        score -= 8.0
    if "hoi doai" in q and "hoi doai" not in row:
        score -= 10.0
    if "vay va no" in q and "vay va no" not in row and "vay" not in row:
        score -= 8.0
    # Synonym: "Quỹ Đầu tư Giá trị Bảo Việt" = BVIF
    if ("lai tien gui" in q or "tien gui" in q) and "lai tien gui" in row:
        score += 8.0
    return score


def _make_query(
    var_name: str,
    row_index: int,
    unit_factor: float,
    is_negative_abs: bool = False,
) -> str:
    """Build an evidence-derived scalar query without changing its sign.

    ``is_negative_abs`` remains in the signature for compatibility with older
    callers, but is intentionally ignored.  Financial-statement signs are data,
    not an error-recovery signal.
    """

    del is_negative_abs
    expr = f"float({var_name}.iloc[{int(row_index)}]['Gia_tri'])"
    if unit_factor != 1.0:
        if unit_factor > 1.0:
            if unit_factor == int(unit_factor):
                expr = f"{expr} * {int(unit_factor)}"
            else:
                expr = f"{expr} * {unit_factor}"
        else:
            inv = int(round(1.0 / unit_factor))
            expr = f"{expr} / {inv}"
    return expr


def _get_unit_factor(value: float, source_unit: str, target_unit: str) -> float:
    """Return the canonical row-level conversion factor.

    ``value`` is retained solely for API compatibility.  Magnitude-based unit
    inference was unsafe and is deliberately not performed.
    """

    del value
    if not target_unit:
        return 1.0
    if pd.isna(source_unit) or not str(source_unit).strip():
        raise UnitConversionError("Selected evidence row has no source unit")
    if resolve_unit(source_unit) is None:
        raise UnitConversionError(f"Unknown selected-row unit: {source_unit!r}")
    return canonical_conversion_factor(source_unit, target_unit)


def _safe_plan(question: str, plan: Optional[QuestionPlan] = None) -> Optional[QuestionPlan]:
    if plan is not None:
        return plan
    try:
        return _QUESTION_PLANNER.analyze(question)
    except Exception:
        # A planner failure must fail closed.  Falling back to lexical row
        # selection when complexity is unknown recreates the original bug.
        return None


def try_multistep_rule_based_answer(
    question: str,
    csv_paths: list,
    plan: Optional[QuestionPlan] = None,
) -> Optional[FallbackResult]:
    """Compatibility entrypoint for the retired analytical fallback.

    Multi-step calculations now belong to the deterministic metric executor.
    This function deliberately returns no answer; in particular it can never
    hide missing metrics behind a plausible-looking row or ad-hoc formula.
    """

    del csv_paths
    resolved_plan = _safe_plan(question, plan)
    if resolved_plan is None or resolved_plan.is_complex:
        return None
    return None


def try_rule_based_answer(
    question: str,
    csv_paths: list,
    min_score: float = 4.0,
    plan: Optional[QuestionPlan] = None,
) -> Optional[FallbackResult]:
    """Conservative, evidence-bound fallback for ``SIMPLE_LOOKUP`` only.

    The result is sourced from exactly one of ``csv_paths`` and its query uses
    the stable variable assigned to that same supplied path.  Zero and negative
    values are returned unchanged.  Unknown units, ambiguous metrics, planner
    failures, and every analytical question fail closed with ``None``.
    """

    resolved_plan = _safe_plan(question, plan)
    if resolved_plan is None or resolved_plan.question_type != QuestionType.SIMPLE_LOOKUP:
        return None

    candidates = _candidate_paths(question, csv_paths)
    if not candidates:
        return None
    target_unit = detect_target_unit(question)
    path_to_var = {path: f"df{i + 1}" for i, path in enumerate(candidates)}
    best: Optional[FallbackResult] = None
    # Legacy callers may pass a lower threshold, but a semantic fallback is not
    # accepted on weak lexical evidence.
    effective_min_score = max(float(min_score), 10.0)

    for path in candidates:
        var_name = path_to_var.get(path)
        if var_name is None:  # Defensive: never invent a ``df1`` mapping.
            continue
        if not os.path.isfile(path):
            continue
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        if "Chi_tieu" not in df.columns or "Gia_tri" not in df.columns:
            continue

        for idx, row in df.iterrows():
            try:
                value = float(row.get("Gia_tri"))
            except (TypeError, ValueError):
                continue
            semantic_bonus = _semantic_metric_match(
                question,
                row.get("Chi_tieu", ""),
                resolved_plan,
            )
            if semantic_bonus <= 0:
                continue
            score = _row_score(question, path, row.get("Chi_tieu", "")) + semantic_bonus
            if score < effective_min_score:
                continue
            try:
                unit_factor = _get_unit_factor(
                    value,
                    row.get("Don_vi", ""),
                    target_unit,
                )
                answer = convert_unit(
                    value,
                    row.get("Don_vi", ""),
                    target_unit,
                )
            except UnitConversionError:
                continue

            query = _make_query(var_name, int(idx), unit_factor)
            candidate = FallbackResult(float(answer), query, path, int(idx), float(score))
            if best is None or candidate.score > best.score:
                best = candidate

    return best


__all__ = [
    "FallbackResult",
    "convert_unit",
    "detect_target_unit",
    "normalize_text",
    "try_multistep_rule_based_answer",
    "try_rule_based_answer",
]
