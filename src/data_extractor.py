"""ViFinQA OCR financial-table extractor.

Convert the OCR ``.txt`` corpus in ``data/raw_vifinqa`` to the canonical
three-column CSV format used by ``data/mock_csv``::

    Chi_tieu,Gia_tri,Don_vi

The raw corpus contains HTML-like tables embedded in OCR text.  The extractor
also includes a plain-text table fallback so it remains useful if a future
version of the dataset loses the HTML tags.

Design goals
------------
* scan the whole corpus recursively (no company limit);
* infer ticker, report year, company name and table title;
* repair common OCR issues (broken rows/cells and shifted numeric columns);
* keep one current-period value per indicator, matching the mock CSV schema;
* normalize monetary values to ``Ty dong`` while preserving percentages and
  obvious non-monetary units;
* avoid collisions between separate and consolidated reports.

The module intentionally uses only the Python standard library so it can run in
competition/offline environments without an extra parser dependency.
"""

from __future__ import annotations

import argparse
import csv
import html
import math
import os
import re
import shutil
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple


CANONICAL_HEADER = ("Chi_tieu", "Gia_tri", "Don_vi")

# Canonical statement names deliberately match data/mock_csv filenames.
STATEMENT_PATTERNS: Sequence[Tuple[str, str]] = (
    (r"bang\s+can\s+doi\s+ke\s+toan", "BangCanDoiKeToan"),
    (r"bao\s+cao\s+tinh\s+hinh\s+tai\s+chinh", "BangCanDoiKeToan"),
    (r"bao\s+cao\s+ket\s+qua(?:\s+hoat\s+dong)?\s+kinh\s+doanh", "BaoCaoKetQuaKinhDoanh"),
    (r"bao\s+cao\s+luu\s+chuyen\s+tien\s+te", "BaoCaoLuuChuyenTienTe"),
    (r"bao\s+cao\s+thay\s+doi\s+von\s+chu\s+so\s+huu", "BaoCaoThayDoiVonChuSoHuu"),
)

CANONICAL_STATEMENT_TITLES = {canonical for _, canonical in STATEMENT_PATTERNS}

DESCRIPTION_HEADER_WORDS = (
    "chi tieu",
    "noi dung",
    "khoan muc",
    "dien giai",
    "tai san",
    "nguon von",
    "ten",
)

REFERENCE_HEADER_WORDS = (
    "ma so",
    "ma",
    "stt",
    "thuyet minh",
    "ghi chu",
    "note",
)

CURRENT_PERIOD_WORDS = (
    "nam nay",
    "cuoi nam",
    "cuoi ky",
    "ky nay",
    "hien tai",
    "31/12",
    "31-12",
)

PREVIOUS_PERIOD_WORDS = (
    "nam truoc",
    "dau nam",
    "dau ky",
    "ky truoc",
    "31/12/20",  # refined with exact report year below
)

MONEY_HINTS = (
    "doanh thu",
    "chi phi",
    "loi nhuan",
    "tai san",
    "nguon von",
    "von chu",
    "no phai tra",
    "tien",
    "phai thu",
    "phai tra",
    "hang ton kho",
    "vay",
    "thue",
    "quy",
    "co tuc",
    "luu chuyen",
    "thu nhap",
    "gia von",
    "du phong",
    "khau hao",
)

PERCENT_HINTS = (
    "ty le",
    "phan tram",
    "quyen bieu quyet",
    "ty trong",
    "lai suat",
)

NON_MONEY_HINTS = (
    "so luong co phieu",
    "so co phieu",
    "so luong",
    "so nguoi",
    "nhan vien",
    "lao dong",
    "so ngay",
    "so thang",
)


@dataclass(frozen=True)
class ReportMetadata:
    ticker: str
    year: int
    scope: str  # "separate", "consolidated", or "unknown"
    company_name: str = ""


@dataclass
class ParsedCell:
    text: str
    rowspan: int = 1
    colspan: int = 1


@dataclass
class ParsedTable:
    rows: List[List[str]]
    title: str
    unit: str
    ordinal: int
    source_path: Path
    metadata: ReportMetadata


@dataclass
class ExtractedRow:
    indicator: str
    value: float
    unit: str


@dataclass
class ExtractionStats:
    files_seen: int = 0
    tables_seen: int = 0
    tables_written: int = 0
    rows_written: int = 0
    files_failed: int = 0


class _SingleTableParser(HTMLParser):
    """Parse one HTML-like table without third-party dependencies."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.raw_rows: List[List[ParsedCell]] = []
        self._row: Optional[List[ParsedCell]] = None
        self._cell: Optional[ParsedCell] = None
        self._cell_chunks: List[str] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        tag = tag.lower()
        if tag == "tr":
            self._row = []
        elif tag in {"td", "th"}:
            amap = {k.lower(): (v or "") for k, v in attrs}
            self._cell = ParsedCell(
                text="",
                rowspan=_safe_positive_int(amap.get("rowspan"), 1),
                colspan=_safe_positive_int(amap.get("colspan"), 1),
            )
            self._cell_chunks = []
        elif tag == "br" and self._cell is not None:
            self._cell_chunks.append(" ")

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell_chunks.append(data)

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"td", "th"} and self._cell is not None:
            self._cell.text = clean_text("".join(self._cell_chunks))
            if self._row is not None:
                self._row.append(self._cell)
            self._cell = None
            self._cell_chunks = []
        elif tag == "tr" and self._row is not None:
            self.raw_rows.append(self._row)
            self._row = None

    def expanded_rows(self) -> List[List[str]]:
        """Expand rowspan/colspan so OCR tables align to stable columns."""
        rows: List[List[str]] = []
        # col -> [remaining future rows, text]
        spans: Dict[int, List[object]] = {}

        for raw_row in self.raw_rows:
            out: List[str] = []
            col = 0

            def consume_span(c: int) -> bool:
                if c not in spans:
                    return False
                remaining, text = spans[c]
                out.append(str(text))
                remaining = int(remaining) - 1
                if remaining <= 0:
                    del spans[c]
                else:
                    spans[c][0] = remaining
                return True

            for cell in raw_row:
                while consume_span(col):
                    col += 1

                for j in range(max(1, cell.colspan)):
                    out.append(cell.text)
                    if cell.rowspan > 1:
                        spans[col + j] = [cell.rowspan - 1, cell.text]
                col += max(1, cell.colspan)

            # Fill any trailing rowspans that belong to this row.
            if spans:
                max_col = max(spans)
                while col <= max_col:
                    if consume_span(col):
                        col += 1
                    else:
                        out.append("")
                        col += 1

            rows.append(out)

        width = max((len(r) for r in rows), default=0)
        return [r + [""] * (width - len(r)) for r in rows]


def _safe_positive_int(value: Optional[str], default: int) -> int:
    try:
        parsed = int(str(value))
        return parsed if parsed > 0 else default
    except (TypeError, ValueError):
        return default


def clean_text(value: object) -> str:
    text = html.unescape(str(value or ""))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\xa0", " ").replace("\u200b", " ")
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n|;")


def strip_accents(value: str) -> str:
    value = value.replace("Đ", "D").replace("đ", "d")
    normalized = unicodedata.normalize("NFD", value)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Mn")


def normalized_key(value: str) -> str:
    value = strip_accents(clean_text(value)).lower()
    value = re.sub(r"[^a-z0-9%/]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def slugify(value: str, max_len: int = 72) -> str:
    raw = strip_accents(clean_text(value))
    words = re.findall(r"[A-Za-z0-9]+", raw)
    if not words:
        return "BangDuLieu"
    slug = "".join(w[:1].upper() + w[1:] for w in words)
    return slug[:max_len].rstrip("_") or "BangDuLieu"


def canonical_indicator(value: str) -> str:
    text = strip_accents(clean_text(value))
    # Remove typical row numbering while keeping semantic Roman section labels
    # when they are the only useful text.
    text = re.sub(r"^\s*(?:\d{1,3}[.)]\s*|[-–—]\s*)", "", text)
    text = re.sub(r"\s+", " ", text).strip(" -–—:;")
    return text


def infer_metadata(path: Path, raw_root: Path, content: str) -> ReportMetadata:
    ticker = "UNK"
    year = 0
    scope = "unknown"

    try:
        rel_parts = path.resolve().relative_to(raw_root.resolve()).parts
    except Exception:
        rel_parts = path.parts

    # Typical corpus layout: financial_statements/TICKER/YEAR/.../*.txt
    parts = list(rel_parts)
    if "financial_statements" in parts:
        idx = parts.index("financial_statements")
        tail = parts[idx + 1 :]
    else:
        tail = parts

    for part in tail:
        if ticker == "UNK" and re.fullmatch(r"[A-Z0-9]{2,6}", part):
            ticker = part
        if not year and re.fullmatch(r"20\d{2}", part):
            year = int(part)

    filename = path.name
    if ticker == "UNK":
        m = re.search(r"(?:^|[_-])([A-Z]{2,6})(?:[_-])", filename)
        if m:
            ticker = m.group(1)
    if not year:
        m = re.search(r"\b(20\d{2})\b", filename.replace("_", " "))
        if m:
            year = int(m.group(1))

    lower_path = str(path).lower()
    if "consolidated" in lower_path or "hop_nhat" in lower_path or "hopnhat" in lower_path:
        scope = "consolidated"
    elif "separate" in lower_path or "rieng" in lower_path:
        scope = "separate"

    company_name = infer_company_name(content)
    return ReportMetadata(ticker=ticker, year=year, scope=scope, company_name=company_name)


def infer_company_name(content: str) -> str:
    head = re.sub(r"<table.*?</table>", " ", content[:12000], flags=re.I | re.S)
    candidates: List[str] = []
    for raw_line in head.splitlines():
        line = clean_text(raw_line)
        key = normalized_key(line)
        if not (5 <= len(line) <= 160):
            continue
        if any(x in key for x in ("bao cao", "dia chi", "muc luc", "kiem toan", "don vi tinh")):
            continue
        if any(x in key for x in ("cong ty", "ngan hang", "tong cong ty", "tap doan")):
            candidates.append(line)
    if not candidates:
        return ""
    # Headers are repeated on pages; the most frequent candidate is usually the company.
    return Counter(candidates).most_common(1)[0][0]


def _context_lines(context: str) -> List[str]:
    context = re.sub(r"<[^>]+>", " ", context)
    lines: List[str] = []
    for raw in context.splitlines():
        line = clean_text(raw)
        if not line or re.fullmatch(r"=+\s*PAGE\s+\d+\s*=+", line, flags=re.I):
            continue
        lines.append(line)
    return lines


def infer_table_title(context: str, table_rows: Sequence[Sequence[str]], ordinal: int) -> str:
    lines = _context_lines(context)[-35:]

    # First, identify the statement family only when its heading is genuinely
    # close to this table.  Looking too far back mislabels footnote tables that
    # happen to follow a primary statement on the same OCR page.
    recent = lines[-10:]
    for distance, line in enumerate(reversed(recent)):
        line_key = normalized_key(line)
        if distance > 7 or len(line_key) > 78:
            continue
        # Primary statement titles are headings that START with the report name.
        # This avoids false positives such as "Thông tin bổ sung cho các khoản
        # mục trình bày trên Bảng cân đối kế toán ..." in the notes.
        for pattern, canonical in STATEMENT_PATTERNS:
            if re.match(rf"^(?:{pattern})(?:\s+(?:rieng|hop nhat))?\s*$", line_key, flags=re.I):
                return canonical

    # Some note tables carry a useful title in their first row.
    if table_rows:
        first = [clean_text(c) for c in table_rows[0] if clean_text(c)]
        if len(first) == 1 and 4 <= len(first[0]) <= 120 and parse_number(first[0]) is None:
            k = normalized_key(first[0])
            if not any(h in k for h in DESCRIPTION_HEADER_WORDS + REFERENCE_HEADER_WORDS):
                return slugify(first[0])

    boilerplate = (
        "cong ty",
        "ngan hang",
        "dia chi",
        "bao cao tai chinh",
        "cho nam tai chinh",
        "don vi tinh",
        "don vi",
        "dom vi",
        "tai ngay",
        "ket thuc tai ngay",
        "da duoc kiem toan",
    )

    # Prefer nearby section headings / note names, walking backwards from the table.
    for line in reversed(lines):
        key = normalized_key(line)
        if not (4 <= len(line) <= 140):
            continue
        if any(token in key for token in boilerplate):
            continue
        if re.fullmatch(r"\d+", key):
            continue
        if re.search(r"\b20\d{2}\b", key) and len(key.split()) < 8:
            continue
        # A line with a leading note number or mostly uppercase is likely a useful title.
        upper_letters = sum(1 for ch in strip_accents(line) if ch.isalpha() and ch.isupper())
        letters = sum(1 for ch in line if ch.isalpha())
        looks_heading = bool(re.match(r"^(?:[IVXLC]+\.|\d+[.)])\s*", strip_accents(line), re.I))
        looks_heading |= letters > 0 and upper_letters / letters >= 0.55
        if looks_heading or len(line.split()) <= 12:
            return f"ThuyetMinh_{slugify(line)}"

    return f"BangDuLieu_{ordinal:03d}"


def infer_unit(context: str, rows: Sequence[Sequence[str]]) -> str:
    # Closest source wins: inspect the final context lines first, then first table rows.
    probe = "\n".join(_context_lines(context)[-12:])
    # OCR variants seen in the corpus include "Đơn vị:", "Đơn vị tính:",
    # "ĐVT:" and even "Dom vi:".
    probe_key = strip_accents(probe).lower()
    m = re.findall(
        r"(?:don\s+vi(?:\s+tinh)?|dom\s+vi(?:\s+tinh)?|dvt)\s*[:\-]?\s*([^\n]{1,50})",
        probe_key,
        flags=re.I,
    )
    if m:
        unit = normalize_unit(m[-1])
        if unit:
            return unit

    first_rows = " ".join(" ".join(r) for r in rows[:3])
    return detect_unit_from_text(first_rows) or ""


def normalize_unit(value: str) -> str:
    key = normalized_key(value)
    if not key:
        return ""
    if "%" in key or "phan tram" in key:
        return "%"
    if "nghin ty" in key:
        return "Nghin ty dong"
    if "ty dong" in key or key == "ty":
        return "Ty dong"
    if "trieu dong" in key or "trieu vnd" in key:
        return "Trieu dong"
    if "nghin dong" in key or "ngan dong" in key or "nghin vnd" in key:
        return "Nghin dong"
    if "vnd/co phieu" in key or "dong/co phieu" in key:
        return "VND/co phieu"
    if key in {"vnd", "vnđ", "dong"} or "vnd" in key:
        return "VND"
    if "usd" in key:
        return "USD"
    if "co phieu" in key:
        return "Co phieu"
    if "nguoi" in key:
        return "Nguoi"
    # Keep a concise unknown source unit instead of inventing one.
    return strip_accents(clean_text(value))[:40]


def detect_unit_from_text(text: str) -> str:
    key = normalized_key(text)
    if "%" in key or "phan tram" in key:
        return "%"
    for needle, unit in (
        ("nghin ty dong", "Nghin ty dong"),
        ("ty dong", "Ty dong"),
        ("trieu dong", "Trieu dong"),
        ("nghin dong", "Nghin dong"),
        ("vnd/co phieu", "VND/co phieu"),
        ("vnd", "VND"),
        ("usd", "USD"),
    ):
        if needle in key:
            return unit
    return ""


def parse_number(value: object) -> Optional[float]:
    """Parse Vietnamese/English formatted numeric OCR cells.

    Supports thousands separators, decimal commas, parentheses for negatives,
    percent signs and a few conservative OCR substitutions inside numeric cells.
    """
    text = clean_text(value)
    if not text or text in {"-", "–", "—", "-/-", "n/a", "N/A"}:
        return None

    negative = bool(re.search(r"^\s*\(.*\)\s*$", text))
    text = text.strip().strip("()")
    text = re.sub(r"(?i)\b(?:VND|VNĐ|USD|đồng|dong|triệu|trieu|nghìn|nghin|tỷ|ty)\b", "", text)
    text = text.replace("%", "")
    text = text.replace("−", "-").replace("–", "-").replace("—", "-")
    text = re.sub(r"\s+", "", text)

    # Numeric cells occasionally contain O/l/I OCR substitutions.  Only replace
    # them when the rest of the token is numeric punctuation.
    if re.fullmatch(r"[-+0-9OoIl.,'’]+", text):
        text = text.translate(str.maketrans({"O": "0", "o": "0", "I": "1", "l": "1"}))

    text = text.replace("'", "").replace("’", "")
    text = re.sub(r"[^0-9,\.\-+]", "", text)
    if not text or text in {"-", "+", ".", ","}:
        return None

    sign = -1.0 if text.startswith("-") else 1.0
    text = text.lstrip("+-")
    if not text:
        return None

    def normalize_separators(token: str) -> str:
        if "." in token and "," in token:
            # Last separator is decimal only if followed by 1-2 digits;
            # otherwise both separators are treated as thousands separators.
            last_dot = token.rfind(".")
            last_comma = token.rfind(",")
            last = max(last_dot, last_comma)
            digits_after = len(token) - last - 1
            if digits_after in {1, 2}:
                decimal_sep = token[last]
                int_part = re.sub(r"[.,]", "", token[:last])
                frac = re.sub(r"[.,]", "", token[last + 1 :])
                return f"{int_part}.{frac}"
            return re.sub(r"[.,]", "", token)

        sep = "." if "." in token else ("," if "," in token else "")
        if not sep:
            return token

        parts = token.split(sep)
        if len(parts) > 2:
            # 1.234.567 / 1,234,567 -> thousands; a final 1-2 digit group is decimal.
            if len(parts[-1]) in {1, 2} and all(len(p) == 3 for p in parts[1:-1]):
                return "".join(parts[:-1]) + "." + parts[-1]
            return "".join(parts)

        left, right = parts
        # A single separator followed by exactly 3 digits is overwhelmingly a
        # thousands separator in Vietnamese financial statements.
        if len(right) == 3 and left:
            return left + right
        if len(right) in {1, 2}:
            return left + "." + right
        return left + right

    normalized = normalize_separators(text)
    try:
        number = float(normalized) * sign
        if negative:
            number = -abs(number)
        return number if math.isfinite(number) else None
    except ValueError:
        return None


def _row_numeric_positions(row: Sequence[str]) -> List[int]:
    return [i for i, cell in enumerate(row) if parse_number(cell) is not None]


def _is_header_like(row: Sequence[str]) -> bool:
    joined = normalized_key(" ".join(row))
    if any(word in joined for word in DESCRIPTION_HEADER_WORDS + REFERENCE_HEADER_WORDS):
        return True
    if any(word in joined for word in CURRENT_PERIOD_WORDS + PREVIOUS_PERIOD_WORDS):
        return True
    return False


def find_description_column(rows: Sequence[Sequence[str]]) -> int:
    if not rows:
        return 0
    width = max(len(r) for r in rows)
    for row in rows[:5]:
        for idx, cell in enumerate(row):
            key = normalized_key(cell)
            if any(word in key for word in DESCRIPTION_HEADER_WORDS) and not any(
                ref == key for ref in REFERENCE_HEADER_WORDS
            ):
                return idx

    best_idx = 0
    best_score = float("-inf")
    sample = rows[: min(len(rows), 40)]
    for idx in range(width):
        alpha_cells = []
        numeric_count = 0
        ref_like = 0
        for row in sample:
            cell = clean_text(row[idx]) if idx < len(row) else ""
            if not cell:
                continue
            if parse_number(cell) is not None:
                numeric_count += 1
            if re.fullmatch(r"(?:[A-Z]?\.?\d+[A-Za-z]?|[IVXLC]+\.?|V?I?\.?\d+)", strip_accents(cell), re.I):
                ref_like += 1
            if re.search(r"[A-Za-zÀ-ỹ]", cell) and parse_number(cell) is None:
                alpha_cells.append(cell)
        avg_len = sum(map(len, alpha_cells)) / len(alpha_cells) if alpha_cells else 0.0
        score = len(alpha_cells) * 3 + avg_len * 0.12 - numeric_count * 2 - ref_like * 2
        if score > best_score:
            best_score = score
            best_idx = idx
    return best_idx


def find_value_column(rows: Sequence[Sequence[str]], description_idx: int, report_year: int) -> int:
    if not rows:
        return min(description_idx + 1, 0)
    width = max(len(r) for r in rows)
    prior_year = report_year - 1 if report_year else 0
    header_rows = rows[: min(5, len(rows))]
    scores: Dict[int, float] = defaultdict(float)

    for idx in range(width):
        header = normalized_key(" ".join(row[idx] for row in header_rows if idx < len(row)))
        if idx == description_idx:
            scores[idx] -= 100
        if any(ref in header for ref in REFERENCE_HEADER_WORDS):
            scores[idx] -= 12
        if report_year and str(report_year) in header:
            scores[idx] += 22
        if prior_year and str(prior_year) in header:
            scores[idx] -= 10
        if any(word in header for word in CURRENT_PERIOD_WORDS):
            scores[idx] += 16
        if any(word in header for word in ("nam truoc", "dau nam", "dau ky", "ky truoc")):
            scores[idx] -= 12

        numeric = 0
        nonempty = 0
        for row in rows[1: min(len(rows), 45)]:
            if idx >= len(row) or not clean_text(row[idx]):
                continue
            nonempty += 1
            if parse_number(row[idx]) is not None:
                numeric += 1
        if nonempty:
            scores[idx] += 8 * numeric / nonempty
        if idx > description_idx:
            scores[idx] += max(0, 3 - 0.2 * (idx - description_idx))

    candidates = [i for i in range(width) if i != description_idx]
    if not candidates:
        return description_idx
    return max(candidates, key=lambda i: scores[i])


def _looks_percentage(indicator: str, raw_value: str, table_unit: str) -> bool:
    key = normalized_key(indicator)
    if "%" in raw_value or table_unit == "%":
        return True
    return any(hint in key for hint in PERCENT_HINTS)


def _looks_non_money(indicator: str, raw_value: str, table_unit: str) -> bool:
    key = normalized_key(indicator)
    if _looks_percentage(indicator, raw_value, table_unit):
        return True
    if table_unit in {"Co phieu", "Nguoi", "USD", "VND/co phieu"}:
        return True
    return any(hint in key for hint in NON_MONEY_HINTS)


def normalize_value_and_unit(
    value: float, raw_value: str, indicator: str, table_unit: str
) -> Tuple[float, str]:
    if _looks_percentage(indicator, raw_value, table_unit):
        return value, "%"
    if _looks_non_money(indicator, raw_value, table_unit):
        key = normalized_key(indicator)
        if table_unit:
            return value, table_unit
        if "co phieu" in key:
            return value, "Co phieu"
        if "nguoi" in key or "nhan vien" in key or "lao dong" in key:
            return value, "Nguoi"
        return value, "Khong ro"

    # For finance rows, normalize VND-family units to billion VND (Ty dong).
    unit = table_unit
    if unit == "Nghin ty dong":
        return value * 1000.0, "Ty dong"
    if unit == "Ty dong":
        return value, "Ty dong"
    if unit == "Trieu dong":
        return value / 1000.0, "Ty dong"
    if unit == "Nghin dong":
        return value / 1_000_000.0, "Ty dong"
    if unit == "VND":
        # EPS/mệnh giá style rows should stay in VND rather than becoming 1e-6 bn.
        key = normalized_key(indicator)
        if any(token in key for token in ("tren co phieu", "moi co phieu", "menh gia")):
            return value, "VND/co phieu" if "co phieu" in key else "VND"
        return value / 1_000_000_000.0, "Ty dong"

    # If no unit was captured but this is a clearly financial row with large VND-like
    # numbers, infer VND conservatively.  Otherwise preserve the unknown scale.
    key = normalized_key(indicator)
    if not unit and abs(value) >= 1_000_000 and any(hint in key for hint in MONEY_HINTS):
        return value / 1_000_000_000.0, "Ty dong"
    return value, unit or "Khong ro"


def repair_rows(rows: Sequence[Sequence[str]], description_idx: int) -> List[List[str]]:
    """Repair common OCR line breaks without borrowing values across periods.

    A text-only row can be either the continuation of the previous numeric row
    (very common when a long indicator wraps) or a prefix of the next row.
    Lowercase continuation lines are preferentially appended backwards; other
    fragments are held briefly and prepended to the next numeric row.
    """
    repaired = [list(r) for r in rows]
    pending_desc = ""

    for i, row in enumerate(repaired):
        if description_idx >= len(row):
            continue
        desc = clean_text(row[description_idx])
        numeric_positions = _row_numeric_positions(row)
        non_ref_numeric = [p for p in numeric_positions if p != description_idx]

        if desc and not non_ref_numeric and not _is_header_like(row):
            prev = repaired[i - 1] if i > 0 else None
            prev_desc = clean_text(prev[description_idx]) if prev and description_idx < len(prev) else ""
            prev_has_numeric = bool(prev and [p for p in _row_numeric_positions(prev) if p != description_idx])
            starts_like_continuation = bool(desc[:1].islower()) or desc.startswith(("(", "-"))
            if prev_has_numeric and prev_desc and starts_like_continuation:
                prev[description_idx] = clean_text(f"{prev_desc} {desc}")
                pending_desc = ""
            elif len(desc) <= 180:
                pending_desc = desc if not pending_desc else f"{pending_desc} {desc}"
            continue

        if non_ref_numeric and (not desc or len(desc) <= 2) and pending_desc:
            row[description_idx] = pending_desc
            pending_desc = ""
        elif non_ref_numeric and desc and pending_desc:
            if not re.match(r"^(?:[A-ZIVXLC]+\.|\d+[.)])", strip_accents(pending_desc), re.I):
                row[description_idx] = f"{pending_desc} {desc}"
            pending_desc = ""
        elif non_ref_numeric:
            pending_desc = ""

    return repaired


def extract_rows_from_table(table: ParsedTable) -> List[ExtractedRow]:
    rows = [list(r) for r in table.rows if any(clean_text(c) for c in r)]
    if not rows:
        return []

    desc_idx = find_description_column(rows)
    rows = repair_rows(rows, desc_idx)
    value_idx = find_value_column(rows, desc_idx, table.metadata.year)

    extracted: List[ExtractedRow] = []
    seen: set[Tuple[str, float, str]] = set()

    for row in rows:
        if desc_idx >= len(row):
            continue
        desc_raw = clean_text(row[desc_idx])
        if not desc_raw:
            continue

        desc_key = normalized_key(desc_raw)
        if any(desc_key == word for word in DESCRIPTION_HEADER_WORDS + REFERENCE_HEADER_WORDS):
            continue

        raw_value = row[value_idx] if value_idx < len(row) else ""
        value = parse_number(raw_value)

        # OCR column-shift fallback is deliberately local.  Never borrow a
        # previous-year value just because the current-year cell is blank.
        # Only accept a numeric cell immediately adjacent to the expected value
        # column, which covers the common one-cell shift caused by malformed HTML.
        if value is None and value_idx >= len(row):
            for p in (value_idx - 1, value_idx + 1):
                if p > desc_idx and 0 <= p < len(row):
                    candidate = parse_number(row[p])
                    if candidate is not None:
                        raw_value = row[p]
                        value = candidate
                        break

        if value is None:
            continue

        indicator = canonical_indicator(desc_raw)
        if not indicator or parse_number(indicator) is not None:
            continue
        # Ignore signature/contact/page-number-like text accidentally aligned with a number.
        ikey = normalized_key(indicator)
        if any(token in ikey for token in ("nguoi lap bieu", "ke toan truong", "tong giam doc")):
            continue

        norm_value, norm_unit = normalize_value_and_unit(value, clean_text(raw_value), indicator, table.unit)
        # Avoid -0.0 in generated CSVs.
        if abs(norm_value) < 1e-15:
            norm_value = 0.0
        item = (indicator, norm_value, norm_unit)
        if item in seen:
            continue
        seen.add(item)
        extracted.append(ExtractedRow(*item))

    return extracted


def parse_html_table(table_html: str) -> List[List[str]]:
    """Parse the corpus' simple HTML-like table markup efficiently.

    The ViFinQA OCR files consistently use flat ``<tr><td>`` structures. A
    regex fast path is substantially faster than constructing an HTMLParser for
    tens of thousands of tables, while still honoring rowspan/colspan. The
    standard-library HTMLParser remains a fallback for malformed edge cases.
    """
    raw_rows: List[List[ParsedCell]] = []
    for row_match in re.finditer(r"<tr\b[^>]*>(.*?)</tr>", table_html, flags=re.I | re.S):
        row_html = row_match.group(1)
        cells: List[ParsedCell] = []
        for cell_match in re.finditer(
            r"<t[dh]\b([^>]*)>(.*?)</t[dh]>", row_html, flags=re.I | re.S
        ):
            attrs, body = cell_match.groups()
            rowspan_match = re.search(r"\browspan\s*=\s*[\"']?(\d+)", attrs, flags=re.I)
            colspan_match = re.search(r"\bcolspan\s*=\s*[\"']?(\d+)", attrs, flags=re.I)
            rowspan = _safe_positive_int(rowspan_match.group(1) if rowspan_match else None, 1)
            colspan = _safe_positive_int(colspan_match.group(1) if colspan_match else None, 1)
            body = re.sub(r"<br\s*/?>", " ", body, flags=re.I)
            text = clean_text(re.sub(r"<[^>]+>", " ", body))
            cells.append(ParsedCell(text=text, rowspan=rowspan, colspan=colspan))
        if cells:
            raw_rows.append(cells)

    if raw_rows:
        parser = _SingleTableParser()
        parser.raw_rows = raw_rows
        return parser.expanded_rows()

    parser = _SingleTableParser()
    try:
        parser.feed(table_html)
        parser.close()
        return parser.expanded_rows()
    except Exception:
        return []


def iter_html_tables(content: str, path: Path, metadata: ReportMetadata) -> Iterator[ParsedTable]:
    table_re = re.compile(r"<table\b[^>]*>.*?</table>", flags=re.I | re.S)
    for ordinal, match in enumerate(table_re.finditer(content), start=1):
        rows = parse_html_table(match.group(0))
        if not rows:
            continue
        context = content[max(0, match.start() - 2200) : match.start()]
        title = infer_table_title(context, rows, ordinal)
        unit = infer_unit(context, rows)
        yield ParsedTable(
            rows=rows,
            title=title,
            unit=unit,
            ordinal=ordinal,
            source_path=path,
            metadata=metadata,
        )


def iter_plain_text_tables(content: str, path: Path, metadata: ReportMetadata, start_ordinal: int = 1) -> Iterator[ParsedTable]:
    """Fallback detector for whitespace/tab separated table blocks.

    A block needs at least two table-like rows.  Each row must split into at
    least two cells and contain a numeric token.  This is not normally needed by
    the current corpus (which has HTML-like tables) but fulfills the line-scan
    requirement and protects against future OCR format changes.
    """
    without_html = re.sub(r"<table\b[^>]*>.*?</table>", "\n", content, flags=re.I | re.S)
    blocks: List[List[List[str]]] = []
    current: List[List[str]] = []

    for raw_line in without_html.splitlines():
        line = clean_text(raw_line)
        if not line:
            if len(current) >= 2:
                blocks.append(current)
            current = []
            continue
        cells = [clean_text(c) for c in re.split(r"\t+|\s{2,}|\s*\|\s*", line) if clean_text(c)]
        numeric = sum(parse_number(c) is not None for c in cells)
        if len(cells) >= 2 and numeric >= 1:
            current.append(cells)
        else:
            if len(current) >= 2:
                blocks.append(current)
            current = []
    if len(current) >= 2:
        blocks.append(current)

    for offset, rows in enumerate(blocks):
        width = max(len(r) for r in rows)
        padded = [r + [""] * (width - len(r)) for r in rows]
        ordinal = start_ordinal + offset
        yield ParsedTable(
            rows=padded,
            title=f"BangText_{ordinal:03d}",
            unit=detect_unit_from_text(" ".join(" ".join(r) for r in rows[:3])),
            ordinal=ordinal,
            source_path=path,
            metadata=metadata,
        )


def _scope_suffix(scope: str) -> str:
    if scope == "consolidated":
        return "_HopNhat"
    # Keep separate/company-parent filenames compatible with existing mock naming.
    if scope == "unknown":
        return "_KhongRo"
    return ""


def output_filename(table: ParsedTable, collision_index: int = 0) -> str:
    ticker = re.sub(r"[^A-Za-z0-9]", "", table.metadata.ticker.upper()) or "UNK"
    year = str(table.metadata.year or "UnknownYear")
    title = re.sub(r"[^A-Za-z0-9_]", "", table.title) or f"BangDuLieu_{table.ordinal:03d}"
    base = f"{ticker}_{year}_{title}{_scope_suffix(table.metadata.scope)}"
    if collision_index:
        base += f"_{collision_index:02d}"
    return base + ".csv"


def dedupe_extracted_rows(rows: Iterable[ExtractedRow]) -> List[ExtractedRow]:
    out: List[ExtractedRow] = []
    seen: set[Tuple[str, float, str]] = set()
    for row in rows:
        key = (row.indicator, row.value, row.unit)
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def write_canonical_csv(path: Path, rows: Sequence[ExtractedRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CANONICAL_HEADER)
        for row in rows:
            # repr-like formatting keeps enough precision while producing clean
            # values such as 45000.0 for integer-valued floats.
            value = float(row.value)
            writer.writerow([row.indicator, value, row.unit])


def extract_tables_from_txt(
    txt_path: os.PathLike[str] | str,
    output_dir: os.PathLike[str] | str,
    company: Optional[str] = None,
    year: Optional[int | str] = None,
    report_type: Optional[str] = None,
    raw_root: Optional[os.PathLike[str] | str] = None,
    include_plain_text_fallback: bool = True,
) -> List[Path]:
    """Extract every useful table from one OCR text report.

    ``company/year/report_type`` remain optional for backwards compatibility
    with the original baseline function signature.  Metadata inferred from the
    path is used when those arguments are omitted.
    """
    path = Path(txt_path)
    out_dir = Path(output_dir)
    root = Path(raw_root) if raw_root is not None else path.parent
    content = path.read_text(encoding="utf-8", errors="ignore")
    metadata = infer_metadata(path, root, content)

    if company:
        metadata = ReportMetadata(str(company).upper(), metadata.year, metadata.scope, metadata.company_name)
    if year:
        try:
            metadata = ReportMetadata(metadata.ticker, int(year), metadata.scope, metadata.company_name)
        except (TypeError, ValueError):
            pass
    if report_type:
        rt = str(report_type).lower()
        scope = "consolidated" if "consolid" in rt or "hop" in rt else "separate" if "separate" in rt or "rieng" in rt else metadata.scope
        metadata = ReportMetadata(metadata.ticker, metadata.year, scope, metadata.company_name)

    tables = list(iter_html_tables(content, path, metadata))
    # Only use the whitespace detector when HTML-like tables are absent.  Mixing
    # both modes on the current corpus would duplicate data and turn prose with
    # dates/numbers into noisy pseudo-tables.
    if include_plain_text_fallback and not tables:
        tables.extend(iter_plain_text_tables(content, path, metadata, start_ordinal=1))

    written: List[Path] = []
    name_counts: Dict[str, int] = defaultdict(int)
    statement_rows: Dict[str, List[ExtractedRow]] = defaultdict(list)
    statement_tables: Dict[str, ParsedTable] = {}

    for table in tables:
        rows = extract_rows_from_table(table)
        if not rows:
            continue
        if table.title in CANONICAL_STATEMENT_TITLES:
            statement_rows[table.title].extend(rows)
            statement_tables.setdefault(table.title, table)
            continue

        natural = output_filename(table, 0)
        name_counts[natural] += 1
        collision_index = name_counts[natural] - 1
        target = out_dir / output_filename(table, collision_index)
        write_canonical_csv(target, rows)
        written.append(target)

    for title, rows in statement_rows.items():
        table = statement_tables[title]
        target = out_dir / output_filename(table, 0)
        write_canonical_csv(target, dedupe_extracted_rows(rows))
        written.append(target)
    return written


def _discover_txt_files(raw_dir: Path) -> List[Path]:
    root = raw_dir / "financial_statements" if (raw_dir / "financial_statements").is_dir() else raw_dir
    return sorted(root.rglob("*.txt"))


def process_all_reports(
    raw_dir: os.PathLike[str] | str,
    processed_dir: os.PathLike[str] | str,
    max_companies: Optional[int] = None,
    limit_files: Optional[int] = None,
    clean_output: bool = False,
    include_plain_text_fallback: bool = True,
    verbose: bool = False,
) -> ExtractionStats:
    """Extract the full OCR warehouse into canonical CSV files.

    ``max_companies`` is retained only for compatibility with the old baseline;
    ``None`` means all companies (the new default).
    """
    raw_root = Path(raw_dir)
    out_dir = Path(processed_dir)
    stats = ExtractionStats()

    if clean_output and out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # File-system existence checks become expensive once processed_csv contains
    # tens of thousands of tables. Keep an in-memory name index instead.
    used_names = {p.name for p in out_dir.glob("*.csv")}

    files = _discover_txt_files(raw_root)
    if max_companies is not None:
        tickers: List[str] = []
        selected: List[Path] = []
        base = raw_root / "financial_statements" if (raw_root / "financial_statements").is_dir() else raw_root
        for p in files:
            try:
                rel = p.relative_to(base)
                ticker = rel.parts[0] if rel.parts else "UNK"
            except ValueError:
                ticker = "UNK"
            if ticker not in tickers:
                if len(tickers) >= max_companies:
                    continue
                tickers.append(ticker)
            selected.append(p)
        files = selected
    if limit_files is not None:
        files = files[: max(0, limit_files)]

    for idx, path in enumerate(files, start=1):
        stats.files_seen += 1
        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            metadata = infer_metadata(path, raw_root, content)
            html_tables = list(iter_html_tables(content, path, metadata))
            stats.tables_seen += len(html_tables)
            tables = html_tables
            if include_plain_text_fallback and not html_tables:
                tables = list(iter_plain_text_tables(content, path, metadata, start_ordinal=1))

            # Aggregate multi-page primary statements into one mock-compatible
            # file; keep note tables separate with collision-safe semantic names.
            local_counts: Dict[str, int] = defaultdict(int)
            statement_rows: Dict[str, List[ExtractedRow]] = defaultdict(list)
            statement_tables: Dict[str, ParsedTable] = {}

            for table in tables:
                rows = extract_rows_from_table(table)
                if not rows:
                    continue
                if table.title in CANONICAL_STATEMENT_TITLES:
                    statement_rows[table.title].extend(rows)
                    statement_tables.setdefault(table.title, table)
                    continue

                natural = output_filename(table, 0)
                local_counts[natural] += 1
                collision = local_counts[natural] - 1
                filename = output_filename(table, collision)
                while filename in used_names:
                    collision += 1
                    filename = output_filename(table, collision)
                used_names.add(filename)
                target = out_dir / filename
                write_canonical_csv(target, rows)
                stats.tables_written += 1
                stats.rows_written += len(rows)

            for title, merged in statement_rows.items():
                table = statement_tables[title]
                rows = dedupe_extracted_rows(merged)
                collision = 0
                filename = output_filename(table, collision)
                while filename in used_names:
                    collision += 1
                    filename = output_filename(table, collision)
                used_names.add(filename)
                target = out_dir / filename
                write_canonical_csv(target, rows)
                stats.tables_written += 1
                stats.rows_written += len(rows)

            if verbose:
                print(f"[{idx}/{len(files)}] {path} -> {stats.tables_written} CSV tables total")
        except Exception as exc:  # continue corpus extraction after one bad OCR file
            stats.files_failed += 1
            print(f"[WARN] Failed to extract {path}: {exc}")

    return stats


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract ViFinQA OCR .txt tables to canonical CSV files.")
    parser.add_argument("--raw-dir", default="data/raw_vifinqa", help="Raw ViFinQA directory (default: data/raw_vifinqa)")
    parser.add_argument("--processed-dir", default="data/processed_csv", help="Output CSV directory")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N txt files (smoke testing)")
    parser.add_argument("--max-companies", type=int, default=None, help="Optional compatibility/debug limit by ticker")
    parser.add_argument("--clean", action="store_true", help="Delete processed-dir before extraction")
    parser.add_argument("--no-text-fallback", action="store_true", help="Disable whitespace-table fallback scan")
    parser.add_argument("--verbose", action="store_true", help="Print per-file progress")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    stats = process_all_reports(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        max_companies=args.max_companies,
        limit_files=args.limit,
        clean_output=args.clean,
        include_plain_text_fallback=not args.no_text_fallback,
        verbose=args.verbose,
    )
    print(
        "Extraction completed: "
        f"files={stats.files_seen}, tables_seen={stats.tables_seen}, "
        f"csv_written={stats.tables_written}, rows_written={stats.rows_written}, "
        f"failed_files={stats.files_failed}."
    )
    return 0 if stats.files_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
