"""Trich xuat bang tai chinh tu corpus OCR ViFinQA sang CSV ba cot.

Module chi dung quy tac xac dinh, khong goi LLM. Moi tep TXT duoc xu ly doc lap
de mot bao cao hong khong lam dung toan bo corpus.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import math
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Optional, Sequence

from bs4 import BeautifulSoup, NavigableString, Tag


CSV_COLUMNS = ("Chi_tieu", "Gia_tri", "Don_vi")
REPORT_TYPES = ("consolidated", "separate", "aggregated")
KNOWN_TABLE_SLUGS = {
    "BangCanDoiKeToan",
    "BaoCaoTinhHinhTaiChinh",
    "BaoCaoKetQuaKinhDoanh",
    "BaoCaoLuuChuyenTienTe",
    "BaoCaoThayDoiVonChuSoHuu",
}

FINANCIAL_KEYWORDS = (
    "tai san",
    "nguon von",
    "no phai tra",
    "von chu so huu",
    "doanh thu",
    "loi nhuan",
    "chi phi",
    "thu nhap",
    "luu chuyen tien",
    "tien va",
    "tien mat",
    "so du",
    "nguyen gia",
    "khau hao",
    "du phong",
    "phai thu",
    "phai tra",
    "hang ton kho",
    "thue",
    "lai co ban",
    "co phieu",
    "tin dung",
    "cho vay",
    "tien gui",
)

HEADER_KEYWORDS = (
    "chi tieu",
    "ma so",
    "thuyet minh",
    "ghi chu",
    "stt",
    "noi dung",
    "nam nay",
    "nam truoc",
    "ky nay",
    "ky truoc",
    "tai ngay",
    "don vi",
)

METADATA_COLUMN_KEYWORDS = (
    "ma so",
    "thuyet minh",
    "ghi chu",
    "stt",
    "so thu tu",
    "code",
    "trang",
    "ty le",
    "phan tram",
)

ADMINISTRATIVE_KEYWORDS = (
    "hoi dong quan tri",
    "ban tong giam doc",
    "ban kiem soat",
    "chu tich",
    "thanh vien",
    "tru so chinh",
    "dia chi",
    "kiem toan vien",
    "giay phep",
    "chuc vu",
)

MISSING_VALUE_TOKENS = {
    "",
    "-",
    "--",
    "—",
    "–",
    "−",
    "n/a",
    "na",
    "nil",
}


@dataclass(frozen=True)
class ReportMetadata:
    """Metadata lay chu yeu tu duong dan cua mot bao cao."""

    ticker: str
    company_name: str
    report_year: int
    report_type: str
    source_txt: Path


@dataclass
class RawTable:
    """Bang trung gian sau khi doc cau truc HTML hoac plain text."""

    title: str
    rows: list[list[str]]
    parser: str
    unit: str = ""
    unit_source: str = ""
    unit_confidence: str = "low"
    continued: bool = False
    source_table_index: int = -1
    context_before: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class ExtractedTable:
    """Bang tai chinh da chon dung mot ky va chuan hoa thanh ba cot."""

    table_title: str
    table_slug: str
    unit: str
    value_period: str
    parser: str
    records: list[dict[str, object]]
    value_column_method: str = ""
    value_column_header: str = ""
    value_column_confidence: str = "low"
    candidate_columns: list[dict[str, object]] = field(default_factory=list)
    unit_source: str = ""
    unit_confidence: str = "low"
    default_unit: str = ""
    report_year: int = 0
    source_table_indices: list[int] = field(default_factory=list)
    header_signature: str = ""
    logical_table_id: str = ""
    continued: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ValueColumnDecision:
    """Ket qua cham diem cot gia tri co the audit duoc."""

    column: Optional[int]
    method: str
    header: str
    value_period: str
    confidence: str
    candidates: list[dict[str, object]]
    warnings: list[str]


@dataclass
class ExtractionDiagnostics:
    """Du lieu can review duoc gom rieng, khong lam thay doi schema CSV."""

    rejected_cells: list[dict[str, object]] = field(default_factory=list)
    quarantine: list[dict[str, object]] = field(default_factory=list)


@dataclass
class ExtractionStats:
    """Thong ke van hanh cua mot lan chay extractor."""

    txt_scanned: int = 0
    tables_detected: int = 0
    csv_written: int = 0
    tables_skipped: int = 0
    errors: int = 0
    warnings: int = 0
    tables_quarantined: int = 0
    cells_rejected: int = 0
    elapsed_seconds: float = 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "txt_scanned": self.txt_scanned,
            "tables_detected": self.tables_detected,
            "csv_written": self.csv_written,
            "tables_skipped": self.tables_skipped,
            "errors": self.errors,
            "warnings": self.warnings,
            "tables_quarantined": self.tables_quarantined,
            "cells_rejected": self.cells_rejected,
            "elapsed_seconds": round(self.elapsed_seconds, 3),
        }


def normalize_space(value: object) -> str:
    """Chuan hoa HTML entity, Unicode va khoang trang OCR."""

    if value is None:
        return ""
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ").replace("\u2007", " ").replace("\u202f", " ")
    text = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def fold_text(value: object) -> str:
    """Bo dau de so khop tu khoa nhung khong sua noi dung xuat CSV."""

    text = normalize_space(value).replace("Đ", "D").replace("đ", "d")
    text = "".join(
        char for char in unicodedata.normalize("NFD", text) if unicodedata.category(char) != "Mn"
    )
    return text.casefold()


def parse_number(value: object) -> Optional[int | float]:
    """Doc so Viet/Anh, dau tru Unicode va so am trong ngoac.

    Dau gach don va o rong tra ve ``None`` thay vi bi bien thanh 0.
    """

    raw_text = "" if value is None else html.unescape(str(value))
    text = normalize_space(value)
    if text.casefold() in MISSING_VALUE_TOKENS:
        return None

    negative = False
    if text.startswith("(") and text.endswith(")"):
        negative = True
        text = normalize_space(text[1:-1])

    text = text.replace("−", "-").replace("–", "-").replace("—", "-").replace("﹣", "-")
    if text.startswith("-"):
        negative = not negative
        text = text[1:].strip()

    text = re.sub(r"(?i)\b(vnd|vnđ|dong|đong|d|eps)\b", "", text)
    text = text.replace("₫", "").replace("%", "").replace(" ", "")
    # Khoang trang/line break nam giua hai token so la dau vet hai gia tri OCR
    # bi dinh. Khong duoc silently xoa khoang trang roi bien thanh mot so moi.
    raw_numeric = unicodedata.normalize("NFKC", raw_text)
    space_grouped_number = bool(
        re.fullmatch(
            r"\s*\(?[-−–—]?\s*\d{1,3}(?:[\s\u00a0\u202f]\d{3})+"
            r"(?:[.,]\d{1,2})?\)?\s*(?:%|VND|VNĐ|đ|₫)?\s*",
            raw_numeric,
            flags=re.IGNORECASE,
        )
    )
    if not space_grouped_number:
        if re.search(r"\d\s+[-−–—(]?\s*\d", raw_numeric):
            return None
        if len(re.findall(r"[-−–—]?\(?\d[\d.,]*\)?", raw_numeric)) > 1:
            return None
    if not text or not re.fullmatch(r"\d[\d.,]*", text):
        return None
    # So VND hop le trong corpus nam gon trong signed int64. Chuoi dai hon
    # thuong la nhieu cot/nhieu dong OCR bi dinh lien, khong duoc phep coi la
    # mot gia tri tai chinh duy nhat.
    if len(re.sub(r"\D", "", text)) > 18:
        return None

    decimal_separator = ""
    if "." in text and "," in text:
        last_dot = text.rfind(".")
        last_comma = text.rfind(",")
        candidate = "." if last_dot > last_comma else ","
        tail = text.rsplit(candidate, 1)[1]
        if len(tail) in (1, 2):
            decimal_separator = candidate
    elif "." in text or "," in text:
        separator = "." if "." in text else ","
        groups = text.split(separator)
        if len(groups) == 2 and len(groups[1]) in (1, 2):
            decimal_separator = separator
        elif len(groups) > 2 and not all(len(group) == 3 for group in groups[1:]):
            if len(groups[-1]) in (1, 2):
                decimal_separator = separator

    integer_text = text
    if decimal_separator:
        integer_text = text.rsplit(decimal_separator, 1)[0]
    grouping_separators = [separator for separator in (".", ",") if separator in integer_text]
    for separator in grouping_separators:
        groups = integer_text.split(separator)
        if not 1 <= len(groups[0]) <= 3 or not all(len(group) == 3 for group in groups[1:]):
            return None

    if decimal_separator:
        thousands_separator = "," if decimal_separator == "." else "."
        normalized = text.replace(thousands_separator, "")
        normalized = normalized.replace(decimal_separator, ".")
    else:
        normalized = text.replace(".", "").replace(",", "")

    try:
        number = float(normalized) if "." in normalized else int(normalized)
    except ValueError:
        return None

    if negative:
        number = -number
    if isinstance(number, float) and math.isfinite(number) and number.is_integer():
        return int(number)
    return number


def detect_unit(text: object) -> str:
    """Nhan dien don vi nguon, uu tien mau cu the hon mau tong quat."""

    raw = normalize_space(text)
    folded = fold_text(raw)
    if not raw:
        return ""
    if re.search(r"(?i)(vnd|dong|đồng)\s*/\s*(co phieu|cổ phiếu)", raw) or "eps" in folded:
        return "VND/co phieu"
    # Longest/specific-first matching is semantically significant.  In
    # particular, ``tram ty dong`` is 100 times ``ty dong`` and must not fall
    # through to the shorter suffix.
    if "nghin ty dong" in folded or "ngan ty dong" in folded or "nghin ty vnd" in folded:
        return "Nghin ty dong"
    if "tram ty dong" in folded or "tram ty vnd" in folded:
        return "Tram ty dong"
    if "trieu usd" in folded:
        return "Trieu USD"
    if "nghin usd" in folded or "ngan usd" in folded:
        return "Nghin USD"
    if "trieu vnd" in folded or "trieu vnđ" in folded:
        return "Trieu VND"
    if "nghin vnd" in folded or "ngan vnd" in folded:
        return "Nghin VND"
    if "ty dong" in folded or "ti dong" in folded:
        return "Ty dong"
    if "trieu dong" in folded:
        return "Trieu dong"
    if "nghin dong" in folded or "ngan dong" in folded:
        return "Nghin dong"
    if re.search(r"(?i)(?:JPY)\b", raw) or any(
        token in folded for token in ("yen nhat", "yen nhật")
    ):
        return "JPY"
    if re.search(r"(?i)(?:USD)\b", raw) or any(
        token in folded for token in ("do la my", "dollar my", "us dollar")
    ):
        return "USD"
    if re.search(r"(?i)(?:EUR)\b", raw) or "euro" in folded:
        return "EUR"
    if "diem phan tram" in folded or re.search(r"(?i)\bpp\b", raw):
        return "Diem phan tram"
    if re.search(r"\blan\b", folded):
        return "Lan"
    if "co phieu" in folded and any(token in folded for token in ("so luong", "binh quan", "don vi")):
        return "Co phieu"
    compact_raw = raw.replace(" ", "")
    attached_vnd = bool(
        re.search(r"(?i)(?:nam|năm|nay|truoc|trước|ky|kỳ|ngay|ngày|cuoi|cuối|dau|đầu)vnd\b", compact_raw)
        or re.search(r"(?i)\d(?:[./-]?\d)*vnd\b", compact_raw)
    )
    attached_upper_vnd = bool(re.search(r"(?:VND|VNĐ)\b", raw))
    if (
        re.search(r"(?i)\b(vnd|vnđ)\b", raw)
        or attached_vnd
        or attached_upper_vnd
        or re.search(r"(?i)\b(dong|đồng)\b", raw)
    ):
        return "VND"
    if "%" in raw or "phan tram" in folded:
        return "%"
    return ""


def infer_row_unit(label: str, raw_value: str, table_unit: str) -> str:
    """Giu don vi bang, chi doi khi o/chi tieu cho bang chung ro rang."""

    folded_label = fold_text(label)
    if "%" in raw_value or "%" in label or any(
        token in folded_label
        for token in ("ty le", "phan tram", "thue suat", "lai suat", "bien loi nhuan")
    ):
        return "%"
    if "eps" in folded_label or any(
        token in folded_label
        for token in (
            "lai co ban tren co phieu",
            "lai suy giam tren co phieu",
            "lai tren co phieu",
            "thu nhap tren moi co phieu",
        )
    ):
        return "VND/co phieu"
    if "so luong co phieu" in folded_label or "co phieu binh quan" in folded_label:
        return "Co phieu"
    return table_unit


def _span_value(cell: Tag, attribute: str) -> int:
    value = normalize_space(cell.get(attribute, "1"))
    if value.isdigit() and int(value) > 0:
        return int(value)
    return 1


def expand_html_table(table: Tag) -> list[list[str]]:
    """Mo rong ``rowspan``/``colspan`` thanh ma tran chu nhat."""

    rows: list[list[str]] = []
    spans: dict[int, tuple[str, int]] = {}

    for tr in table.find_all("tr"):
        direct_cells = tr.find_all(["td", "th"], recursive=False)
        cells = direct_cells if direct_cells else tr.find_all(["td", "th"])
        if not cells and not spans:
            continue

        row: list[str] = []
        column = 0

        def consume_span(target_column: int) -> None:
            span_text, remaining = spans[target_column]
            row.append(span_text)
            if remaining <= 1:
                del spans[target_column]
            else:
                spans[target_column] = (span_text, remaining - 1)

        for cell in cells:
            while column in spans:
                consume_span(column)
                column += 1

            cell_text = normalize_space(cell.get_text(" ", strip=True))
            colspan = _span_value(cell, "colspan")
            rowspan = _span_value(cell, "rowspan")
            for _ in range(colspan):
                while column in spans:
                    consume_span(column)
                    column += 1
                row.append(cell_text)
                if rowspan > 1:
                    spans[column] = (cell_text, rowspan - 1)
                column += 1

        if spans:
            max_column = max(spans)
            while column <= max_column:
                if column in spans:
                    consume_span(column)
                else:
                    row.append("")
                column += 1

        if any(row):
            rows.append(row)

    width = max((len(row) for row in rows), default=0)
    return [row + [""] * (width - len(row)) for row in rows]


def canonical_table_slug(title: str) -> str:
    """Chuan hoa ten bang thanh slug khong dau, on dinh."""

    folded = fold_text(title)
    # Chi coi la statement core khi cum tu nam o dau/gan dau tieu de. Cac note
    # nhu "... ghi nhan tren bang can doi ke toan" van la bang tai chinh nhung
    # khong duoc lam phong dai core coverage hay bi merge vao statement chinh.
    core_prefix = re.sub(
        r"^\s*(?:phu luc\s+)?(?:\d+(?:\.\d+)*[.)]?\s+)?", "", folded
    )
    # Some OCR reports put the legal company name on the same line immediately
    # before a core-statement title.  Accept that narrow corporate prefix while
    # still rejecting note titles such as "ghi nhan trong bao cao ket qua...".
    if re.match(r"^(?:cong ty|tap doan|tong cong ty|ngan hang)\b", core_prefix):
        embedded_core = re.search(
            r"\b(?:bang can doi ke toan|bao cao tinh hinh tai chinh|"
            r"bao cao (?:ket qua hoat dong kinh doanh|ket qua kinh doanh|"
            r"(?:luu|lu) chuyen tien te|thay doi von chu so huu))\b",
            core_prefix,
        )
        if embedded_core and embedded_core.start() <= 140:
            core_prefix = core_prefix[embedded_core.start():]
    if re.match(r"^(?:bang\s+)?can doi ke toan\b|^bang can doi ke toan\b", core_prefix):
        return "BangCanDoiKeToan"
    if re.match(r"^(?:bao cao\s+)?tinh hinh tai chinh\b", core_prefix):
        return "BaoCaoTinhHinhTaiChinh"
    if re.match(
        r"^(?:bao cao\s+)?(?:ket qua hoat dong kinh doanh|ket qua kinh doanh)\b",
        core_prefix,
    ):
        return "BaoCaoKetQuaKinhDoanh"
    if re.match(r"^(?:bao cao\s+)?(?:luu|lu) chuyen tien te\b", core_prefix):
        return "BaoCaoLuuChuyenTienTe"
    if re.match(r"^(?:bao cao\s+)?thay doi von chu so huu\b", core_prefix):
        return "BaoCaoThayDoiVonChuSoHuu"

    cleaned = re.sub(r"\b(tiep theo|tiep|continued|hop nhat|tong hop|rieng le)\b", " ", folded)
    words = re.findall(r"[a-z0-9]+", cleaned)
    ignored = {"don", "vi", "tinh", "nam", "tai", "ngay", "bang"}
    words = [word for word in words if word not in ignored]
    if not words:
        return "BangTaiChinh"
    slug = "".join(word.capitalize() for word in words[:10])
    return slug[:64] or "BangTaiChinh"


def is_continuation_title(title: str) -> bool:
    folded = fold_text(title)
    return bool(
        re.search(
            r"(?:[([]\s*(?:tiep theo|tiep|continued)\s*[)\]]|(?:^|\s)(?:tiep theo|continued)\s*[)\]]?\s*$)",
            folded,
        )
    )


def _has_nearby_continuation_marker(
    title: str, context_nearest_first: Sequence[str]
) -> bool:
    """Use explicit page-context markers when the visual title omits them."""

    if is_continuation_title(title):
        return True
    title_slug = canonical_table_slug(title)
    if title_slug not in KNOWN_TABLE_SLUGS:
        return False
    nearby = list(context_nearest_first[:10])
    for marker_index, item in enumerate(nearby):
        if len(item) > 220 or not is_continuation_title(item):
            continue
        folded = fold_text(item)
        if canonical_table_slug(item) == title_slug:
            return True
        if title_slug == "BaoCaoLuuChuyenTienTe" and re.search(
            r"phuong phap\s+(?:truc|gian)\s+tiep", folded
        ):
            return True

        generic = bool(
            re.fullmatch(r"[([]?\s*(?:tiep theo|tiep|continued)\s*[)\]]?", folded)
        )
        split_period = bool(
            re.match(
                r"^(?:tai ngay|cho nam|nam ket thuc|ngay|ky ket thuc)\b",
                folded,
            )
        )
        if not (generic or split_period):
            continue
        neighbors = nearby[max(0, marker_index - 3) : marker_index + 4]
        if any(canonical_table_slug(neighbor) == title_slug for neighbor in neighbors):
            return True
    return False


def _base_table_title(title: str) -> str:
    cleaned = re.sub(
        r"(?i)(?:\s*[\[(]\s*(?:tiếp theo|tiếp|continued)\s*[\])]\s*|\s*(?:tiếp theo|continued)\s*$)",
        " ",
        title,
        count=1,
    )
    return normalize_space(cleaned).strip("-:;,. ") or "Bảng tài chính"


def _nearby_text_before(table: Tag, limit: int = 24) -> list[str]:
    candidates: list[str] = []
    node = table.previous_element
    while node is not None and len(candidates) < limit:
        if isinstance(node, NavigableString):
            parent_table = node.find_parent("table")
            if parent_table is None:
                # BeautifulSoup thuong gom toan bo van ban giua hai bang vao mot
                # NavigableString. Tach tung dong va doc tu duoi len de giu dung
                # thu tu "gan bang nhat truoc".
                for raw_line in reversed(str(node).splitlines()):
                    value = normalize_space(raw_line)
                    if value and not value.startswith("====="):
                        candidates.append(value)
                    if len(candidates) >= limit:
                        break
        node = node.previous_element
    return candidates


def _looks_like_table_title(text: str) -> bool:
    folded = fold_text(text)
    slug = canonical_table_slug(text)
    if slug in KNOWN_TABLE_SLUGS:
        return True
    if re.match(r"^\d+(?:\.\d+)*\.?\s+\D", folded):
        return len(text) <= 160
    title_terms = ("bao cao", "bang", "thuyet minh", "tai san", "doanh thu", "chi phi", "so du")
    return len(text) <= 160 and any(term in folded for term in title_terms)


def _title_from_context(context_nearest_first: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    for width in (1, 2, 3):
        for start in range(0, min(len(context_nearest_first), 12) - width + 1):
            ordered = list(reversed(context_nearest_first[start : start + width]))
            candidate = normalize_space(" ".join(ordered))
            if _looks_like_table_title(candidate):
                return candidate[:180]

    for row in rows[:3]:
        candidate = normalize_space(" ".join(cell for cell in row if cell))
        if _looks_like_table_title(candidate):
            return candidate[:180]
    return "Bảng tài chính"


def _header_band_rows(rows: Sequence[Sequence[str]], limit: int = 8) -> list[list[str]]:
    """Return only the leading structural header rows, never ordinary data rows.

    OCR exports often omit ``th`` tags, so the boundary is inferred from the
    first row containing a numeric amount. Stand-alone years and dates remain
    valid header evidence; tax rates or other numeric data do not.
    """

    band: list[list[str]] = []
    for row in rows[:limit]:
        has_data_number = False
        nonempty = [normalize_space(cell) for cell in row if normalize_space(cell)]
        for raw_cell in row:
            cell = normalize_space(raw_cell)
            parsed = parse_number(cell)
            if parsed is None:
                continue
            folded = fold_text(cell)
            # ``parse_number`` intentionally accepts values carrying a unit,
            # so period headers such as ``2022 VND`` would otherwise be
            # mistaken for the first data row.  Strip only a recognized unit
            # suffix here; arbitrary text after a number remains body data.
            period_token = re.sub(
                r"\s+(?:(?:nghin|ngan|trieu|tram\s+ty|ty)\s+)?"
                r"(?:vnd|dong|usd|eur|jpy)$|\s*%$",
                "",
                folded,
            ).strip()
            is_year = bool(re.fullmatch(r"(?:19|20)\d{2}", period_token))
            is_date = bool(
                re.fullmatch(
                    r"(?:0?[1-9]|[12]\d|3[01])[./-](?:0?[1-9]|1[0-2])[./-](?:19|20)\d{2}",
                    period_token,
                )
            )
            if not (is_year or is_date):
                has_data_number = True
                break
        if has_data_number:
            break
        row_folded = fold_text(" ".join(nonempty))
        row_has_header_signal = bool(
            re.search(r"\b(?:19|20)\d{2}\b", row_folded)
            or any(token in row_folded for token in HEADER_KEYWORDS)
            or any(
                token in row_folded
                for token in ("chi tieu", "ma so", "thuyet minh", "don vi", "dvt")
            )
        )
        # Once a header has been seen, a label-only/category row starts the
        # body even when its selected value is blank (common in bank notes).
        if band and nonempty and not row_has_header_signal:
            first_nonempty_column = next(
                (index for index, cell in enumerate(row) if normalize_space(cell)), -1
            )
            if first_nonempty_column == 0 or len(set(nonempty)) == 1:
                break
        band.append(list(row))
    return band


def _unit_from_html_evidence(
    table: Tag,
    title: str,
    rows: Sequence[Sequence[str]],
    context_nearest_first: Sequence[str],
) -> tuple[str, str, str]:
    """Doc tung nguon don vi rieng de khong tron VND va cot ty le %."""

    caption = table.find("caption")
    caption_text = normalize_space(caption.get_text(" ", strip=True)) if caption else ""
    evidence: list[tuple[str, str, str]] = []
    if caption_text:
        evidence.append(("caption", caption_text, "high"))
    title_folded = fold_text(title)
    title_has_explicit_unit = any(
        token in title_folded for token in ("don vi", "dvt", "tinh bang")
    )
    if title_has_explicit_unit:
        evidence.append(("title", title, "high"))

    # Column-specific percentages/EPS are row units, not a table default. Only
    # explicit or monetary units in the actual leading header band qualify.
    for row in _header_band_rows(rows):
        for cell in row:
            text = normalize_space(cell)
            unit = detect_unit(text)
            explicit = any(
                token in fold_text(text) for token in ("don vi", "dvt", "tinh bang")
            )
            if text and unit and (explicit or unit not in {"%", "Co phieu", "VND/co phieu"}):
                evidence.append(("header", text, "high"))

    # An explicit source declaration immediately before the table is stronger
    # than a column-specific special unit that happens to occur in a header.
    for distance, text in enumerate(context_nearest_first[:16]):
        folded = fold_text(text)
        explicit = any(token in folded for token in ("don vi", "dvt", "tinh bang"))
        confidence = "high" if explicit and distance <= 5 else "medium"
        if explicit:
            evidence.append(("preceding", text, confidence))

    for distance, text in enumerate(context_nearest_first[:16]):
        folded = fold_text(text)
        explicit = any(token in folded for token in ("don vi", "dvt", "tinh bang"))
        if not explicit:
            evidence.append(("preceding", text, "medium"))

    for source, text, confidence in evidence:
        unit = detect_unit(text)
        if unit:
            return unit, source, confidence
    return "", "unknown", "low"


def extract_html_tables(content: str) -> list[RawTable]:
    """Doc cac bang HTML bang BeautifulSoup, gom ca the HTML long nhau."""

    soup = BeautifulSoup(content, "html.parser")
    tables: list[RawTable] = []
    for source_table_index, table in enumerate(soup.find_all("table")):
        if table.find_parent("table") is not None:
            continue
        rows = expand_html_table(table)
        if not rows:
            continue
        context = _nearby_text_before(table)
        title = _title_from_context(context, rows)
        unit, unit_source, unit_confidence = _unit_from_html_evidence(
            table, title, rows, context
        )
        tables.append(
            RawTable(
                title=title,
                rows=rows,
                parser="html",
                unit=unit,
                unit_source=unit_source,
                unit_confidence=unit_confidence,
                continued=_has_nearby_continuation_marker(title, context),
                source_table_index=source_table_index,
                context_before=list(context),
            )
        )
    return tables


def _split_plain_line(line: str) -> list[str]:
    prepared = unicodedata.normalize("NFKC", html.unescape(line)).replace("\u00a0", " ")
    if "|" in prepared:
        parts = re.split(r"\s*\|\s*", prepared.strip().strip("|"))
    elif "\t" in prepared:
        parts = re.split(r"\t+", prepared.strip())
    else:
        parts = re.split(r"\s{2,}", prepared.strip())
    return [normalize_space(part) for part in parts]


def _plain_header_signal(cells: Sequence[str]) -> bool:
    folded = fold_text(" ".join(cells))
    return bool(re.search(r"\b(19|20)\d{2}\b", folded)) or any(
        keyword in folded for keyword in HEADER_KEYWORDS
    )


def _finalize_plain_block(
    block: list[list[str]], context: Sequence[str], output: list[RawTable]
) -> None:
    numeric_rows = sum(any(parse_number(cell) is not None for cell in row[1:]) for row in block)
    if numeric_rows < 2:
        return
    width = max(len(row) for row in block)
    rows = [row + [""] * (width - len(row)) for row in block]
    nearest_first = list(reversed([normalize_space(item) for item in context if normalize_space(item)]))
    title = _title_from_context(nearest_first, rows)
    unit = detect_unit(" ".join(list(context[-6:]) + [" ".join(" ".join(row) for row in rows[:4])]))
    output.append(
        RawTable(
            title=title,
            rows=rows,
            parser="plain_text",
            unit=unit,
            unit_source="header" if unit else "unknown",
            unit_confidence="medium" if unit else "low",
            continued=_has_nearby_continuation_marker(title, nearest_first),
            source_table_index=len(output),
            context_before=list(nearest_first),
        )
    )


def extract_plain_text_tables(content: str) -> list[RawTable]:
    """Nhan dien bang fixed-width, tab hoac pipe nam ngoai cac bang HTML."""

    without_tables = re.sub(r"(?is)<table\b[^>]*>.*?</table\s*>", "\n", content)
    without_tables = re.sub(r"(?i)<br\s*/?>", "\n", without_tables)
    without_tables = re.sub(r"(?i)</?(p|div|section|article|li|ul|ol|h[1-6])\b[^>]*>", "\n", without_tables)
    plain = html.unescape(re.sub(r"<[^>]+>", " ", without_tables))

    tables: list[RawTable] = []
    block: list[list[str]] = []
    context: list[str] = []

    for raw_line in plain.splitlines():
        line = raw_line.rstrip()
        cells = _split_plain_line(line) if line.strip() else []
        structured = len(cells) >= 2
        numeric_after_label = structured and any(parse_number(cell) is not None for cell in cells[1:])
        header_signal = structured and _plain_header_signal(cells)

        if structured and (numeric_after_label or header_signal):
            block.append(cells)
            continue

        single_text = normalize_space(line)
        continuation_line = (
            bool(block)
            and bool(single_text)
            and len(single_text) <= 120
            and not single_text.startswith("=====")
        )
        if continuation_line:
            block.append([single_text])
            continue

        if block:
            _finalize_plain_block(block, context, tables)
            block = []
        if single_text:
            context.append(single_text)
            context = context[-12:]

    if block:
        _finalize_plain_block(block, context, tables)
    return tables


def _candidate_shape(rows: Sequence[Sequence[str]]) -> tuple[int, int, int]:
    numeric_cells = 0
    numeric_rows = 0
    total_cells = 0
    for row in rows:
        row_numeric = 0
        for cell in row:
            if normalize_space(cell):
                total_cells += 1
            if parse_number(cell) is not None or len(_split_unique_thousands_concatenation(cell)) == 2:
                numeric_cells += 1
                row_numeric += 1
        if row_numeric:
            numeric_rows += 1
    return numeric_cells, numeric_rows, total_cells


def is_financial_candidate(candidate: RawTable) -> tuple[bool, str]:
    """Loai bang muc luc, nhan su va bang chu khong co so lieu tai chinh."""

    if len(candidate.rows) < 2:
        return False, "fewer_than_two_rows"
    numeric_cells, numeric_rows, total_cells = _candidate_shape(candidate.rows)
    if numeric_cells < 2 or numeric_rows < 2:
        return False, "insufficient_numeric_rows"

    first_rows = fold_text(" ".join(" ".join(row) for row in candidate.rows[:3]))
    all_text = fold_text(candidate.title + " " + " ".join(" ".join(row) for row in candidate.rows[:30]))
    header_cells = [fold_text(cell).strip(" .:-") for cell in candidate.rows[0]]
    if "trang" in header_cells and any(
        token in first_rows for token in ("noi dung", "muc luc", "bao cao")
    ):
        return False, "table_of_contents"

    known_statement = canonical_table_slug(candidate.title) in KNOWN_TABLE_SLUGS
    financial_hits = sum(keyword in all_text for keyword in FINANCIAL_KEYWORDS)
    administrative_hits = sum(keyword in all_text for keyword in ADMINISTRATIVE_KEYWORDS)
    if administrative_hits >= 2 and financial_hits < 2:
        return False, "administrative_or_staff_table"
    first_header = fold_text(" / ".join(candidate.rows[0]))
    categorical_header = any(
        token in first_header
        for token in ("nhom", "tinh hinh qua han", "phan loai", "xep hang")
    )
    small_code_rows = 0
    prose_rows = 0
    for row in candidate.rows[1:12]:
        parsed = parse_number(row[0]) if row else None
        if parsed is not None and float(parsed).is_integer() and 0 <= abs(float(parsed)) <= 20:
            small_code_rows += 1
        if sum(len(normalize_space(cell)) for cell in row[1:]) >= 40:
            prose_rows += 1
    if categorical_header and small_code_rows >= 2 and prose_rows >= 2:
        return False, "categorical_code_table"
    if not known_statement and financial_hits == 0:
        return False, "no_financial_context"

    density = numeric_cells / max(total_cells, 1)
    if density < 0.08 and not known_statement:
        return False, "numeric_density_too_low"
    return True, ""


def _column_values(rows: Sequence[Sequence[str]], column: int) -> list[int | float]:
    values: list[int | float] = []
    for row in rows:
        if column < len(row):
            parsed = parse_number(row[column])
            if parsed is not None:
                values.append(parsed)
    return values


def _column_numeric_evidence(rows: Sequence[Sequence[str]], column: int) -> int:
    """Dem ca so parse duoc va o OCR co mot cach tach grouping duy nhat."""

    count = 0
    for row in rows:
        if column >= len(row):
            continue
        if parse_number(row[column]) is not None:
            count += 1
        elif len(_split_unique_thousands_concatenation(row[column])) == 2:
            count += 1
    return count


def _header_text_for_column(rows: Sequence[Sequence[str]], column: int) -> str:
    parts: list[str] = []
    for row in rows[:6]:
        if column >= len(row):
            continue
        cell = normalize_space(row[column])
        if not cell:
            continue
        folded = fold_text(cell)
        if parse_number(cell) is None or re.search(r"\b(19|20)\d{2}\b", folded):
            parts.append(cell)
    return normalize_space(" ".join(dict.fromkeys(parts)))


def build_header_paths(rows: Sequence[Sequence[str]]) -> list[str]:
    """Dung header path theo cot sau khi rowspan/colspan da duoc mo rong."""

    width = max((len(row) for row in rows), default=0)
    header_rows = _header_band_rows(rows)
    paths: list[str] = []
    for column in range(width):
        parts: list[str] = []
        for row in header_rows:
            if column >= len(row):
                continue
            cell = normalize_space(row[column])
            if not cell:
                continue
            folded = fold_text(cell)
            numeric_like = bool(re.fullmatch(r"[\d.,()\-−–—\s]+", cell))
            looks_header = (
                parse_number(cell) is None
                or bool(re.search(r"\b(?:19|20)\d{2}\b", folded))
                or any(token in folded for token in HEADER_KEYWORDS)
            )
            if numeric_like and not re.search(r"\b(?:19|20)\d{2}\b", folded):
                break
            if not looks_header:
                break
            if not parts or fold_text(parts[-1]) != folded:
                parts.append(cell)
        paths.append(normalize_space(" / ".join(parts)))
    return paths


def _period_semantics(header: str, report_year: int) -> dict[str, bool]:
    folded = fold_text(header)
    leaf = fold_text(header.split(" / ")[-1])
    year = str(report_year)
    prior_year = str(report_year - 1)
    leaf_years = set(re.findall(r"\b(?:19|20)\d{2}\b", leaf))
    period_scope = leaf if leaf_years else folded
    exact_year = bool(re.search(rf"(?<!\d){year}(?!\d)", period_scope))
    exact_prior = bool(re.search(rf"(?<!\d){prior_year}(?!\d)", period_scope))
    role_scope = leaf if any(
        token in leaf
        for token in (
            "nam nay", "ky nay", "cuoi nam", "cuoi ky", "tai ngay", "hien tai",
            "nam truoc", "ky truoc", "cung ky", "dau nam", "dau ky",
        )
    ) else folded
    current = any(
        token in role_scope
        for token in ("nam nay", "ky nay", "cuoi nam", "cuoi ky", "tai ngay", "hien tai")
    )
    prior = any(
        token in role_scope
        for token in ("nam truoc", "ky truoc", "cung ky", "dau nam", "dau ky")
    )
    end_date = bool(
        re.search(rf"31[./-]12[./-]{year}", period_scope)
        or re.search(rf"(?:ngay\s+)?31\s+thang\s+12\s+nam\s+{year}", period_scope)
    )
    start_date = bool(
        re.search(rf"0?1[./-]0?1[./-]{year}", period_scope)
        or re.search(
            rf"(?:ngay\s+)?0?1\s+thang\s+0?1\s+nam\s+{year}", period_scope
        )
    )
    ratio = "%" in leaf or any(token in leaf for token in ("ty le", "phan tram"))
    metadata = any(keyword in folded for keyword in METADATA_COLUMN_KEYWORDS)
    return {
        "exact_year": exact_year,
        "exact_prior_year": exact_prior,
        "current": current,
        "prior": prior,
        "end_date": end_date,
        "start_date": start_date,
        "ratio": ratio,
        "metadata": metadata,
    }


def decide_value_column(rows: Sequence[Sequence[str]], report_year: int) -> ValueColumnDecision:
    """Cham diem cot gia tri va giai thich tung candidate trong manifest."""

    width = max((len(row) for row in rows), default=0)
    headers = build_header_paths(rows)
    candidates: list[dict[str, object]] = []
    for column in range(width):
        values = _column_values(rows, column)
        numeric_evidence = _column_numeric_evidence(rows, column)
        if numeric_evidence < 1:
            continue
        header = headers[column] if column < len(headers) else ""
        semantics = _period_semantics(header, report_year)
        folded_header = fold_text(header)
        categorical_header = bool(
            re.search(r"(?:^| / )(nhom|loai|hang|muc|cap|bac)(?:$| / )", folded_header)
        ) or any(
            token in folded_header
            for token in ("tinh hinh qua han", "phan loai", "xep hang")
        )
        text_count = sum(
            column < len(row)
            and bool(normalize_space(row[column]))
            and parse_number(row[column]) is None
            for row in rows
        )
        if column == 0 and text_count > len(values):
            continue

        score = 0.0
        signals: list[str] = []
        excluded = False
        if semantics["metadata"]:
            score -= 500.0
            signals.append("metadata_column")
            excluded = True
        if categorical_header:
            score -= 500.0
            signals.append("categorical_code_column")
            excluded = True
        if semantics["ratio"]:
            score -= 180.0
            signals.append("ratio_column")
        if semantics["exact_year"]:
            score += 180.0
            signals.append("exact_report_year")
        if semantics["end_date"]:
            score += 80.0
            signals.append("period_end_date")
        if semantics["current"]:
            score += 70.0
            signals.append("current_period_label")
        if semantics["start_date"]:
            score -= 100.0
            signals.append("period_start_date")
        if semantics["prior"] or semantics["exact_prior_year"]:
            score -= 120.0
            signals.append("prior_period_label")
        if any(
            token in folded_header
            for token in ("gia tri ghi so", "gia tri", "so du", "nguyen gia", "tong cong")
        ):
            score += 45.0
            signals.append("primary_amount_label")
        if any(token in folded_header for token in ("du phong", "chenh lech", "bien dong")):
            score -= 20.0
            signals.append("secondary_amount_label")

        # Coverage chi la tie-breaker: OCR thieu o o ky hien tai khong duoc phep
        # day score sang cot nam truoc day du hon.
        score += min(numeric_evidence, 20) * 0.5
        if not header:
            score -= 20.0
            signals.append("missing_header")
        candidates.append(
            {
                "index": column,
                "header": header,
                "score": round(score, 3),
                "numeric_count": len(values),
                "ocr_split_count": numeric_evidence - len(values),
                "excluded": excluded,
                "signals": signals,
            }
        )

    eligible = [item for item in candidates if not item["excluded"]]
    if not eligible:
        return ValueColumnDecision(
            None, "none", "", "", "low", candidates, ["no_reliable_value_column"]
        )
    eligible.sort(key=lambda item: (-float(item["score"]), int(item["index"])))
    best = eligible[0]
    runner_score = float(eligible[1]["score"]) if len(eligible) > 1 else -999.0
    margin = float(best["score"]) - runner_score
    signals = set(best["signals"])
    explicit_current = bool(
        signals.intersection({"exact_report_year", "period_end_date", "current_period_label"})
    )
    conflict = bool(
        signals.intersection(
            {"ratio_column", "prior_period_label", "period_start_date"}
        )
    )
    if explicit_current and not conflict and margin >= 25:
        confidence = "high"
        method = "scored_explicit_period"
    elif not conflict and (
        explicit_current or "primary_amount_label" in signals
    ):
        confidence = "medium"
        method = "scored_structural"
    else:
        confidence = "low"
        method = "scored_ambiguous"

    header = str(best["header"])
    value_period = header or f"heuristic_current_period_{report_year}"
    warnings: list[str] = []
    if confidence != "high":
        warnings.append(
            "value_column_{}:selected={};margin={:.2f}".format(
                confidence, best["index"], margin
            )
        )
    return ValueColumnDecision(
        int(best["index"]),
        method,
        header,
        value_period,
        confidence,
        candidates,
        warnings,
    )


def select_value_column(
    rows: Sequence[Sequence[str]], report_year: int
) -> tuple[Optional[int], str, list[str]]:
    """Chon duy nhat cot cua nam bao cao, loai ma so/thuyet minh/STT."""

    decision = decide_value_column(rows, report_year)
    return decision.column, decision.value_period, decision.warnings


def _label_index(row: Sequence[str], value_column: int) -> Optional[int]:
    choices: list[tuple[int, int]] = []
    for index, cell in enumerate(row):
        if index == value_column:
            continue
        text = normalize_space(cell)
        if not text or parse_number(text) is not None:
            continue
        folded = fold_text(text)
        if folded in MISSING_VALUE_TOKENS or not re.search(r"[a-zA-ZÀ-ỹĐđ]", text):
            continue
        if folded in {"a", "b", "c", "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x"}:
            continue
        if re.fullmatch(r"[a-zivx]+[.)-]?", folded):
            continue
        score = len(text) + (10 if index < value_column else 0)
        choices.append((score, index))
    if not choices:
        return None
    return max(choices)[1]


def _single_text_fragment(row: Sequence[str]) -> tuple[Optional[int], str]:
    nonempty = [(index, normalize_space(cell)) for index, cell in enumerate(row) if normalize_space(cell)]
    textual = [(index, cell) for index, cell in nonempty if parse_number(cell) is None]
    if len(nonempty) != 1 or len(textual) != 1:
        return None, ""
    return textual[0]


def _looks_like_continuation_fragment(text: str) -> bool:
    folded = fold_text(text)
    if len(text) < 3 or len(text) > 120 or text.endswith(":"):
        return False
    if text.isupper() or canonical_table_slug(text) in KNOWN_TABLE_SLUGS:
        return False
    if any(keyword in folded for keyword in HEADER_KEYWORDS):
        return False
    # Parent/section labels often introduce several bullet children. Only merge
    # a preceding single-cell line when morphology indicates dropped prose.
    if (
        text[:1].isupper()
        and not re.search(r"[,;(-]\s*$", text)
        and not fold_text(text).endswith(
            ("tu", "cua", "va", "voi", "cho", "trong", "tren", "duoi", "bang")
        )
    ):
        return False
    return True


def merge_wrapped_label_rows(
    source_rows: Sequence[Sequence[str]], value_column: int
) -> tuple[list[list[str]], list[str]]:
    """Gop dong rot chu chi khi dong don va dong ke co bang chung dinh luong."""

    rows = [list(row) for row in source_rows]
    warnings: list[str] = []
    index = 0
    while index < len(rows):
        current = rows[index]
        fragment_index, fragment = _single_text_fragment(current)
        if fragment_index is None or not _looks_like_continuation_fragment(fragment):
            index += 1
            continue

        previous_has_value = (
            index > 0
            and value_column < len(rows[index - 1])
            and parse_number(rows[index - 1][value_column]) is not None
        )
        next_has_value = (
            index + 1 < len(rows)
            and value_column < len(rows[index + 1])
            and parse_number(rows[index + 1][value_column]) is not None
        )

        if next_has_value:
            target = rows[index + 1]
            label_index = _label_index(target, value_column)
            if label_index is not None:
                target[label_index] = normalize_space(f"{fragment} {target[label_index]}")
                del rows[index]
                warnings.append("merged_wrapped_label_before_value_row")
                continue

        if previous_has_value and fragment[:1].islower():
            target = rows[index - 1]
            label_index = _label_index(target, value_column)
            if label_index is not None:
                target[label_index] = normalize_space(f"{target[label_index]} {fragment}")
                del rows[index]
                warnings.append("merged_wrapped_label_after_value_row")
                continue
        index += 1
    return rows, warnings


def _is_header_label(label: str) -> bool:
    folded = fold_text(label).strip(" .:-")
    if any(folded == keyword for keyword in HEADER_KEYWORDS):
        return True
    return bool(re.fullmatch(r"(tai ngay|nam|ky)\s*(19|20)?\d{0,4}", folded))


def _resolve_unit_for_column(
    candidate: RawTable, decision: ValueColumnDecision
) -> tuple[str, str, str]:
    selected_header = fold_text(decision.header)
    if (
        "so luong" in selected_header
        and "co phieu" in fold_text(candidate.title)
        and "lai tren co phieu" not in fold_text(candidate.title)
    ):
        return "Co phieu", "semantic_header", "high"
    if candidate.unit and candidate.unit_source in {"caption", "title"}:
        return candidate.unit, candidate.unit_source, candidate.unit_confidence
    header_unit = detect_unit(decision.header)
    if header_unit:
        return header_unit, "header", "high"
    if candidate.unit and candidate.unit_source == "header":
        return candidate.unit, "header", candidate.unit_confidence
    for distance, context in enumerate(candidate.context_before[:16]):
        unit = detect_unit(context)
        if not unit:
            continue
        explicit = any(
            token in fold_text(context) for token in ("don vi", "dvt", "tinh bang")
        )
        confidence = "high" if explicit and distance <= 5 else "medium"
        return unit, "preceding", confidence
    return "", "unknown", "low"


def _prefer_quantity_column(
    candidate: RawTable, decision: ValueColumnDecision
) -> ValueColumnDecision:
    """Prefer the current share-count axis when a share table also has VND."""

    title = fold_text(candidate.title)
    if "co phieu" not in title or "lai tren co phieu" in title:
        return decision
    eligible = []
    for item in decision.candidates:
        header = fold_text(item.get("header", ""))
        signals = set(item.get("signals", []))
        if "so luong" not in header:
            continue
        if signals.intersection({"prior_period_label", "period_start_date"}):
            continue
        if not signals.intersection(
            {"exact_report_year", "period_end_date", "current_period_label"}
        ):
            continue
        eligible.append(item)
    if not eligible:
        return decision
    selected = max(
        eligible, key=lambda item: (float(item["score"]), -int(item["index"]))
    )
    header = str(selected["header"])
    return ValueColumnDecision(
        int(selected["index"]),
        "semantic_share_quantity",
        header,
        header,
        "high",
        decision.candidates,
        list(dict.fromkeys((*decision.warnings, "selected_share_quantity_axis"))),
    )


def _split_unique_thousands_concatenation(raw_value: str) -> list[str]:
    """Tach duy nhat hai so grouping dau cham khi mot group bi dinh.

    Vi du ``89.962.600.00573.083.857.692`` chi co mot diem tach hop le:
    ``89.962.600.005`` va ``73.083.857.692``. Khong tach chuoi khong co
    bang chung grouping duy nhat.
    """

    compact = normalize_space(raw_value).replace(" ", "")
    sign = ""
    if compact.startswith("(") and compact.endswith(")"):
        sign, compact = "parenthesized", compact[1:-1]
    elif compact[:1] in "-−–—":
        sign, compact = "negative", compact[1:]
    if "," in compact or not re.fullmatch(r"\d+(?:\.\d+)+", compact):
        return []
    groups = compact.split(".")
    solutions: list[list[str]] = []
    for index, group in enumerate(groups):
        if not 4 <= len(group) <= 6:
            continue
        left_group, right_group = group[:3], group[3:]
        if not 1 <= len(right_group) <= 3:
            continue
        first_groups = groups[:index] + [left_group]
        second_groups = [right_group] + groups[index + 1 :]
        first_valid = (
            1 <= len(first_groups[0]) <= 3
            and all(len(item) == 3 for item in first_groups[1:])
        )
        second_valid = (
            1 <= len(second_groups[0]) <= 3
            and all(len(item) == 3 for item in second_groups[1:])
        )
        if first_valid and second_valid:
            values = [".".join(first_groups), ".".join(second_groups)]
            if sign == "parenthesized":
                values = [f"({value})" for value in values]
            elif sign == "negative":
                values = [f"-{value}" for value in values]
            solutions.append(values)
    return solutions[0] if len(solutions) == 1 else []


def _split_compound_indicator(label: str) -> list[str]:
    match = re.match(r"^(.*?)\s*-\s*(Trong\s+đó\s*:.*)$", label, flags=re.IGNORECASE)
    if not match:
        return []
    first = normalize_space(match.group(1))
    second = normalize_space(match.group(2))
    return [first, second] if first and second else []


def _is_suspicious_numeric_cell(raw_value: str) -> bool:
    raw = normalize_space(raw_value)
    if not raw or not re.fullmatch(r"[\d.,()\-−–—\s]+", raw):
        return False
    if parse_number(raw_value) is not None:
        return False
    digit_count = len(re.sub(r"\D", "", raw))
    numeric_tokens = re.findall(r"[-−–—]?\(?\d[\d.,]*\)?", raw)
    malformed_grouping = any(
        separator in raw
        and len(raw.split(separator)) > 2
        and not all(len(group.strip("()-−–—")) == 3 for group in raw.split(separator)[1:])
        for separator in (".", ",")
    )
    return (
        digit_count > 18
        or bool(re.search(r"\d\s+[-−–—(]?\s*\d", raw_value))
        or bool(re.search(r"\d[\d.,]*[-−–—]\s*$", raw))
        or len(numeric_tokens) > 1
        or malformed_grouping
    )


def _metadata_value(metadata: Optional[ReportMetadata], name: str, default: object) -> object:
    return getattr(metadata, name, default) if metadata is not None else default


def _rejected_cell_entry(
    candidate: RawTable,
    metadata: Optional[ReportMetadata],
    source_row: int,
    source_column: int,
    raw_cell: str,
    candidate_split: Sequence[str],
) -> dict[str, object]:
    return {
        "source_txt": _display_path(metadata.source_txt) if metadata else "",
        "ticker": _metadata_value(metadata, "ticker", ""),
        "report_year": _metadata_value(metadata, "report_year", 0),
        "table_title": candidate.title,
        "source_table_index": candidate.source_table_index,
        "source_row": source_row,
        "source_column": source_column,
        "raw_cell": raw_cell,
        "reason": "ambiguous_ocr_numeric_concatenation",
        "candidate_split": list(candidate_split),
        "confidence": "low",
    }


def _trace_suspicious_cells(
    candidate: RawTable,
    metadata: Optional[ReportMetadata],
    diagnostics: Optional[ExtractionDiagnostics],
) -> None:
    if diagnostics is None:
        return
    existing = {
        (
            int(entry.get("source_table_index", -1)),
            int(entry.get("source_row", -1)),
            int(entry.get("source_column", -1)),
        )
        for entry in diagnostics.rejected_cells
    }
    for row_index, row in enumerate(candidate.rows):
        for column_index, raw_cell in enumerate(row):
            key = (candidate.source_table_index, row_index, column_index)
            if key in existing or not _is_suspicious_numeric_cell(raw_cell):
                continue
            diagnostics.rejected_cells.append(
                _rejected_cell_entry(
                    candidate,
                    metadata,
                    row_index,
                    column_index,
                    normalize_space(raw_cell),
                    _split_unique_thousands_concatenation(raw_cell),
                )
            )


def convert_candidate(
    candidate: RawTable,
    report_year: int,
    *,
    diagnostics: Optional[ExtractionDiagnostics] = None,
    metadata: Optional[ReportMetadata] = None,
    decision: Optional[ValueColumnDecision] = None,
) -> tuple[Optional[ExtractedTable], str]:
    accepted, reason = is_financial_candidate(candidate)
    if not accepted:
        _trace_suspicious_cells(candidate, metadata, diagnostics)
        return None, reason

    column_decision = decision or _prefer_quantity_column(
        candidate, decide_value_column(candidate.rows, report_year)
    )
    value_column = column_decision.column
    if value_column is None:
        _trace_suspicious_cells(candidate, metadata, diagnostics)
        return None, "no_reliable_value_column"
    if column_decision.confidence == "low":
        if diagnostics is not None:
            diagnostics.quarantine.append(
                {
                    "source_txt": _display_path(metadata.source_txt) if metadata else "",
                    "ticker": _metadata_value(metadata, "ticker", ""),
                    "report_year": _metadata_value(metadata, "report_year", report_year),
                    "report_type": _metadata_value(metadata, "report_type", "unknown"),
                    "table_title": candidate.title,
                    "table_slug": canonical_table_slug(candidate.title),
                    "source_table_index": candidate.source_table_index,
                    "reason": "low_confidence_value_column",
                    "value_column_method": column_decision.method,
                    "value_column_header": column_decision.header,
                    "value_column_confidence": column_decision.confidence,
                    "candidate_columns": column_decision.candidates,
                    "warnings": column_decision.warnings,
                }
            )
        _trace_suspicious_cells(candidate, metadata, diagnostics)
        return None, "low_confidence_value_column"

    rows, merge_warnings = merge_wrapped_label_rows(candidate.rows, value_column)
    warnings = list(column_decision.warnings)
    warnings.extend(candidate.warnings)
    warnings.extend(merge_warnings)
    records: list[dict[str, object]] = []
    table_unit, unit_source, unit_confidence = _resolve_unit_for_column(
        candidate, column_decision
    )
    traced_cells: set[tuple[int, int]] = set()

    for row_index, row in enumerate(rows):
        if value_column >= len(row):
            continue
        raw_value = normalize_space(row[value_column])
        value = parse_number(raw_value)
        if value is None:
            split_values = _split_unique_thousands_concatenation(raw_value)
            label_index = _label_index(row, value_column)
            labels = (
                _split_compound_indicator(normalize_space(row[label_index]))
                if label_index is not None
                else []
            )
            parsed_splits = [parse_number(item) for item in split_values]
            if (
                len(labels) == 2
                and len(split_values) == 2
                and all(item is not None for item in parsed_splits)
            ):
                for split_label, split_raw, split_value in zip(
                    labels, split_values, parsed_splits
                ):
                    records.append(
                        {
                            "Chi_tieu": split_label,
                            "Gia_tri": split_value,
                            "Don_vi": infer_row_unit(split_label, split_raw, table_unit),
                        }
                    )
                warnings.append("recovered_unique_ocr_numeric_split")
                traced_cells.add((row_index, value_column))
            elif _is_suspicious_numeric_cell(raw_value):
                warnings.append("rejected_ambiguous_ocr_numeric_concatenation")
                traced_cells.add((row_index, value_column))
                if diagnostics is not None:
                    diagnostics.rejected_cells.append(
                        _rejected_cell_entry(
                            candidate,
                            metadata,
                            row_index,
                            value_column,
                            raw_value,
                            split_values,
                        )
                    )
            continue
        label_index = _label_index(row, value_column)
        if label_index is None:
            continue
        label = normalize_space(row[label_index])
        if not label or _is_header_label(label):
            continue
        records.append(
            {
                "Chi_tieu": label,
                "Gia_tri": value,
                "Don_vi": infer_row_unit(label, raw_value, table_unit),
            }
        )

    # Trace ca OCR cell o cot comparative/khac, du chung khong duoc chon de xuat.
    for row_index, row in enumerate(rows):
        for column_index, raw_cell in enumerate(row):
            if (row_index, column_index) in traced_cells:
                continue
            if not _is_suspicious_numeric_cell(raw_cell):
                continue
            split_values = _split_unique_thousands_concatenation(raw_cell)
            warnings.append("rejected_ambiguous_ocr_numeric_concatenation")
            if diagnostics is not None:
                diagnostics.rejected_cells.append(
                    _rejected_cell_entry(
                        candidate,
                        metadata,
                        row_index,
                        column_index,
                        normalize_space(raw_cell),
                        split_values,
                    )
                )

    if len(records) < 2:
        return None, "fewer_than_two_normalized_rows"

    units = {str(record["Don_vi"]) for record in records if record["Don_vi"]}
    manifest_unit = next(iter(units)) if len(units) == 1 else ("mixed" if units else "")
    if not table_unit:
        warnings.append("unit_not_reliably_detected")

    title = _base_table_title(candidate.title)
    return (
        ExtractedTable(
            table_title=title,
            table_slug=canonical_table_slug(title),
            unit=manifest_unit,
            value_period=column_decision.value_period,
            parser=candidate.parser,
            records=records,
            value_column_method=column_decision.method,
            value_column_header=column_decision.header,
            value_column_confidence=column_decision.confidence,
            candidate_columns=column_decision.candidates,
            unit_source=unit_source,
            unit_confidence=unit_confidence,
            default_unit=table_unit,
            report_year=report_year,
            source_table_indices=[candidate.source_table_index],
            header_signature="|".join(fold_text(item) for item in build_header_paths(candidate.rows)),
            continued=candidate.continued,
            warnings=list(dict.fromkeys(warnings)),
        ),
        "",
    )


def _header_roles(signature: str) -> set[str]:
    folded = fold_text(signature)
    roles: set[str] = set()
    for token in (
        "ma so",
        "thuyet minh",
        "nam nay",
        "nam truoc",
        "ky nay",
        "ky truoc",
        "tai ngay",
        "cuoi nam",
        "dau nam",
    ):
        if token in folded:
            roles.add(token)
    roles.update(re.findall(r"\b(?:19|20)\d{2}\b", folded))
    return roles


def _period_profile(value_period: str, report_year: int = 0) -> tuple[set[int], str]:
    folded = fold_text(value_period)
    years = {int(item) for item in re.findall(r"\b(?:19|20)\d{2}\b", folded)}
    is_start_date = bool(
        re.search(r"\b0?1[./-]0?1[./-](?:19|20)\d{2}\b", folded)
        or re.search(
            r"(?:ngay\s+)?0?1\s+thang\s+0?1\s+nam\s+(?:19|20)\d{2}",
            folded,
        )
    )
    is_end_date = bool(
        re.search(r"\b31[./-]12[./-](?:19|20)\d{2}\b", folded)
        or re.search(
            r"(?:ngay\s+)?31\s+thang\s+12\s+nam\s+(?:19|20)\d{2}",
            folded,
        )
    )
    if is_start_date or any(token in folded for token in ("dau nam", "dau ky")):
        role = "start"
    elif is_end_date or any(token in folded for token in ("cuoi nam", "cuoi ky", "tai ngay")):
        role = "end"
    elif any(token in folded for token in ("nam truoc", "ky truoc", "cung ky")):
        role = "prior"
    elif any(token in folded for token in ("nam nay", "ky nay")):
        role = "current"
    elif report_year and years == {report_year}:
        role = "current"
    elif report_year and years == {report_year - 1}:
        role = "prior"
    else:
        role = "dated" if years else "unknown"
    return years, role


def _compatible_periods(previous: str, current: str, report_year: int = 0) -> bool:
    previous_years, previous_role = _period_profile(previous, report_year)
    current_years, current_role = _period_profile(current, report_year)
    if previous_years and current_years and previous_years.isdisjoint(current_years):
        return False
    current_roles = {"current", "end"}
    prior_roles = {"prior", "start"}
    if (
        previous_role in current_roles and current_role in prior_roles
    ) or (
        previous_role in prior_roles and current_role in current_roles
    ):
        return False
    return bool(
        previous == current
        or previous_years.intersection(current_years)
        and previous_role == current_role
        or (
            previous_role in current_roles
            and current_role in current_roles
        )
        or (
            previous_role in prior_roles
            and current_role in prior_roles
        )
        or (previous_role == current_role and previous_role != "unknown")
    )


def _default_merge_unit(table: ExtractedTable) -> str:
    if table.default_unit:
        return table.default_unit
    if table.unit not in {"", "mixed", "%", "Co phieu", "VND/co phieu"}:
        return table.unit
    return ""


def _compatible_adjacent_parts(previous: ExtractedTable, current: ExtractedTable) -> bool:
    if previous.table_slug != current.table_slug:
        return False
    # Implicit structural merging is intentionally limited to core statements.
    # A non-core note may merge only when the source marks the next adjacent
    # part explicitly as a continuation and all remaining evidence agrees.
    if previous.table_slug not in KNOWN_TABLE_SLUGS and not current.continued:
        return False
    previous_index = previous.source_table_indices[-1] if previous.source_table_indices else -99
    current_index = current.source_table_indices[0] if current.source_table_indices else -1
    if current_index != previous_index + 1:
        return False
    previous_default_unit = _default_merge_unit(previous)
    current_default_unit = _default_merge_unit(current)
    if (
        previous_default_unit
        and current_default_unit
        and previous_default_unit != current_default_unit
    ):
        return False
    if previous.value_column_confidence == "low" or current.value_column_confidence == "low":
        return False
    previous_roles = _header_roles(previous.header_signature)
    current_roles = _header_roles(current.header_signature)
    if (
        not current.continued
        and previous_roles
        and current_roles
        and not previous_roles.intersection(current_roles)
    ):
        return False
    report_year = previous.report_year or current.report_year
    if previous.report_year and current.report_year and previous.report_year != current.report_year:
        return False
    period_compatible = _compatible_periods(
        previous.value_period, current.value_period, report_year
    )
    if not period_compatible:
        return False
    return current.continued or bool(previous_roles.intersection(current_roles))


def _boundary_duplicate_count(
    previous_records: Sequence[dict[str, object]], current_records: Sequence[dict[str, object]]
) -> int:
    max_window = min(5, len(previous_records), len(current_records))
    for size in range(max_window, 0, -1):
        left = [
            (str(row["Chi_tieu"]), row["Gia_tri"], str(row["Don_vi"]))
            for row in previous_records[-size:]
        ]
        right = [
            (str(row["Chi_tieu"]), row["Gia_tri"], str(row["Don_vi"]))
            for row in current_records[:size]
        ]
        if left == right:
            return size
    return 0


def merge_continuation_tables(tables: Sequence[ExtractedTable]) -> list[ExtractedTable]:
    """Chi gop cac part lien tiep co header/ky/don vi tuong thich."""

    merged: list[ExtractedTable] = []
    for table in tables:
        if merged and _compatible_adjacent_parts(merged[-1], table):
            target = merged[-1]
            duplicate_count = _boundary_duplicate_count(target.records, table.records)
            target.records.extend(table.records[duplicate_count:])
            target.parser = target.parser if target.parser == table.parser else "mixed"
            target.warnings.extend(table.warnings)
            target.warnings.append(
                "merged_continuation_table"
                if table.continued
                else "merged_structurally_contiguous_table"
            )
            if duplicate_count:
                target.warnings.append(f"removed_continuation_boundary_duplicates={duplicate_count}")
            target.warnings = list(dict.fromkeys(target.warnings))
            target.source_table_indices.extend(table.source_table_indices)
            if not target.default_unit and table.default_unit:
                target.default_unit = table.default_unit
            units = {str(row["Don_vi"]) for row in target.records if row["Don_vi"]}
            target.unit = next(iter(units)) if len(units) == 1 else ("mixed" if units else "")
            continue
        if table.continued:
            table.warnings.append("continuation_without_compatible_preceding_table")
        merged.append(table)
    return merged


def _inherit_continuation_units(tables: Sequence[ExtractedTable]) -> None:
    """Ke thua don vi chi giua hai part lien ke da co cung logical evidence."""

    for index in range(1, len(tables)):
        current = tables[index]
        previous = tables[index - 1]
        inherit_unit = _default_merge_unit(previous)
        if current.default_unit or not inherit_unit:
            continue
        if not _compatible_adjacent_parts(previous, current):
            continue
        current.unit = inherit_unit
        current.default_unit = inherit_unit
        current.unit_source = "continuation"
        current.unit_confidence = "medium"
        for record in current.records:
            if not record.get("Don_vi"):
                record["Don_vi"] = inherit_unit
        units = {str(row["Don_vi"]) for row in current.records if row.get("Don_vi")}
        current.unit = (
            next(iter(units)) if len(units) == 1 else ("mixed" if units else "")
        )
        current.warnings = [
            warning for warning in current.warnings if warning != "unit_not_reliably_detected"
        ]
        current.warnings.append("inherited_unit_from_verified_continuation")


def _logical_table_id(source_txt: str, table: ExtractedTable) -> str:
    first_index = table.source_table_indices[0] if table.source_table_indices else -1
    payload = "|".join(
        (source_txt.casefold(), str(first_index), table.table_slug, fold_text(table.value_period))
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def _extract_raw_candidates(content: str) -> list[RawTable]:
    """Combine parsers with one deterministic source-table index namespace."""

    candidates = extract_html_tables(content)
    plain_candidates = extract_plain_text_tables(content)
    for source_table_index, candidate in enumerate(candidates + plain_candidates):
        candidate.source_table_index = source_table_index
    candidates.extend(plain_candidates)
    return candidates


def extract_tables_from_text(
    content: str, report_year: int
) -> tuple[list[ExtractedTable], int, int]:
    """API thuan tien cho unit test va cho xu ly tung bao cao."""

    candidates = _extract_raw_candidates(content)
    extracted: list[ExtractedTable] = []
    skipped = 0
    for candidate in candidates:
        table, _ = convert_candidate(candidate, report_year)
        if table is None:
            skipped += 1
            continue
        extracted.append(table)
    _inherit_continuation_units(extracted)
    return merge_continuation_tables(extracted), len(candidates), skipped


def load_company_map(csv_path: Optional[Path]) -> dict[str, str]:
    """Doc code_stock.csv ma khong phu thuoc ten cot bi BOM."""

    if csv_path is None or not csv_path.exists():
        return {}
    mapping: dict[str, str] = {}
    with csv_path.open("r", encoding="utf-8-sig", errors="replace", newline="") as handle:
        reader = csv.reader(handle)
        rows = iter(reader)
        header = next(rows, [])
        if len(header) < 2:
            return mapping
        for row in rows:
            if len(row) < 2:
                continue
            ticker = normalize_space(row[0]).upper()
            company = normalize_space(row[1])
            if ticker and company:
                mapping[ticker] = company
    return mapping


def find_company_map(raw_dir: Path) -> Optional[Path]:
    for directory in (raw_dir, raw_dir.parent, raw_dir.parent.parent):
        candidate = directory / "code_stock.csv"
        if candidate.exists():
            return candidate
    return None


def extract_report_metadata(path: Path, company_map: dict[str, str]) -> ReportMetadata:
    """Uu tien ticker/nam trong cau truc financial_statements/TICKER/YEAR."""

    parts = list(path.parts)
    lower_parts = [part.casefold() for part in parts]
    ticker = ""
    year: Optional[int] = None
    if "financial_statements" in lower_parts:
        index = lower_parts.index("financial_statements")
        if index + 1 < len(parts):
            ticker = re.sub(r"[^A-Za-z0-9]", "", parts[index + 1]).upper()
        for part in parts[index + 2 :]:
            if re.fullmatch(r"(19|20)\d{2}", part):
                year = int(part)
                break

    filename = path.stem
    if not ticker:
        ticker_match = re.match(r"([A-Za-z0-9]{2,10})[_-]", filename)
        ticker = ticker_match.group(1).upper() if ticker_match else "UNKNOWN"
    if year is None:
        year_match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", str(path))
        year = int(year_match.group(1)) if year_match else 0

    folded_path = fold_text(str(path))
    report_type = "unknown"
    for candidate in REPORT_TYPES:
        if candidate in folded_path:
            report_type = candidate
            break

    return ReportMetadata(
        ticker=ticker,
        company_name=company_map.get(ticker, ""),
        report_year=year,
        report_type=report_type,
        source_txt=path,
    )


class OutputNameAllocator:
    """Cap ten file theo thu tu on dinh; lan chay lai ghi dung cung ten."""

    def __init__(self) -> None:
        self._counts: Counter[str] = Counter()

    def reserve(self, metadata: ReportMetadata, table_slug: str) -> str:
        base = f"{metadata.ticker}_{metadata.report_year}_{table_slug}_{metadata.report_type}"
        # Windows mac dinh khong phan biet hoa/thuong. Hai slug OCR nhu
        # ``LuUChuyen`` va ``LuuChuyen`` phai duoc xem la va cham cung ten.
        collision_key = base.casefold()
        self._counts[collision_key] += 1
        occurrence = self._counts[collision_key]
        suffix = "" if occurrence == 1 else f"_{occurrence:02d}"
        return f"{base}{suffix}.csv"


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def write_csv(path: Path, records: Sequence[dict[str, object]]) -> None:
    """Ghi atomically CSV dung chinh xac schema ma retriever/agent dang doc."""

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    temporary.replace(path)


def _load_manifest(path: Path) -> dict[str, dict[str, object]]:
    existing: dict[str, dict[str, object]] = {}
    if not path.exists():
        return existing
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            csv_path = str(record.get("csv_path", ""))
            if csv_path:
                existing[csv_path] = record
    return existing


def write_manifest(
    path: Path,
    entries: Sequence[dict[str, object]],
    preserve_existing: bool = False,
    replace_sources: Optional[set[str]] = None,
) -> None:
    records = _load_manifest(path) if preserve_existing else {}
    if replace_sources:
        records = {
            key: value
            for key, value in records.items()
            if str(value.get("source_txt", "")) not in replace_sources
        }
    for entry in entries:
        records[str(entry["csv_path"])] = entry
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for csv_path in sorted(records):
            handle.write(json.dumps(records[csv_path], ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _diagnostic_key(entry: dict[str, object]) -> str:
    if "raw_cell" in entry:
        return "|".join(
            str(entry.get(name, ""))
            for name in (
                "source_txt",
                "source_table_index",
                "source_row",
                "source_column",
                "raw_cell",
            )
        )
    return "|".join(
        str(entry.get(name, ""))
        for name in ("source_txt", "source_table_index", "reason")
    )


def write_jsonl(
    path: Path,
    entries: Sequence[dict[str, object]],
    *,
    preserve_existing: bool = False,
    replace_sources: Optional[set[str]] = None,
) -> None:
    records: dict[str, dict[str, object]] = {}
    if preserve_existing and path.exists():
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records[_diagnostic_key(entry)] = entry
    if replace_sources:
        records = {
            key: value
            for key, value in records.items()
            if str(value.get("source_txt", "")) not in replace_sources
        }
    for entry in entries:
        records[_diagnostic_key(entry)] = entry
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for key in sorted(records):
            handle.write(json.dumps(records[key], ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _clean_output(output_dir: Path) -> int:
    removed = 0
    for csv_path in output_dir.glob("*.csv"):
        csv_path.unlink()
        removed += 1
    for generated_log in ("_manifest.jsonl", "_rejected_cells.jsonl", "_quarantine.jsonl"):
        path = output_dir / generated_log
        if path.exists():
            path.unlink()
            removed += 1
    return removed


def _iter_txt_files(
    raw_dir: Path,
    company_map: dict[str, str],
    ticker: Optional[str],
    year: Optional[int],
    limit_files: Optional[int],
) -> Iterable[tuple[Path, ReportMetadata]]:
    emitted = 0
    ticker_filter = ticker.upper() if ticker else None
    for path in sorted(raw_dir.rglob("*.txt"), key=lambda item: item.as_posix().casefold()):
        metadata = extract_report_metadata(path, company_map)
        if ticker_filter and metadata.ticker != ticker_filter:
            continue
        if year is not None and metadata.report_year != year:
            continue
        yield path, metadata
        emitted += 1
        if limit_files is not None and emitted >= limit_files:
            return


def process_all_reports(
    raw_dir: str | Path,
    processed_dir: str | Path,
    *,
    limit_files: Optional[int] = None,
    ticker: Optional[str] = None,
    year: Optional[int] = None,
    clean: bool = False,
    verbose: bool = False,
) -> ExtractionStats:
    """Quet de quy toan bo TXT; mac dinh khong gioi han cong ty hay bao cao."""

    started = time.perf_counter()
    raw_path = Path(raw_dir)
    output_path = Path(processed_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw directory does not exist: {raw_path}")
    if limit_files is not None and limit_files <= 0:
        raise ValueError("limit_files must be a positive integer")

    output_path.mkdir(parents=True, exist_ok=True)
    if clean:
        removed = _clean_output(output_path)
        if verbose:
            print(f"[clean] removed {removed} generated files")

    company_map_path = find_company_map(raw_path)
    company_map = load_company_map(company_map_path)
    allocator = OutputNameAllocator()
    stats = ExtractionStats()
    manifest_entries: list[dict[str, object]] = []
    rejected_entries: list[dict[str, object]] = []
    quarantine_entries: list[dict[str, object]] = []
    processed_sources: set[str] = set()
    existing_manifest = _load_manifest(output_path / "_manifest.jsonl")

    for txt_path, metadata in _iter_txt_files(
        raw_path, company_map, ticker, year, limit_files
    ):
        stats.txt_scanned += 1
        if verbose:
            print(f"[scan] {_display_path(txt_path)}")
        try:
            content = txt_path.read_text(encoding="utf-8", errors="replace")
            candidates = _extract_raw_candidates(content)
            diagnostics = ExtractionDiagnostics()
            accepted_tables: list[ExtractedTable] = []
            skipped = 0
            for candidate in candidates:
                table, _ = convert_candidate(
                    candidate,
                    metadata.report_year,
                    diagnostics=diagnostics,
                    metadata=metadata,
                )
                if table is None:
                    skipped += 1
                else:
                    accepted_tables.append(table)
            _inherit_continuation_units(accepted_tables)
            tables = merge_continuation_tables(accepted_tables)
            detected = len(candidates)
            rejected_entries.extend(diagnostics.rejected_cells)
            quarantine_entries.extend(diagnostics.quarantine)
            stats.cells_rejected += len(diagnostics.rejected_cells)
            stats.tables_quarantined += len(diagnostics.quarantine)
            stats.tables_detected += detected
            stats.tables_skipped += skipped
            for table in tables:
                source_display = _display_path(txt_path)
                table.logical_table_id = _logical_table_id(source_display, table)
                filename = allocator.reserve(metadata, table.table_slug)
                csv_path = output_path / filename
                write_csv(csv_path, table.records)
                stats.csv_written += 1
                stats.warnings += len(table.warnings)
                manifest_entries.append(
                    {
                        "csv_path": _display_path(csv_path),
                        "source_txt": _display_path(txt_path),
                        "ticker": metadata.ticker,
                        "company_name": metadata.company_name,
                        "report_year": metadata.report_year,
                        "report_type": metadata.report_type,
                        "table_title": table.table_title,
                        "table_slug": table.table_slug,
                        "unit": table.unit,
                        "unit_source": table.unit_source,
                        "unit_confidence": table.unit_confidence,
                        "value_period": table.value_period,
                        "value_column_method": table.value_column_method,
                        "value_column_header": table.value_column_header,
                        "value_column_confidence": table.value_column_confidence,
                        "candidate_columns": table.candidate_columns,
                        "source_table_index": table.source_table_indices[0]
                        if table.source_table_indices
                        else -1,
                        "source_table_indices": table.source_table_indices,
                        "logical_table_id": table.logical_table_id,
                        "parser": table.parser,
                        "row_count": len(table.records),
                        "warnings": table.warnings,
                    }
                )
            processed_sources.add(_display_path(txt_path))
        except Exception as exc:
            stats.errors += 1
            if verbose:
                print(f"[error] {_display_path(txt_path)}: {type(exc).__name__}: {exc}")

    partial_run = limit_files is not None or ticker is not None or year is not None
    if partial_run and not clean:
        regenerated_paths = {str(entry["csv_path"]) for entry in manifest_entries}
        stale_paths = {
            csv_path
            for csv_path, entry in existing_manifest.items()
            if str(entry.get("source_txt", "")) in processed_sources
            and csv_path not in regenerated_paths
        }
        resolved_output = output_path.resolve()
        for stale_path in stale_paths:
            candidate = Path(stale_path)
            if not candidate.is_absolute():
                candidate = (Path.cwd() / candidate).resolve()
            else:
                candidate = candidate.resolve()
            if candidate.parent == resolved_output and candidate.suffix.casefold() == ".csv":
                candidate.unlink(missing_ok=True)
    write_manifest(
        output_path / "_manifest.jsonl",
        manifest_entries,
        preserve_existing=partial_run and not clean,
        replace_sources=processed_sources if partial_run and not clean else None,
    )
    # Diagnostic logs belong to the current extraction scope. A partial run does
    # not silently append stale incidents from an older algorithm version.
    write_jsonl(
        output_path / "_rejected_cells.jsonl",
        rejected_entries,
        preserve_existing=partial_run and not clean,
        replace_sources=processed_sources if partial_run and not clean else None,
    )
    write_jsonl(
        output_path / "_quarantine.jsonl",
        quarantine_entries,
        preserve_existing=partial_run and not clean,
        replace_sources=processed_sources if partial_run and not clean else None,
    )
    stats.elapsed_seconds = time.perf_counter() - started
    return stats


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract deterministic three-column financial CSVs from ViFinQA OCR TXT files."
    )
    parser.add_argument(
        "--raw-dir",
        default="data/raw_vifinqa/financial_statements",
        help="Directory recursively scanned for *.txt files.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed_csv",
        help="Directory receiving CSV files and _manifest.jsonl.",
    )
    parser.add_argument("--limit-files", type=int, help="Optional smoke-test limit; no default limit.")
    parser.add_argument("--ticker", help="Only process one ticker (case-insensitive).")
    parser.add_argument("--year", type=int, help="Only process one report year.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Explicitly remove generated CSVs and manifest from output-dir before extraction.",
    )
    parser.add_argument("--verbose", action="store_true", help="Print each source path and error.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_argument_parser().parse_args(argv)
    stats = process_all_reports(
        args.raw_dir,
        args.output_dir,
        limit_files=args.limit_files,
        ticker=args.ticker,
        year=args.year,
        clean=args.clean,
        verbose=args.verbose,
    )
    labels = (
        ("TXT scanned", stats.txt_scanned),
        ("Tables detected", stats.tables_detected),
        ("CSV written", stats.csv_written),
        ("Tables skipped", stats.tables_skipped),
        ("Errors", stats.errors),
        ("Warnings", stats.warnings),
        ("Tables quarantined", stats.tables_quarantined),
        ("Cells rejected", stats.cells_rejected),
        ("Elapsed seconds", f"{stats.elapsed_seconds:.3f}"),
    )
    print("Extraction summary")
    for label, value in labels:
        print(f"- {label}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
