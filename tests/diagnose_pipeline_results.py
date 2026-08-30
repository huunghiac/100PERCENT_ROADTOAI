#!/usr/bin/env python3
"""Audit a ViFinQA submission against its final evidence.

The diagnostic intentionally treats the submission and log as untrusted data:
``pandas_query`` is parsed as a restricted expression before evaluation, and log
contents are only inspected as text.  The report includes both a strict
answer/query comparison and the historical 1% comparison so that old baselines
remain reproducible without weakening the acceptance criterion.

Example:

    python tests/diagnose_pipeline_results.py submission.json \
        --log pipeline.log \
        --repo-root . \
        --output data/quality_reports/submission_baseline.json
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib
import inspect
import json
import math
import os
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd


DEFAULT_STRICT_TOLERANCE = 1e-9
DEFAULT_LEGACY_TOLERANCE = 1e-2
DEFAULT_PERCENTAGE_MAX_ABS = 100.0

LOG_SEGMENT_RE = re.compile(
    r"^--- \[(\d+)/(\d+)\] ID=(\d+): (.*?) ---\s*$", re.MULTILINE
)
EVIDENCE_NAME_RE = re.compile(r"^([A-Za-z0-9]+)_(20\d{2})_")
YEAR_RE = re.compile(r"\b(20\d{2})\b")
YEAR_RANGE_RE = re.compile(r"\b(20\d{2})\s*[-\u2013\u2014]\s*(20\d{2})\b")


def _ids(items: Iterable[Mapping[str, Any]]) -> list[int]:
    return sorted(int(item["id"]) for item in items)


def _rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def _normalise_text(text: str) -> str:
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^A-Za-z0-9]+", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _jsonable(val) for key, val in asdict(value).items()}
    if isinstance(value, Mapping):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(val) for val in value]
    if hasattr(value, "value") and not isinstance(value, (str, bytes)):
        try:
            return _jsonable(value.value)
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _field(obj: Any, *names: str) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return None


class EvidenceResolver:
    """Resolve flat submission paths into the repository's nested CSV tree."""

    def __init__(
        self,
        repo_root: Path,
        processed_csv: Path,
        submission_dir: Path,
        evidence_roots: Sequence[Path] = (),
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.processed_csv = processed_csv.resolve()
        self.submission_dir = submission_dir.resolve()
        self.evidence_roots = [root.resolve() for root in evidence_roots]
        self._basename_index: dict[str, list[Path]] | None = None

    def _build_index(self) -> None:
        index: dict[str, list[Path]] = {}
        if self.processed_csv.exists():
            for path in self.processed_csv.rglob("*.csv"):
                index.setdefault(path.name, []).append(path.resolve())
        self._basename_index = index

    def resolve(self, csv_path: str) -> Path:
        normalised = csv_path.replace("\\", "/")
        relative = Path(*[part for part in normalised.split("/") if part])
        basename = relative.name
        candidates = [
            self.repo_root / relative,
            self.submission_dir / relative,
            self.processed_csv / relative,
            self.processed_csv / basename,
        ]
        candidates.extend(root / relative for root in self.evidence_roots)
        candidates.extend(root / basename for root in self.evidence_roots)
        for candidate in candidates:
            if candidate.is_file():
                return candidate.resolve()

        if self._basename_index is None:
            self._build_index()
        matches = self._basename_index.get(basename, []) if self._basename_index else []
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(
                f"Cannot resolve evidence {csv_path!r} under {self.processed_csv}"
            )
        raise FileNotFoundError(
            f"Ambiguous evidence basename {basename!r}: "
            + ", ".join(str(path) for path in matches[:5])
        )


ALLOWED_FUNCTIONS: dict[str, Any] = {
    "abs": abs,
    "float": float,
    "int": int,
    "len": len,
    "max": max,
    "min": min,
    "round": round,
    "sum": sum,
    # ComplexSolver emits compact ``pd.Series({...})`` expressions.  Keep
    # submission evaluation restricted by exposing only that constructor,
    # rather than the complete pandas module and its file-I/O entry points.
    "pd": SimpleNamespace(Series=pd.Series),
}

ALLOWED_AST_NODES = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.Call,
    ast.Attribute,
    ast.Subscript,
    ast.keyword,
    ast.BinOp,
    ast.UnaryOp,
    ast.BoolOp,
    ast.Compare,
    ast.IfExp,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Slice,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.USub,
    ast.UAdd,
    ast.Not,
    ast.Invert,
    ast.And,
    ast.Or,
    ast.BitAnd,
    ast.BitOr,
    ast.Eq,
    ast.NotEq,
    ast.Lt,
    ast.LtE,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.NotIn,
)

DANGEROUS_ATTRIBUTES = {
    "compile",
    "eval",
    "exec",
    "open",
    "popen",
    "read_csv",
    "read_excel",
    "read_json",
    "read_pickle",
    "system",
    "to_clipboard",
    "to_csv",
    "to_excel",
    "to_feather",
    "to_hdf",
    "to_json",
    "to_parquet",
    "to_pickle",
    "to_sql",
}


def _validate_expression(query: str, evidence_variables: set[str]) -> ast.Expression:
    tree = ast.parse(query, mode="eval")
    allowed_names = evidence_variables | set(ALLOWED_FUNCTIONS)
    for node in ast.walk(tree):
        if not isinstance(node, ALLOWED_AST_NODES):
            raise ValueError(f"Disallowed AST node: {type(node).__name__}")
        if isinstance(node, ast.Name) and node.id not in allowed_names:
            raise ValueError(f"Unknown or disallowed name: {node.id}")
        if isinstance(node, ast.Attribute):
            attribute = node.attr.lower()
            if attribute.startswith("_") or attribute in DANGEROUS_ATTRIBUTES:
                raise ValueError(f"Disallowed attribute: {node.attr}")
        if isinstance(node, ast.Call):
            for keyword in node.keywords:
                if keyword.arg == "inplace":
                    if not isinstance(keyword.value, ast.Constant) or keyword.value.value:
                        raise ValueError("inplace mutation is not allowed")
    return tree


@dataclass
class QueryEvaluation:
    executed: bool
    result: float | None
    error_type: str | None
    error: str | None
    referenced_variables: list[str]
    evidence_variables: list[str]
    missing_variables: list[str]
    unused_variables: list[str]
    resolved_evidence: list[dict[str, str]]


def evaluate_query(item: Mapping[str, Any], resolver: EvidenceResolver) -> QueryEvaluation:
    evidence = item.get("evidence") or []
    evidence_variables = [str(entry.get("variable", "")) for entry in evidence]
    resolved: list[dict[str, str]] = []
    frames: dict[str, pd.DataFrame] = {}
    referenced: set[str] = set()
    try:
        if len(evidence_variables) != len(set(evidence_variables)):
            raise ValueError("Duplicate evidence variable")
        if any(not re.fullmatch(r"[A-Za-z_]\w*", variable) for variable in evidence_variables):
            raise ValueError("Invalid evidence variable")

        query = str(item.get("pandas_query", ""))
        tree = _validate_expression(query, set(evidence_variables))
        referenced = {
            node.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Name) and node.id in set(evidence_variables)
        }
        for entry in evidence:
            variable = str(entry["variable"])
            source = resolver.resolve(str(entry["csv_path"]))
            frames[variable] = pd.read_csv(source)
            resolved.append(
                {
                    "variable": variable,
                    "csv_path": str(entry["csv_path"]),
                    "resolved_path": str(source),
                }
            )

        globals_scope = {"__builtins__": {}, **ALLOWED_FUNCTIONS}
        value = eval(compile(tree, "<pandas_query>", "eval"), globals_scope, frames)
        if hasattr(value, "item"):
            value = value.item()
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Query result is not a numeric scalar: {type(value).__name__}")
        result = float(value)
        if not math.isfinite(result):
            raise ValueError(f"Query result is not finite: {result!r}")
        return QueryEvaluation(
            executed=True,
            result=result,
            error_type=None,
            error=None,
            referenced_variables=sorted(referenced),
            evidence_variables=evidence_variables,
            missing_variables=sorted(referenced - set(evidence_variables)),
            unused_variables=sorted(set(evidence_variables) - referenced),
            resolved_evidence=resolved,
        )
    except Exception as exc:
        return QueryEvaluation(
            executed=False,
            result=None,
            error_type=type(exc).__name__,
            error=str(exc),
            referenced_variables=sorted(referenced),
            evidence_variables=evidence_variables,
            missing_variables=sorted(referenced - set(evidence_variables)),
            unused_variables=sorted(set(evidence_variables) - referenced),
            resolved_evidence=resolved,
        )


class HeuristicQuestionAnalyzer:
    """Entity/type fallback used when a repository planner is unavailable."""

    _LEGAL_PATTERNS = (
        r"^(?:ctcp|cong ty co phan|cong ty tnhh|cong ty|tong cong ty co phan|"
        r"tong cong ty|ngan hang tmcp|tap doan)\s+",
        r"\s+(?:ctcp|cong ty co phan)$",
    )
    _MULTI_CONTEXT_RE = re.compile(
        r"\b(?:nhom|cac doanh nghiep|cac cong ty|trong so|phan nhom|tap hop|"
        r"bao gom|gom cac|[2-9]\s+doanh nghiep|nganh)\b"
    )
    _COMPLEX_TERMS = (
        "trung vi",
        "cao nhat",
        "thap nhat",
        "binh quan",
        "cagr",
        "tang truong",
        "muc thay doi",
        "chenh lech",
        "dong thoi",
        "gia su",
        "kich ban",
        "neu ",
        "trong giai doan",
        "nam sau",
        "nam ke tiep",
        "vong quay",
        "he so",
        "ty so",
        "bien loi nhuan",
        "roe",
        "roa",
        "cfo",
        "so ngay ton kho",
        "toc do tang",
    )

    _COMMON_ALIASES = {
        "bao viet": "BVH",
        "binh son": "BSR",
        "dabaco": "DBC",
        "dai duong": "OGC",
        "dam ca mau": "DCM",
        "dam phu my": "DPM",
        "dau khi ca mau": "DCM",
        "dau tu hai phat": "HPX",
        "dien luc gelex": "GEE",
        "do thi kinh bac": "KBC",
        "duong quang ngai": "QNS",
        "hai phat dau tu": "HPX",
        "hoa phat": "HPG",
        "hoa sen": "HSG",
        "masan": "MSN",
        "masan high tech materials": "MSR",
        "masan meatlife": "MML",
        "minh phu": "MPC",
        "nam kim": "NKG",
        "pvtrans": "PVT",
        "sao mai": "ASM",
        "sunshine homes": "SSH",
        "tap doan fpt": "FPT",
        "tap doan gelex": "GEX",
        "tap doan xang dau": "PLX",
        "the gioi di dong": "MWG",
        "thuy san minh phu": "MPC",
        "van phu invest": "VPI",
        "vicem ha tien": "HT1",
        "viglacera": "VGC",
        "vinamilk": "VNM",
        "vincom retail": "VRE",
        "vingroup": "VIC",
    }

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.tickers: set[str] = set()
        aliases: set[tuple[str, str]] = set()
        code_stock = repo_root / "data" / "raw_vifinqa" / "code_stock.csv"
        if code_stock.is_file():
            with code_stock.open("r", encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    ticker = str(row.get("Mã CK", "")).strip().upper()
                    company = _normalise_text(str(row.get("Tên công ty", "")))
                    if not ticker:
                        continue
                    self.tickers.add(ticker)
                    candidates = {company}
                    changed = True
                    while changed:
                        changed = False
                        for candidate in list(candidates):
                            for pattern in self._LEGAL_PATTERNS:
                                shorter = re.sub(pattern, "", candidate).strip()
                                if shorter and shorter not in candidates:
                                    candidates.add(shorter)
                                    changed = True
                    aliases.update((candidate, ticker) for candidate in candidates if len(candidate) >= 4)
        processed = repo_root / "data" / "processed_csv"
        if processed.is_dir():
            self.tickers.update(
                path.name.upper()
                for path in processed.iterdir()
                if path.is_dir() and re.fullmatch(r"[A-Za-z0-9]{2,5}", path.name)
            )
        aliases.update(
            (_normalise_text(alias), ticker)
            for alias, ticker in self._COMMON_ALIASES.items()
        )
        self.aliases = sorted(aliases, key=lambda item: len(item[0]), reverse=True)

    @staticmethod
    def years(question: str) -> list[str]:
        years: list[str] = []
        for start_text, end_text in YEAR_RANGE_RE.findall(question):
            start, end = int(start_text), int(end_text)
            if start <= end and end - start <= 10:
                years.extend(str(year) for year in range(start, end + 1))
        years.extend(YEAR_RE.findall(question))
        return list(dict.fromkeys(years))

    def _alias_matches(self, normalised_question: str) -> list[tuple[int, int, str, str]]:
        matches: list[tuple[int, int, str, str]] = []
        padded = f" {normalised_question} "
        for alias, ticker in self.aliases:
            pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
            matches.extend(
                (match.start(), match.end(), alias, ticker)
                for match in re.finditer(pattern, padded)
            )
        chosen: list[tuple[int, int, str, str]] = []
        for match in sorted(matches, key=lambda value: (-(value[1] - value[0]), value[0])):
            overlaps_other = any(
                max(match[0], other[0]) < min(match[1], other[1])
                and match[3] != other[3]
                for other in chosen
            )
            if not overlaps_other:
                chosen.append(match)
        return sorted(chosen, key=lambda value: value[0])

    def tickers_for(self, question: str) -> list[str]:
        normalised = _normalise_text(question)
        alias_matches = self._alias_matches(normalised)
        multi_context = bool(self._MULTI_CONTEXT_RE.search(normalised))

        parenthesised = [
            ticker.upper()
            for ticker in re.findall(r"\(([A-Za-z][A-Za-z0-9]{1,4})\)", question)
            if ticker.upper() in self.tickers
        ]
        token_matches = list(re.finditer(r"(?<![a-z0-9])([a-z][a-z0-9]{1,4})(?![a-z0-9])", normalised))
        bare: list[str] = []
        for match in token_matches:
            ticker = match.group(1).upper()
            if ticker not in self.tickers:
                continue
            # A ticker-like brand inside a longer company alias is not a second entity.
            inside_other_alias = any(
                alias_start <= match.start() + 1
                and match.end() + 1 <= alias_end
                and alias_ticker != ticker
                for alias_start, alias_end, _, alias_ticker in alias_matches
            )
            if not inside_other_alias:
                bare.append(ticker)

        alias_tickers = [match[3] for match in alias_matches]
        explicit = list(dict.fromkeys(parenthesised + bare))
        if not multi_context and explicit:
            # Simple related-party lookups often name a counterparty as well as the
            # reporting entity.  An explicit ticker identifies the latter.
            return explicit
        return list(dict.fromkeys(explicit + alias_tickers))

    def analyze(self, question: str) -> dict[str, Any]:
        normalised = _normalise_text(question)
        tickers = self.tickers_for(question)
        years = self.years(question)
        multi_context = bool(self._MULTI_CONTEXT_RE.search(normalised))
        complex_question = (
            multi_context
            or len(years) > 1
            or any(term in normalised for term in self._COMPLEX_TERMS)
        )
        return {
            "question_type": "COMPLEX_ANALYTICAL" if complex_question else "SIMPLE_LOOKUP",
            "is_complex": complex_question,
            "tickers": tickers,
            "years": years,
            "source": "heuristic",
        }


class QuestionAnalyzerAdapter:
    """Use a repository planner when available, with a deterministic fallback."""

    def __init__(self, repo_root: Path) -> None:
        self.repo_root = repo_root
        self.heuristic = HeuristicQuestionAnalyzer(repo_root)
        self.planner: Any = None
        self.planner_name: str | None = None
        self.planner_error: str | None = None
        self._load_planner()

    def _load_planner(self) -> None:
        src_dir = self.repo_root / "src"
        for entry in (str(self.repo_root), str(src_dir)):
            if entry not in sys.path:
                sys.path.insert(0, entry)

        modules = ["question_planner", "planner", "question_analyzer"]
        if src_dir.is_dir():
            modules.extend(
                path.stem
                for path in src_dir.glob("*.py")
                if "planner" in path.stem or "question_analy" in path.stem
            )
        errors: list[str] = []
        for module_name in dict.fromkeys(modules):
            try:
                module = importlib.import_module(module_name)
            except Exception as exc:
                errors.append(f"{module_name}: {type(exc).__name__}: {exc}")
                continue
            classes = [
                value
                for name, value in vars(module).items()
                if inspect.isclass(value)
                and ("planner" in name.lower() or "questionanalyzer" in name.lower())
                and value.__module__ == module.__name__
            ]
            for planner_class in classes:
                try:
                    planner = planner_class()
                except Exception as exc:
                    errors.append(
                        f"{module_name}.{planner_class.__name__}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    continue
                if any(callable(getattr(planner, method, None)) for method in ("analyze", "plan", "parse")):
                    self.planner = planner
                    self.planner_name = f"{module_name}.{planner_class.__name__}"
                    return
        if errors:
            self.planner_error = "; ".join(errors[-5:])

    def analyze(self, question: str) -> dict[str, Any]:
        fallback = self.heuristic.analyze(question)
        if self.planner is None:
            return fallback
        try:
            plan = None
            for method_name in ("analyze", "plan", "parse"):
                method = getattr(self.planner, method_name, None)
                if callable(method):
                    plan = method(question)
                    break
            if plan is None:
                return fallback
            question_type = _field(plan, "question_type", "type", "kind")
            if hasattr(question_type, "value"):
                question_type = question_type.value
            if hasattr(question_type, "name"):
                question_type = question_type.name
            question_type_text = str(question_type or fallback["question_type"]).upper()
            tickers = _field(plan, "tickers", "entities", "companies") or fallback["tickers"]
            years = _field(plan, "years", "periods") or fallback["years"]
            is_complex = _field(plan, "is_complex", "complex")
            if is_complex is None:
                is_complex = question_type_text not in {"SIMPLE", "SIMPLE_LOOKUP", "LOOKUP"}
            return {
                "question_type": question_type_text,
                "is_complex": bool(is_complex),
                "tickers": [str(value).upper() for value in tickers],
                "years": [str(value) for value in years],
                "source": self.planner_name,
                "raw_plan": _jsonable(plan),
            }
        except Exception as exc:
            fallback["planner_failure"] = f"{type(exc).__name__}: {exc}"
            return fallback


def _parse_evidence_entities(evidence: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[str]]:
    tickers: list[str] = []
    years: list[str] = []
    for entry in evidence:
        basename = Path(str(entry.get("csv_path", "")).replace("\\", "/")).name
        match = EVIDENCE_NAME_RE.match(basename)
        if not match:
            continue
        tickers.append(match.group(1).upper())
        years.append(match.group(2))
    return list(dict.fromkeys(tickers)), list(dict.fromkeys(years))


def _target_unit(question: str) -> str | None:
    matches = list(re.finditer(r"bao nhiêu\s+([^?.!,;]+)", question, re.IGNORECASE))
    tail = _normalise_text(matches[-1].group(1)) if matches else ""
    if "diem phan tram" in tail:
        return "percentage_point"
    if "phan tram" in tail:
        return "percent"
    if "lan" in tail or "vong" in tail:
        return "times"
    if "tram ty dong" in tail:
        return "hundred_billion_vnd"
    if "nghin ty dong" in tail:
        return "trillion_vnd"
    if "ty dong" in tail:
        return "billion_vnd"
    if "trieu dong" in tail:
        return "million_vnd"
    if "nghin dong" in tail:
        return "thousand_vnd"
    if "trieu usd" in tail:
        return "million_usd"
    return None


def parse_pipeline_log(path: Path, submission_ids: set[int]) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = list(LOG_SEGMENT_RE.finditer(text))
    segments: dict[int, str] = {}
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        segments[int(match.group(3))] = text[match.end() : end]

    relevant = {item_id: segment for item_id, segment in segments.items() if item_id in submission_ids}
    fallback_ids: list[int] = []
    fallback_details: list[dict[str, Any]] = []
    generation_failure_ids: list[int] = []
    retry_failure_ids: list[int] = []
    ticker_none_ids: list[int] = []
    csv_zero_ids: list[int] = []
    answer_zero_line_ids: list[int] = []
    prompt_echo_ids: list[int] = []
    suspected_truncation_ids: list[int] = []
    retrieved_csv_counts: dict[int, int] = {}

    for item_id, segment in relevant.items():
        fallback_match = re.search(
            r"^\s*Fallback:\s*([^\s]+).*?row=(\d+), file=([^\)\r\n]+)",
            segment,
            re.MULTILINE,
        )
        if fallback_match:
            fallback_ids.append(item_id)
            fallback_details.append(
                {
                    "id": item_id,
                    "answer": fallback_match.group(1),
                    "row": int(fallback_match.group(2)),
                    "file": fallback_match.group(3).strip(),
                }
            )
        if "Error: Failed after" in segment:
            generation_failure_ids.append(item_id)
        if "[Agent Self-Correction]" in segment:
            retry_failure_ids.append(item_id)
        if re.search(r"^\s*Ticker=None\b", segment, re.MULTILINE):
            ticker_none_ids.append(item_id)
        if re.search(r"\bCSVs=0\b", segment):
            csv_zero_ids.append(item_id)
        if re.search(r"^\s*Answer: 0(?:\.0+)?\s*$", segment, re.MULTILINE):
            answer_zero_line_ids.append(item_id)
        csv_count = re.search(r"\bCSVs=(\d+)", segment)
        if csv_count:
            retrieved_csv_counts[item_id] = int(csv_count.group(1))

        raw_blocks = re.findall(
            r"\[Agent RAW\] (.*?)(?=\n\[Agent\] Extracted code:)", segment, re.DOTALL
        )
        strong_echo = any(
            re.search(r"(?:Ví dụ|## CÂU HỎI|\*\*Câu hỏi:\*\*)", block, re.IGNORECASE)
            for block in raw_blocks
        )
        suspicious_clip = any(
            len(block) == 500 and not block.lstrip().startswith("```") for block in raw_blocks
        )
        if strong_echo:
            prompt_echo_ids.append(item_id)
        if strong_echo or suspicious_clip:
            suspected_truncation_ids.append(item_id)

    return {
        "available": True,
        "path": str(path.resolve()),
        "segments_total": len(segments),
        "segments_matching_submission": len(relevant),
        "fallback_ids": sorted(fallback_ids),
        "fallback_details": sorted(fallback_details, key=lambda value: value["id"]),
        "generation_failure_ids": sorted(generation_failure_ids),
        "retry_failure_ids": sorted(retry_failure_ids),
        "ticker_none_ids": sorted(ticker_none_ids),
        "csv_zero_ids": sorted(csv_zero_ids),
        "answer_zero_line_ids": sorted(answer_zero_line_ids),
        "prompt_echo_ids": sorted(prompt_echo_ids),
        "suspected_prompt_truncation_ids": sorted(suspected_truncation_ids),
        "retrieved_csv_counts": {str(key): value for key, value in sorted(retrieved_csv_counts.items())},
    }


def _category(ids: Sequence[int], total: int) -> dict[str, Any]:
    unique = sorted(set(int(item_id) for item_id in ids))
    return {"count": len(unique), "rate": _rate(len(unique), total), "ids": unique}


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    submission_path = args.submission.resolve()
    repo_root = args.repo_root.resolve()
    processed_csv = args.processed_csv
    if not processed_csv.is_absolute():
        processed_csv = repo_root / processed_csv
    evidence_roots = [path if path.is_absolute() else repo_root / path for path in args.evidence_root]

    with submission_path.open("r", encoding="utf-8") as handle:
        submission = json.load(handle)
    if not isinstance(submission, list):
        raise ValueError("Submission root must be a JSON array")
    submission_ids = {int(item["id"]) for item in submission}
    if len(submission_ids) != len(submission):
        raise ValueError("Submission IDs must be unique")

    resolver = EvidenceResolver(
        repo_root=repo_root,
        processed_csv=processed_csv,
        submission_dir=submission_path.parent,
        evidence_roots=evidence_roots,
    )
    analyzer = QuestionAnalyzerAdapter(repo_root)

    per_question: list[dict[str, Any]] = []
    for item in submission:
        item_id = int(item["id"])
        question = str(item.get("question", ""))
        answer = float(item.get("answer", 0.0))
        evidence = item.get("evidence") or []
        analysis = analyzer.analyze(question)
        evaluation = evaluate_query(item, resolver)
        evidence_tickers, evidence_years = _parse_evidence_entities(evidence)
        requested_tickers = list(dict.fromkeys(str(value).upper() for value in analysis["tickers"]))
        requested_years = list(dict.fromkeys(str(value) for value in analysis["years"]))

        strict_match = bool(
            evaluation.executed
            and math.isclose(
                float(evaluation.result),
                answer,
                rel_tol=args.strict_rel_tol,
                abs_tol=args.strict_abs_tol,
            )
        )
        legacy_match = bool(
            evaluation.executed
            and math.isclose(
                float(evaluation.result),
                answer,
                rel_tol=args.legacy_rel_tol,
                abs_tol=args.legacy_abs_tol,
            )
        )
        exact_match = bool(evaluation.executed and float(evaluation.result) == answer)
        missing_tickers = sorted(set(requested_tickers) - set(evidence_tickers))
        extra_tickers = sorted(set(evidence_tickers) - set(requested_tickers))
        missing_years = sorted(set(requested_years) - set(evidence_years))
        extra_years = sorted(set(evidence_years) - set(requested_years))
        target_unit = _target_unit(question)
        percentage_out_of_range = target_unit in {"percent", "percentage_point"} and abs(answer) > args.percentage_max_abs
        validation = item.get("validation")
        solver_validation = validation.get("solver", {}) if isinstance(validation, Mapping) else {}
        fallback_marker_available = bool(
            isinstance(validation, Mapping)
            and (
                "single_row_fallback_used" in validation
                or (
                    isinstance(solver_validation, Mapping)
                    and "single_row_fallback_used" in solver_validation
                )
            )
        )
        recorded_fallback = bool(
            fallback_marker_available
            and (
                validation.get("single_row_fallback_used")
                or (
                    isinstance(solver_validation, Mapping)
                    and solver_validation.get("single_row_fallback_used")
                )
            )
        )

        per_question.append(
            {
                "id": item_id,
                "question": question,
                "question_type": analysis["question_type"],
                "classification": "complex" if analysis["is_complex"] else "simple",
                "analysis_source": analysis["source"],
                "requested_tickers": requested_tickers,
                "requested_years": requested_years,
                "evidence_tickers": evidence_tickers,
                "evidence_years": evidence_years,
                "missing_tickers": missing_tickers,
                "extra_tickers": extra_tickers,
                "missing_years": missing_years,
                "extra_years": extra_years,
                "target_unit": target_unit,
                "answer": answer,
                "answer_is_zero": answer == 0.0,
                "percentage_out_of_range": percentage_out_of_range,
                "evidence_count": len(evidence),
                "query_executed": evaluation.executed,
                "query_result": evaluation.result,
                "query_error_type": evaluation.error_type,
                "query_error": evaluation.error,
                "strict_answer_query_match": strict_match,
                "legacy_answer_query_match": legacy_match,
                "exact_answer_query_match": exact_match,
                "referenced_variables": evaluation.referenced_variables,
                "evidence_variables": evaluation.evidence_variables,
                "missing_variables": evaluation.missing_variables,
                "unused_variables": evaluation.unused_variables,
                "resolved_evidence": evaluation.resolved_evidence,
                "fallback_marker_available": fallback_marker_available,
                "recorded_fallback": recorded_fallback,
            }
        )

    total = len(per_question)
    simple_ids = _ids(row for row in per_question if row["classification"] == "simple")
    complex_ids = _ids(row for row in per_question if row["classification"] == "complex")

    log_report: dict[str, Any]
    if args.log is not None:
        log_report = parse_pipeline_log(args.log.resolve(), submission_ids)
    else:
        log_report = {"available": False}

    recorded_fallback_ids = _ids(row for row in per_question if row["recorded_fallback"])
    fallback_ids = sorted(set(log_report.get("fallback_ids", [])) | set(recorded_fallback_ids))
    fallback_available = bool(
        log_report.get("available")
        or any(row["fallback_marker_available"] for row in per_question)
    )
    generation_failure_ids = log_report.get("generation_failure_ids", [])
    retry_failure_ids = log_report.get("retry_failure_ids", [])

    fallback_source_not_final: list[int] = []
    if log_report.get("available"):
        item_by_id = {row["id"]: row for row in per_question}
        for detail in log_report.get("fallback_details", []):
            row = item_by_id.get(detail["id"])
            if row is None:
                continue
            final_basenames = {
                Path(entry["csv_path"].replace("\\", "/")).name
                for entry in row["resolved_evidence"]
            }
            if Path(detail["file"]).name not in final_basenames:
                fallback_source_not_final.append(detail["id"])

    executed_ids = _ids(row for row in per_question if row["query_executed"])
    execution_failure_ids = _ids(row for row in per_question if not row["query_executed"])
    strict_match_ids = _ids(row for row in per_question if row["strict_answer_query_match"])
    strict_mismatch_ids = _ids(row for row in per_question if not row["strict_answer_query_match"])
    legacy_match_ids = _ids(row for row in per_question if row["legacy_answer_query_match"])
    legacy_mismatch_ids = _ids(row for row in per_question if not row["legacy_answer_query_match"])
    exact_match_ids = _ids(row for row in per_question if row["exact_answer_query_match"])
    empty_evidence_ids = _ids(row for row in per_question if row["evidence_count"] == 0)
    zero_answer_ids = _ids(row for row in per_question if row["answer_is_zero"])
    single_company_ids = _ids(row for row in per_question if len(row["requested_tickers"]) == 1)
    multi_company_ids = _ids(row for row in per_question if len(row["requested_tickers"]) > 1)
    no_company_ids = _ids(row for row in per_question if not row["requested_tickers"])
    single_year_ids = _ids(row for row in per_question if len(row["requested_years"]) == 1)
    multi_year_ids = _ids(row for row in per_question if len(row["requested_years"]) > 1)
    no_year_ids = _ids(row for row in per_question if not row["requested_years"])
    single_company_single_year_ids = _ids(
        row
        for row in per_question
        if len(row["requested_tickers"]) == 1 and len(row["requested_years"]) == 1
    )
    percentage_target_ids = _ids(
        row for row in per_question if row["target_unit"] in {"percent", "percentage_point"}
    )
    percentage_out_of_range_ids = _ids(row for row in per_question if row["percentage_out_of_range"])
    percentage_zero_ids = _ids(
        row
        for row in per_question
        if row["target_unit"] in {"percent", "percentage_point"} and row["answer_is_zero"]
    )
    missing_ticker_ids = _ids(row for row in per_question if row["missing_tickers"])
    extra_ticker_ids = _ids(row for row in per_question if row["extra_tickers"])
    missing_year_ids = _ids(row for row in per_question if row["missing_years"])
    extra_year_ids = _ids(row for row in per_question if row["extra_years"])
    exact_entity_year_ids = _ids(
        row
        for row in per_question
        if not (
            row["missing_tickers"]
            or row["extra_tickers"]
            or row["missing_years"]
            or row["extra_years"]
        )
    )
    unused_evidence_ids = _ids(row for row in per_question if row["unused_variables"])
    missing_variable_ids = _ids(row for row in per_question if row["missing_variables"])

    entity_extraction_failure_ids = sorted(
        set(no_company_ids)
        | set(log_report.get("ticker_none_ids", []))
        | set(log_report.get("csv_zero_ids", []))
    )

    def average_evidence(ids: Sequence[int]) -> float:
        selected = [row["evidence_count"] for row in per_question if row["id"] in set(ids)]
        return sum(selected) / len(selected) if selected else 0.0

    retrieved_csv_counts = {
        int(key): value for key, value in log_report.get("retrieved_csv_counts", {}).items()
    }

    def average_retrieved(ids: Sequence[int]) -> float | None:
        selected = [retrieved_csv_counts[item_id] for item_id in ids if item_id in retrieved_csv_counts]
        return sum(selected) / len(selected) if selected else None

    categories = {
        "query_execution": {
            "success": _category(executed_ids, total),
            "failure": _category(execution_failure_ids, total),
            "failures": [
                {
                    "id": row["id"],
                    "error_type": row["query_error_type"],
                    "error": row["query_error"],
                }
                for row in per_question
                if not row["query_executed"]
            ],
        },
        "answer_query_consistency": {
            "strict": {
                "rel_tol": args.strict_rel_tol,
                "abs_tol": args.strict_abs_tol,
                "match": _category(strict_match_ids, total),
                "mismatch": _category(strict_mismatch_ids, total),
            },
            "legacy": {
                "rel_tol": args.legacy_rel_tol,
                "abs_tol": args.legacy_abs_tol,
                "match": _category(legacy_match_ids, total),
                "mismatch": _category(legacy_mismatch_ids, total),
            },
            "exact_match": _category(exact_match_ids, total),
        },
        "empty_evidence": _category(empty_evidence_ids, total),
        "zero_answer": _category(zero_answer_ids, total),
        "fallback": {
            "available": fallback_available,
            "recorded_in_submission": _category(recorded_fallback_ids, total),
            "overall": _category(fallback_ids, total),
            "simple": _category(sorted(set(fallback_ids) & set(simple_ids)), len(simple_ids)),
            "complex": _category(sorted(set(fallback_ids) & set(complex_ids)), len(complex_ids)),
            "source_not_in_final_evidence": _category(fallback_source_not_final, len(fallback_ids)),
        },
        "generation_failure": {
            "available": bool(log_report.get("available")),
            "overall": _category(generation_failure_ids, total),
            "simple": _category(
                sorted(set(generation_failure_ids) & set(simple_ids)), len(simple_ids)
            ),
            "complex": _category(
                sorted(set(generation_failure_ids) & set(complex_ids)), len(complex_ids)
            ),
            "any_retry_failure": _category(retry_failure_ids, total),
        },
        "question_shape": {
            "simple": _category(simple_ids, total),
            "complex": _category(complex_ids, total),
            "single_company": _category(single_company_ids, total),
            "multi_company": _category(multi_company_ids, total),
            "no_company": _category(no_company_ids, total),
            "single_year": _category(single_year_ids, total),
            "multi_year": _category(multi_year_ids, total),
            "no_year": _category(no_year_ids, total),
            "single_company_single_year": _category(single_company_single_year_ids, total),
        },
        "percentage_answer_validation": {
            "threshold_abs": args.percentage_max_abs,
            "target_questions": _category(percentage_target_ids, total),
            "out_of_range": _category(percentage_out_of_range_ids, len(percentage_target_ids)),
            "zero_answers_review_separately": _category(percentage_zero_ids, len(percentage_target_ids)),
        },
        "evidence_entity_alignment": {
            "exact_ticker_year_set": _category(exact_entity_year_ids, total),
            "missing_requested_ticker": _category(missing_ticker_ids, total),
            "extra_or_wrong_ticker": _category(extra_ticker_ids, total),
            "missing_requested_year": _category(missing_year_ids, total),
            "extra_or_wrong_year": _category(extra_year_ids, total),
        },
        "query_evidence_variables": {
            "missing_variable": _category(missing_variable_ids, total),
            "unused_evidence": _category(unused_evidence_ids, total),
        },
        "entity_extraction_failure": _category(entity_extraction_failure_ids, total),
        "prompt_truncation": {
            "available": bool(log_report.get("available")),
            "confirmed_prompt_echo": _category(log_report.get("prompt_echo_ids", []), total),
            "suspected_union": _category(
                log_report.get("suspected_prompt_truncation_ids", []), total
            ),
            "definition": (
                "Confirmed when logged model output echoes few-shot/prompt headings; suspected "
                "also includes a non-fenced 500-character log clip."
            ),
        },
    }

    summary = {
        "total_questions": total,
        "query_execution_rate": categories["query_execution"]["success"]["rate"],
        "answer_query_consistency_rate_strict": categories["answer_query_consistency"]["strict"]["match"]["rate"],
        "answer_query_consistency_rate_legacy": categories["answer_query_consistency"]["legacy"]["match"]["rate"],
        "empty_evidence_count": len(empty_evidence_ids),
        "zero_answer_count": len(zero_answer_ids),
        "fallback_rate_overall": categories["fallback"]["overall"]["rate"],
        "fallback_rate_simple": categories["fallback"]["simple"]["rate"],
        "fallback_rate_complex": categories["fallback"]["complex"]["rate"],
        "generation_failure_rate": categories["generation_failure"]["overall"]["rate"],
        "entity_extraction_failure_count": len(entity_extraction_failure_ids),
        "prompt_truncation_count": categories["prompt_truncation"]["suspected_union"]["count"],
        "average_evidence_count": {
            "overall": average_evidence([row["id"] for row in per_question]),
            "simple": average_evidence(simple_ids),
            "complex": average_evidence(complex_ids),
        },
        "average_retrieved_csv_count_from_log": {
            "overall": average_retrieved([row["id"] for row in per_question]),
            "simple": average_retrieved(simple_ids),
            "complex": average_retrieved(complex_ids),
        },
    }

    return {
        "metadata": {
            "submission_path": str(submission_path),
            "log_path": str(args.log.resolve()) if args.log else None,
            "repo_root": str(repo_root),
            "processed_csv": str(processed_csv.resolve()),
            "planner": analyzer.planner_name,
            "planner_load_error": analyzer.planner_error,
            "classification_fallback": "HeuristicQuestionAnalyzer",
            "strict_tolerance": {
                "relative": args.strict_rel_tol,
                "absolute": args.strict_abs_tol,
            },
            "legacy_tolerance": {
                "relative": args.legacy_rel_tol,
                "absolute": args.legacy_abs_tol,
            },
        },
        "summary": summary,
        "categories": categories,
        "log": log_report,
        "per_question": sorted(per_question, key=lambda row: row["id"]),
    }


def _print_summary(report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    total = summary["total_questions"]
    print(f"Questions: {total}")
    print(f"Query execution: {summary['query_execution_rate']:.2%}")
    print(
        "Answer/query consistency: "
        f"strict={summary['answer_query_consistency_rate_strict']:.2%}, "
        f"legacy={summary['answer_query_consistency_rate_legacy']:.2%}"
    )
    print(
        f"Empty evidence: {summary['empty_evidence_count']} | "
        f"Zero answers: {summary['zero_answer_count']}"
    )
    print(
        "Fallback rate: "
        f"overall={summary['fallback_rate_overall']:.2%}, "
        f"simple={summary['fallback_rate_simple']:.2%}, "
        f"complex={summary['fallback_rate_complex']:.2%}"
    )
    print(f"Generation failure: {summary['generation_failure_rate']:.2%}")
    print(f"Prompt truncation signals: {summary['prompt_truncation_count']}")
    averages = summary["average_evidence_count"]
    print(
        "Average final evidence: "
        f"overall={averages['overall']:.3f}, simple={averages['simple']:.3f}, "
        f"complex={averages['complex']:.3f}"
    )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    script_repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Execute and audit ViFinQA submission queries against final evidence."
    )
    parser.add_argument("submission", type=Path, help="Path to submission JSON")
    parser.add_argument("--log", type=Path, help="Optional pipeline log")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=script_repo_root,
        help="Repository root (default: inferred from this script)",
    )
    parser.add_argument(
        "--processed-csv",
        type=Path,
        default=Path("data/processed_csv"),
        help="Nested processed CSV root, absolute or relative to repo root",
    )
    parser.add_argument(
        "--evidence-root",
        action="append",
        type=Path,
        default=[],
        help="Additional evidence root; may be repeated",
    )
    parser.add_argument("--output", type=Path, help="Write the complete JSON report here")
    parser.add_argument("--strict-rel-tol", type=float, default=DEFAULT_STRICT_TOLERANCE)
    parser.add_argument("--strict-abs-tol", type=float, default=DEFAULT_STRICT_TOLERANCE)
    parser.add_argument("--legacy-rel-tol", type=float, default=DEFAULT_LEGACY_TOLERANCE)
    parser.add_argument("--legacy-abs-tol", type=float, default=DEFAULT_LEGACY_TOLERANCE)
    parser.add_argument(
        "--percentage-max-abs", type=float, default=DEFAULT_PERCENTAGE_MAX_ABS
    )
    parser.add_argument(
        "--stdout-json",
        action="store_true",
        help="Print the full JSON report instead of the compact summary",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_report(args)
    if args.output:
        output = args.output
        if not output.is_absolute():
            output = args.repo_root.resolve() / output
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        print(f"Report written: {output.resolve()}")
    if args.stdout_json:
        print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    else:
        _print_summary(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
