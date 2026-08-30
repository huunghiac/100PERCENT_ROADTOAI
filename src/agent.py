"""Small-context pandas code agent used only for simple ViFinQA lookups.

Analytical questions are solved by :mod:`complex_solver`. This module keeps
the legacy model backend, but guarantees that the complete question and core
constraints are never tokenizer-truncated. Evidence is the only section that
may be shortened to meet a prompt budget.
"""

from __future__ import annotations

import io
import os
import re
import sys
import traceback
from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

try:
    from .units import detect_target_unit
except ImportError:  # pragma: no cover - legacy script imports
    from units import detect_target_unit

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    _HAS_TRANSFORMERS = True
except ImportError:  # pragma: no cover - optional inference dependency
    _HAS_TRANSFORMERS = False

try:
    import requests as _requests
except ImportError:  # pragma: no cover - optional inference dependency
    _requests = None


class PromptBudgetError(ValueError):
    """The immutable prompt sections alone exceed the model context."""


@dataclass(frozen=True)
class PromptReport:
    token_budget: int
    estimated_tokens: int
    question_preserved: bool
    evidence_truncated: bool
    raw_csv_tables: int
    semantic_context: bool


class PandasAgent:
    """Generate auditable pandas expressions for simple lookups only."""

    _SYSTEM_PROMPT = (
        "Bạn là bộ sinh biểu thức Pandas cho dữ liệu BCTC Việt Nam. "
        "Chỉ trả về đúng một code block Python. Các DataFrame df1, df2, ... "
        "đã được nạp; không đọc file, không import, không đoán dòng, không dùng "
        "answer có sẵn. Chọn đúng Chi_tieu và đơn vị của chính dòng đã chọn. "
        "Giữ nguyên dấu số liệu. Kết thúc bằng print(answer), một scalar số."
    )

    _ONE_EXAMPLE = """Ví dụ ngắn:
```python
row = df1[df1['Chi_tieu'].str.fullmatch(r'Doanh thu thuần', case=False, na=False)].iloc[0]
answer = float(row['Gia_tri'])
print(answer)
```"""

    _PREVIEW_STOPWORDS = {
        "của", "cho", "và", "vào", "cuối", "trong", "năm", "là", "bao", "nhiêu",
        "đồng", "triệu", "tỷ", "nghìn", "ngày", "tháng", "đến", "tại", "với",
        "công", "ty", "ctcp", "tnhh", "tmcp", "ngân", "hàng", "tổng", "tập", "đoàn",
        "mẹ", "hợp", "nhất", "riêng", "báo", "cáo", "đơn", "vị", "theo", "các",
    }

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
        base_url: str = "http://localhost:11434",
        backend: str = "auto",
        torch_dtype=None,
        max_new_tokens: int = 512,
        prompt_token_budget: int = 5632,
    ) -> None:
        self.max_new_tokens = max_new_tokens
        self.prompt_token_budget = int(prompt_token_budget)
        self.last_prompt_report: PromptReport | None = None
        if backend == "auto":
            backend = "transformers" if _HAS_TRANSFORMERS else "ollama"
        self.backend = backend

        if backend == "transformers":
            if not _HAS_TRANSFORMERS:
                raise ImportError("pip install transformers torch accelerate")
            print(f"[Agent] Loading {model_name} via transformers ...")
            if torch_dtype is None:
                torch_dtype = torch.float16
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            if torch.cuda.is_available():
                max_memory = {
                    index: f"{int(torch.cuda.get_device_properties(index).total_memory * 0.90 / 1e9)}GiB"
                    for index in range(torch.cuda.device_count())
                }
                max_memory["cpu"] = "32GiB"
            else:
                max_memory = None
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto",
                max_memory=max_memory,
                trust_remote_code=True,
            )
            self.model.eval()
            print(f"[Agent] Model loaded. Device map: {getattr(self.model, 'hf_device_map', 'single-gpu')}")
        else:
            self.model_name = model_name
            self.api_url = f"{base_url}/api/chat"
            self.tokenizer = None
            self.model = None

    def clean_response(self, text: str) -> str:
        cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
        cleaned = re.sub(r"<think>.*", "", cleaned, flags=re.DOTALL).strip()
        solution = re.search(r"#\s*BEGIN\s+SOLUTION\s*\n(.*?)#\s*END\s+SOLUTION", cleaned, re.DOTALL)
        code = solution.group(1).strip() if solution else ""
        if not code:
            for block in re.findall(r"```(?:python)?\s*(.*?)\s*```", cleaned, re.DOTALL | re.IGNORECASE):
                candidate = block.strip()
                if any(marker in candidate for marker in ("df1", "print(", "answer", "result")):
                    code = candidate
                    break
        if not code:
            lines: list[str] = []
            started = False
            for line in cleaned.splitlines():
                stripped = line.strip()
                looks_like_code = bool(
                    re.match(r"^[A-Za-z_]\w*\s*=", stripped)
                    or re.search(r"\bdf[1-9]\d*\b|print\(", stripped)
                )
                if looks_like_code:
                    started = True
                    lines.append(line)
                elif started and (not stripped or stripped.startswith("#")):
                    lines.append(line)
                elif started:
                    break
            code = "\n".join(lines).strip()
        if not code:
            return "# GENERATION_FAILED"
        code = re.split(r"\n\s*(?:Explanation|Giải thích|Notes?):", code, maxsplit=1)[0].strip()
        if "## CÂU HỎI" in code or "QUY TẮC BẮT BUỘC" in code or "```python" in code:
            return "# GENERATION_FAILED"
        example_match = re.search(
            r"```python\s*(.*?)\s*```", self._ONE_EXAMPLE, re.DOTALL | re.IGNORECASE
        )
        if example_match is not None:
            normalize_code = lambda value: re.sub(r"\s+", "", value).casefold()
            if normalize_code(code) == normalize_code(example_match.group(1)):
                # A verbatim few-shot echo is not evidence that the model
                # solved the current question.  Fail closed so the simple
                # deterministic fallback can handle it.
                return "# GENERATION_FAILED"
        if "print(" not in code and not re.search(r"\b(?:answer|result)\s*=", code):
            return "# GENERATION_FAILED"
        return code

    def _question_keywords(self, question: str) -> list[str]:
        text = re.sub(r"\b20\d{2}\b", " ", question.lower())
        tokens = re.findall(r"[\wÀ-ỹ]+", text, flags=re.UNICODE)
        return [token for token in tokens if len(token) > 1 and token not in self._PREVIEW_STOPWORDS]

    @staticmethod
    def _resolve_path(path: str) -> str | None:
        candidates = [path]
        if path.startswith("data/"):
            candidates.append(path.replace("data/", "", 1))
        basename = os.path.basename(path)
        ticker = basename.split("_")[0] if "_" in basename else ""
        candidates.append(os.path.join("data", "processed_csv", ticker, basename))
        return next((candidate for candidate in candidates if os.path.exists(candidate)), None)

    def get_csv_preview(
        self,
        csv_paths: Sequence[str],
        question: str = "",
        *,
        max_rows_per_table: int = 4,
        max_tables: int = 4,
        path_to_variable: Mapping[str, str] | None = None,
    ) -> str:
        """Build a compact, query-focused preview without dataframe heads."""

        keywords = self._question_keywords(question)
        blocks: list[str] = []
        for index, path in enumerate(list(csv_paths)[:max_tables]):
            if path_to_variable is None:
                variable = f"df{index + 1}"
            else:
                normalized = path.replace("\\", "/")
                variable = path_to_variable.get(path) or path_to_variable.get(normalized)
                if variable is None:
                    raise ValueError(
                        f"Evidence path has no official dataframe variable: {path}"
                    )
            resolved = self._resolve_path(path)
            if not resolved:
                continue
            try:
                frame = pd.read_csv(resolved)
            except Exception as exc:
                blocks.append(f"{variable} | {os.path.basename(path)} | read_error={type(exc).__name__}")
                continue
            if "Chi_tieu" not in frame.columns:
                blocks.append(f"{variable} | {os.path.basename(path)} | columns={list(frame.columns)}")
                continue
            labels = frame["Chi_tieu"].astype(str)
            scores: list[tuple[int, int]] = []
            folded = labels.str.casefold()
            for row_index in frame.index:
                label = folded.loc[row_index]
                score = sum(len(keyword) for keyword in keywords if keyword.casefold() in label)
                if score:
                    scores.append((score, int(row_index)))
            selected = [row for _, row in sorted(scores, reverse=True)[:max_rows_per_table]]
            if not selected:
                selected = list(frame.index[: min(2, max_rows_per_table)])
            columns = [name for name in ("Chi_tieu", "Gia_tri", "Don_vi") if name in frame.columns]
            rows = frame.loc[selected, columns].copy()
            rows.insert(0, "row", selected)
            blocks.append(
                f"{variable} | file={os.path.basename(path)} | rows={len(frame)}\n"
                + rows.to_csv(index=False).strip()
            )
        return "\n\n".join(blocks)

    def _detect_unit_request(self, question: str) -> str | None:
        return detect_target_unit(question) or None

    def _extract_csv_units(self, csv_paths: Sequence[str]) -> dict[str, str]:
        """Compatibility helper returning all units, never a dominant guess."""

        result: dict[str, str] = {}
        for index, path in enumerate(csv_paths):
            resolved = self._resolve_path(path)
            if not resolved:
                result[f"df{index + 1}"] = ""
                continue
            try:
                frame = pd.read_csv(resolved, usecols=["Don_vi"])
                units = list(dict.fromkeys(frame["Don_vi"].dropna().astype(str)))
                result[f"df{index + 1}"] = " | ".join(units[:8])
            except Exception:
                result[f"df{index + 1}"] = ""
        return result

    def _build_unit_guidance(self, csv_units: Mapping[str, str], target_unit: str | None) -> str:
        del csv_units
        requested = target_unit or "đơn vị của dòng nguồn"
        return (
            f"Đơn vị đích: {requested}. Luôn đọc Don_vi của đúng row được chọn; "
            "không suy từ 20 dòng đầu. trăm tỷ=1e11 VND, nghìn tỷ=1e12 VND; "
            "% là tỷ số×100, điểm phần trăm là hiệu hai tỷ lệ, lần không phải tiền."
        )

    def _render_chat(self, messages: Sequence[Mapping[str, str]]) -> str:
        if self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)

    def _count_tokens(self, messages: Sequence[Mapping[str, str]]) -> int:
        rendered = self._render_chat(messages)
        if self.tokenizer is not None:
            encoded = self.tokenizer(rendered, add_special_tokens=False, truncation=False)
            ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded
            return len(ids)
        return (len(rendered.encode("utf-8")) + 2) // 3

    def _build_messages(
        self,
        question: str,
        csv_paths: Sequence[str],
        error_log: str | None = None,
        *,
        semantic_context: str | None = None,
        path_to_variable: Mapping[str, str] | None = None,
    ) -> list[dict[str, str]]:
        # Never let a failed build expose the previous question's report to
        # pipeline diagnostics.
        self.last_prompt_report = None
        if not question or not question.strip():
            raise ValueError("question must be non-empty")
        evidence = semantic_context if semantic_context is not None else self.get_csv_preview(
            csv_paths, question, path_to_variable=path_to_variable
        )
        unit_guidance = self._build_unit_guidance({}, self._detect_unit_request(question))
        error = f"\nLỗi lần trước (chỉ sửa lỗi này): {error_log[:300]}" if error_log else ""
        prefix = (
            f"## CÂU HỎI NGUYÊN VẸN\n{question}\n\n"
            f"## ĐƠN VỊ\n{unit_guidance}\n\n"
            f"{self._ONE_EXAMPLE}\n\n## EVIDENCE RÚT GỌN\n"
        )
        suffix = f"{error}\n\nChỉ xuất code giải đúng câu hỏi nguyên vẹn ở đầu prompt."

        def messages_for(current_evidence: str) -> list[dict[str, str]]:
            return [
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user", "content": prefix + current_evidence + suffix},
            ]

        immutable_tokens = self._count_tokens(messages_for(""))
        if immutable_tokens > self.prompt_token_budget:
            raise PromptBudgetError(
                f"Question and constraints require {immutable_tokens} tokens, budget={self.prompt_token_budget}"
            )
        candidate = evidence
        messages = messages_for(candidate)
        truncated = False
        while candidate and self._count_tokens(messages) > self.prompt_token_budget:
            truncated = True
            candidate = candidate[: max(0, int(len(candidate) * 0.80))]
            messages = messages_for(candidate + "\n[EVIDENCE_TRUNCATED]")
        count = self._count_tokens(messages)
        if count > self.prompt_token_budget:
            raise PromptBudgetError(f"Unable to fit evidence within {self.prompt_token_budget} tokens")
        if question not in messages[1]["content"]:
            raise AssertionError("Question was altered while constructing prompt")
        self.last_prompt_report = PromptReport(
            token_budget=self.prompt_token_budget,
            estimated_tokens=count,
            question_preserved=True,
            evidence_truncated=truncated,
            raw_csv_tables=0 if semantic_context is not None else min(len(csv_paths), 4),
            semantic_context=semantic_context is not None,
        )
        return messages

    def _generate_transformers(self, messages: Sequence[Mapping[str, str]]) -> str:
        text = self._render_chat(messages)
        inputs = self.tokenizer(text, return_tensors="pt", truncation=False)
        prompt_length = int(inputs["input_ids"].shape[1])
        if prompt_length > self.prompt_token_budget:
            raise PromptBudgetError(
                f"Tokenizer produced {prompt_length} tokens after budget guard ({self.prompt_token_budget})"
            )
        first_device = next(iter(self.model.parameters())).device
        inputs = {name: value.to(first_device) for name, value in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(outputs[0][prompt_length:], skip_special_tokens=True)

    def _generate_ollama(self, messages: Sequence[Mapping[str, str]]) -> str:
        if _requests is None:
            raise ImportError("requests not installed for ollama backend")
        payload = {
            "model": self.model_name,
            "messages": list(messages),
            "stream": False,
            "options": {
                "num_ctx": self.prompt_token_budget + self.max_new_tokens,
                "num_predict": self.max_new_tokens,
            },
        }
        response = _requests.post(self.api_url, json=payload, timeout=None)
        response.raise_for_status()
        body = response.json()
        return body.get("message", {}).get("content", body.get("response", ""))

    def generate_code(
        self,
        question: str,
        csv_paths: Sequence[str],
        error_log: str | None = None,
        *,
        semantic_context: str | None = None,
        path_to_variable: Mapping[str, str] | None = None,
    ) -> str:
        messages = self._build_messages(
            question,
            csv_paths,
            error_log,
            semantic_context=semantic_context,
            path_to_variable=path_to_variable,
        )
        try:
            raw = self._generate_transformers(messages) if self.backend == "transformers" else self._generate_ollama(messages)
            return self.clean_response(raw)
        except Exception as exc:
            print(f"[Agent Warning] generation failed ({self.backend}): {exc}")
            return "# GENERATION_FAILED"

    def execute_code(
        self,
        code: str,
        csv_paths: Sequence[str] | None = None,
        *,
        path_to_variable: Mapping[str, str] | None = None,
    ):
        if not code or "GENERATION_FAILED" in code:
            return None, "Generation failed; no executable query was produced."
        old_stdout, capture = sys.stdout, io.StringIO()
        sys.stdout = capture
        try:
            scope = {"pd": pd, "re": re}
            for index, path in enumerate(csv_paths or []):
                resolved = self._resolve_path(path)
                if resolved:
                    if path_to_variable is None:
                        variable = f"df{index + 1}"
                    else:
                        normalized = path.replace("\\", "/")
                        variable = path_to_variable.get(path) or path_to_variable.get(normalized)
                        if variable is None:
                            return None, f"Evidence path has no official dataframe variable: {path}"
                    scope[variable] = pd.read_csv(resolved)
            stripped = code.strip()
            if "\n" not in stripped and not stripped.startswith(("print", "import")):
                value = eval(stripped, scope)
                return str(value), None
            exec(code, scope)
            output = capture.getvalue().strip()
            if output:
                return output.splitlines()[-1].strip(), None
            for name in ("answer", "result"):
                if name in scope:
                    return str(scope[name]), None
            return None, "Output rỗng; cần answer/result hoặc print(answer)."
        except Exception:
            return None, traceback.format_exc()
        finally:
            sys.stdout = old_stdout

    def run_agent(
        self,
        question: str,
        csv_paths: Sequence[str],
        max_retries: int = 3,
        *,
        semantic_context: str | None = None,
        path_to_variable: Mapping[str, str] | None = None,
    ):
        if not csv_paths and not semantic_context:
            return None, "", "No evidence found by retriever."
        error_log = None
        last_code = ""
        for _ in range(max_retries):
            last_code = self.generate_code(
                question,
                csv_paths,
                error_log,
                semantic_context=semantic_context,
                path_to_variable=path_to_variable,
            )
            answer, error = self.execute_code(
                last_code,
                csv_paths,
                path_to_variable=path_to_variable,
            )
            if error is None and answer is not None:
                return answer, last_code, None
            error_log = error
        return None, last_code, f"Failed after {max_retries} retries. Last error: {error_log}"


__all__ = ["PandasAgent", "PromptBudgetError", "PromptReport"]
