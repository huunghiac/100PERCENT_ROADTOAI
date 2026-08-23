import json
import pandas as pd
import os
import re
import sys
import io
import traceback

# ---------- Backend detection ----------
try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _HAS_TRANSFORMERS = True
except ImportError:
    _HAS_TRANSFORMERS = False

try:
    import requests as _requests
except ImportError:
    _requests = None


class PandasAgent:
    def __init__(self,
                 model_name="Qwen/Qwen2.5-Coder-7B-Instruct",
                 base_url="http://localhost:11434",
                 backend="auto",
                 torch_dtype=None,
                 max_new_tokens=768):
        """
        backend: "auto" | "transformers" | "ollama"
        """
        self.max_new_tokens = max_new_tokens
        if backend == "auto":
            backend = "transformers" if _HAS_TRANSFORMERS else "ollama"
        self.backend = backend

        if self.backend == "transformers":
            if not _HAS_TRANSFORMERS:
                raise ImportError("pip install transformers torch accelerate")
            print(f"[Agent] Loading {model_name} via transformers ...")
            if torch_dtype is None:
                torch_dtype = torch.float16
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
            # Detect GPU memory → ép accelerate dùng hết VRAM trước khi fallback CPU
            if torch.cuda.is_available():
                max_memory = {
                    i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.90 / 1e9)}GiB"
                    for i in range(torch.cuda.device_count())
                }
                max_memory["cpu"] = "32GiB"
            else:
                max_memory = None
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch_dtype,
                device_map="auto", max_memory=max_memory,
                trust_remote_code=True,
            )
            self.model.eval()
            print(f"[Agent] Model loaded. Device map: {self.model.hf_device_map}")
        else:
            self.model_name = model_name
            self.api_url = f"{base_url}/api/generate"
            self.tokenizer = None
            self.model = None

    def clean_response(self, text: str) -> str:
        # Xóa <think>...</think> block (có hoặc không đóng tag)
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL).strip()
        code_match = re.search(r'```(?:python)?\s*(.*?)\s*```', cleaned, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return cleaned

    def get_csv_preview(self, csv_paths: list) -> str:
        context = []
        for path in csv_paths:
            real_path = path if os.path.exists(path) else path.replace("data/", "", 1)
            if not os.path.exists(real_path):
                continue
            try:
                df = pd.read_csv(real_path)
                n = min(10, len(df))
                preview = (
                    f"--- File: {path} ---\n"
                    f"Shape: {df.shape[0]} rows x {df.shape[1]} cols\n"
                    f"Columns: {list(df.columns)}\n"
                    f"Data (first {n} rows):\n{df.head(n).to_string()}\n"
                )
                context.append(preview)
            except Exception as e:
                context.append(f"--- File: {path} (Error: {e}) ---\n")
        return "\n".join(context)

    def _detect_unit_request(self, question: str):
        q = question.lower()
        if "nghìn tỷ" in q:
            return "nghìn tỷ đồng"
        if "tỷ đồng" in q:
            return "tỷ đồng"
        if "triệu đồng" in q:
            return "triệu đồng"
        if "nghìn đồng" in q:
            return "nghìn đồng"
        if "phần trăm" in q or "%" in q:
            return "%"
        return None


    def _build_prompt(self, question, csv_paths, error_log=None):
        csv_context = self.get_csv_preview(csv_paths)
        unit_request = self._detect_unit_request(question)
        unit_note = ""
        if unit_request:
            unit_note = (
                f"\nUNIT CONVERSION (CRITICAL):\n"
                f"- The question asks for the answer in: {unit_request}\n"
                f'- Check Don_vi column for CSV unit (VND, Triệu VND, Nghìn VND, etc.)\n'
                f'- If CSV unit is VND and question asks "triệu đồng": divide by 1,000,000\n'
                f'- If CSV unit is VND and question asks "tỷ đồng": divide by 1,000,000,000\n'
                f'- If CSV unit is VND and question asks "nghìn tỷ đồng": divide by 1,000,000,000,000\n'
                f'- If CSV unit is "Triệu VND"/"Trieu VND" and question asks "tỷ đồng": divide by 1,000\n'
                f'- If CSV unit is "Triệu VND"/"Trieu VND" and question asks "triệu đồng": no conversion\n'
                f'- If CSV unit already matches question unit: no conversion needed\n'
                f'- Round to 2 decimal places for tỷ, 0 for triệu/VND\n'
            )
        prompt = (
            'You are a Python/Pandas expert. Write ONLY Python code to answer the question.\n\n'
            'RULES:\n'
            '- Read CSV: pd.read_csv("exact_file_path")\n'
            '- CSV has columns: Chi_tieu (indicator name in Vietnamese WITH DIACRITICS), Gia_tri (numeric value), Don_vi (unit)\n'
            '- Search ALL provided CSV files. Do not stop at first file unless a strong match is found.\n'
            '- NEVER use the whole question inside str.contains(). Use only short indicator keywords from Chi_tieu.\n'
            '- Prefer flexible matching: multiple keywords or regex with .* between words. Example: "Lợi nhuận.*sau thuế" matches "LỢI NHUẬN KẾ TOÁN SAU THUẾ TNDN".\n'
            '- Check match.empty before reading values[0]. If no match found in all files, print(0.0).\n'
            '- The final output MUST be exactly one number. No text, no units.\n'
            '- Output code inside ```python ... ``` block.\n'
            f'{unit_note}\n'
            'USE THIS TEMPLATE STYLE:\n'
            '```python\nimport pandas as pd\nimport re\n\n'
            f'target_unit = "{unit_request or ""}"  # unit asked by the question; keep empty if none\n\n'
            'def convert_unit(value, source_unit, target_unit):\n'
            '    value = float(value)\n'
            '    u = str(source_unit).lower()\n'
            '    t = str(target_unit).lower()\n'
            '    is_vnd = "vnd" in u or "đồng" in u\n'
            '    is_trieu = "trieu" in u or "triệu" in u\n'
            '    is_ty = "ty" in u or "tỷ" in u\n'
            '    if "nghìn tỷ" in t:\n'
            '        if is_trieu: return value / 1_000_000\n'
            '        if is_vnd and not is_trieu and not is_ty: return value / 1_000_000_000_000\n'
            '        return value\n'
            '    if "tỷ" in t:\n'
            '        if is_trieu: return value / 1000\n'
            '        if is_vnd and not is_trieu and not is_ty: return value / 1_000_000_000\n'
            '        return value\n'
            '    if "triệu" in t:\n'
            '        if is_ty: return value * 1000\n'
            '        if is_vnd and not is_trieu and not is_ty: return value / 1_000_000\n'
            '        return value\n'
            '    return value\n\n'
            'def find_rows(df, regex=None, keywords_all=None):\n'
            '    s = df["Chi_tieu"].astype(str)\n'
            '    mask = pd.Series(True, index=df.index)\n'
            '    if regex:\n'
            '        mask &= s.str.contains(regex, case=False, na=False, regex=True)\n'
            '    if keywords_all:\n'
            '        for kw in keywords_all:\n'
            '            mask &= s.str.contains(kw, case=False, na=False, regex=False)\n'
            '    return df.loc[mask]\n\n'
            'files = ["data/processed_csv/AAA/file1.csv", "data/processed_csv/AAA/file2.csv"]\n'
            'answer = None\n'
            'for f in files:\n'
            '    df = pd.read_csv(f)\n'
            '    m = find_rows(df, regex=r"Lợi nhuận.*sau thuế")\n'
            '    if not m.empty:\n'
            '        row = m.iloc[0]\n'
            '        answer = convert_unit(row["Gia_tri"], row.get("Don_vi", ""), target_unit)\n'
            '        break\n'
            'print(round(answer, 2) if answer is not None else 0.0)\n```\n\n'
            f'NOW SOLVE THIS:\nQuestion: {question}\n\nAvailable data:\n{csv_context}\n'
        )
        if error_log:
            prompt += f"\nPREVIOUS ERROR (fix this):\n{error_log}\n"
        return prompt

    def _generate_transformers(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=3584)
        first_device = next(iter(self.model.parameters())).device
        inputs = {k: v.to(first_device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens,
                do_sample=False, temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _generate_ollama(self, prompt):
        if _requests is None:
            raise ImportError("requests not installed for ollama backend")
        payload = {
            "model": self.model_name, "prompt": prompt, "stream": False,
            "options": {"num_ctx": 4096, "num_predict": self.max_new_tokens},
        }
        res = _requests.post(self.api_url, json=payload, timeout=None)
        res.raise_for_status()
        return res.json().get("response", "")

    def generate_code(self, question, csv_paths, error_log=None):
        prompt = self._build_prompt(question, csv_paths, error_log)
        try:
            if self.backend == "transformers":
                raw_text = self._generate_transformers(prompt)
            else:
                raw_text = self._generate_ollama(prompt)
            return self.clean_response(raw_text)
        except Exception as e:
            print(f"[Agent Warning] Error generating code ({self.backend}): {e}")
            return 'import pandas as pd\nprint(0.0)'

    def execute_code(self, code):
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        try:
            exec_globals = {"pd": pd, "os": os, "re": re}
            exec(code, exec_globals)
            sys.stdout = old_stdout
            result = new_stdout.getvalue().strip()
            if not result:
                return None, "Output rỗng. Đảm bảo có print(kết_quả) ở cuối."
            last_line = result.strip().split("\n")[-1].strip()
            return last_line, None
        except Exception:
            sys.stdout = old_stdout
            return None, traceback.format_exc()

    def run_agent(self, question, csv_paths, max_retries=3):
        """Trả về: (answer_str, pandas_code_str, error_str_or_None)"""
        if not csv_paths:
            return "0.0", "", "No CSV files found by retriever."
        error_log = None
        last_code = ""
        for attempt in range(max_retries):
            code = self.generate_code(question, csv_paths, error_log)
            last_code = code
            ans, err = self.execute_code(code)
            if err is None and ans is not None:
                return ans, code, None
            error_log = err
            print(f"[Agent Self-Correction] Attempt {attempt+1}/{max_retries} failed.")
        return "0.0", last_code, f"Failed after {max_retries} retries. Last error: {error_log}"


if __name__ == "__main__":
    agent = PandasAgent(backend="ollama", model_name="deepseek-r1:14b")
    test_units = [
        ("Doanh thu thuần năm 2022 là bao nhiêu tỷ đồng?", "tỷ đồng"),
        ("Lợi nhuận sau thuế là bao nhiêu triệu đồng?", "triệu đồng"),
        ("Tỷ lệ sở hữu là bao nhiêu %?", "%"),
    ]
    print("=== Unit detection test ===")
    for q, expected in test_units:
        result = agent._detect_unit_request(q)
        status = "✓" if result == expected else f"✗ (got {result})"
        print(f"  {status} '{q[:50]}...' -> {result}")
    print(f"\n=== Agent ready (backend={agent.backend}) ===")

