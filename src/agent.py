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
            print(f"[Agent] Model loaded. Device map: {getattr(self.model, 'hf_device_map', 'single-gpu')}")
        else:
            self.model_name = model_name
            self.api_url = f"{base_url}/api/chat"
            self.tokenizer = None
            self.model = None

    def clean_response(self, text: str) -> str:
        # Xóa <think>...</think> block (có hoặc không đóng tag)
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL).strip()
        cleaned = re.sub(r'^\s*NEW\s+CODE:\s*', '', cleaned, flags=re.IGNORECASE)

        code = None

        # 0) Ưu tiên: tìm block # BEGIN SOLUTION ... # END SOLUTION
        sol_match = re.search(r'#\s*BEGIN\s+SOLUTION\s*\n(.*?)#\s*END\s+SOLUTION', cleaned, re.DOTALL)
        if sol_match:
            code = sol_match.group(1).strip()

        # 1) Tìm tất cả markdown code blocks, lấy block đầu tiên có nội dung thực
        if not code:
            blocks = re.findall(r'```(?:python)?\s*(.*?)\s*```', cleaned, re.DOTALL)
            for block in blocks:
                b = block.strip()
                if len(b) > 10 and any(m in b for m in ['df1', 'df2', 'print(', 'answer', '.iloc', '.str.', 'float(', '# BEGIN']):
                    code = b
                    # Nếu block chứa BEGIN/END SOLUTION, cắt lấy phần trong
                    inner = re.search(r'#\s*BEGIN\s+SOLUTION\s*\n(.*?)#\s*END\s+SOLUTION', b, re.DOTALL)
                    if inner:
                        code = inner.group(1).strip()
                    break

        # 2) Có import pandas → lấy từ đó trở đi
        if not code and "import pandas" in cleaned:
            code = cleaned[cleaned.find("import pandas"):].strip()

        # 3) Thử lấy code trần: tìm dòng đầu tiên có dấu hiệu code Python
        if not code:
            code_lines = []
            found_code = False
            for line in cleaned.split("\n"):
                stripped = line.strip()
                if not stripped:
                    if found_code:
                        code_lines.append(line)
                    continue
                is_code = any(marker in stripped for marker in [
                    "df1", "df2", "df3", "float(", "int(", "abs(",
                    ".iloc", ".loc[", ".str.", "print(", "= pd.",
                    "answer", "result", "val ", "val=",
                    "m_", "m =", "row ", "row=",
                ])
                if not is_code:
                    is_code = bool(re.match(r'^[a-zA-Z_]\w*\s*=', stripped))
                if is_code:
                    found_code = True
                    code_lines.append(line)
                elif found_code:
                    if not stripped.startswith("#"):
                        break
                    code_lines.append(line)
            code = "\n".join(code_lines).strip()

        if not code:
            return 'import pandas as pd\n# GENERATION_FAILED\nprint(0.0)'

        # Cắt bỏ text giải thích sau code nếu model lỡ viết thêm.
        code = re.split(r'\n\s*(?:Explanation|Giải thích|Notes?):', code, maxsplit=1)[0].strip()
        # Đảm bảo có print hoặc assignment answer/result
        if "print(" not in code and "answer" not in code and "result" not in code:
            code += "\nprint(0.0)"
        return code

    _PREVIEW_STOPWORDS = {
        "của", "cho", "và", "vào", "cuối", "trong", "năm", "là", "bao", "nhiêu",
        "đồng", "triệu", "tỷ", "nghìn", "ngày", "tháng", "đến", "tại", "với",
        "công", "ty", "ctcp", "tnhh", "tmcp", "ngân", "hàng", "tổng", "tập", "đoàn",
        "mẹ", "hợp", "nhất", "riêng", "báo", "cáo", "đơn", "vị", "theo", "các",
    }

    def _question_keywords(self, question: str) -> list:
        if not question:
            return []
        q = re.sub(r'\b20\d{2}\b', ' ', question.lower())
        q = re.sub(r'\b[A-Z]{2,4}\b', ' ', q)
        tokens = re.findall(r"[\wÀ-ỹ]+", q, flags=re.UNICODE)
        return [t for t in tokens if len(t) > 1 and t not in self._PREVIEW_STOPWORDS]

    def get_csv_preview(self, csv_paths: list, question: str = None) -> str:
        """
        Trích xuất context thông minh:
        - Metadata (Ticker, Năm, Loại BCTC)
        - Preview head(3)
        - Dòng chỉ tiêu liên quan trực tiếp
        """
        context = []
        noisy_keywords = {"chi", "phí", "tiền", "số", "dư", "khác", "khoản", "hoạt", "động", "tính", "hỏi", "cho", "biết"}
        keywords = [k for k in self._question_keywords(question or "") if k not in noisy_keywords]
        
        for i, path in enumerate(csv_paths):
            var_name = f"df{i+1}"
            real_path = path if os.path.exists(path) else path.replace("data/", "", 1)
            if not os.path.exists(real_path):
                bn = os.path.basename(path)
                ticker = bn.split("_")[0] if "_" in bn else ""
                cand = os.path.join("data", "processed_csv", ticker, bn)
                if os.path.exists(cand):
                    real_path = cand
                else:
                    continue
            try:
                df = pd.read_csv(real_path)
                flat_name = os.path.basename(path)
                parts = flat_name.replace(".csv", "").split("_")
                tk_info = parts[0] if len(parts) > 0 else ""
                yr_info = parts[1] if len(parts) > 1 else ""
                type_info = parts[-1] if len(parts) > 2 else ""

                preview = [
                    f"--- Variable: {var_name} [Ticker: {tk_info}, Year: {yr_info}, Type: {type_info}] (File: {flat_name}) ---",
                    f"Columns: {list(df.columns)}",
                    f"Total rows: {len(df)}",
                    f"Sample rows:\n{df.head(3).to_string()}"
                ]
                
                if "Chi_tieu" in df.columns:
                    s = df["Chi_tieu"].astype(str).str.lower()
                    matched_indices = set()
                    for kw in keywords:
                        mask = s.str.contains(kw, case=False, na=False, regex=False)
                        for idx in df.index[mask]:
                            matched_indices.add(idx)
                            
                    if matched_indices:
                        sorted_indices = sorted(list(matched_indices))[:10]
                        rel = df.loc[sorted_indices]
                        preview.append(f"Relevant indicator rows in {var_name}:\n{rel.to_string()}")
                        
                context.append("\n".join(preview) + "\n")
            except Exception as e:
                flat_name = f"data/{os.path.basename(path)}"
                context.append(f"--- Table variable: {var_name} (File: {flat_name}, Error: {e}) ---\n")
                
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

    # ---- Unit helpers ----
    _UNIT_VND_FACTOR = {
        "VND": 1,
        "Dong": 1,
        "dong": 1,
        "Nghin VND": 1_000,
        "Nghin dong": 1_000,
        "Trieu VND": 1_000_000,
        "Trieu dong": 1_000_000,
        "Ty dong": 1_000_000_000,
        "%": None,           # không quy đổi
        "VND/co phieu": 1,
        "Co phieu": None,
        "USD": None,
        "EUR": None,
        "JPY": None,
        "mixed": None,
    }

    _TARGET_VND_FACTOR = {
        "nghìn tỷ đồng": 1_000_000_000_000,
        "tỷ đồng": 1_000_000_000,
        "triệu đồng": 1_000_000,
        "nghìn đồng": 1_000,
    }

    def _extract_csv_units(self, csv_paths: list) -> dict:
        """Trả về dict {var_name: dominant_unit_string} cho từng CSV."""
        result = {}
        for i, path in enumerate(csv_paths):
            var_name = f"df{i+1}"
            real_path = path if os.path.exists(path) else path.replace("data/", "", 1)
            if not os.path.exists(real_path):
                bn = os.path.basename(path)
                ticker = bn.split("_")[0] if "_" in bn else ""
                cand = os.path.join("data", "processed_csv", ticker, bn)
                if os.path.exists(cand):
                    real_path = cand
                else:
                    result[var_name] = ""
                    continue
            try:
                df = pd.read_csv(real_path, usecols=["Don_vi"], nrows=20)
                units = df["Don_vi"].dropna().astype(str).value_counts()
                result[var_name] = units.index[0] if len(units) > 0 else ""
            except Exception:
                result[var_name] = ""
        return result

    def _build_unit_guidance(self, csv_units: dict, target_unit: str) -> str:
        """
        Sinh hướng dẫn quy đổi đơn vị chính xác dựa trên đơn vị gốc THỰC TẾ
        trong từng CSV và đơn vị yêu cầu của câu hỏi.
        """
        if target_unit == "%":
            return "Câu hỏi yêu cầu TỶ LỆ PHẦN TRĂM (%). Tính tỷ số rồi nhân 100. KHÔNG chia cho bất kỳ hệ số nào."

        target_factor = self._TARGET_VND_FACTOR.get(target_unit)
        lines = []

        for var_name, src_unit in csv_units.items():
            src_factor = self._UNIT_VND_FACTOR.get(src_unit)
            if src_factor is None or target_factor is None:
                lines.append(f"- {var_name}: đơn vị gốc = '{src_unit}'. Giữ nguyên giá trị, KHÔNG chia/nhân.")
                continue

            if src_factor == target_factor:
                lines.append(
                    f"- {var_name}: đơn vị gốc = '{src_unit}' — CÂU HỎI cũng yêu cầu {target_unit} "
                    f"→ GIỮ NGUYÊN giá trị. KHÔNG chia thêm."
                )
            elif src_factor < target_factor:
                divisor = target_factor // src_factor
                lines.append(
                    f"- {var_name}: đơn vị gốc = '{src_unit}' — câu hỏi yêu cầu {target_unit} "
                    f"→ CHIA giá trị cho {divisor:_}."
                )
            else:
                multiplier = src_factor // target_factor
                lines.append(
                    f"- {var_name}: đơn vị gốc = '{src_unit}' — câu hỏi yêu cầu {target_unit} "
                    f"→ NHÂN giá trị với {multiplier:_}."
                )

        if not lines:
            if target_unit:
                return f"Câu hỏi yêu cầu: {target_unit}. Kiểm tra cột Don_vi trong bảng để quy đổi phù hợp."
            return "Giữ nguyên đơn vị gốc trong bảng."

        header = f"Câu hỏi yêu cầu đáp án theo: {target_unit}.\n" if target_unit else "Giữ nguyên đơn vị gốc:\n"
        return header + "\n".join(lines)


    _SYSTEM_PROMPT = """\
Bạn là chuyên gia phân tích dữ liệu tài chính BCTC Việt Nam bằng Python Pandas.

QUY TẮC BẮT BUỘC:
- Trả lời bằng ĐÚNG MỘT code block ```python ... ``` duy nhất.
- Các DataFrame df1, df2, ... ĐÃ ĐƯỢC NẠP SẴN. KHÔNG import thư viện. KHÔNG dùng pd.read_csv().
- Luôn kiểm tra bảng nào chứa chỉ tiêu cần tìm. Nếu chỉ tiêu không có trong df1, hãy tìm trong df2 hoặc các bảng khác.
- Tìm chỉ tiêu trong cột 'Chi_tieu' bằng .str.contains(r'...', case=False, na=False).
- Lấy giá trị số từ cột 'Gia_tri': float(row['Gia_tri']).
- Nếu bảng có nhiều cột số khác (ví dụ: tỷ lệ %, số lượng), dùng cột phù hợp với câu hỏi.
- Kết thúc bằng print(answer) — answer là MỘT số float/int duy nhất.

TUYỆT ĐỐI KHÔNG:
- Viết "# Your code here" hoặc bất kỳ placeholder nào.
- Lặp lại code block nhiều lần.
- Viết text giải thích bên ngoài code block.
- Để code block trống hoặc chỉ chứa comment."""

    def _build_messages(self, question, csv_paths, error_log=None):
        preview = self.get_csv_preview(csv_paths, question)
        target_unit = self._detect_unit_request(question)
        csv_units = self._extract_csv_units(csv_paths)
        unit_guidance = self._build_unit_guidance(csv_units, target_unit)

        error_context = ""
        if error_log:
            error_context = f"""
LỖI TỪ LẦN CHẠY TRƯỚC (cần sửa):
```
{error_log[:500]}
```
Viết lại code mới sửa lỗi trên. KHÔNG lặp lại code cũ.
"""

        user_content = f"""## BẢNG DỮ LIỆU (đã load sẵn):
{preview}

## HƯỚNG DẪN QUY ĐỔI ĐƠN VỊ (BẮT BUỘC TUÂN THỦ):
{unit_guidance}

QUAN TRỌNG: Đọc kỹ đơn vị gốc ở trên. Nếu đơn vị gốc là 'Trieu VND' và câu hỏi yêu cầu 'triệu đồng' thì GIỮ NGUYÊN giá trị, KHÔNG chia thêm.

## VÍ DỤ:

Ví dụ 1 — Tra cứu đơn giản (đơn vị gốc VND, hỏi tỷ đồng → chia 1_000_000_000):
```python
m = df1[df1['Chi_tieu'].str.contains(r'doanh thu thuần', case=False, na=False)]
val = float(m.iloc[0]['Gia_tri'])
answer = val / 1_000_000_000
print(answer)
```

Ví dụ 2 — Chỉ tiêu nằm ở bảng thứ 2 (df2):
```python
m = df2[df2['Chi_tieu'].str.contains(r'phải thu ngắn hạn', case=False, na=False)]
val = float(m.iloc[0]['Gia_tri'])
answer = val / 1_000_000_000
print(answer)
```

Ví dụ 3 — Đơn vị gốc Trieu VND, hỏi triệu đồng → GIỮ NGUYÊN:
```python
m = df1[df1['Chi_tieu'].str.contains(r'cho vay khách hàng', case=False, na=False)]
answer = float(m.iloc[0]['Gia_tri'])
print(answer)
```

Ví dụ 4 — Đơn vị gốc Trieu VND, hỏi tỷ đồng → chia 1000:
```python
m = df1[df1['Chi_tieu'].str.contains(r'tổng tài sản', case=False, na=False)]
answer = float(m.iloc[0]['Gia_tri']) / 1000
print(answer)
```

Ví dụ 5 — Tính tỷ lệ % / Biên lợi nhuận liên bảng (df1=KQKD, df2=KQKD hoặc CĐKT):
```python
m_lnst = df1[df1['Chi_tieu'].str.contains(r'lợi nhuận sau thuế', case=False, na=False)]
m_dtt = df1[df1['Chi_tieu'].str.contains(r'doanh thu thuần', case=False, na=False)]
answer = float(m_lnst.iloc[0]['Gia_tri']) / float(m_dtt.iloc[0]['Gia_tri']) * 100
print(answer)
```

Ví dụ 6 — Tăng trưởng qua các năm (df1 = năm 2020, df2 = năm 2021):
```python
v2020 = float(df1[df1['Chi_tieu'].str.contains(r'doanh thu thuần', case=False, na=False)].iloc[0]['Gia_tri'])
v2021 = float(df2[df2['Chi_tieu'].str.contains(r'doanh thu thuần', case=False, na=False)].iloc[0]['Gia_tri'])
answer = (v2021 - v2020) / v2020 * 100
print(answer)
```

Ví dụ 7 — So sánh giữa 2 công ty (df1 = Ticker A, df2 = Ticker B):
```python
v_a = float(df1[df1['Chi_tieu'].str.contains(r'tổng tài sản', case=False, na=False)].iloc[0]['Gia_tri'])
v_b = float(df2[df2['Chi_tieu'].str.contains(r'tổng tài sản', case=False, na=False)].iloc[0]['Gia_tri'])
answer = (v_a - v_b) / 1_000_000_000
print(answer)
```
{error_context}
## CÂU HỎI:
{question}
"""
        return [
            {"role": "system", "content": self._SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    def _generate_transformers(self, messages):
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=6144)
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

    def _generate_ollama(self, messages):
        if _requests is None:
            raise ImportError("requests not installed for ollama backend")
        payload = {
            "model": self.model_name, "messages": messages, "stream": False,
            "options": {"num_ctx": 4096, "num_predict": self.max_new_tokens},
        }
        res = _requests.post(self.api_url, json=payload, timeout=None)
        res.raise_for_status()
        data = res.json()
        return data.get("message", {}).get("content", data.get("response", ""))

    def generate_code(self, question, csv_paths, error_log=None):
        messages = self._build_messages(question, csv_paths, error_log)
        try:
            if self.backend == "transformers":
                raw_text = self._generate_transformers(messages)
            else:
                raw_text = self._generate_ollama(messages)
            print(f"[Agent RAW] {raw_text[:500]}")
            code = self.clean_response(raw_text)
            if "GENERATION_FAILED" in code:
                print(f"[Agent] clean_response → GENERATION_FAILED. Full raw ({len(raw_text)} chars):\n{raw_text[:1000]}")
            else:
                print(f"[Agent] Extracted code:\n{code}")
            return code
        except Exception as e:
            print(f"[Agent Warning] Error generating code ({self.backend}): {e}")
            return 'import pandas as pd\nprint(0.0)'

    def execute_code(self, code, csv_paths=None):
        """
        Thực thi code Pandas. Nạp sẵn các biến df1, df2, ... từ csv_paths vào scope
        để hỗ trợ cả biểu thức đơn (eval) lẫn script đầy đủ (exec).
        """
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout
        orig_read_csv = pd.read_csv

        def _custom_read_csv(filepath_or_buffer, *args, **kwargs):
            if isinstance(filepath_or_buffer, str) and not os.path.exists(filepath_or_buffer):
                bn = os.path.basename(filepath_or_buffer)
                ticker = bn.split("_")[0] if "_" in bn else ""
                cand = os.path.join("data", "processed_csv", ticker, bn)
                if os.path.exists(cand):
                    filepath_or_buffer = cand
            return orig_read_csv(filepath_or_buffer, *args, **kwargs)

        pd.read_csv = _custom_read_csv
        try:
            exec_globals = {"pd": pd, "os": os, "re": re}
            
            # Load DataFrames for df1, df2, ...
            if csv_paths:
                for i, p in enumerate(csv_paths):
                    real_p = p if os.path.exists(p) else p.replace("data/", "", 1)
                    if not os.path.exists(real_p):
                        bn = os.path.basename(p)
                        ticker = bn.split("_")[0] if "_" in bn else ""
                        cand = os.path.join("data", "processed_csv", ticker, bn)
                        if os.path.exists(cand):
                            real_p = cand
                    if os.path.exists(real_p):
                        try:
                            exec_globals[f"df{i+1}"] = pd.read_csv(real_p)
                        except Exception:
                            pass

            # Thử eval trước nếu là biểu thức 1 dòng
            code_clean = code.strip()
            if "\n" not in code_clean and not code_clean.startswith("import") and not code_clean.startswith("print"):
                try:
                    val = eval(code_clean, exec_globals)
                    sys.stdout = old_stdout
                    pd.read_csv = orig_read_csv
                    return str(val), None
                except Exception:
                    pass

            exec(code, exec_globals)
            sys.stdout = old_stdout
            pd.read_csv = orig_read_csv
            result = new_stdout.getvalue().strip()
            if not result:
                # Kiểm tra nếu có biến result hoặc answer trong globals
                if "result" in exec_globals:
                    return str(exec_globals["result"]), None
                if "answer" in exec_globals:
                    return str(exec_globals["answer"]), None
                return None, "Output rỗng. Đảm bảo có print(kết_quả) hoặc trả về giá trị."
            last_line = result.strip().split("\n")[-1].strip()
            return last_line, None
        except Exception:
            sys.stdout = old_stdout
            pd.read_csv = orig_read_csv
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
            ans, err = self.execute_code(code, csv_paths=csv_paths)
            print(f"[Agent] Attempt {attempt+1}/{max_retries}: ans={ans}, err={err[:200] if err else None}")
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

