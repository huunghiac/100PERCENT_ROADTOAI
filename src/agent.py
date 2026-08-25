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
            self.api_url = f"{base_url}/api/generate"
            self.tokenizer = None
            self.model = None

    def clean_response(self, text: str) -> str:
        # Xóa <think>...</think> block (có hoặc không đóng tag)
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        cleaned = re.sub(r'<think>.*', '', cleaned, flags=re.DOTALL).strip()
        cleaned = re.sub(r'^\s*NEW\s+CODE:\s*', '', cleaned, flags=re.IGNORECASE)
        code_match = re.search(r'```(?:python)?\s*(.*?)\s*```', cleaned, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        elif "import pandas" in cleaned:
            code = cleaned[cleaned.find("import pandas"):].strip()
        else:
            return 'import pandas as pd\n# GENERATION_FAILED\nprint(0.0)'
        # Cắt bỏ text giải thích sau code nếu model lỡ viết thêm.
        code = re.split(r'\n\s*(?:Explanation|Giải thích|Notes?):', code, maxsplit=1)[0].strip()
        return code if "print(" in code else code + "\nprint(0.0)"

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
        - Metadata & cột
        - Preview head(5)
        - Toàn bộ các dòng match với từng từ khóa chỉ tiêu trong câu hỏi
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
                flat_name = f"data/{os.path.basename(path)}"
                preview = [
                    f"--- Table variable: {var_name} (File: {flat_name}) ---",
                    f"Columns: {list(df.columns)}",
                    f"Total rows: {len(df)}",
                    f"Sample rows:\n{df.head(4).to_string()}"
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


    def _build_prompt(self, question, csv_paths, error_log=None):
        preview = self.get_csv_preview(csv_paths, question)
        target_unit = self._detect_unit_request(question)

        error_context = ""
        if error_log:
            error_context = f"""
## LỖI Ở LẦN THỬ TRƯỚC:
{error_log}
Hãy sửa lại code để khắc phục lỗi trên. Kiểm tra kỹ tên cột, chỉ tiêu tìm kiếm và cách tính toán.
"""

        prompt = f"""Bạn là chuyên gia phân tích dữ liệu tài chính Báo cáo tài chính (BCTC) Việt Nam và lập trình Python Pandas.
Nhiệm vụ: Viết code Python Pandas ngắn gọn để tính toán đáp án chính xác cho câu hỏi tài chính.

## CÁC BẢNG DỮ LIỆU ĐÃ ĐƯỢC LOAD SẴN VÀO CÁC BIẾN (DataFrames):
{preview}

## YÊU CẦU BẮT BUỘC:
1. Các biến DataFrame `df1`, `df2`, ... đã được nạp sẵn tương ứng với các bảng trên. KHÔNG cần import lại thư viện, KHÔNG cần dùng pd.read_csv.
2. Tìm chỉ tiêu trong cột 'Chi_tieu' bằng cách so khớp chuỗi không phân biệt hoa thường, ví dụ:
   `df1[df1['Chi_tieu'].str.contains(r'doanh thu thuần', case=False, na=False)]`
3. Lấy giá trị số từ cột 'Gia_tri'. Đảm bảo ép kiểu float: `float(row['Gia_tri'])`.
4. Quy đổi đơn vị theo đúng yêu cầu trong câu hỏi ({target_unit if target_unit else 'theo đơn vị chuẩn'}):
   - Đơn vị gốc của bảng thường là VND (đồng). Nếu câu hỏi yêu cầu "triệu đồng" -> chia 1_000_000.
   - Nếu câu hỏi yêu cầu "tỷ đồng" -> chia 1_000_000_000.
   - Nếu câu hỏi yêu cầu "nghìn tỷ đồng" -> chia 1_000_000_000_000.
   - Nếu câu hỏi về tỷ lệ %, biên lợi nhuận -> tính tỷ số rồi nhân 100.
5. Đối với câu hỏi tính toán đa bước (Tổng nợ = Nợ ngắn hạn + Nợ dài hạn; Biên LN = LNST / DTT * 100; Tăng trưởng = (Năm sau - Năm trước) / Năm trước * 100):
   Trích xuất từng biến thành phần và thực hiện phép tính tương ứng.
6. Kết thúc bằng: `print(answer)` hoặc `answer = ...` (giá trị là số float/int duy nhất).

## VÍ DỤ MẪU:

Ví dụ 1 (Tra cứu đơn):
```python
m = df1[df1['Chi_tieu'].str.contains(r'doanh thu thuần', case=False, na=False)]
val = float(m.iloc[0]['Gia_tri'])
answer = val / 1_000_000_000  # tỷ đồng
print(answer)
```

Ví dụ 2 (Cộng dồn 2 chỉ tiêu):
```python
m_ngan = df1[df1['Chi_tieu'].str.contains(r'nợ ngắn hạn', case=False, na=False)]
m_dai = df1[df1['Chi_tieu'].str.contains(r'nợ dài hạn', case=False, na=False)]
val_ngan = float(m_ngan.iloc[0]['Gia_tri'])
val_dai = float(m_dai.iloc[0]['Gia_tri'])
answer = (val_ngan + val_dai) / 1_000_000_000  # tỷ đồng
print(answer)
```

Ví dụ 3 (Tính tỷ lệ %):
```python
m_lnst = df1[df1['Chi_tieu'].str.contains(r'lợi nhuận sau thuế', case=False, na=False)]
m_dtt = df1[df1['Chi_tieu'].str.contains(r'doanh thu thuần', case=False, na=False)]
lnst = float(m_lnst.iloc[0]['Gia_tri'])
dtt = float(m_dtt.iloc[0]['Gia_tri'])
answer = (lnst / dtt) * 100
print(answer)
```
{error_context}
## CÂU HỎI:
{question}

Hãy viết code Python Pandas bên trong khối ```python ... ```:"""
        return prompt

    def _generate_transformers(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=6144)
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

