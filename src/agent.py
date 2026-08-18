import json
import pandas as pd
import os
import re
import sys
import io
import traceback
import requests


class PandasAgent:
    def __init__(self, model_name="deepseek-r1:14b", base_url="http://localhost:11434"):
        """
        AI Agent sinh và thực thi mã Pandas dùng Ollama local model.
        Tự động bóc tách thẻ <think>...</think> và hỗ trợ Self-Correction tối đa N lần.
        """
        self.model_name = model_name
        self.api_url = f"{base_url}/api/generate"

    def clean_response(self, text: str) -> str:
        """
        Loại bỏ <think>...</think>.
        Trích xuất code Python trong ```python ... ``` hoặc ``` ... ```.
        """
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
        code_match = re.search(r'```(?:python)?\s*(.*?)\s*```', cleaned, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        return cleaned

    def get_csv_preview(self, csv_paths: list) -> str:
        """Đọc tiêu đề cột và tối đa 30 dòng đầu mỗi CSV."""
        context = []
        for path in csv_paths:
            real_path = path if os.path.exists(path) else path.replace("data/", "", 1)
            if not os.path.exists(real_path):
                continue
            try:
                df = pd.read_csv(real_path)
                n = min(30, len(df))
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
        """Detect đơn vị câu hỏi yêu cầu trả lời."""
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

    def generate_code(self, question: str, csv_paths: list, error_log: str = None) -> str:
        """Tạo prompt few-shot và gửi request đến Ollama API."""
        csv_context = self.get_csv_preview(csv_paths)
        unit_request = self._detect_unit_request(question)

        # Thêm hướng dẫn quy đổi đơn vị dựa vào unit_request
        unit_note = ""
        if unit_request:
            unit_note = f"""
UNIT CONVERSION (CRITICAL):
- The question asks for the answer in: {unit_request}
- Check the Don_vi column or file header to see the CSV's unit (VND, Triệu VND, Nghìn VND, etc.)
- If CSV unit is VND and question asks "triệu đồng": divide by 1,000,000
- If CSV unit is VND and question asks "tỷ đồng": divide by 1,000,000,000
- If CSV unit is VND and question asks "nghìn tỷ đồng": divide by 1,000,000,000,000
- If CSV unit is "Triệu VND"/"Trieu VND" and question asks "tỷ đồng": divide by 1,000
- If CSV unit is "Triệu VND"/"Trieu VND" and question asks "triệu đồng": no conversion needed
- If CSV unit already matches question unit: no conversion needed
- Always round to reasonable precision (2 decimal places for tỷ, 0 for triệu/VND)
"""

        prompt = f"""You are a Python/Pandas expert. Write ONLY Python code to answer the question.

RULES:
- Read CSV: pd.read_csv("exact_file_path")
- CSV has columns: Chi_tieu (indicator name), Gia_tri (numeric value), Don_vi (unit)
- Use str.contains() with case=False, na=False on Chi_tieu column to find rows
- The final line MUST be: print(numeric_answer)
- Print ONLY a single number. No text, no units.
- Keep <think> under 50 words.
- Output code inside ```python ... ``` block.
- If Gia_tri has parentheses like (123), it means negative: -123
{unit_note}
EXAMPLE 1:
Question: Doanh thu thuần năm 2022 của AAA là bao nhiêu tỷ đồng?
File: data/processed_csv/AAA_2022_BaoCaoKetQuaKinhDoanh_separate.csv
```python
import pandas as pd
df = pd.read_csv("data/processed_csv/AAA_2022_BaoCaoKetQuaKinhDoanh_separate.csv")
val = df.loc[df["Chi_tieu"].str.contains("Doanh thu thuan", case=False, na=False), "Gia_tri"].values[0]
unit = df.loc[df["Chi_tieu"].str.contains("Doanh thu thuan", case=False, na=False), "Don_vi"].values[0]
if "VND" == str(unit).strip():
    val = val / 1e9
print(round(val, 2))
```

EXAMPLE 2:
Question: Lợi nhuận sau thuế của ACB năm 2023 là bao nhiêu triệu đồng?
File: data/processed_csv/ACB_2023_BaoCaoKetQuaHoatDong_consolidated.csv (Don_vi = Trieu VND)
```python
import pandas as pd
df = pd.read_csv("data/processed_csv/ACB_2023_BaoCaoKetQuaHoatDong_consolidated.csv")
val = df.loc[df["Chi_tieu"].str.contains("Loi nhuan sau thue", case=False, na=False), "Gia_tri"].values[0]
print(val)
```

NOW SOLVE THIS:
Question: {question}

Available data:
{csv_context}
"""
        if error_log:
            prompt += f"\nPREVIOUS ERROR (fix this):\n{error_log}\n"

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "num_ctx": 4096,
                "num_predict": 512
            }
        }

        try:
            res = requests.post(self.api_url, json=payload, timeout=None)
            res.raise_for_status()
            raw_text = res.json().get("response", "")
            return self.clean_response(raw_text)
        except Exception as e:
            print(f"[Agent Warning] Error calling Ollama API: {e}")
            return "import pandas as pd\nprint(0.0)"

    def execute_code(self, code: str):
        """Thực thi mã Python sinh ra bằng exec() và bắt kết quả từ stdout."""
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

            # Lấy dòng cuối cùng (model có thể print nhiều dòng debug)
            last_line = result.strip().split("\n")[-1].strip()
            return last_line, None
        except Exception:
            sys.stdout = old_stdout
            return None, traceback.format_exc()

    def run_agent(self, question: str, csv_paths: list, max_retries: int = 3):
        """
        Vòng lặp Agent: Sinh code -> Thực thi -> Tự sửa lỗi.
        Trả về: (answer_str, pandas_code_str, error_str_or_None)
        """
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
            print(f"[Agent Self-Correction] Attempt {attempt+1}/{max_retries} failed. Retrying...")

        return "0.0", last_code, f"Failed after {max_retries} retries. Last error: {error_log}"


if __name__ == "__main__":
    agent = PandasAgent()
    # Test unit detection
    test_units = [
        ("Doanh thu thuần năm 2022 là bao nhiêu tỷ đồng?", "tỷ đồng"),
        ("Lợi nhuận sau thuế là bao nhiêu triệu đồng?", "triệu đồng"),
        ("Tỷ lệ sở hữu là bao nhiêu %?", "%"),
        ("Vốn là bao nhiêu nghìn tỷ đồng?", "nghìn tỷ đồng"),
    ]
    print("=== Unit detection test ===")
    for q, expected in test_units:
        result = agent._detect_unit_request(q)
        status = "✓" if result == expected else f"✗ (got {result})"
        print(f"  {status} '{q[:50]}...' -> {result}")
    print("\n=== Agent ready (needs Ollama running for full test) ===")

