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
        AI Agent sinh và thực thi mã Pandas dùng Ollama local model (DeepSeek-R1:14b).
        Tự động bóc tách thẻ <think>...</think> và hỗ trợ cơ chế Self-Correction tối đa 3 lần.
        """
        self.model_name = model_name
        self.api_url = f"{base_url}/api/generate"

    def clean_response(self, text: str) -> str:
        """
        Loại bỏ phần suy luận trong thẻ <think>...</think> của DeepSeek-R1.
        Trích xuất duy nhất đoạn code Python trong ```python ... ``` hoặc ``` ... ```.
        """
        # 1. Bóc thẻ <think>...</think>
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()

        # 2. Tìm khối code Python
        code_match = re.search(r'```(?:python)?\s*(.*?)\s*```', cleaned, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()
        
        # Fallback nếu model không viết thẻ ```
        return cleaned

    def get_csv_preview(self, csv_paths: list) -> str:
        """
        Đọc tiêu đề cột và tối đa 30 dòng đầu của mỗi file CSV.
        Chỉ gửi preview cho LLM, không gửi toàn bộ file.
        """
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
                    f"Dtypes: {df.dtypes.to_dict()}\n"
                    f"Data (first {n} rows):\n{df.head(n).to_string()}\n"
                )
                context.append(preview)
            except Exception as e:
                context.append(f"--- File: {path} (Error: {e}) ---\n")
        return "\n".join(context)

    def generate_code(self, question: str, csv_paths: list, error_log: str = None) -> str:
        """Tạo prompt few-shot và gửi request đến Ollama API."""
        csv_context = self.get_csv_preview(csv_paths)

        prompt = f"""You are a Python/Pandas expert. Write ONLY Python code to answer the question.

RULES:
- Read CSV files with: pd.read_csv("exact_file_path")
- The final line MUST be: print(numeric_answer)
- Print ONLY a single number. No text, no units, no explanation.
- Keep your <think> reasoning very short (under 50 words).
- Output code inside ```python ... ``` block.

EXAMPLE 1:
Question: Doanh thu thuần năm 2018 của VJC là bao nhiêu tỷ đồng?
File: data/mock_csv/VJC_2018_BaoCaoKetQuaKinhDoanh.csv (Columns: Chi_tieu, Gia_tri, Don_vi)
```python
import pandas as pd
df = pd.read_csv("data/mock_csv/VJC_2018_BaoCaoKetQuaKinhDoanh.csv")
result = df.loc[df["Chi_tieu"].str.contains("Doanh thu thuan", case=False, na=False), "Gia_tri"].values[0]
print(result)
```

EXAMPLE 2:
Question: Lợi nhuận sau thuế năm 2023 của FPT là bao nhiêu tỷ đồng?
File: data/processed_csv/FPT_2023_BaoCaoKetQuaKinhDoanh_consolidated.csv (Columns: Chi_tieu, Gia_tri, Don_vi)
```python
import pandas as pd
df = pd.read_csv("data/processed_csv/FPT_2023_BaoCaoKetQuaKinhDoanh_consolidated.csv")
result = df.loc[df["Chi_tieu"].str.contains("Loi nhuan sau thue", case=False, na=False), "Gia_tri"].values[0]
print(result)
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
        """
        Thực thi mã Python sinh ra bằng exec() và bắt kết quả từ stdout.
        """
        old_stdout = sys.stdout
        new_stdout = io.StringIO()
        sys.stdout = new_stdout

        try:
            # Tạo môi trường thực thi an toàn
            exec_globals = {"pd": pd, "os": os}
            exec(code, exec_globals)
            sys.stdout = old_stdout
            result = new_stdout.getvalue().strip()
            
            if not result:
                return None, "Output rỗng. Vui lòng đảm bảo có lệnh print(kết_quả) ở cuối script."
            return result, None
        except Exception:
            sys.stdout = old_stdout
            return None, traceback.format_exc()

    def run_agent(self, question: str, csv_paths: list, max_retries: int = 3):
        """
        Vòng lặp Agent: Sinh code -> Bóc thẻ <think> -> Thực thi -> Tự sửa lỗi nếu văng Exception.
        """
        if not csv_paths:
            return "0.0", "No CSV files found by retriever."

        error_log = None
        for attempt in range(max_retries):
            code = self.generate_code(question, csv_paths, error_log)
            ans, err = self.execute_code(code)

            if err is None and ans is not None:
                return ans, None

            error_log = err
            print(f"[Agent Self-Correction] Attempt {attempt+1}/{max_retries} failed. Retrying...")

        # Nếu sau 3 lần vẫn lỗi, trả về đáp án mặc định
        return "0.0", f"Failed after {max_retries} retries. Last error: {error_log}"

if __name__ == "__main__":
    agent = PandasAgent()
    mock_paths = ["data/mock_csv/VJC_2018_BaoCaoKetQuaKinhDoanh.csv"]
    question = "Doanh thu thuần của VJC năm 2018 là bao nhiêu tỷ đồng?"
    print("Testing Agent execution on mock path...")
    ans, err = agent.run_agent(question, mock_paths, max_retries=1)
    print(f"Result: {ans} | Error: {err}")

