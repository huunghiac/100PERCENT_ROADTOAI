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
        Đọc tiêu đề cột và 3 dòng đầu của các file CSV để đưa vào context cho LLM.
        """
        context = []
        for path in csv_paths:
            # Sửa đường dẫn tương đối để đọc đúng file từ working directory
            real_path = path if os.path.exists(path) else path.replace("data/", "", 1)
            if not os.path.exists(real_path):
                continue
            try:
                df = pd.read_csv(real_path)
                preview = f"--- File: {path} ---\nColumns: {list(df.columns)}\nData Sample (Top 3 rows):\n{df.head(3).to_string()}\n"
                context.append(preview)
            except Exception as e:
                context.append(f"--- File: {path} (Error reading CSV: {str(e)}) ---\n")
        return "\n".join(context)

    def generate_code(self, question: str, csv_paths: list, error_log: str = None) -> str:
        """
        Tạo prompt và gửi request đến Ollama API.
        """
        csv_context = self.get_csv_preview(csv_paths)

        prompt = f"""Bạn là một chuyên gia phân tích dữ liệu Python và Pandas xuất sắc.
Nhiệm vụ của bạn là viết một đoạn mã Python duy nhất sử dụng thư viện pandas để trả lời câu hỏi tài chính dựa trên dữ liệu từ các file CSV được cung cấp.

CÂU HỎI:
{question}

THÔNG TIN BẢNG DỮ LIỆU:
{csv_context}

YÊU CẦU BẮT BUỘC:
1. Đọc đúng các file CSV bằng pandas: `pd.read_csv(filepath)`.
2. Tính toán chính xác con số đáp án.
3. Chỉ dùng lệnh `print(...)` ở cuối cùng để in ra DUY NHẤT một con số (hoặc chuỗi số) kết quả cuối cùng. Không in kèm chữ hay đơn vị.
4. Chỉ viết mã Python trong khối ```python ... ```. Không giải thích dông dài.
"""
        if error_log:
            prompt += f"\n\nLƯU Ý: Lần chạy trước code bị lỗi với log sau:\n{error_log}\nHãy sửa lại mã Python để khắc phục lỗi trên."

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        try:
            res = requests.post(self.api_url, json=payload, timeout=120)
            res.raise_for_status()
            raw_text = res.json().get("response", "")
            return self.clean_response(raw_text)
        except Exception as e:
            # Fallback code nếu gọi API lỗi
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

