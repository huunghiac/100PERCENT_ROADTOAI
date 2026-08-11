# Bộ Quy Tắc Viết Code (Code Skills & Constraints)

## 1. Nguyên Tắc Lập Trình (Coding Principles)
- **Code Đầy Đủ (Complete Code):** Không sử dụng `pass`, `...`, hoặc placeholder dở dang. Mọi hàm viết ra phải có logic thực thi hoàn chỉnh.
- **Mô-đun Hóa Dễ Tuning:** Tách biệt rõ ràng các thành phần (`retriever.py`, `agent.py`, `pipeline.py`). Viết docstring tiếng Việt ngắn gọn để đồng đội (Data & Retrieval Engineer) dễ đọc và tự tinh chỉnh sau này.

## 2. Quy Định Chuẩn Ban Tổ Chức (BTC Schema Rules)
- **File Nộp:** `submission.json` và thư mục `data/` đóng gói trực tiếp trong `submission.zip` (không chứa thư mục cha).
- **Cấu Trúc JSON:**
  ```json
  [
    {
      "id": 1,
      "answer": "1500.5",
      "evidence": [
        {"csv_path": "data/mock_csv/VJC_2018_BaoCaoKetQuaKinhDoanh.csv"}
      ]
    }
  ]
  ```
- **Đường Dẫn Evidence:** `csv_path` bắt buộc là đường dẫn tương đối bắt đầu bằng `data/` (dùng dấu `/`, không dùng `\`).

## 3. Tích Hợp Ollama & DeepSeek-R1:14b
- **Endpoint:** Call HTTP API Ollama local (`http://localhost:11434/api/generate`).
- **Xử Lý Model Output:**
  - Bắt buộc dùng Regex loại bỏ toàn bộ thẻ suy luận: `re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)`.
  - Bóc tách mã Python thuần túy nằm trong block ` ```python ... ``` ` hoặc ` ``` ... ``` `.

## 4. Vòng Lặp Tự Sửa Lỗi (Self-Correction Exec Loop)
- Thực thi code sinh ra bằng `exec()`.
- Chặn stdout để lấy kết quả in (`print()`).
- Bắt lỗi Exception: Nếu văng lỗi hoặc kết quả rỗng, gửi lại traceback lỗi cho LLM yêu cầu sửa lại code.
- Cho phép thử lại tối đa 3 lần mỗi câu hỏi.
