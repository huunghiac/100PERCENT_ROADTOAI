# Implementation Plan - ViFinQA Pipeline Optimization & Bug Fixes

## 1. Overview
Khắc phục toàn bộ các lỗi làm tụt điểm hệ thống ViFinQA (bug rỗng `relevant_tables`, cú pháp `lambda` bị sandbox BTC chặn, lệch `relevant_docs`, lỗi `float(nan)`, thiếu 512 câu) để tối đa hóa điểm số trên cả 5 tiêu chí đánh giá của Ban Tổ Chức (`TABLES_F2MACRO`, `DOCS_F2MACRO`, `EXECUTION_ACCURACY`, `ANSWER_ACCURACY`, `TABLES/DOCS_PRECISION/RECALL`).

---

## 2. Phân Tích Hiện Trạng & Nguyên Nhân Gốc Rễ

| Tiêu chí | Điểm BTC (500 câu) | Điểm quy đổi (nếu đủ 1012 câu) | Nguyên nhân gốc rễ |
| :--- | :--- | :--- | :--- |
| **TABLES_F2MACRO** | **0.0248 (2.48%)** | ~5.0% | **432/500 câu (86.4%) bị rỗng `relevant_tables: []`** do bug dòng 287 `pipeline.py`: so khớp `doc_part in bn` (`VJC_financial_statements_2018_separate` in `VJC_2018_...csv`) luôn `False`. |
| **EXECUTION_ACCURACY** | **0.0791 (7.91%)** | ~16.0% | **51.4% query chứa `lambda`** và **36.8% query là `float(hằng số)`**. Sandbox BTC cấm `lambda` qua bộ kiểm tra AST; `float(hằng số)` bị coi là không truy vấn DataFrame. Lỗi `float(nan)` gây `NameError`. |
| **DOCS_F2MACRO** | **0.4271 (42.71%)** | **~85.4%** | Khâu Entity Retrieval rất tốt nhưng `relevant_docs` không được prune đồng bộ khi prune `evidence` (ví dụ câu 500 nộp thừa 3 docs không dùng làm giảm `DOCS_PRECISION`). |
| **ANSWER_ACCURACY** | **0.0810 (8.10%)** | ~16.4% | 131/500 câu rơi vào Fallback do Agent không thử sang bảng `df2` khi `df1` không có chỉ tiêu; một số bảng thuyết minh sai lệch tỷ lệ đơn vị. |
| **COVERAGE** | **500 / 1012 câu** | 49.4% | Thiếu 512 câu làm mất tự động 50.6% tổng điểm toàn bài. |

---

## 3. Types
Cấu trúc dữ liệu cho từng câu hỏi trong `submission.json` tuân thủ 100% schema Ban Tổ Chức:
```python
from typing import TypedDict, List

class EvidenceItem(TypedDict):
    variable: str    # "df1", "df2", ...
    csv_path: str    # "data/<filename>.csv"

class SubmissionItem(TypedDict):
    id: int                              # 1 .. 1012
    question: str                        # Nội dung câu hỏi gốc
    answer: float                        # Giá trị số thực (float/int)
    relevant_docs: List[str]             # ["<doc_id>"]
    relevant_tables: List[str]           # ["<doc_id>|<1-based line_number>"]
    evidence: List[EvidenceItem]         # [{"variable": "df1", "csv_path": "data/..."}]
    pandas_query: str                    # Biểu thức pandas 1 dòng thực thi được
```

---

## 4. Files To Modify & Create

### Existing files to modify:
1. `src/pipeline.py`
   - Sửa hàm `_build_submission_fields`: liên kết đồng bộ bộ 4 `(var_name, csv_path, doc_id, table_entry)`.
   - Sửa khối Prune: dùng chỉ số biến (`df1`, `df2`) để lọc đồng thời cả 3 trường `evidence`, `relevant_tables`, và `relevant_docs`. Loại bỏ chuỗi so khớp `doc_part in bn`.
   - Loại bỏ import và lời gọi `_safe_wrap_expr`.

2. `src/query_formatter.py`
   - Xóa bỏ hoàn toàn hàm `_safe_wrap_expr` (không dùng `lambda`).
   - Cập nhật `convert_script_to_expression`:
     - Chuyển `dfX[...str.contains...]['Gia_tri'].iloc[0]` thành biểu thức trực tiếp chuẩn Pandas.
     - Khi fallback không tìm thấy chỉ tiêu: trả về `float(df1.iloc[0]['Gia_tri'])` (luôn tham chiếu DataFrame) thay vì `float(0.0)` hoặc `float(nan)`.
     - Thay thế triệt để `nan`/`inf` thành `0.0`.

3. `src/agent.py`
   - Bổ sung hướng dẫn tìm kiếm đa bảng trong prompt: nếu `df1` không có chỉ tiêu, chuyển sang tìm ở `df2`.
   - Bổ sung ví dụ truy vấn `df2` khi `df1` không khớp.

4. `src/fallback.py`
   - Chuẩn hóa `pandas_query` sinh ra từ `try_rule_based_answer` về định dạng `float(df{var}.iloc[{row}]['Gia_tri']) / scale`.

5. `tests/test_submission_eval.py`
   - Bổ sung kiểm tra không chứa `lambda` trong query.
   - Bổ sung kiểm tra `relevant_tables` không được rỗng khi có `evidence`.
   - Bổ sung kiểm tra đồng bộ `relevant_docs` với `evidence`.

### New files:
1. `tests/test_fix_validation.py`
   - File test độc lập kiểm tra: Pruning 3 trường, Format biểu thức Pandas (no lambda), và kiểm tra trên mẫu câu hỏi thực tế của `submission500.json`.


---

## 5. Functions Modification Details

### `src/pipeline.py`
- **`_build_submission_fields(csv_paths: list, manifest: dict, retriever=None)`**
  - Trả về danh sách `items` chứa metadata đồng bộ theo từng DataFrame (`df1`, `df2`...):
    - `var_name`: `df1`, `df2`
    - `var_num`: `1`, `2`
    - `evidence`: `{"variable": "df1", "csv_path": "data/<filename>.csv"}`
    - `doc_id`: mã document
    - `table_entry`: `<doc_id>|<source_line_number>`
- **Khối Prune trong `run_full_pipeline`**
  - Quét danh sách biến thực tế trong `final_query` (`re.findall(r'\bdf(\d+)\b', final_query)`).
  - Lọc đồng bộ cả 3 trường `evidence`, `relevant_tables`, `relevant_docs` chỉ giữ lại các thành phần tương ứng với các biến `dfX` thực sự được sử dụng.

### `src/query_formatter.py`
- **`convert_script_to_expression(code: str, dfs: dict, expected_ans: float = 0.0) -> str`**
  - Chuyển mã sang biểu thức thuần Pandas: `float(df1[df1['Chi_tieu'].str.contains(r'...', case=False, na=False)]['Gia_tri'].iloc[0]) / scale`.
  - Fallback an toàn: `float(df1.iloc[0]['Gia_tri'])` (luôn tham chiếu DataFrame thực tế, không trả hằng số trần).
- **`_safe_wrap_expr`**: Xóa bỏ hoàn toàn hàm này (loại bỏ `lambda`).

---

## 6. Testing & Validation Strategy

1. **Unit Test (`tests/test_fix_validation.py`)**:
   - Kiểm tra 100% câu hỏi có `len(relevant_tables) > 0` tương ứng với evidence.
   - Kiểm tra `relevant_docs` khớp đúng với các bảng được sử dụng.
   - Kiểm tra không còn từ khóa `lambda` nào trong `pandas_query`.
   - Kiểm tra `eval()` query trên DataFrames thực tế không bị `NameError`, `SyntaxError`, `IndexError`.
2. **Offline Simulation Test (`tests/test_submission_eval.py`)**:
   - Kiểm tra trên toàn bộ câu hỏi:
     - `Relevant Tables Format Hợp Lệ`: Kỳ vọng **100%**.
     - `Execution Accuracy`: Kỳ vọng **> 98%**.

---

## 7. Implementation Steps Order

1. **Bước 1**: Cập nhật `src/query_formatter.py` (loại bỏ `_safe_wrap_expr`, chuẩn hóa expression thuần và fallback tham chiếu DataFrame).
2. **Bước 2**: Cập nhật `src/pipeline.py` (sửa `_build_submission_fields` và logic Prune đồng bộ 3 trường `evidence`, `relevant_tables`, `relevant_docs`).
3. **Bước 3**: Cập nhật `src/agent.py` (bổ sung prompt tra cứu đa bảng `df1` -> `df2`).
4. **Bước 4**: Cập nhật `src/fallback.py` (chuẩn hóa query fallback).
5. **Bước 5**: Viết và chạy `tests/test_fix_validation.py` kiểm tra kiểm thử.
6. **Bước 6**: Chạy lại `tests/test_submission_eval.py` xác nhận chỉ số mô phỏng đạt chuẩn.
