# KẾ HOẠCH TRIỂN KHAI TỐI ƯU HÓA PIPELINE VIFINQA (DYNAMIC TOP-K & TARGETED RETRIEVAL)

## I. TỔNG QUAN HIỆN TRẠNG & PHÂN TÍCH 1012 CÂU HỎI

### 1. Phân bố 1012 câu hỏi thực tế
- **Đơn Ticker, Đơn Năm (44.5% - 450 câu)**: Chỉ tiêu đơn lẻ trong 1 báo cáo tài chính.
- **1 Ticker, Đa Năm (28.0% - 283 câu)**: Tăng trưởng, biến động, so sánh qua các năm (ví dụ: VHM 2018-2022).
- **Đa Ticker, Đơn Năm (20.4% - 206 câu)**: So sánh giữa các công ty trong cùng năm (ví dụ: SSI, HCM, VND năm 2021).
- **Đa Ticker, Đa Năm (7.2% - 73 câu)**: So sánh phức hợp nhiều công ty qua nhiều thời kỳ.

### 2. Các điểm nghẽn đã xác định
- **Top-K cố định gây nhiễu**: Gán cứng `top_k=2` khiến câu đa năm (4 năm) lấy tới 8 bảng gây tràn prompt, trong khi câu đơn lại bị thừa 1 bảng rác làm giảm `TABLES_PRECISION`.
- **BM25 nhầm lẫn giữa Báo cáo chính và Thuyết minh**: BM25 đếm token đơn lẻ khiến bảng Thuyết minh (nhiều từ lặp) vượt điểm bảng Báo cáo chính (CĐKT/KQKD/LCTT) chứa chỉ tiêu gốc.
- **Lệch bảng dẫn đến sai đáp số**: Khi Retriever chọn sai bảng, Agent cố gắng đoán hoặc fallback, làm giảm cả `TABLES_F2MACRO` và `ANSWER_ACCURACY`.

---

## II. THIẾT KẾ KIẾN TRÚC CẢI TIẾN

### 1. `src/retriever.py`: Bộ định tuyến thông minh & Dynamic Top-K (3 Tầng)

#### Tầng 1: Exact Indicator Match (Khớp cụm từ chỉ tiêu trực tiếp)
- Trích xuất cụm từ chỉ tiêu sau khi loại bỏ Ticker, Năm, Stopwords.
- Quét nhanh trong cache cột `Chi_tieu` của toàn bộ CSV thuộc Ticker + Năm.
- Nếu tìm thấy bảng chứa chính xác cụm từ chỉ tiêu (hoặc chuỗi con độ dài >= 6 ký tự) $\rightarrow$ Boost điểm cực đại (+15.0), đưa thẳng lên Top 1.

#### Tầng 2: Core Financial Statement Prioritizer (Ưu tiên Báo cáo chính)
- Nhận diện chỉ tiêu vĩ mô kinh điển:
  - **KQKD**: Doanh thu thuần, Lợi nhuận gộp, Lợi nhuận sau thuế, Chi phí tài chính, Chi phí bán hàng, Chi phí QLDN, Lãi cơ bản trên cổ phiếu (EPS).
  - **CĐKT**: Tổng tài sản, Nợ phải trả, Vốn chủ sở hữu, Tiền và tương đương tiền, Hàng tồn kho, Phải thu ngắn hạn, Vay ngắn hạn/dài hạn, Vốn cổ phần.
  - **LCTT**: Lưu chuyển tiền thuần từ hoạt động kinh doanh/đầu tư/tài chính.
- Khi gặp các chỉ tiêu này, ưu tiên tuyệt đối file `BangCanDoiKeToan`, `BaoCaoKetQuaKinhDoanh`, `BaoCaoLuuChuyenTienTe` thay vì các bảng thuyết minh lẻ.

#### Tầng 3: Intent-Aware Adaptive Dynamic Top-K
- **Đơn Ticker, Đơn Năm**:
  - Có Exact Match hoặc BM25 score top 1 vượt trội (margin > 30% so với top 2): Trả về đúng **1 bảng** (`top_k=1`).
  - Câu hỏi Tỷ số tài chính (ROE, ROA, Biên LN, Đòn bẩy, D/E): Lấy đúng **2 bảng** (1 KQKD + 1 CĐKT).
  - BM25 phân vân (margin <= 30%): Trả về **2 bảng** (`top_k=2`).
- **1 Ticker, Đa Năm ($N$ năm)**:
  - Xác định loại báo cáo phù hợp (ví dụ: KQKD).
  - Lấy đúng **1 bảng cùng loại tốt nhất cho MỖI năm** $\rightarrow$ Tổng trả về đúng $N$ bảng (`df1`..`dfN`).
- **Đa Ticker ($M$ tickers), Đơn Năm**:
  - Xác định loại báo cáo phù hợp.
  - Lấy đúng **1 bảng cùng loại tốt nhất cho MỖI ticker** $\rightarrow$ Tổng trả về đúng $M$ bảng.
- **Đa Ticker, Đa Năm**:
  - Lấy đúng **1 bảng cùng loại cho mỗi cặp (Ticker, Năm)**.

---

### 2. `src/agent.py`: Nâng cấp Prompt & Code Generator cho Đa DataFrames
- **Nhận diện ngữ cảnh động**: Cung cấp mô tả biến DataFrame rõ ràng trong Prompt:
  - `df1: [Ticker A - Năm 2021 - Báo cáo X]`
  - `df2: [Ticker A - Năm 2022 - Báo cáo X]`
- **Hỗ trợ biểu thức tính toán liên bảng**:
  - Công thức tăng trưởng: `(float(df2[...]...['Gia_tri'].iloc[0]) - float(df1[...]...['Gia_tri'].iloc[0])) / float(df1[...]...['Gia_tri'].iloc[0]) * 100`
  - Công thức tỷ số: `float(df1[...]...['Gia_tri'].iloc[0]) / float(df2[...]...['Gia_tri'].iloc[0])`
- **Chống lỗi gãy biểu thức AST**:
  - Đảm bảo cú pháp Python hợp lệ 100%, không lambda, không hàm ngoài built-in/pandas.

---

### 3. `src/pipeline.py` & `src/query_formatter.py`: Đồng bộ Pruning & Fallback
- **Pruning chuẩn xác**:
  - Regex trích xuất tất cả `df\d+` xuất hiện trong `pandas_query`.
  - Giữ lại đúng các bảng và doc_ids tương ứng trong `relevant_tables` và `relevant_docs`.
  - Bảng nào không được dùng để tính kết quả sẽ tự động bị loại khỏi danh sách nộp.
- **Fallback an toàn**:
  - Nếu Agent fail cả 2 lần: Fallback tự động trích xuất dòng khớp tốt nhất trên `df1` và sinh query hợp lệ `float(df1[...]['Gia_tri'].iloc[0])`.

---

## III. KẾ HOẠCH THỰC HIỆN & KIỂM THỬ

### Bước 1: Triển khai cải tiến `src/retriever.py`
- Tích hợp Exact Indicator Match + Core Financial Statement Priority.
- Cài đặt cơ chế Dynamic Adaptive Top-K.

### Bước 2: Triển khai cập nhật `src/agent.py` & `src/pipeline.py`
- Cập nhật prompt template hỗ trợ linh hoạt 1 đến $N$ DataFrames.
- Kiểm tra tính tương thích của AST validator và Pruning module.

### Bước 3: Kiểm thử toàn diện trên bộ Test Suite
- Chạy unit tests: `tests/test_fix_validation.py`, `tests/test_pipeline_prune.py`.
- Viết test suite chuyên biệt `tests/test_retriever_dynamic.py` kiểm tra:
  - 10 câu đơn ticker/năm (kết quả phải trả về 1 bảng chính xác).
  - 10 câu đa năm (trả về đúng 1 bảng/năm).
  - 10 câu đa ticker (trả về đúng 1 bảng/ticker).
  - 10 câu tỷ số (trả về đúng 1 KQKD + 1 CĐKT).

### Bước 4: Chạy End-to-End Pipeline & Đóng gói
- Chạy pipeline trên toàn bộ 1012 câu hỏi.
- Validate `submission.json` qua `tests/test_submission_eval.py`.
- Đóng gói file nộp bài `submission.zip`.
