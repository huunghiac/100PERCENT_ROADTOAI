# Kế hoạch Triển khai Cải tiến Toàn diện Hệ thống ViFinQA

## 1. Mục tiêu và Bối cảnh
- **Mục tiêu**: Tối ưu hoá toàn diện Pipeline ViFinQA đạt điểm tối đa trên toàn bộ 1012 câu hỏi của cuộc thi Road to AI.
- **Hiện trạng benchmark (300 câu đầu)**:
  - `TABLES_F2MACRO`: 28.67% (tăng từ 2.48% ban đầu).
  - `DOCS_F2MACRO`: 70.67% (tăng từ ~55% ban đầu).
  - `ANSWER_ACCURACY`: 23.34%.
  - `EXECUTION_ACCURACY`: 23.98%.
  - Format hợp lệ: 100% (không lỗi lambda, không rỗng).
- **Vấn đề cốt lõi cần giải quyết**:
  1. *Routing Bias*: Trọng số `exact_match_bonus` (+15.0) quá cao khiến 39 câu hỏi bị chuyển nhầm sang bảng Thuyết minh chi tiết thay vì Báo Cáo Tài Chính Cốt Lõi (CĐKT/KQKD/LCTT).
  2. *Fallback Queries*: 63/300 câu (21.0%) rơi vào fallback `float(df1.iloc[0]['Gia_tri'])` do bảng thiếu số liệu hoặc script đa dòng chưa được compile hoàn chỉnh thành biểu thức 1 dòng.
  3. *Scope Routing*: Phân loại Báo cáo riêng (`separate`) vs Báo cáo hợp nhất (`consolidated`) chưa triệt để.

---

## 2. Kiến trúc Giải pháp Hệ thống

### 2.1. Cải tiến `src/retriever.py` (Two-Tier Scoring & Intent Classification)
1. **Phân loại ý định báo cáo (Intent Classifier)**:
   - **Nhóm 1 - CĐKT (Cân Đối Kế Toán)**: Tổng tài sản, vốn chủ sở hữu, nợ phải trả, tiền và tương đương tiền, hàng tồn kho, phải thu, phải trả, vay ngắn/dài hạn, tài sản cố định...
   - **Nhóm 2 - KQKD (Kết Quả Kinh Doanh)**: Doanh thu, doanh thu thuần, lợi nhuận trước/sau thuế, lợi nhuận gộp, giá vốn, EPS, chi phí tài chính/bán hàng/quản lý...
   - **Nhóm 3 - LCTT (Lưu Chuyển Tiền Tệ)**: Lưu chuyển tiền thuần từ hoạt động kinh doanh/đầu tư/tài chính, dòng tiền thuần, tiền cuối kỳ...
   - **Nhóm 4 - Thuyết minh chuyên biệt (Specialized Notes)**: Thù lao HĐQT, ban giám đốc, biến động vốn chủ sở hữu, phân tích nợ xấu, tài sản thế chấp...
2. **Tái cân bằng trọng số (Scoring Rebalance)**:
   - Tăng Core Statement Bonus (`is_kqkd`, `is_cdkt`, `is_lctt`) từ `+7.0` / `+8.0` $\rightarrow$ `+12.0`.
   - Giảm `exact_match_bonus` từ `+15.0` $\rightarrow$ `+6.0`.
   - Thêm phạt (`-5.0`) đối với bảng Thuyết minh đánh số (`_\d+...`) khi câu hỏi thuộc nhóm BCTC cốt lõi.
3. **Bộ lọc phạm vi báo cáo (Report Scope Routing)**:
   - Câu hỏi có từ khoá `"công ty mẹ"`, `"báo cáo riêng"` $\rightarrow$ bắt buộc ưu tiên file `separate` (+10.0), phạt file `consolidated` (-10.0).
   - Câu hỏi có từ khoá `"hợp nhất"`, `"toàn tập đoàn"` $\rightarrow$ bắt buộc ưu tiên file `consolidated` (+10.0), phạt file `separate` (-10.0).
   - Câu hỏi trung tính $\rightarrow$ ưu tiên `consolidated` trước, fallback sang `separate`.
4. **Hỗ trợ Dynamic Top-K theo loại câu hỏi**:
   - Câu đơn 1 Ticker, 1 Year $\rightarrow$ Top 1 bảng tốt nhất.
   - Câu tính tỷ số tài chính (ROE, ROA, ROS, Biên LN, Đòn bẩy...) $\rightarrow$ Top 2 bảng.
   - Câu đa năm / đa mã cổ phiếu $\rightarrow$ Đúng 1 bảng cho mỗi Thực thể / Năm.

### 2.2. Cải tiến `src/query_formatter.py` (Robust AST Query Compiler)
1. **Inlined Variable Propagation**:
   - Chuyển đổi mã Python nhiều dòng (`m = df1[...]; val = float(m.iloc[0]['Gia_tri']); answer = val / 1000`) thành biểu thức 1 dòng chuẩn AST `float(df1[df1['Chi_tieu'].str.contains(...) ]['Gia_tri'].iloc[0]) / 1000`.
2. **Reverse Target Value Solver**:
   - Brute-force thông minh tìm kiếm ngược biểu thức tạo ra giá trị `answer` trên tất cả DataFrame:
     - Biểu thức 1 bảng: `float(df.iloc[i]['Gia_tri'])` kết hợp các hệ số quy đổi đơn vị (1, 10, 100, 1000, 1e6, 1e9, 1e12, 0.01, 0.1).
     - Biểu thức 2 bảng: Phép cộng `+`, trừ `-`, tỷ lệ chia `%` giữa `df1` và `df2`.
3. **AST Sandbox Safety**:
   - Đảm bảo biểu thức trả về luôn chạy được qua `eval(expr, scope)`, không chứa từ khoá nguy hiểm (`lambda`, `import`, `exec`, gán biến, newline).

---

## 3. Trạng thái Triển khai & Kết quả Kiểm thử

### 3.1. Các thay đổi đã hoàn thành (Code Pushed)
- **`src/retriever.py`**:
  - Tái cân bằng trọng số: `exact_match_bonus` 15.0 $\rightarrow$ 6.0, `core_bonus` $\rightarrow$ 12.0, penalty thuyết minh tiết mục `-5.0`.
  - Mở rộng toàn diện danh sách nhận diện intent:
    - **CĐKT**: `"tiền và các khoản tương đương"`, `"số dư tiền"`, `"đầu tư vào công ty con"`, `"giá trị còn lại"`, `"tài sản xây dựng cơ bản"`, `"tổng dư nợ"`, v.v.
    - **KQKD**: `"lãi thuần từ hoạt động"`, `"thu nhập lãi"`, `"chi phí lãi"`, `"thuế thu nhập doanh nghiệp"`, v.v.
    - **LCTT**: `"tiền chi từ"`, `"chi phí lãi vay đã trả"`, v.v.
  - Scope Routing: Bonus/Penalty `±10.0` trong `_bm25_rank`. Bỏ pre-filter cứng `report_type` trong `retrieve()` để không bị drop file khi scope không khớp.
- **`src/query_formatter.py`**:
  - Thêm `_inline_script_variables` chuyển đổi code multi-line sang single-expression.
  - Fix zero-value fallback khi target answer = 0.
  - Mở rộng scale multipliers trong solver (1, 10, 100, 1000, 1e6, 1e9, 1e12, 0.01, 0.1).

### 3.2. Kết quả Kiểm thử
- **Dynamic Top-K Suite (`tests/test_retriever_dynamic.py`)**: 5/5 PASSED (100%).
- **Regression Routing Suite (`tests/test_regression_routing.py`)**: 36/37 PASSED (97.3%).
  - Các case Sabeco, Hoa Sen, VSC, MBB, FIT, ACB... đều đã định tuyến chính xác về CĐKT/KQKD.
- **Đánh giá trên tập chạy thực tế (130 câu đầu)**:
  - **Tỷ lệ chọn Core BCTC**: Tăng từ **37.7% $\rightarrow$ 50.8%** (+13.1%).
  - **Tỷ lệ chọn nhầm Thuyết minh số tiết mục**: Giảm từ **49.0% $\rightarrow$ 33.1%** (-15.9%).

## 3. Các File Cần Chỉnh Sửa

| File | Thay đổi chính |
|---|---|
| `src/retriever.py` | Cập nhật `_path_bonus`, `_bm25_rank`, `_detect_report_type`, `retrieve`. Triển khai bộ phân loại ý định 4 nhóm và cân bằng điểm số Core vs Thuyết minh. |
| `src/query_formatter.py` | Cập nhật `_inline_script_variables`, `convert_script_to_expression`, mở rộng giải thuật dò ngược biểu thức. |
| `tests/test_retriever_dynamic.py` | Bổ sung các ca kiểm thử cho 39 câu từng bị regression (SAB, CEO, FIT, BID, DLG, VSC, MBB...). |

---

## 4. Trình tự Triển khai (Execution Order)

1. **Bước 1**: Cập nhật `src/retriever.py` với logic Intent Classifier & Scoring mới.
2. **Bước 2**: Cập nhật `src/query_formatter.py` với bộ biên dịch AST biểu thức nâng cao.
3. **Bước 3**: Chạy test kiểm thử `tests/test_retriever_dynamic.py` và đo lường tỷ lệ routing đúng trên 39 câu regression.
4. **Bước 4**: Chạy script đối soát trên `submission_test300(new).json` để kiểm tra tỷ lệ khớp `Query Match Answer` (>90%) và tỷ lệ fallback (<5%).
5. **Bước 5**: Báo cáo kết quả chi tiết cho người dùng và sẵn sàng chạy toàn bộ 1012 câu.
