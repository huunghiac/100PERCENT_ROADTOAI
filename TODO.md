# Danh Sách Nhiệm Vụ (TODO Checklist)

## Giai đoạn 1: Chuẩn bị Dữ liệu & Mock Baseline (Thành viên Data)
- [x] Tạo file `TEAM_PLAN.md` phân công công việc.
- [x] Tạo file `implementation_plan.md` quy trình kỹ thuật.
- [x] Tạo file `skill.md` bộ quy tắc viết code.
- [x] Tạo file mock CSV mẫu `data/mock_csv/VJC_2018_BaoCaoKetQuaKinhDoanh.csv`.
- [x] Data Engineer baseline `src/data_extractor.py` (Trích xuất bảng HTML từ `.txt` sang `.csv`).

## Giai đoạn 2: Tối ưu Truy hồi Bảng (Thành viên Retrieval)
- [x] Viết baseline Regex bóc tách Mã Cổ Phiếu + Năm trong `src/retriever.py`.
- [ ] Retrieval Engineer: Nâng cấp `src/retriever.py` thêm thuật toán BM25 / Vector Search để khớp từ khóa chỉ số tài chính (Schema Linking).
- [ ] Kiểm thử đo độ chính xác Top-3 Recall trên dữ liệu thật.

## Giai đoạn 3: Phát triển AI Agent & Self-Correction (Lead - Bạn)
- [x] Tích hợp kết nối Ollama local API `deepseek-r1:14b` trong `src/agent.py`.
- [x] Viết hàm Regex bóc tách thẻ `<think>...</think>` và lấy khối code Python.
- [x] Xây dựng cơ chế chạy `exec()` bắt traceback lỗi tự sửa tối đa 3 lần.

## Giai đoạn 4: Ghép nối Pipeline & Đóng gói Bài nộp (Lead - Bạn)
- [x] Viết `src/pipeline.py` hoàn chỉnh: Duyệt `questions.jsonl` -> Gọi `retriever.py` -> Gọi `agent.py`.
- [x] Tự động tạo file `submission.json` chuẩn cấu trúc BTC (id, answer, evidence).
- [x] Tự động đóng gói file `submission.zip` chứa `submission.json` và thư mục `data/`.
- [x] Viết `src/evaluator.py` tính chỉ số Precision, Recall, F2, Execution/Answer Accuracy theo BTC.
- [x] Thêm cơ chế checkpointing cho `src/pipeline.py` khi xử lý file câu hỏi lớn.
- [x] Cập nhật toàn bộ tài liệu hướng dẫn và danh sách kiểm tra.

