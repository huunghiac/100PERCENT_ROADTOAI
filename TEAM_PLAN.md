# Kế Hoạch Triển Khai Giải ViFinQA (Text-to-Pandas)`
`
## Mục tiêu`
Xây dựng AI Agent nhận câu hỏi tài chính (tiếng Việt), truy hồi đúng bảng dữ liệu từ kho BCTC (.txt), sinh mã Pandas tính toán và trả về đáp án đúng. Hạn chế hallucination.`
`
## Quy định Mô hình & Dữ liệu`
- **Giới hạn mô hình:** Chỉ dùng LLM nguồn mở **≤ 14B** (VD: Qwen2.5-7B/14B, Llama-3-8B) cho cả khâu sinh dữ liệu huấn luyện và khâu suy luận Agent. Không dùng mô hình đóng.`
- **Phương án sinh dữ liệu nội bộ (Synthetic Data):**`
  1. *Nguồn:* Kho BCTC 100 công ty × 10 năm của BTC.`
  2. *Cách sinh:* Sinh bộ (Câu hỏi, Truy vấn Pandas, Đáp án) trên các bảng CSV bóc tách. Đáp án được kiểm định tất định bằng cách chạy trực tiếp code Pandas trên bảng gốc. Không dùng nhãn từ tập test (questions.jsonl).`
  3. *Model:* Chỉ dùng LLM nguồn mở ≤ 14B.`
`
## Cấu trúc thư mục chuẩn`
``	ext`
Road-to-AI/`
├── data/`
│   ├── raw_vifinqa/         # BCTC text tải từ Ban tổ chức (.txt)`
│   ├── processed_csv/       # Kết quả bóc tách tự động (.csv)`
│   ├── mock_csv/            # Data giả để làm độc lập (.csv)`
│   └── synthetic_train/     # Tập dữ liệu tự sinh (Synthetic Data)`
├── src/`
│   ├── data_extractor.py    # Code bóc bảng từ OCR txt (Data Engineer)`
│   ├── synthetic_generator.py # Code sinh dữ liệu tự động <=14B (Data/Agent)`
│   ├── retriever.py         # Code tìm kiếm file CSV (Retrieval Engineer)`
│   ├── agent.py             # Code sinh & chạy Pandas (Agent Lead)`
│   └── pipeline.py          # Script nối luồng và đóng gói (Lead)`
├── questions.jsonl          # 1012 câu hỏi test (Ban tổ chức)`
├── submission.json          # File kết quả cuối `
├── submission.zip           # File ZIP nộp bài`
├── TEAM_PLAN.md             # File kế hoạch làm việc`
└── introduction.md          # Đề bài & quy định cập nhật`
``
`
## Phân Công Chi Tiết & Nhiệm Vụ Chưa Làm (Task Allocation)`
`
### 1. Thành viên Dữ liệu (Data Engineer)`
- [ ] **Hoàn thiện src/data_extractor.py:** Viết thuật toán quét regex bóc tách toàn bộ file OCR .txt sang .csv chuẩn (xử lý gộp dòng, lệch cột).`
- [ ] **Tạo dữ liệu kiểm thử nội bộ:** Tạo thủ công/bán tự động 10-20 file .csv chuẩn đặt tại data/mock_csv/ để phục vụ làm mock data cho team.`
- [ ] **Hỗ trợ sinh Synthetic Data:** Viết script đọc các bảng .csv nguồn để chuẩn bị schema cho module sinh câu hỏi.`
`
### 2. Thành viên Truy hồi (Retrieval Engineer)`
- [ ] **Phát triển Entity Extractor (src/retriever.py):** Viết module bóc tách Mã Chứng Khoán (ticker) và Năm (year) từ câu hỏi (dùng Regex hoặc PhoBERT NER).`
- [ ] **Xây dựng Hard-Filter:** Lọc chỉ giữ lại các file .csv trùng Mã CP và Năm.`
- [ ] **Xây dựng Semantic Search:** Cài đặt BM25 hoặc VectorDB (ChromaDB/FAISS với embedding model nhỏ) để so sánh từ khóa khoản mục trong câu hỏi với tiêu đề/cột của bảng.`
- [ ] **Đánh giá Top-3 Recall:** Đánh giá độ chính xác truy hồi bảng trên tập dữ liệu thử nghiệm.`
`
### 3. Thành viên AI Agent & Quản lý (Lead - Bạn)`
- [ ] **Dựng bộ suy luận LLM ≤14B local/API (src/agent.py):** Kết nối framework (Ollama / vLLM / HuggingFace API) chạy model Qwen2.5-7B-Instruct hoặc Qwen2.5-14B.`
- [ ] **Xây dựng Prompt Engine & Code Executor:** Thiết kế prompt cho LLM ≤14B sinh code Pandas. Dùng exec() thực thi code và bắt ngoại lệ.`
- [ ] **Xây dựng vòng lặp Self-Correction (Tự sửa lỗi):** Khi exec() báo lỗi, bắt stderr đưa ngược lại cho LLM sửa (cho phép retry 3-5 lần).`
- [ ] **Xây dựng Module Sinh dữ liệu tự động (src/synthetic_generator.py):** Viết script cho LLM ≤14B sinh cặp (câu hỏi, code pandas, đáp án số) trên các bảng .csv bóc tách được.`
- [ ] **Viết Pipeline Đóng gói (src/pipeline.py):** Đảm bảo tự động hóa hoàn toàn từ đọc questions.jsonl -> Truy hồi -> Sinh code -> Xuất submission.json -> Nén submission.zip.`

Luồng chạy thực tế:

1. Data Extractor: Chuyển `.txt` sang `.csv`.
2. Retriever: Nhận câu hỏi, tìm top 1-3 file `.csv` liên quan.
3. Agent (Chạy cuối): Nhận câu hỏi + `.csv`, sinh mã Pandas, thực thi `exec()`, xuất kết quả ra `submission.json`.

Lúc phát triển (Development): 3 người làm song song. Agent dùng file `data/mock_csv/` code trước. Không cần chờ 2 người kia làm xong mới bắt đầu.
