# Kế Hoạch Triển Khai Giải Giải ViFinQA (Text-to-Pandas)

## Mục tiêu
Xây dựng AI Agent nhận câu hỏi tài chính (tiếng Việt), truy hồi đúng bảng dữ liệu từ kho BCTC (.txt), sinh mã Pandas tính toán và trả về đáp án đúng. Hạn chế hallucination.

## Cấu trúc thư mục chuẩn
```text
Road-to-AI/
├── data/
│   ├── raw_vifinqa/         # BCTC text tải từ Ban tổ chức (.txt)
│   ├── processed_csv/       # Kết quả bóc tách tự động (.csv)
│   └── mock_csv/            # Data giả để làm độc lập (.csv)
├── src/
│   ├── data_extractor.py    # Code bóc bảng từ OCR txt (Data Engineer)
│   ├── retriever.py         # Code tìm kiếm file CSV (Retrieval Engineer)
│   ├── agent.py             # Code sinh & chạy Pandas (Agent Lead)
│   └── pipeline.py          # Script nối luồng và đóng gói (Lead)
├── questions.jsonl          # 1012 câu hỏi test (Ban tổ chức)
├── submission.json          # File kết quả cuối 
├── submission.zip           # File ZIP nộp bài
└── TEAM_PLAN.md             # File kế hoạch làm việc
```

## Phương pháp làm việc song song (Độc lập, không chờ đợi)

Để tránh nút thắt cổ chai (bottleneck), 3 thành viên làm việc độc lập dựa trên giao thức dữ liệu chung.

### Bước 1: Thống nhất Mock Data & Sinh bộ dữ liệu (Data & Lead)
1. **Tạo Mock Data:** Thành viên Dữ liệu (Data) tạo thủ công 5 file CSV chuẩn xác từ 5 file `.txt` mẫu. Đẩy vào thư mục `data/mock_csv/`. Khung chuẩn (Schema): `Tên chỉ tiêu`, `Giá trị`, `Đơn vị`. 
2. **Sinh bộ dữ liệu nội bộ (Phương án dự kiến):** 
   - Nguồn dữ liệu gốc: Kho BCTC từ BTC.
   - Cách sinh: Sinh tự động các cặp (câu hỏi, truy vấn pandas, đáp án) bằng quy trình nội bộ. Đáp án được kiểm định tất định bằng cách chạy query trên bảng nguồn (không dùng nhãn từ tập test).
   - Mô hình sử dụng: **Chỉ dùng LLM nguồn mở ≤ 14B**. Không dùng mô hình đóng.

### Bước 2: Phát triển độc lập

**Thành viên Dữ liệu (Data Engineer):**
- Không quan tâm câu hỏi. Tập trung viết code trong `data_extractor.py`.
- Đầu vào: `.txt` trong `data/raw_vifinqa/`.
- Đầu ra: File `.csv` trong `data/processed_csv/`. 
- Đảm bảo CSV sinh ra giống với format của `data/mock_csv/`.

**Thành viên Truy hồi (Retrieval Engineer):**
- Nhận mock data. Viết hàm trong `retriever.py`.
- Đầu vào: Câu hỏi `q` và thư mục `data/mock_csv/`.
- Đầu ra: Mảng đường dẫn. Ví dụ: `["data/mock_csv/VJC_2018_BangCanDoi.csv"]`.
- Không cần chờ toàn bộ CSV. Viết thuật toán đúng, khi có data thật thuật toán tự chạy được.

**Thành viên AI Agent & Quản lý (Lead - Bạn):**
- Gắn cứng (hard-code) đường dẫn mock data vào `agent.py`. Không gọi hàm truy hồi.
- Bơm đường dẫn `["data/mock_csv/VJC_2018_BangCanDoi.csv"]` vào LLM.
- Viết vòng lặp thực thi mã Pandas bằng `exec()`. Đóng gói logic tự sửa lỗi (Self-Correction).
- Viết script tạo `submission.json` và nén `submission.zip`.

### Bước 3: Ráp nối (Pipeline)
Khi 3 file code độc lập hoàn tất:
- Chạy `data_extractor.py` xuất toàn bộ CSV ra `data/processed_csv/`.
- Chạy `pipeline.py`: Duyệt 1012 câu hỏi -> Gọi hàm Truy hồi lấy top 3 bảng từ `processed_csv/` -> Gọi hàm Agent sinh code & kết quả.
- Đóng gói file nộp. 

## Lưu ý từ Ban Tổ Chức & Quy định Mô hình
- **Giới hạn mô hình:** Chỉ được phép sử dụng mô hình LLM mã nguồn mở **≤ 14B tham số**. Không dùng mô hình đóng.
- Đường dẫn `csv_path` trong evidence phải là tương đối, bắt đầu bằng `data/`.
- Thiếu file hoặc thiếu câu -> Không được chấm.
- Mỗi ngày được nộp tối đa 10 bài. Private Phase chỉ được 5 bài tổng.