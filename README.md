# Hướng Dẫn Phát Triển Hệ Thống ViFinQA (Text-to-Pandas)

Dự án tự động hóa việc tra cứu và tính toán chỉ số tài chính từ Báo cáo tài chính (BCTC) Việt Nam dạng OCR (.txt) sang mã Pandas thực thi được.

---

## 1. Cấu Trúc Thư Mục Dự Án

`	ext
Road-to-AI/
├── data/
│   ├── raw_vifinqa/         # Kho BCTC dạng văn bản (.txt) từ Ban Tổ Chức
│   ├── processed_csv/       # Kết quả bóc tách tự động (.csv) từ file .txt
│   ├── mock_csv/            # Dữ liệu giả (Mock CSV) phục vụ làm độc lập
│   └── synthetic_train/     # Tập dữ liệu tự sinh (Synthetic Dataset)
├── src/
│   ├── data_extractor.py    # Module bóc tách bảng từ OCR .txt (Data Engineer)
│   ├── retriever.py         # Module truy hồi file CSV phù hợp (Retrieval Engineer)
│   ├── agent.py             # Module sinh mã Pandas & tự sửa lỗi (Agent Lead)
│   ├── synthetic_generator.py # Module sinh dữ liệu tự động với LLM <= 14B (Lead)
│   └── pipeline.py          # Script nối luồng tự động & nén submission.zip (Lead)
├── questions.jsonl          # 1012 câu hỏi kiểm thử từ BTC (không đáp án)
├── submission.json          # File kết quả đầu ra
├── submission.zip           # File ZIP cuối cùng nộp cho BTC
├── README.md                # Hướng dẫn dự án & Kế hoạch làm việc
└── introduction.md          # Đề bài, luật chơi & phương pháp đánh giá chi tiết
`

---

## 2. Nguyên Tắc Làm Việc 

Để đảm bảo tiến độ, **3 thành viên phát triển độc lập** dựa trên Mock Data (data/mock_csv/). 

### Bước 1: Khởi Tạo Mock Data
- Thành viên Data tạo sẵn 2-5 file .csv làm mẫu đặt tại data/mock_csv/.
- Cấu trúc tiêu chuẩn của bảng CSV: Chỉ tiêu, Giá trị, Đơn vị.

---

## 3. Phân Công Chi Tiết & Hướng Dẫn Bắt Đầu

### 👤 Hồng Hà: Dữ liệu (Data Engineer) -> Làm thêm Mock Data ở bước 1 nữa nha, t mới làm 1 file mãu thôi
- **Nhiệm vụ:** Viết module src/data_extractor.py bóc toàn bộ kho .txt trong data/raw_vifinqa/ thành .csv nằm trong data/processed_csv/.
- **Nhiệm vụ cụ thể:**
  1. Quét dòng trong file .txt, nhận diện khối dòng chứa dữ liệu bảng (số liệu, khoảng trắng, phân cách).
  2. Trích xuất tên bảng, tên công ty, năm báo cáo.
  3. Xử lý nhiễu OCR (gộp dòng rớt chữ, lệch cột).
  4. Đảm bảo format CSV sinh ra giống với data/mock_csv/.
---

### 👤 Khánh Ngọc: Truy hồi (Retrieval Engineer)
- **Nhiệm vụ:** Viết module src/retriever.py nhận vào câu hỏi và trả về 1-3 đường dẫn file .csv phù hợp nhất.
- **Nhiệm vụ cụ thể:**
  1. Bóc tách Mã chứng khoán (Ticker - VD: VNM, VJC) và Năm (Year - VD: 2018, 2023) từ câu hỏi.
  2. Viết bộ lọc cứng (Hard-filter) để thu hẹp phạm vi tìm kiếm trong đúng công ty và năm đó.
  3. Dùng BM25 hoặc VectorDB (ChromaDB/FAISS) khớp từ khóa chỉ số trong câu hỏi với tên bảng/cột.
  4. Trả về danh sách file: ['data/mock_csv/VJC_2018_BaoCaoKetQuaKinhDoanh.csv'].

---

### 👤 Hữu Nghĩa: AI Agent & Trưởng Nhóm (Lead Agent)
- **Nhiệm vụ:** Viết src/agent.py, src/synthetic_generator.py và src/pipeline.py.
- **Nhiệm vụ cụ thể:**
  1. Dựng bộ suy luận LLM nguồn mở **≤ 14B** (Ollama deepseek-r1:14b).
  2. Viết Prompt truyền câu hỏi + thông tin CSV cho LLM sinh code Pandas.
  3. Dùng exec() chạy code Pandas. Nếu có lỗi, bắt stderr gửi lại cho LLM sửa (Self-Correction retry 3-5 lần).
  4. Viết synthetic_generator.py sinh cặp (Câu hỏi, Code Pandas, Đáp án) trên CSV nguồn.
  5. Viết pipeline.py chạy tự động toàn bộ 1012 câu trong questions.jsonl và tự động nén submission.zip.

---

## 4. Quy Trình Ráp Luồng & Nộp Bài (Submission Pipeline)

Khi cả 3 module hoàn thành:
1. Data Engineer chạy data_extractor.py để phủ toàn bộ CSV vào data/processed_csv/.
2. Lead chạy pipeline.py:
   - Đọc từng câu hỏi trong questions.jsonl.
   - retriever.py tìm top 3 bảng .csv.
   - agent.py sinh mã Pandas & tính kết quả.
   - Ghi kết quả vào submission.json.
   - Đóng gói file submission.zip chứa submission.json và thư mục data/ chứa các CSV liên quan.
3. Tải file submission.zip lên trang cuộc thi.

---

## 5. Quy Định Quan Trọng 

1. **Giới hạn LLM ≤ 14B:** Chỉ dùng mô hình open-source ≤ 14B (deepseek-r1:14b). Không dùng API OpenAI/Claude.
2. **Quy chuẩn Zip:** submission.json và thư mục data/ phải nằm ở cấp ngoài cùng của file ZIP (không nằm trong thư mục bọc ngoài).
3. **Giới hạn nộp:** Tối đa 10 lần/ngày. Vòng Private tối đa 5 lần.
