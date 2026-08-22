# Hướng Dẫn Phát Triển Hệ Thống ViFinQA (Text-to-Pandas)

Dự án tự động hóa việc tra cứu và tính toán chỉ số tài chính từ Báo cáo tài chính (BCTC) Việt Nam dạng OCR (.txt) sang mã Pandas thực thi được.

---

## 1. Cấu Trúc Thư Mục Dự Án

```text
Road-to-AI/
├── data/
│   ├── raw_vifinqa/         # Kho BCTC dạng văn bản (.txt) và questions.jsonl từ Ban Tổ Chức
│   ├── processed_csv/       # Kết quả bóc tách tự động (.csv) từ file .txt
│   ├── mock_csv/            # Dữ liệu giả cũ, không dùng cho đáp án cuối
│   └── synthetic_train/     # Tập dữ liệu tự sinh (Synthetic Dataset)
├── src/
│   ├── data_extractor.py    # Module bóc tách bảng từ OCR .txt (Data Engineer)
│   ├── retriever.py         # Module truy hồi file CSV phù hợp (Retrieval Engineer)
│   ├── agent.py             # Module sinh mã Pandas & tự sửa lỗi (Agent Lead)
│   └── pipeline.py          # Script nối luồng tự động & nén submission.zip (Lead)
├── data/raw_vifinqa/questions.jsonl # 1012 câu hỏi kiểm thử từ BTC (không đáp án)
├── submission.json          # File kết quả đầu ra
├── submission.zip           # File ZIP cuối cùng nộp cho BTC
├── README.md                # Hướng dẫn dự án & Kế hoạch làm việc
└── introduction.md          # Đề bài, luật chơi & phương pháp đánh giá chi tiết
```

---

## 2. Nguyên Tắc Làm Việc

Để đảm bảo tiến độ, **3 thành viên phát triển độc lập** dựa trên Mock Data (data/mock_csv/).

### Bước 1: Khởi Tạo Mock Data
- Thành viên Data tạo sẵn 2-5 file `.csv` làm mẫu đặt tại `data/mock_csv/`.
- Cấu trúc tiêu chuẩn của bảng CSV: `Chỉ tiêu`, `Giá trị`, `Đơn vị`.

---

## 3. Phân Công Chi Tiết & Hướng Dẫn Bắt Đầu

### 👤 Hồng Hà: Dữ liệu (Data Engineer) -> Làm thêm Mock Data ở bước 1 nữa nha, t mới làm 1 file mẫu thôi
- **Nhiệm vụ:** Viết module `src/data_extractor.py` bóc toàn bộ kho `.txt` trong `data/raw_vifinqa/` thành `.csv` nằm trong `data/processed_csv/`.
- **Nhiệm vụ cụ thể:**
  1. Quét dòng trong file `.txt`, nhận diện khối dòng chứa dữ liệu bảng (số liệu, khoảng trắng, phân cách).
  2. Trích xuất tên bảng, tên công ty, năm báo cáo.
  3. Xử lý nhiễu OCR (gộp dòng rớt chữ, lệch cột).
  4. Đảm bảo format CSV sinh ra giống với `data/mock_csv/`.

---

### 👤 Khánh Ngọc: Truy hồi (Retrieval Engineer)
- **Nhiệm vụ:** Viết module `src/retriever.py` nhận vào câu hỏi và trả về 1-3 đường dẫn file `.csv` phù hợp nhất.
- **Nhiệm vụ cụ thể:**
  1. Bóc tách Mã chứng khoán (Ticker - VD: VNM, VJC) và Năm (Year - VD: 2018, 2023) từ câu hỏi.
  2. Viết bộ lọc cứng (Hard-filter) để thu hẹp phạm vi tìm kiếm trong đúng công ty và năm đó.
  3. Dùng BM25 hoặc VectorDB (ChromaDB/FAISS) khớp từ khóa chỉ số trong câu hỏi với tên bảng/cột.
  4. Trả về danh sách file: `["data/mock_csv/VJC_2018_BaoCaoKetQuaKinhDoanh.csv"]`.

---

### 👤 Hữu Nghĩa: AI Agent & Trưởng Nhóm (Lead Agent)
- **Nhiệm vụ:** Viết `src/agent.py`, `src/synthetic_generator.py` và `src/pipeline.py`.
- **Nhiệm vụ cụ thể:**
  1. Dựng bộ suy luận LLM nguồn mở **≤ 14B** (Ollama deepseek-r1:14b).
  2. Viết Prompt truyền câu hỏi + thông tin CSV cho LLM sinh code Pandas.
  3. Dùng `exec()` chạy code Pandas. Nếu có lỗi, bắt stderr gửi lại cho LLM sửa (Self-Correction retry 3-5 lần).
  4. Viết `synthetic_generator.py` sinh cặp (Câu hỏi, Code Pandas, Đáp án) trên CSV nguồn.
  5. Viết `pipeline.py` chạy tự động toàn bộ 1012 câu trong `questions.jsonl` và tự động nén `submission.zip`.

---

## 4. Quy Trình Ráp Luồng & Nộp Bài (Submission Pipeline)

Khi cả 3 module hoàn thành:
1. Data Engineer chạy `data_extractor.py` để phủ toàn bộ CSV vào `data/processed_csv/`.
2. Lead chạy `pipeline.py`:
   - Đọc từng câu hỏi trong `data/raw_vifinqa/questions.jsonl`.
   - `retriever.py` tìm top 1-3 bảng `.csv` liên quan.
   - `agent.py` sinh mã Pandas và tính kết quả.
   - Ghi kết quả vào `submission.json` theo đúng schema bên dưới.
   - Đóng gói `submission.zip` chứa `submission.json` và thư mục `data/` chứa đầy đủ CSV được tham chiếu.
3. Tải file `submission.zip` lên trang cuộc thi.

### 4.1 Cấu trúc `submission.zip`

```text
submission.zip
├── submission.json
└── data/
    ├── <bảng_1>.csv
    ├── <bảng_2>.csv
    └── ...
```

Yêu cầu:
- `submission.json` và `data/` phải nằm ở root của file ZIP.
- `data/` phải chứa đầy đủ CSV được tham chiếu bởi `evidence[].csv_path`.
- `csv_path` trong JSON phải là đường dẫn tương đối bắt đầu bằng `data/`.

### 4.2 Cú pháp `submission.json`

```json
[
  {
    "id": 1,
    "question": "Doanh thu thuần của Công ty CP Sữa Việt Nam (VNM) năm 2023 là bao nhiêu?",
    "answer": 63075000000.0,
    "relevant_docs": ["AAA_financial_statements_2015_consolidated"],
    "relevant_tables": ["AAA_financial_statements_2015_consolidated|350"],
    "evidence": [
      {
        "variable": "df1",
        "csv_path": "data/AAA_financial_statements_2015_consolidated_table_1.csv"
      }
    ],
    "pandas_query": "df1[(df1.company=='VNM') & (df1.year==2023)]['net_revenue'].values[0]"
  }
]
```

Trường dữ liệu:
- `id`: Mã định danh câu hỏi, kiểu `integer`.
- `question`: Nội dung câu hỏi tài chính, kiểu `string`.
- `answer`: Kết quả số liệu, kiểu `float`.
- `relevant_docs`: Danh sách mã báo cáo liên quan. Mã báo cáo lấy từ tên file/thư mục cuối trong đường dẫn tài liệu sau khi bỏ `.txt`. Ví dụ `ocr_filter\AAA\2015\AAA_financial_statements_2015_consolidated` → `AAA_financial_statements_2015_consolidated`.
- `relevant_tables`: Danh sách bảng liên quan trực tiếp, định dạng `<id_báo_cáo>|<vị trí bảng trong báo cáo>`. Ví dụ `AAA_financial_statements_2015_consolidated|350`.
- `evidence`: Danh sách CSV dùng để chạy `pandas_query`.
  - `variable`: Tên biến DataFrame hợp lệ Python, không trùng nhau trong cùng câu hỏi, ví dụ `df1`, `df2`.
  - `csv_path`: Đường dẫn tương đối tới CSV trong thư mục `data/` của gói nộp bài.
- `pandas_query`: Câu lệnh Pandas dạng `string`, có thể chạy lại trên dữ liệu đã chuẩn hoá để tạo ra `answer`.

---

## 5. Quy Định Quan Trọng

1. **Giới hạn LLM ≤ 14B:** Chỉ dùng mô hình open-source ≤ 14B (deepseek-r1:14b). Không dùng API OpenAI/Claude.
2. **Quy chuẩn Zip:** `submission.json` và thư mục `data/` phải nằm ở cấp ngoài cùng của file ZIP (không nằm trong thư mục bọc ngoài).
3. **Giới hạn nộp:** Tối đa 10 lần/ngày. Vòng Private tối đa 5 lần.
