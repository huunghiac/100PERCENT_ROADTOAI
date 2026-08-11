# Tổng Quan Đề Bài & Quy Định Cuộc Thi ViFinQA (Text-to-Pandas)`
`
## 1. Giới Thiệu Đề Bài`
- **Bối cảnh:** Tra cứu chỉ số tài chính (doanh thu, lợi nhuận, ROE, ROA,...) từ các Báo cáo tài chính (BCTC) dạng bảng của 100 công ty niêm yết qua 10 năm tại Việt Nam.`
- **Nhiệm vụ cốt lõi:** Xây dựng trợ lý AI có khả năng:`
  1. **Table Retrieval (Truy hồi bảng dữ liệu):** Nhận diện chính xác 1-3 bảng dữ liệu trong kho BCTC chứa thông tin liên quan tới câu hỏi.`
  2. **Text-to-Pandas Query Generation (Sinh truy vấn Pandas):** Hiểu tiếng Việt tài chính, chuyển logic thành code Pandas thực thi được để tính ra đáp án chính xác.`
  3. **Grounding & Minh bạch:** Trích dẫn rõ nguồn gốc bảng dữ liệu tham chiếu (tên công ty, năm, tên bảng).`
`
## 2. Dữ Liệu Ban Tổ Chức Cung Cấp & Quy Định Model`
- **Dữ liệu BCTC:** 100 công ty × 10 năm dạng file OCR .txt.`
- **Dữ liệu kiểm thử (Test set):** 1012 câu hỏi questions.jsonl (không có đáp án chuẩn, BTC giữ bộ đáp án kín).`
- **Giới hạn Mô hình (LLM Constraint):** Chỉ được sử dụng các mô hình LLM nguồn mở **≤ 14B** (ví dụ: Qwen2.5-7B, Qwen2.5-14B, Llama-3-8B). **Tuyệt đối không dùng mô hình đóng API** (GPT-4, Claude, Gemini Pro) ở bất kỳ khâu nào để đảm bảo tính tái lập và hợp lệ khi nộp báo cáo (working notes paper).`
`
## 3. Phương Pháp Đánh Giá (Evaluation Specification)`
`
### 3.1 Truy hồi thông tin (Table Retrieval)`
Hiệu suất hệ thống trên nhiệm vụ truy hồi bảng dữ liệu được đánh giá bằng các chỉ số Độ chính xác (Precision), Độ bao phủ (Recall) và điểm F2 macro. Sử dụng macro-average (tính chỉ số đánh giá cho từng truy vấn rồi lấy trung bình) để tính điểm đánh giá cuối cùng.`
`
- **Độ chính xác (Precision):** Precision = trung bình của (số bảng dữ liệu truy hồi đúng cho mỗi truy vấn) / (số bảng dữ liệu đã truy hồi cho mỗi truy vấn)`
- **Độ bao phủ (Recall):** Recall = trung bình của (số bảng dữ liệu truy hồi đúng cho mỗi truy vấn) / (số bảng dữ liệu liên quan của mỗi truy vấn)`
- **Độ đo F2:** F2 = (5 × Precision × Recall) / (4 × Precision + Recall) (Ưu tiên cao độ bao phủ Recall để tránh bỏ sót bảng căn cứ).`
`
### 3.2 Độ chính xác kết quả (Answer Accuracy)`
Độ chính xác của số liệu đầu ra so với đáp án chuẩn, tính trong ngưỡng sai số cho phép do Ban Tổ chức (BTC) công bố.`
`
- Answer Accuracy = (số query có kết quả khớp đáp án chuẩn, trong ngưỡng sai số) / (tổng số query)`
`
### 3.3 Độ chính xác pandas query (Execution Accuracy)`
Hiệu suất hệ thống trên nhiệm vụ sinh mã truy vấn và tính toán trên bảng dữ liệu tài chính được đánh giá bằng chỉ số Execution Accuracy. Sử dụng macro-average để tính điểm đánh giá cuối cùng.`
`
- Execution Accuracy = (số code chạy được và cho kết quả đúng) / (tổng số query)`
`
## 4. Quy Định Nộp Bài (Submission Format)`
Bài nộp đóng gói dưới dạng file submission.zip:`
``	ext`
submission.zip`
├── submission.json`
└── data/`
    ├── <bảng_1>.csv`
    ├── <bảng_2>.csv`
    └── ...`
``
- submission.json và thư mục data/ phải ở cấp ngoài cùng (không bọc trong thư mục cha).`
- Mọi csv_path trong file .json phải là đường dẫn tương đối bắt đầu bằng data/.`
- Giới hạn: 10 lần nộp/ngày. Vòng Private tối đa 5 lần nộp tổng cộng.`
`
---`
`
## 5. Phương Án Sinh Dữ Liệu Dự Kiến (Synthetic Data Pipeline)`
Do BTC không cung cấp tập dữ liệu Train/Dev, nhóm sẽ áp dụng quy trình tự sinh dữ liệu:`
`
1. **Nguồn dữ liệu gốc:** Kho báo cáo tài chính 100 công ty × 10 năm do BTC cung cấp.`
2. **Quy trình sinh:**`
   - Quét dữ liệu bảng .csv trích xuất được.`
   - Sinh các bộ ba: **(Câu hỏi tiếng Việt, Truy vấn Pandas, Đáp án)** từ đúng bảng nguồn đó.`
   - **Kiểm định tất định (Deterministic Validation):** Đáp án được xác nhận bằng cách thực thi trực tiếp truy vấn Pandas trên bảng nguồn. Tuyệt đối không dùng bất kỳ nhãn nào của tập kiểm thử (questions.jsonl).`
3. **Mô hình khâu sinh:** Chỉ sử dụng **LLM nguồn mở ≤ 14B**, không dùng mô hình đóng.`
