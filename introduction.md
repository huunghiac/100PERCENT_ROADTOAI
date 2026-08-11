````shell
## 1. Bối cảnh
Nhà đầu tư, chuyên viên phân tích và doanh nghiệp tại Việt Nam thường mất nhiều thời gian tra cứu thủ công các chỉ số tài chính (doanh thu, lợi nhuận, ROE, ROA, tỉ lệ nợ/vốn chủ sở hữu, tăng trưởng theo giai đoạn...) nằm rải rác trong hàng trăm báo cáo tài chính (BCTC) dạng bảng của các công ty niêm yết. Trợ lý AI Text-to-Pandas được xây dựng nhằm hỗ trợ tự động hoá việc tra cứu, tổng hợp và tính toán các chỉ số này từ dữ liệu BCTC gốc.

So với các bài toán Text-to-SQL trên dữ liệu tiếng Anh, nguồn tài nguyên và nghiên cứu về Text-to-Pandas trên dữ liệu tài chính tiếng Việt vẫn còn hạn chế.

## 2. Nhiệm vụ cốt lõi
Cuộc thi hướng tới việc xây dựng các hệ thống AI có khả năng:
- **Truy hồi bảng dữ liệu (Table Retrieval):** Xác định đúng bảng dữ liệu chứa số liệu cần thiết từ kho báo cáo tài chính.
- **Sinh truy vấn Pandas (Text-to-Pandas):** Tự động sinh và thực thi câu lệnh pandas trên các bảng đã truy hồi để trả lời chính xác câu hỏi.

## 3. Mục tiêu cụ thể của Hệ thống AI
Các đội thi cần xây dựng hệ thống đáp ứng:
1. **Truy hồi dữ liệu chính xác:** Đúng công ty, năm, bảng. Ưu tiên grounding chính xác trên dữ liệu dạng bảng.
2. **Hiểu truy vấn tài chính bằng tiếng Việt:** Xử lý câu hỏi so sánh, chỉ số dẫn xuất.
3. **Sinh truy vấn pandas & tính toán chính xác:** Code pandas chạy được, đúng schema, trả về đúng số liệu và đơn vị.
4. **Dẫn nguồn minh bạch:** Trích dẫn rõ công ty, năm, tên báo cáo, tên bảng (để kiểm chứng).
5. **Kiểm soát nội dung sai lệch (Hallucination):** Hạn chế AI sinh số liệu/bảng bịa đặt.

## 4. Dữ liệu Cuộc thi
- **Kho báo cáo tài chính:** BCTC OCR định dạng `.txt` của 100 công ty niêm yết trong 10 năm (Bảng CĐKT, KQKD, LCTT, Thuyết minh). Làm nguồn gốc để truy hồi.
- **Bộ dữ liệu kiểm thử (Test set):** File `questions.jsonl`. Gồm id và câu hỏi (Không kèm đáp án). Dùng để chấm điểm.
- **Lưu ý:** BTC KHÔNG cung cấp tập Train/Dev. Không cung cấp pipeline xử lý dữ liệu sẵn. Đội thi tự làm sạch, trích xuất bảng từ `.txt`. Các nguồn dữ liệu hợp pháp khác được phép sử dụng.

## 5. Quy định về Mô hình & Phương pháp Sinh dữ liệu
- **Giới hạn Mô hình:** Chỉ được phép sử dụng mô hình LLM mã nguồn mở có kích thước **≤ 14B tham số** . **Tuyệt đối không dùng mô hình đóng** (như GPT-4, Claude).

- **Phương án sinh dữ liệu dự kiến của đội:**
  1. **Nguồn dữ liệu gốc:** Kho báo cáo tài chính do chính BTC cung cấp (100 công ty × 10 năm).
  2. **Cách sinh:** Tự động sinh cặp (câu hỏi, truy vấn pandas, đáp án) mới trên kho gốc thông qua quy trình sinh đề của đội. Đáp án được kiểm định tất định bằng cách thực thi truy vấn trên đúng bảng nguồn, không dùng bất kỳ nhãn nào của tập kiểm thử.
  3. **Mô hình dùng trong khâu sinh:** Tuân thủ triệt để giới hạn - chỉ dùng LLM nguồn mở ≤ 14B.

## 6. Định dạng Nộp bài (Submission)
- **Cấu trúc nộp:** Phải đóng gói thành 1 file `submission.zip`.
  ```text
  submission.zip
  ├── submission.json
  └── data/
      ├── <bảng_1>.csv
      ├── <bảng_2>.csv
      └── ...
````

- File `.json` và thư mục `data/` phải nằm ngoài cùng (không có thư mục cha).
- Đường dẫn `csv_path` trong `.json` phải là đường dẫn tương đối, bắt đầu bằng `data/`.
- Bài nộp thiếu file CSV hoặc thiếu câu hỏi sẽ __không được chấm__.
- __Giới hạn:__ 10 bài/ngày. Vòng Private (Private Phase) tối đa 5 bài tổng cộng.
- __Quy định chót:__ Kết quả chỉ chính thức khi đội nộp báo cáo mô tả phương pháp (Working notes paper). BTC có quyền kiểm tra và loại bài vi phạm." 
