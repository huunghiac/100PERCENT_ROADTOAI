import os
import re
import glob

class TableRetriever:
    def __init__(self, csv_dir="data/processed_csv"):
        """
        Nhiệm vụ: Tìm bảng CSV phù hợp dựa trên Mã Chứng Khoán và Năm từ câu hỏi.
        Mô-đun cơ bản cho Retrieval Engineer tự tuning nâng cao (BM25/Vector Search) sau này.
        """
        self.csv_dir = csv_dir

    def extract_entities(self, question: str):
        """
        Trích xuất Mã Chứng Khoán (VD: VNM, VJC) và Năm (VD: 2018, 2023) từ câu hỏi.
        """
        # Bắt mã chứng khoán trong ngoặc đơn hoặc viết hoa 3 chữ cái
        ticker_match = re.search(r'\b([A-Z]{3})\b', question)
        # Bắt năm 4 chữ số
        year_match = re.search(r'\b(20\d{2})\b', question)

        ticker = ticker_match.group(1) if ticker_match else None
        year = year_match.group(1) if year_match else None

        return ticker, year

    def retrieve(self, question: str, top_k: int = 3) -> list:
        """
        Đầu vào: Câu hỏi tiếng Việt.
        Đầu ra: Danh sách đường dẫn tương đối dạng 'data/...' của file CSV.
        """
        ticker, year = self.extract_entities(question)

        # Nếu không có folder processed_csv hoặc rỗng -> Fallback sang mock_csv
        if not os.path.exists(self.csv_dir) or not os.listdir(self.csv_dir):
            return ["data/mock_csv/VJC_2018_BaoCaoKetQuaKinhDoanh.csv"]

        if not ticker or not year:
            # Nếu không bắt được entity, lấy ngẫu nhiên top_k file trong csv_dir làm fallback
            files = glob.glob(f"{self.csv_dir}/*.csv")[:top_k]
            if not files:
                return ["data/mock_csv/VJC_2018_BaoCaoKetQuaKinhDoanh.csv"]
            return [f.replace("\\", "/") for f in files]

        # Khớp file chứa Ticker và Year
        pattern = f"{self.csv_dir}/{ticker}_{year}_*.csv"
        matching_files = glob.glob(pattern)

        if not matching_files:
            # Thử tìm chỉ theo Ticker nếu không thấy cả Ticker + Year
            matching_files = glob.glob(f"{self.csv_dir}/{ticker}_*.csv")

        if not matching_files:
            return ["data/mock_csv/VJC_2018_BaoCaoKetQuaKinhDoanh.csv"]

        # Chuẩn hóa đường dẫn dạng data/...
        results = []
        for f in matching_files[:top_k]:
            clean_path = f.replace("\\", "/")
            if not clean_path.startswith("data/"):
                clean_path = f"data/{clean_path}"
            results.append(clean_path)

        return results

if __name__ == "__main__":
    retriever = TableRetriever()
    test_q = "Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet (VJC) là bao nhiêu triệu đồng?"
    print("Test Retriever Output:", retriever.retrieve(test_q))

