import os
import re
import json
import glob
from rank_bm25 import BM25Okapi


class TableRetriever:
    def __init__(self, csv_dir="data/processed_csv",
                 manifest_path="data/processed_csv/_manifest.jsonl"):
        """
        Tìm bảng CSV phù hợp cho câu hỏi bằng 2 tầng:
          Tầng 1 – Lọc cứng theo ticker + year (glob filename).
          Tầng 2 – Xếp hạng BM25 trên table_title/table_slug của các file đã lọc.
        """
        self.csv_dir = csv_dir
        self.manifest = {}
        self._load_manifest(manifest_path)

    def _load_manifest(self, path: str):
        if not os.path.exists(path):
            return
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                    csv_path = entry.get("csv_path", "").replace("\\", "/")
                    self.manifest[csv_path] = entry
                except json.JSONDecodeError:
                    continue

    def extract_entities(self, question: str):
        """Trích xuất Mã CK (2-4 ký tự in hoa) và Năm (20xx) từ câu hỏi."""
        paren = re.search(r'\(([A-Z]{2,4})\)', question)
        if paren:
            ticker = paren.group(1)
        else:
            noise = {"CTCP", "TNHH", "TMCP", "VND", "USD", "BTC", "JSC"}
            candidates = re.findall(r'\b([A-Z]{2,4})\b', question)
            ticker = None
            for c in candidates:
                if c not in noise:
                    ticker = c
                    break
        year_match = re.search(r'\b(20\d{2})\b', question)
        year = year_match.group(1) if year_match else None
        return ticker, year

    def _bm25_rank(self, question: str, csv_paths: list, top_k: int) -> list:
        """Xếp hạng csv_paths theo BM25 dựa trên metadata manifest."""
        if not csv_paths:
            return []
        corpus_tokens = []
        valid_paths = []
        for p in csv_paths:
            entry = self.manifest.get(p, {})
            doc = " ".join([
                entry.get("table_title", ""),
                entry.get("table_slug", ""),
                entry.get("company_name", ""),
                entry.get("report_type", ""),
            ])
            tokens = doc.lower().split()
            if tokens:
                corpus_tokens.append(tokens)
                valid_paths.append(p)
        if not corpus_tokens:
            return csv_paths[:top_k]
        bm25 = BM25Okapi(corpus_tokens)
        query_tokens = question.lower().split()
        scores = bm25.get_scores(query_tokens)
        ranked = sorted(zip(valid_paths, scores), key=lambda x: x[1], reverse=True)
        return [p for p, _ in ranked[:top_k]]

    def _detect_report_type(self, question: str):
        """'công ty mẹ' -> separate, 'hợp nhất' -> consolidated, else None."""
        q = question.lower()
        if "công ty mẹ" in q:
            return "separate"
        if "hợp nhất" in q:
            return "consolidated"
        return None

    def retrieve(self, question: str, top_k: int = 3) -> list:
        """
        Đầu vào: Câu hỏi tiếng Việt.
        Đầu ra: Danh sách đường dẫn 'data/...' của file CSV, tối đa top_k.
        """
        ticker, year = self.extract_entities(question)
        report_type = self._detect_report_type(question)

        if not os.path.exists(self.csv_dir) or not os.listdir(self.csv_dir):
            print(f"[Retriever] WARNING: csv_dir '{self.csv_dir}' empty or missing.")
            return []

        # Tầng 1: Glob filter
        if ticker and year:
            matching = glob.glob(f"{self.csv_dir}/{ticker}_{year}_*.csv")
        elif ticker:
            matching = glob.glob(f"{self.csv_dir}/{ticker}_*.csv")
        else:
            matching = glob.glob(f"{self.csv_dir}/*.csv")

        if not matching:
            print(f"[Retriever] No CSV found for ticker={ticker} year={year}")
            return []

        matching = [f.replace("\\", "/") for f in matching]

        # Lọc report_type nếu phát hiện
        if report_type:
            filtered = [f for f in matching if report_type in f]
            if filtered:
                matching = filtered

        # Tầng 2: BM25 ranking
        ranked = self._bm25_rank(question, matching, top_k)
        return ranked if ranked else matching[:top_k]


if __name__ == "__main__":
    retriever = TableRetriever()
    test_questions = [
        "Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet (VJC) là bao nhiêu triệu đồng?",
        "Số dư cho vay khách hàng ngành Thương mại của công ty mẹ Ngân hàng TMCP Á Châu (ACB) cuối năm 2022 là bao nhiêu triệu đồng?",
        "Lợi nhuận sau thuế của CTCP Chứng khoán FPT năm 2023 là bao nhiêu tỷ đồng?",
    ]
    for q in test_questions:
        ticker, year = retriever.extract_entities(q)
        results = retriever.retrieve(q, top_k=3)
        print(f"Q: {q[:60]}...")
        print(f"   Ticker={ticker}  Year={year}")
        print(f"   Results: {results}\n")

