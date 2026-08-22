import os
import re
import json
import glob
from rank_bm25 import BM25Okapi


class TableRetriever:
    def __init__(self, csv_dir="data/processed_csv",
                 manifest_path="data/processed_csv/_manifest.jsonl"):
        """
        Tìm bảng CSV phù hợp cho câu hỏi bằng 3 tầng:
          Tầng 0 – Entity extraction: ticker (ngoặc đơn > bare match > company name) + year.
          Tầng 1 – Lọc cứng theo ticker + year + report_type (glob filename).
          Tầng 2 – Xếp hạng BM25 trên table_title / table_slug / company_name metadata.
        """
        self.csv_dir = csv_dir
        self.manifest = {}
        self.name_to_ticker = {}
        self.ticker_set = set()
        self._load_manifest(manifest_path)
        self._build_name_index()

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

    # ---- Company-name → ticker index (built once) ----
    def _normalize_name(self, name: str) -> str:
        n = name.strip().lower()
        n = re.sub(r'\s*-\s*(ctcp|tnhh|tjsc)\s*$', '', n)
        n = re.sub(r'^(ctcp|tnhh|tổng công ty cổ phần|tổng công ty|công ty cổ phần|công ty)\s+', '', n)
        return n.strip()

    def _build_name_index(self):
        seen = {}
        for entry in self.manifest.values():
            ticker = entry.get("ticker", "")
            name = entry.get("company_name", "")
            if ticker and name:
                self.ticker_set.add(ticker)
                if ticker not in seen:
                    seen[ticker] = set()
                seen[ticker].add(name)
        for ticker, names in seen.items():
            for raw_name in names:
                self.name_to_ticker[self._normalize_name(raw_name)] = ticker
                self.name_to_ticker[raw_name.lower().strip()] = ticker

    # ---- Noise tickers ----
    _NOISE_TICKERS = {
        "CTCP", "TNHH", "TMCP", "VND", "USD", "BTC", "JSC", "HĐQT",
        "TCTD", "NHNN", "BIDV", "CKPT", "CNTT",
        "EPS", "CFO", "DOH", "LDR", "ROE", "ROA", "NIM", "CIR",
        "CAR", "NPL", "COD",
    }

    def _extract_ticker_from_name(self, question: str):
        q_lower = question.lower()
        best_ticker = None
        best_len = 0
        for name_key, ticker in self.name_to_ticker.items():
            if name_key in q_lower and len(name_key) > best_len:
                best_len = len(name_key)
                best_ticker = ticker
        return best_ticker

    def extract_entities(self, question: str):
        """
        Trích xuất Mã CK và Năm từ câu hỏi.
        Ưu tiên:
          P1: Ticker trong ngoặc đơn (VJC)
          P2: Company name substring match (longest match wins)
          P3: Bare uppercase match known tickers (fallback)
        """
        ticker = None
        # P1: Trong ngoặc đơn
        paren = re.search(r'\(([A-Z][A-Z0-9]{1,3})\)', question)
        if paren and paren.group(1) not in self._NOISE_TICKERS:
            if paren.group(1) in self.ticker_set:
                ticker = paren.group(1)
        # P2: Company name substring match (takes priority over bare ticker)
        if not ticker:
            ticker = self._extract_ticker_from_name(question)
        # P3: Bare uppercase match known tickers
        if not ticker:
            for c in re.findall(r'\b([A-Z][A-Z0-9]{1,3})\b', question):
                if c in self._NOISE_TICKERS:
                    continue
                if c in self.ticker_set:
                    ticker = c
                    break
        # Year
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
        Đầu ra: Danh sách đường dẫn CSV, tối đa top_k.
        """
        ticker, year = self.extract_entities(question)
        report_type = self._detect_report_type(question)

        if not os.path.exists(self.csv_dir) or not os.listdir(self.csv_dir):
            print(f"[Retriever] WARNING: csv_dir '{self.csv_dir}' empty or missing.")
            return []

        # Tầng 1: Glob filter (CSV nằm trong subdir theo ticker)
        if ticker and year:
            matching = glob.glob(f"{self.csv_dir}/{ticker}/{ticker}_{year}_*.csv")
            # Fallback: bỏ year nếu ticker+year không ra
            if not matching:
                matching = glob.glob(f"{self.csv_dir}/{ticker}/{ticker}_*.csv")
        elif ticker:
            matching = glob.glob(f"{self.csv_dir}/{ticker}/{ticker}_*.csv")
        else:
            matching = []

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
        ("Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet (VJC) là bao nhiêu triệu đồng?", "VJC"),
        ("Số dư cho vay khách hàng ngành Thương mại của công ty mẹ Ngân hàng TMCP Á Châu (ACB) cuối năm 2022 là bao nhiêu triệu đồng?", "ACB"),
        ("Lợi nhuận sau thuế của CTCP Chứng khoán FPT năm 2023 là bao nhiêu tỷ đồng?", "FTS"),
        ("Chi phí dự phòng của Ngân hàng TMCP Sài Gòn Tài Lộc trong năm 2020 là bao nhiêu triệu đồng?", "STB"),
        ("Chi phí tài chính của công ty mẹ CTCP Phát triển Sunshine Homes năm 2021 là bao nhiêu triệu đồng?", "SSH"),
        ("Tổng tài sản của STB là bao nhiêu triệu đồng vào cuối năm 2016?", "STB"),
        ("Số dư dự phòng rủi ro cho vay khách hàng của Ngân hàng TMCP Quân đội là bao nhiêu triệu đồng vào cuối năm 2020?", "MBB"),
        ("Tổng giá trị thuần khoản đầu tư góp vốn vào đơn vị khác của Tập đoàn Bảo Việt là bao nhiêu triệu đồng đến ngày 31 tháng 12 năm 2020?", "BVH"),
    ]
    ok = 0
    for q, expected in test_questions:
        ticker, year = retriever.extract_entities(q)
        results = retriever.retrieve(q, top_k=3)
        status = "✓" if ticker == expected else f"✗ (expected {expected})"
        if ticker == expected:
            ok += 1
        print(f"{status} Ticker={ticker} Year={year} | Q: {q[:70]}...")
        print(f"   Results: {results}\n")
    print(f"Entity extraction: {ok}/{len(test_questions)} correct")

