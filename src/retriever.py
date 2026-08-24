import os
import re
import json
import glob
import pandas as pd
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

    def extract_all_entities(self, question: str):
        """
        Trích xuất TẤT CẢ Tickers và Năm từ câu hỏi (hỗ trợ so sánh nhiều công ty / nhiều năm).
        """
        tickers = []
        q_lower = question.lower()

        # 1. Tickers trong ngoặc: (VJC), (ACB)
        parens = re.findall(r'\(([A-Z][A-Z0-9]{1,3})\)', question)
        for p in parens:
            if p not in self._NOISE_TICKERS and p in self.ticker_set and p not in tickers:
                tickers.append(p)

        # 2. Match company names
        for name_key, ticker in self.name_to_ticker.items():
            if name_key in q_lower and ticker not in tickers:
                tickers.append(ticker)

        # 3. Bare uppercase match (e.g. "nhóm MSN, MCH, DBC, ASM và OGC")
        for c in re.findall(r'\b([A-Z][A-Z0-9]{1,3})\b', question):
            if c not in self._NOISE_TICKERS and c in self.ticker_set and c not in tickers:
                tickers.append(c)

        # 4. Years: 2016-2020 range hoặc các năm riêng lẻ
        years = []
        # Pattern 2016-2020
        range_match = re.search(r'\b(20\d{2})\s*[-–]\s*(20\d{2})\b', question)
        if range_match:
            start_y, end_y = int(range_match.group(1)), int(range_match.group(2))
            if start_y <= end_y and (end_y - start_y) <= 10:
                for y in range(start_y, end_y + 1):
                    years.append(str(y))

        # Individual years
        for y in re.findall(r'\b(20\d{2})\b', question):
            if y not in years:
                years.append(y)

        # Primary single ticker & year for backward compatibility
        ticker = tickers[0] if tickers else None
        year = years[0] if years else None
        return ticker, year, tickers, years

    def extract_entities(self, question: str):
        ticker, year, _, _ = self.extract_all_entities(question)
        return ticker, year

    _QUESTION_STOPWORDS = {
        "của", "cho", "và", "vào", "cuối", "trong", "năm", "là", "bao", "nhiêu",
        "đồng", "triệu", "tỷ", "nghìn", "ngày", "tháng", "đến", "tại", "với",
        "công", "ty", "ctcp", "tnhh", "tmcp", "ngân", "hàng", "tổng", "tập", "đoàn",
        "mẹ", "hợp", "nhất", "riêng", "báo", "cáo", "đơn", "vị",
    }

    def _tokenize(self, text: str) -> list:
        return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)

    def _clean_query_tokens(self, question: str) -> list:
        """Giữ token chỉ tiêu, bỏ ticker/year/company/unit/question filler đã được filter ở tầng 1."""
        ticker, year = self.extract_entities(question)
        q = question.lower()
        if ticker:
            q = re.sub(rf"\b{re.escape(ticker.lower())}\b", " ", q)
            for name_key, mapped_ticker in self.name_to_ticker.items():
                if mapped_ticker == ticker:
                    q = q.replace(name_key, " ")
        if year:
            q = q.replace(year, " ")
        tokens = [t for t in self._tokenize(q) if t not in self._QUESTION_STOPWORDS and len(t) > 1]
        return tokens or self._tokenize(question)

    def _path_bonus(self, question_tokens: list, path: str) -> float:
        """Generic financial-statement prior from indicator intent, not question ID."""
        p = path.lower()
        qt = set(question_tokens)
        bonus = 0.0
        if {"lợi", "nhuận"} & qt or {"doanh", "thu"} <= qt or "chi" in qt:
            if "baocaoketqua" in p or "ketquakinhdoanh" in p or "ketquahoatdong" in p:
                bonus += 3.0
        # Chỉ boost LCTT khi câu hỏi thật sự hỏi "lưu chuyển"; token "tiền" đơn lẻ quá nhiễu.
        if {"lưu", "chuyển"} <= qt or "luuchuyen" in qt:
            if "luuchuyentiente" in p or "lưuchuyểntiềntệ" in p:
                bonus += 5.0
        if "tài" in qt and ("sản" in qt or "san" in p):
            if "bangcandoi" in p or "tinhhinhtaichinh" in p:
                bonus += 3.0
        if "dự" in qt or "phòng" in qt or "duphong" in p:
            if "duphong" in p or "chiphihoatdong" in p:
                bonus += 2.0
        if "vay" in qt or "cho" in qt:
            if "chovay" in p or "khachhang" in p:
                bonus += 2.0
        return bonus

    def _bm25_rank(self, question: str, csv_paths: list, top_k: int) -> list:
        """Xếp hạng csv_paths theo BM25 đã clean query + boost nội dung Chi_tieu."""
        if not csv_paths:
            return []
        query_tokens = self._clean_query_tokens(question)
        corpus_tokens = []
        valid_paths = []
        chi_tieu_cache = {}
        for p in csv_paths:
            entry = self.manifest.get(p, {})
            metadata = " ".join([
                entry.get("table_title", ""),
                entry.get("table_slug", ""),
                entry.get("report_type", ""),
            ])
            chi_tieu_text = ""
            real_path = p if os.path.exists(p) else p.replace("data/", "", 1)
            if os.path.exists(real_path):
                try:
                    df = pd.read_csv(real_path, usecols=["Chi_tieu"], nrows=80)
                    values = df["Chi_tieu"].dropna().astype(str).tolist()
                    chi_tieu_text = " ".join(values)
                    chi_tieu_cache[p] = " ".join(values).lower()
                except Exception:
                    chi_tieu_cache[p] = ""
            # Chi_tieu weighted higher than metadata because user asks about indicators.
            doc = f"{metadata} {chi_tieu_text} {chi_tieu_text} {chi_tieu_text}"
            tokens = self._tokenize(doc)
            if tokens:
                corpus_tokens.append(tokens)
                valid_paths.append(p)
        if not corpus_tokens:
            return csv_paths[:top_k]

        bm25 = BM25Okapi(corpus_tokens)
        base_scores = bm25.get_scores(query_tokens)
        scored = []
        for p, score in zip(valid_paths, base_scores):
            text = chi_tieu_cache.get(p, "")
            bonus = self._path_bonus(query_tokens, p)
            hits = sum(1 for t in query_tokens if t in text)
            if query_tokens and hits == len(set(query_tokens)):
                bonus += 6.0
            elif hits >= max(2, len(set(query_tokens)) // 2):
                bonus += 2.0
            scored.append((p, float(score) + bonus))
        ranked = sorted(scored, key=lambda x: x[1], reverse=True)
        return [p for p, _ in ranked[:top_k]]

    def _detect_report_type(self, question: str):
        """'công ty mẹ' -> separate, 'hợp nhất' -> consolidated, else None."""
        q = question.lower()
        if "công ty mẹ" in q:
            return "separate"
        if "hợp nhất" in q:
            return "consolidated"
        return None

    def retrieve(self, question: str, top_k: int = None) -> list:
        """
        Đầu vào: Câu hỏi tiếng Việt.
        Đầu ra: Danh sách đường dẫn CSV (dynamic / adaptive).
        - Nếu 1 ticker, 1 năm: trả về top 1-2 bảng chính xác nhất (tránh nhiễu context).
        - Nếu nhiều ticker hoặc nhiều năm (multi-year/multi-company): tự động lấy top bảng cho từng thực thể.
        """
        _, _, tickers, years = self.extract_all_entities(question)
        report_type = self._detect_report_type(question)

        if not os.path.exists(self.csv_dir) or not os.listdir(self.csv_dir):
            print(f"[Retriever] WARNING: csv_dir '{self.csv_dir}' empty or missing.")
            return []

        # TH1: Multi-company hoặc Multi-year -> Dynamic gather
        is_multi = len(tickers) > 1 or len(years) > 1
        if is_multi:
            target_tickers = tickers if tickers else [None]
            target_years = years if years else [None]
            gathered_paths = []

            for t in target_tickers:
                for y in target_years:
                    matching = []
                    if t and y:
                        matching = glob.glob(f"{self.csv_dir}/{t}/{t}_{y}_*.csv")
                        if not matching:
                            matching = glob.glob(f"{self.csv_dir}/{t}/{t}_*.csv")
                    elif t:
                        matching = glob.glob(f"{self.csv_dir}/{t}/{t}_*.csv")
                    elif y:
                        matching = glob.glob(f"{self.csv_dir}/*/*_{y}_*.csv")

                    matching = [f.replace("\\", "/") for f in matching]
                    if report_type:
                        filtered = [f for f in matching if report_type in f]
                        if filtered:
                            matching = filtered

                    # Lấy top 1 bảng tốt nhất cho từng (ticker, year)
                    if matching:
                        best = self._bm25_rank(question, matching, top_k=1)
                        for p in best:
                            if p not in gathered_paths:
                                gathered_paths.append(p)

            if gathered_paths:
                max_k = top_k if top_k is not None else 10
                return gathered_paths[:max_k]

        # TH2: Đơn ticker / đơn year (hoặc không nhận diện được)
        ticker = tickers[0] if tickers else None
        year = years[0] if years else None

        if ticker and year:
            matching = glob.glob(f"{self.csv_dir}/{ticker}/{ticker}_{year}_*.csv")
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

        if report_type:
            filtered = [f for f in matching if report_type in f]
            if filtered:
                matching = filtered

        # Đối với câu đơn: mặc định top_k = 2 (đủ 1 chính + 1 dự phòng), trừ khi có truyền top_k ngoài
        k = top_k if top_k is not None else 2
        ranked = self._bm25_rank(question, matching, top_k=k)
        return ranked if ranked else matching[:k]


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

