import os
import re
import json
import glob
import pandas as pd
from rank_bm25 import BM25Okapi


class TableRetriever:
    def __init__(self, csv_dir="data/processed_csv",
                 manifest_path="data/processed_csv/_manifest.jsonl",
                 line_map_path="data/table_line_map.json"):
        """
        Tìm bảng CSV phù hợp cho câu hỏi bằng 3 tầng:
          Tầng 0 – Entity extraction: ticker (ngoặc đơn > bare match > company name) + year.
          Tầng 1 – Lọc cứng theo ticker + year + report_type (glob filename).
          Tầng 2 – Xếp hạng BM25 trên table_title / table_slug / company_name metadata.
        """
        self.csv_dir = csv_dir
        self.manifest = {}
        self.line_map = {}
        self.name_to_ticker = {}
        self.ticker_set = set()
        self._load_manifest(manifest_path)
        self._load_line_map(line_map_path)
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

    def _load_line_map(self, path: str):
        if not os.path.exists(path):
            return
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.line_map = json.load(f)
        except Exception:
            self.line_map = {}

    def get_source_line_number(self, doc_id: str, table_index: int) -> int:
        """Lấy số dòng 1-based bắt đầu bảng trong file txt OCR."""
        key = f"{doc_id}|{table_index}"
        if key in self.line_map:
            return self.line_map[key]
        return table_index

    # ---- Company-name → ticker index (built once) ----
    def _normalize_name(self, name: str) -> str:
        n = name.strip().lower()
        n = re.sub(r'\s*-\s*(ctcp|tnhh|tjsc)\s*$', '', n)
        n = re.sub(r'^(ctcp|tnhh|tổng công ty cổ phần|tổng công ty|công ty cổ phần|công ty)\s+', '', n)
        return n.strip()

    def _build_name_index(self):
        seen = {}
        # 1. Từ manifest
        for entry in self.manifest.values():
            ticker = entry.get("ticker", "")
            name = entry.get("company_name", "")
            if ticker and name:
                self.ticker_set.add(ticker)
                if ticker not in seen:
                    seen[ticker] = set()
                seen[ticker].add(name)

        # 2. Từ code_stock.csv nếu có
        for cs_path in ["data/raw_vifinqa/code_stock.csv", "data/code_stock.csv"]:
            if os.path.exists(cs_path):
                try:
                    df_cs = pd.read_csv(cs_path)
                    for _, row in df_cs.iterrows():
                        tk = str(row.iloc[0]).strip()
                        nm = str(row.iloc[1]).strip()
                        if tk and nm:
                            self.ticker_set.add(tk)
                            if tk not in seen:
                                seen[tk] = set()
                            seen[tk].add(nm)
                except Exception:
                    pass

        # 3. Từ tên folder trong csv_dir
        if os.path.exists(self.csv_dir):
            for d in os.listdir(self.csv_dir):
                if os.path.isdir(os.path.join(self.csv_dir, d)) and len(d) in [3, 4] and d.isupper():
                    self.ticker_set.add(d)

        for ticker, names in seen.items():
            for raw_name in names:
                self.name_to_ticker[self._normalize_name(raw_name)] = ticker
                self.name_to_ticker[raw_name.lower().strip()] = ticker

    # ---- Hardcoded ticker aliases for entities that manifest name_to_ticker misses ----
    _TICKER_ALIASES = {
        # Chứng khoán (tên chứa ticker công ty mẹ → dễ nhầm)
        "chứng khoán fpt": "FTS",
        "ctcp chứng khoán fpt": "FTS",
        "chứng khoán bản việt": "VCI",
        "chứng khoán ssi": "SSI",
        "chứng khoán mb": "MBS",
        # Ngân hàng viết tắt phổ biến
        "bidv": "BID",
        "vietcombank": "VCB",
        "vietinbank": "CTG",
        "mbbank": "MBB",
        "mb bank": "MBB",
        "vpbank": "VPB",
        "techcombank": "TCB",
        "hdbank": "HDB",
        "seabank": "SSB",
        "sacombank": "STB",
        "abbank": "ABB",
        "bắc á bank": "BAB",
        "bac a bank": "BAB",
        "kienlongbank": "KLB",
        "kiên long bank": "KLB",
        "nam á bank": "NAB",
        "nam a bank": "NAB",
        # Bất động sản / công ty con tập đoàn
        "vinhomes": "VHM",
        "vincom retail": "VRE",
        "sunshine homes": "SSH",
        # Tập đoàn
        "tập đoàn bảo việt": "BVH",
        "bảo việt": "BVH",
        "tập đoàn vingroup": "VIC",
        "vingroup": "VIC",
        "tập đoàn masan": "MSN",
        "masan": "MSN",
        "tập đoàn hòa phát": "HPG",
        "hòa phát": "HPG",
    }

    # ---- Noise tickers ----
    _NOISE_TICKERS = {
        "CTCP", "TNHH", "TMCP", "VND", "USD", "BTC", "JSC", "HĐQT",
        "TCTD", "NHNN", "CKPT", "CNTT",
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
        Ưu tiên: (1) hardcoded alias (longest match) → (2) paren ticker → (3) manifest name → (4) bare uppercase.
        """
        tickers = []
        q_lower = question.lower()

        # 0. Hardcoded aliases — longest match first (chống nhầm "Chứng khoán FPT" → FPT thay vì FTS)
        alias_sorted = sorted(self._TICKER_ALIASES.items(), key=lambda x: len(x[0]), reverse=True)
        for alias_name, alias_ticker in alias_sorted:
            if alias_name in q_lower and alias_ticker not in tickers:
                tickers.append(alias_ticker)

        # 1. Tickers trong ngoặc: (VJC), (ACB)
        parens = re.findall(r'\(([A-Z][A-Z0-9]{1,3})\)', question)
        for p in parens:
            if p not in self._NOISE_TICKERS and p in self.ticker_set and p not in tickers:
                tickers.append(p)

        # 2. Match company names from manifest (longest match wins)
        for name_key, ticker in self.name_to_ticker.items():
            if name_key in q_lower and ticker not in tickers:
                # Kiểm tra alias đã cover ticker này chưa — tránh thêm FPT khi FTS đã match
                already_covered = False
                for alias_name, alias_ticker in alias_sorted:
                    if alias_name in q_lower and alias_ticker != ticker:
                        # alias match rồi, nếu name_key là substring của alias thì skip
                        if name_key in alias_name or alias_name in name_key:
                            already_covered = True
                            break
                if not already_covered:
                    tickers.append(ticker)

        # 3. Bare uppercase match (e.g. "nhóm MSN, MCH, DBC, ASM và OGC")
        for c in re.findall(r'\b([A-Z][A-Z0-9]{1,3})\b', question):
            if c not in self._NOISE_TICKERS and c in self.ticker_set and c not in tickers:
                # Kiểm tra: nếu alias đã resolve ticker khác cho cùng context thì skip
                # VD: "Chứng khoán FPT" → FTS đã có, skip bare "FPT"
                skip = False
                for alias_name, alias_ticker in alias_sorted:
                    if alias_name in q_lower and c.lower() in alias_name and alias_ticker != c:
                        skip = True
                        break
                if not skip:
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
        "mẹ", "hợp", "nhất", "riêng", "báo", "cáo", "đơn", "vị", "hội", "đồng", "quản", "trị",
        "thành", "viên", "ông", "bà", "tổng", "giám", "đốc",
    }

    def _tokenize(self, text: str) -> list:
        return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)

    def _clean_query_tokens(self, question: str) -> list:
        """Giữ token chỉ tiêu, bỏ ALL tickers/years/company names/unit/question filler."""
        _, _, all_tickers, all_years = self.extract_all_entities(question)
        q = question.lower()
        for tk in all_tickers:
            q = re.sub(rf"\b{re.escape(tk.lower())}\b", " ", q)
            for name_key, mapped_ticker in self.name_to_ticker.items():
                if mapped_ticker == tk:
                    q = q.replace(name_key, " ")
        for yr in all_years:
            q = q.replace(yr, " ")
        tokens = [t for t in self._tokenize(q) if t not in self._QUESTION_STOPWORDS and len(t) > 1]
        return tokens or self._tokenize(question)

    def _extract_query_phrase(self, question: str) -> str:
        """Trích xuất cụm từ khóa chỉ tiêu chính."""
        tokens = self._clean_query_tokens(question)
        return " ".join(tokens)

    def _path_bonus(self, question_tokens: list, path: str, question: str = "") -> float:
        """
        Boost điểm BM25 dựa trên ý định câu hỏi vs loại báo cáo trong tên file CSV.
        Ưu tiên Báo Cáo Tài Chính Chính (KQKD/CĐKT/LCTT) cho các chỉ tiêu cốt lõi.
        """
        p = path.lower()
        qt = set(question_tokens)
        q_raw = question.lower() if question else " ".join(question_tokens)
        bonus = 0.0

        # --- Detect loại báo cáo trong filename ---
        is_kqkd = "baocaoketqua" in p or "ketquakinhdoanh" in p or "ketquahoatdong" in p
        is_lctt = "luuchuyentiente" in p or "lưuchuyểntiềntệ" in p or "baocaoluuchuyen" in p
        is_cdkt = "bangcandoi" in p or "candoiketoan" in p or "tinhhinhtaichinh" in p

        # -----------------------------------------------------------------------
        # TẦNG A: Phân loại ý định BCTC (Intent Classification)
        # -----------------------------------------------------------------------
        # Tín hiệu KQKD
        kqkd_signal = False
        _kqkd_phrases = [
            "doanh thu", "lợi nhuận", "lãi gộp", "lỗ thuần", "lãi thuần",
            "thu nhập lãi thuần", "thu nhập hoạt động", "thu nhập từ",
            "chi phí tài chính", "chi phí bán hàng", "chi phí quản lý",
            "chi phí hoạt động", "chi phí lãi vay", "chi phí dự phòng",
            "giá vốn hàng bán", "giá vốn hàng hóa",
            "lãi cơ bản trên cổ phiếu", "lãi pha loãng",
            "lãi thuần từ hoạt động", "lãi từ hoạt động",
            "thu nhập lãi", "chi phí lãi",
            "thuế thu nhập doanh nghiệp",
        ]
        if any(ph in q_raw for ph in _kqkd_phrases):
            kqkd_signal = True
        if "eps" in qt or ({"lãi", "cổ", "phiếu"} <= qt):
            kqkd_signal = True

        # Tín hiệu LCTT
        lctt_signal = False
        _lctt_phrases = [
            "lưu chuyển tiền", "luuchuyen", "dòng tiền", "tiền thuần từ",
            "tiền thu từ", "tiền chi từ", "tiền cuối kỳ", "tiền đầu kỳ",
            "lưu chuyển thuần", "hoạt động kinh doanh", "hoạt động đầu tư",
            "hoạt động tài chính",
            "chi phí lãi vay đã trả",
        ]
        if any(ph in q_raw for ph in _lctt_phrases):
            lctt_signal = True

        # Tín hiệu CĐKT
        cdkt_signal = False
        _cdkt_phrases = [
            "tổng tài sản", "tổng cộng tài sản", "tài sản ngắn hạn",
            "tài sản dài hạn", "tài sản cố định", "tài sản vô hình",
            "vốn chủ sở hữu", "vốn cổ phần", "thặng dư vốn cổ phần",
            "lợi nhuận chưa phân phối", "tổng vốn chủ",
            "nợ phải trả", "tổng nợ", "nợ ngắn hạn", "nợ dài hạn",
            "hàng tồn kho", "phải thu", "phải trả",
            "tiền và các khoản tương đương", "tiền và tương đương",
            "tiền mặt", "tiền gửi tại", "số dư tiền",
            "vay ngắn hạn", "vay dài hạn", "tổng vay", "dư nợ vay",
            "tổng nguồn vốn", "tổng cộng nguồn vốn",
            "đầu tư vào công ty con", "đầu tư vào các công ty",
            "tài sản xây dựng cơ bản", "xây dựng cơ bản dở dang",
            "giá trị còn lại", "nguyên giá",
            "cho vay khách hàng", "tổng dư nợ",
            "quyền sử dụng đất",
            "lãi vay phải trả", "lãi phải trả", "chi phí phải trả",
            "thuế thu nhập doanh nghiệp phải trả", "thuế phải trả",
            "số dư cho vay", "dư nợ cho vay", "cho vay đối với",
            "trái phiếu phát hành", "kỳ phiếu", "chứng chỉ tiền gửi",
            "bất động sản đầu tư", "giá trị hợp lý",
        ]
        if any(ph in q_raw for ph in _cdkt_phrases):
            cdkt_signal = True

        # Tín hiệu Thuyết minh chuyên biệt
        # (Câu hỏi hỏi số liệu chi tiết, không phải số tổng hợp BCTC chính)
        note_signal = False
        _note_phrases = [
            "thù lao", "hđqt", "hội đồng quản trị", "ban giám đốc",
            "tỷ lệ sở hữu", "biểu quyết", "cổ đông lớn",
            "giao dịch bên liên quan", "bên liên quan",
            "phân tích nợ xấu", "tài sản thế chấp", "cầm cố",
        ]
        if any(ph in q_raw for ph in _note_phrases):
            note_signal = True

        # Cờ bảng số tiết mục (numbered thuyết minh: _4TienVa_, _5PhanTich_,...)
        import re as _re
        is_numbered_note = bool(_re.search(r'_\d+[a-z]', p.lower()))

        # Tin hieu Phan tich chat luong no (NPL)
        npl_signal = False
        _npl_phrases = [
            "phân tích chất lượng nợ", "nợ đủ tiêu chuẩn", "nợ cần chú ý",
            "nợ dưới tiêu chuẩn", "nợ nghi ngờ", "nợ xấu", "phân loại nợ",
            "nợ có khả năng mất vốn",
        ]
        if any(ph in q_raw for ph in _npl_phrases):
            npl_signal = True

        # Tin hieu hoat dong thu nhap ngan hang chuyen biet
        bank_activity_signal = False
        _bank_activity_phrases = [
            "lãi thuần từ hoạt động dịch vụ",
            "lãi thuần từ hoạt động kinh doanh ngoại tệ",
            "lãi thuần từ hoạt động mua bán chứng khoán",
            "lãi thuần từ hoạt động khác",
            "thu nhập từ hoạt động dịch vụ",
            "lãi thuần từ góp vốn",
        ]
        if any(ph in q_raw for ph in _bank_activity_phrases):
            bank_activity_signal = True

        # Tin hieu tien gui NHNN / TCTD chuyen biet
        bank_deposit_signal = False
        _bank_deposit_phrases = [
            "tiền gửi tại nhnn", "tiền gửi tại ngân hàng nhà nước",
            "tiền gửi tại tctd", "số dư kỳ phiếu", "trái phiếu trung hạn",
            "chứng chỉ tiền gửi",
        ]
        if any(ph in q_raw for ph in _bank_deposit_phrases):
            bank_deposit_signal = True

        # -----------------------------------------------------------------------
        # TẦNG B: Áp dụng điểm thưởng / phạt theo ý định BCTC
        # -----------------------------------------------------------------------

        # --- KQKD: ưu tiên file Báo Cáo Kết Quả Kinh Doanh ---
        if kqkd_signal and is_kqkd:
            bonus += 12.0
        # Penalty mạnh nếu câu KQKD nhưng file là LCTT
        if kqkd_signal and is_lctt and not is_kqkd:
            bonus -= 5.0
        # Penalty nhẹ nếu câu KQKD nhưng file là bảng số tiết mục
        if kqkd_signal and is_numbered_note and not is_kqkd and not is_cdkt:
            bonus -= 5.0

        # --- LCTT: ưu tiên file Báo Cáo Lưu Chuyển Tiền Tệ ---
        if lctt_signal and is_lctt:
            bonus += 12.0
        # Penalty mạnh nếu câu LCTT nhưng file không phải LCTT
        if lctt_signal and not is_lctt:
            bonus -= 5.0

        # --- CĐKT: ưu tiên file Bảng Cân Đối Kế Toán ---
        if cdkt_signal and is_cdkt:
            bonus += 12.0
        # Penalty nếu câu CĐKT nhưng file là LCTT
        if cdkt_signal and is_lctt and not is_cdkt:
            bonus -= 5.0
        # Penalty nếu câu CĐKT nhưng file là bảng số tiết mục
        if cdkt_signal and is_numbered_note and not is_cdkt:
            bonus -= 5.0

        # --- Thuyết minh chuyên biệt: ưu tiên file Thuyết Minh nếu tín hiệu rõ ---
        if note_signal and ("thuyetminh" in p or "dautu" in p or "congtycon" in p):
            bonus += 10.0

        # --- Thuyết minh chuyên biệt (giữ nguyên logic cũ, tinh chỉnh) ---
        # Dự phòng
        if {"dự", "phòng"} <= qt:
            if "duphong" in p or "chiphihoatdong" in p:
                bonus += 3.0
        # Cho vay khách hàng
        if {"cho", "vay"} <= qt:
            if "chovay" in p or "khachhang" in p:
                bonus += 3.0
        # Thuyết minh đầu tư công ty con / tỷ lệ sở hữu / quyền biểu quyết
        if {"sở", "hữu"} <= qt or {"công", "con"} <= qt or "biểu" in qt:
            if "thuyetminh" in p or "dautu" in p or "congtycon" in p:
                bonus += 4.0
        # Tiền gửi tại TCTD
        if "tctd" in qt or ({"tiền", "gửi"} <= qt):
            if "tiengui" in p or "tctd" in p or "tuongduongtien" in p:
                bonus += 3.0
        # Cam kết ngoại bảng / giao dịch hối đoái
        if {"cam", "kết"} <= qt or {"hối", "đoái"} <= qt:
            if "camket" in p or "nghiavu" in p or "ngoaibang" in p or "congcu" in p:
                bonus += 3.0
        # Thuế hiện hành
        if "thuế" in qt and ("hiện" in qt or "hành" in qt):
            if "thue" in p or "thuethu" in p or is_kqkd:
                bonus += 3.0
        # Giá vốn hàng hóa (thuyết minh riêng)
        if {"giá", "vốn"} <= qt and {"hàng", "hóa"} <= qt:
            if "giavon" in p:
                bonus += 4.0
        # Vay và nợ thuê chính
        if "vay" in qt and ("ngắn" in qt or "dài" in qt):
            if "vayvano" in p or "vayvanothuechinh" in p:
                bonus += 3.0
        # Quỹ khen thưởng, phúc lợi
        if "quỹ" in qt and ("khen" in qt or "phúc" in qt):
            if "vonchusohuu" in p or "quy" in p:
                bonus += 2.0
        # Chi phí quản lý doanh nghiệp (thuyết minh)
        if {"quản", "lý"} <= qt or {"quản", "lý", "doanh", "nghiệp"} <= qt:
            if "chiphiquanly" in p or "chiphiquanlydoanhnghiep" in p:
                bonus += 3.0

        # --- Phân tích chất lượng nợ (NPL) ---
        if npl_signal:
            p_slug = p.split("/")[-1].lower()
            if any(kw in p_slug for kw in ["phantich", "chatluong", "phanloai", "nochova"]):
                bonus += 12.0
            elif is_cdkt or is_kqkd:
                bonus -= 3.0

        # --- Lãi thuần từ hoạt động chuyên biệt (ngân hàng) ---
        if bank_activity_signal:
            p_slug = p.split("/")[-1].lower()
            if any(kw in p_slug for kw in ["laithuan", "hoatdong", "dichvu", "ngoaite", "chungkhoan", "gopvon"]):
                bonus += 10.0
            if is_cdkt or is_lctt:
                bonus -= 3.0

        # --- Tiền gửi NHNN / TCTD chuyên biệt ---
        if bank_deposit_signal:
            p_slug = p.split("/")[-1].lower()
            if any(kw in p_slug for kw in ["tiengui", "nhnn", "tctd", "tuongduongtien"]):
                bonus += 12.0

        # --- Penalty nhẹ cho trang phụ (_02, _03...) để ưu tiên trang chính ---
        if _re.search(r'_0[2-9]\.csv$', p.lower()):
            bonus -= 2.0

        return bonus

    def _bm25_rank(self, question: str, csv_paths: list, top_k: int = 2) -> list:
        """Xếp hạng csv_paths kết hợp Exact Indicator Match + Core Statement + BM25."""
        if not csv_paths:
            return []
        query_tokens = self._clean_query_tokens(question)
        query_phrase = self._extract_query_phrase(question)
        q_raw_clean = re.sub(r'[^\w\s]', '', question.lower())

        corpus_tokens = []
        valid_paths = []
        chi_tieu_cache = {}
        exact_match_paths = []

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
                    df = pd.read_csv(real_path, usecols=["Chi_tieu"])
                    values = [str(x) for x in df["Chi_tieu"].dropna().tolist()]
                    chi_tieu_text = " ".join(values)
                    chi_tieu_cache[p] = chi_tieu_text.lower()

                    for val in values:
                        val_clean = re.sub(r'[^\w\s]', '', str(val).lower()).strip()
                        if len(val_clean) >= 6 and (val_clean in q_raw_clean or (query_phrase and query_phrase in val_clean)):
                            if p not in exact_match_paths:
                                exact_match_paths.append(p)
                            break
                except Exception:
                    chi_tieu_cache[p] = ""

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
            bonus = self._path_bonus(query_tokens, p, question=question)

            # Exact Indicator Match: giảm từ 15.0 -> 6.0 để tránh lấn át Core BCTC
            if p in exact_match_paths:
                bonus += 6.0

            hits = sum(1 for t in query_tokens if t in text)
            if query_tokens and hits == len(set(query_tokens)):
                bonus += 6.0
            elif hits >= max(2, len(set(query_tokens)) // 2):
                bonus += 2.0

            # Scope Routing: separate / consolidated bonus+penalty
            p_lower = p.lower()
            q_lower_scope = question.lower()
            need_separate = "công ty mẹ" in q_lower_scope or "báo cáo riêng" in q_lower_scope
            need_consolidated = "hợp nhất" in q_lower_scope or "toàn tập đoàn" in q_lower_scope

            if need_separate:
                if "separate" in p_lower:
                    bonus += 10.0
                elif "consolidated" in p_lower or "_aggregated" in p_lower:
                    bonus -= 10.0
            elif need_consolidated:
                if "consolidated" in p_lower or "_aggregated" in p_lower:
                    bonus += 10.0
                elif "separate" in p_lower:
                    bonus -= 10.0

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
        Đầu ra: Danh sách đường dẫn CSV thích ứng (Dynamic Adaptive Top-K).
        - 1 Ticker, 1 Year: Trả về 1 bảng tốt nhất (hoặc 2 nếu câu tỷ số).
        - Multi-Year: Trả về đúng 1 bảng/năm.
        - Multi-Ticker: Trả về đúng 1 bảng/ticker.
        Scope Routing (separate/consolidated) được xử lý bên trong _bm25_rank bằng
        bonus/penalty thay vì lọc cứng, để tránh mất file khi không tồn tại bản mong muốn.
        """
        _, _, tickers, years = self.extract_all_entities(question)
        q_lower = question.lower()

        if not os.path.exists(self.csv_dir) or not os.listdir(self.csv_dir):
            print(f"[Retriever] WARNING: csv_dir '{self.csv_dir}' empty or missing.")
            return []

        _ratio_keywords = {"hệ số", "tỷ số", "tỉ số", "biên lợi nhuận", "biên ln", "biên gộp",
                           "đòn bẩy", "roe", "roa", "ros", "nim", "cir", "npl", "d/e"}
        is_ratio = any(kw in q_lower for kw in _ratio_keywords)

        # TH1: Multi-company hoặc Multi-year -> Lấy đúng 1 bảng/thực thể
        is_multi = len(tickers) > 1 or len(years) > 1
        if is_multi:
            target_tickers = tickers if tickers else [None]
            target_years = years if years else [None]
            gathered_paths = []

            per_entity_k = 2 if is_ratio else 1

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

                    if matching:
                        best = self._bm25_rank(question, matching, top_k=per_entity_k)
                        for p in best:
                            if p not in gathered_paths:
                                gathered_paths.append(p)

            if gathered_paths:
                max_k = top_k if top_k is not None else 10
                return gathered_paths[:max_k]

        # TH2: Đơn Ticker, Đơn Year
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

        if top_k is not None:
            effective_k = top_k
        elif is_ratio:
            effective_k = 2
        else:
            effective_k = 1

        ranked = self._bm25_rank(question, matching, top_k=max(effective_k, 2))
        return ranked[:effective_k] if ranked else matching[:effective_k]


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

