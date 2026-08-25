import os
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class FallbackResult:
    answer: float
    pandas_query: str
    csv_path: str
    row_index: int
    score: float


_STOPWORDS = {
    "cua", "cho", "va", "vao", "cuoi", "trong", "nam", "la", "bao", "nhieu",
    "dong", "trieu", "ty", "nghin", "ngay", "thang", "den", "tai", "voi",
    "cong", "ty", "ctcp", "tnhh", "tmcp", "ngan", "hang", "tong", "tap", "doan",
    "me", "hop", "nhat", "rieng", "bao", "cao", "don", "vi", "theo", "cac",
    "co", "phan", "phai", "duoc", "hoi", "so", "du",
}

_PHRASES = [
    ("chi phi quan ly doanh nghiep", ["chi", "phi", "quan", "ly", "doanh", "nghiep"]),
    ("chi phi luong", ["chi", "phi", "luong"]),
    ("khac theo luong", ["khac", "luong"]),
    ("chi phi phat", ["chi", "phi", "phat"]),
    ("tien va cac khoan tuong duong tien", ["tien", "tuong", "duong", "tien"]),
    ("von co phan", ["von", "co", "phan"]),
    ("tong tai san", ["tong", "tai", "san"]),
    ("chung khoan kinh doanh", ["chung", "khoan", "kinh", "doanh"]),
    ("thue tndn con phai nop", ["thue", "tndn", "con", "phai", "nop"]),
    ("thue tndn phai nop", ["thue", "tndn", "phai", "nop"]),
    ("vay ngan han", ["vay", "ngan", "han"]),
    ("vay dai han", ["vay", "dai", "han"]),
    ("no ngan han", ["no", "ngan", "han"]),
    ("no dai han", ["no", "dai", "han"]),
    ("luu chuyen tien thuan tu hoat dong kinh doanh", ["luu", "chuyen", "tien", "thuan", "hoat", "dong", "kinh", "doanh"]),
    ("luu chuyen tien thuan", ["luu", "chuyen", "tien", "thuan"]),
    ("nguyen gia", ["nguyen", "gia"]),
    ("chi phi du phong rui ro tin dung", ["chi", "phi", "du", "phong", "rui", "ro", "tin", "dung"]),
    ("chi phi du phong", ["chi", "phi", "du", "phong"]),
    ("loi nhuan sau thue", ["loi", "nhuan", "sau", "thue"]),
    ("loi nhuan gop", ["loi", "nhuan", "gop"]),
    ("doanh thu thuan", ["doanh", "thu", "thuan"]),
    ("tong quy luong", ["tong", "quy", "luong"]),
    ("chi phi dich vu mua ngoai", ["chi", "phi", "dich", "vu", "mua", "ngoai"]),
    ("phai thu ngan han khac", ["phai", "thu", "ngan", "han", "khac"]),
    ("thue thu nhap doanh nghiep", ["thue", "thu", "nhap", "doanh", "nghiep"]),
    ("thue thu nhap doanh nghiep phai tra", ["thue", "thu", "nhap", "doanh", "nghiep", "phai", "tra"]),
    ("thue tndn phai tra", ["thue", "tndn", "phai", "tra"]),
    ("ty le quyen bieu quyet", ["ty", "le", "quyen", "bieu", "quyet"]),
    ("ty le bieu quyet", ["ty", "le", "bieu", "quyet"]),
    ("thu lao", ["thu", "lao"]),
    ("thanh vien hdqt", ["thanh", "vien", "hdqt"]),
    ("cam ket cho thue hoat dong", ["cam", "ket", "cho", "thue", "hoat", "dong"]),
    ("loi the thuong mai", ["loi", "the", "thuong", "mai"]),
    ("cho vay khach hang", ["cho", "vay", "khach", "hang"]),
    ("chi phi khac", ["chi", "phi", "khac"]),
    ("thu nhap khac", ["thu", "nhap", "khac"]),
    ("chung khoan no", ["chung", "khoan", "no"]),
    ("bat dong san dau tu", ["bat", "dong", "san", "dau", "tu"]),
    ("gia tri con lai", ["gia", "tri", "con", "lai"]),
    # --- NEW PHRASES ---
    ("gia von hang hoa", ["gia", "von", "hang", "hoa"]),
    ("gia von hang ban", ["gia", "von", "hang", "ban"]),
    ("gia von cung cap dich vu", ["gia", "von", "cung", "cap", "dich", "vu"]),
    ("tien gui tai cac tctd khac", ["tien", "gui", "tctd", "khac"]),
    ("tien gui tai tctd", ["tien", "gui", "tctd"]),
    ("tien gui tai cac to chuc tin dung", ["tien", "gui", "to", "chuc", "tin", "dung"]),
    ("du no cho vay", ["du", "no", "cho", "vay"]),
    ("thuong mai dich vu", ["thuong", "mai", "dich", "vu"]),
    ("vay va no thue tai chinh", ["vay", "no", "thue", "tai", "chinh"]),
    ("vay va no", ["vay", "va", "no"]),
    ("du phong rui ro cho vay khach hang", ["du", "phong", "rui", "ro", "cho", "vay", "khach", "hang"]),
    ("du phong rui ro cho vay", ["du", "phong", "rui", "ro", "cho", "vay"]),
    ("so du du phong", ["so", "du", "du", "phong"]),
    ("chi phi thue hien hanh", ["chi", "phi", "thue", "hien", "hanh"]),
    ("thue thu nhap doanh nghiep hien hanh", ["thue", "thu", "nhap", "hien", "hanh"]),
    ("ty le so huu", ["ty", "le", "so", "huu"]),
    ("von gop truc tiep", ["von", "gop", "truc", "tiep"]),
    ("lai vay phai tra", ["lai", "vay", "phai", "tra"]),
    ("cam ket giao dich hoi doai", ["cam", "ket", "giao", "dich", "hoi", "doai"]),
    ("cam ket giao dich", ["cam", "ket", "giao", "dich"]),
    ("luu chuyen tien tu hoat dong kinh doanh", ["luu", "chuyen", "hoat", "dong", "kinh", "doanh"]),
    ("nam hien hanh", ["nam", "hien", "hanh"]),
    ("quy dau tu gia tri bao viet", ["quy", "dau", "tu", "gia", "tri", "bao", "viet"]),
    ("dau tu vao cong ty con va bvif", ["dau", "tu", "cong", "ty", "con", "bvif"]),
    ("bvif", ["bvif"]),
]


def normalize_text(text) -> str:
    text = "" if text is None else str(text).lower()
    text = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    text = text.replace("đ", "d")
    return re.sub(r"\s+", " ", text).strip()


def detect_target_unit(question: str) -> str:
    q = normalize_text(question)
    if "nghin ty" in q:
        return "nghìn tỷ đồng"
    if "ty dong" in q:
        return "tỷ đồng"
    if "trieu dong" in q:
        return "triệu đồng"
    if "nghin dong" in q:
        return "nghìn đồng"
    if "%" in question or "phan tram" in q:
        return "%"
    return ""


def convert_unit(value, source_unit, target_unit: str) -> float:
    value = float(value)
    u = "" if pd.isna(source_unit) else normalize_text(source_unit)
    t = normalize_text(target_unit)
    if u == "" or u == "nan":
        # CSV mixes blank units: large raw numbers are VND; small bank-note values are usually already target unit.
        if abs(value) >= 1_000_000_000:
            u = "vnd"
        else:
            u = t or "vnd"
    if "%" in u or "%" in t:
        return value
    is_vnd = "vnd" in u or "dong" in u
    is_trieu = "trieu" in u
    is_ty = "ty" in u
    if "nghin ty" in t:
        if is_trieu:
            return value / 1_000_000
        if is_vnd and not is_trieu and not is_ty:
            return value / 1_000_000_000_000
        return value
    if "ty" in t:
        if is_trieu:
            return value / 1000
        if is_vnd and not is_trieu and not is_ty:
            return value / 1_000_000_000
        return value
    if "trieu" in t:
        if is_ty:
            return value * 1000
        if is_vnd and not is_trieu and not is_ty:
            return value / 1_000_000
        return value
    if "nghin" in t:
        if is_vnd and not is_trieu and not is_ty:
            return value / 1000
        if is_trieu:
            return value * 1000
    return value



def _question_tokens(question: str) -> list:
    q = re.sub(r"\b20\d{2}\b", " ", normalize_text(question))
    toks = re.findall(r"[a-z0-9]+", q)
    return [t for t in toks if len(t) > 1 and t not in _STOPWORDS]


def _extract_ticker_year(question: str):
    year_match = re.search(r"\b(20\d{2})\b", question)
    tickers = re.findall(r"\b[A-Z]{2,5}\b", question)
    ticker = tickers[-1] if tickers else None
    return ticker, year_match.group(1) if year_match else None


def _candidate_paths(question: str, csv_paths: list) -> list:
    paths = list(dict.fromkeys(csv_paths))
    ticker, year = _extract_ticker_year(question)
    if ticker and year:
        folder = os.path.join("data", "processed_csv", ticker)
        if os.path.isdir(folder):
            prefix = f"{ticker}_{year}_"
            extra = [os.path.join(folder, name).replace("\\", "/") for name in os.listdir(folder) if name.startswith(prefix) and name.endswith(".csv")]
            paths.extend(extra)
    return list(dict.fromkeys(paths))


def _row_score(question: str, path: str, chi_tieu: str) -> float:
    q = normalize_text(question)
    row = normalize_text(chi_tieu)
    p = normalize_text(path)
    qtoks = _question_tokens(question)
    if not row:
        return 0.0
    score = 0.0
    for tok in qtoks:
        if tok in row:
            score += 1.0
    for phrase, toks in _PHRASES:
        phrase_in_q = phrase in q or all(t in q for t in toks)
        phrase_in_row = phrase in row or all(t in row for t in toks)
        if phrase_in_q and phrase_in_row:
            score += 8.0 + min(len(toks), 5)
    if ("cuoi nam" in q or "31/12" in q) and ("cuoi nam" in row or "cuoi ky" in row or "con phai nop" in row or row.startswith("tong") or row.startswith("i.")):
        score += 4.0
    if "dau nam" in row and ("cuoi nam" in q or "31/12" in q):
        score -= 5.0
    if "cong ty me" in q or "rieng" in q:
        if "separate" in p or "rieng" in p:
            score += 1.5
        if "consolidated" in p or "hop nhat" in p:
            score -= 2.0
    if "luu chuyen" not in q and "luuchuyentiente" in p:
        score -= 3.0
    if "luu chuyen" in q and "luuchuyentiente" in p:
        score += 4.0
    if "tong" in q and row.startswith("tong"):
        score += 3.0
    if "phai thu" in q and "phai tra" in row:
        score -= 8.0
    if "phai tra" in q and "phai thu" in row:
        score -= 8.0
    if "vay ngan han" in q and "cho vay ngan han" in row:
        score -= 10.0
    if "loi the thuong mai" in q and "loi the thuong mai" not in row:
        score -= 10.0
    if "chi phi du phong" in q and "truoc chi phi du phong" in row:
        score -= 8.0
    if ("so du" in q or "phai tra" in q or "phai nop" in q) and row.startswith("chi phi"):
        score -= 8.0
    if "quyen bieu quyet" in q and "quyen bieu quyet" not in row and "bieu quyet" not in row:
        score -= 5.0
    if "thuong mai" in q and "thuong mai" not in row:
        score -= 10.0
    if "chu thi binh" in q and "chu thi binh" not in row:
        score -= 10.0
    if "cam ket" in q and "cam ket" not in row:
        score -= 10.0
    if "gia tri con lai" in q and "gia tri con lai" not in row and "con lai" not in row:
        score -= 8.0
    # --- NEW GUARDS ---
    if "gia von hang hoa" in q and "hang hoa" not in row:
        score -= 10.0
    if "du phong rui ro" in q and "du phong" not in row and "rui ro" not in row:
        score -= 8.0
    if "so du" in q and "du phong" in q:
        # Prefer rows with "so du tai ngay 31" for balance-type questions
        if "so du tai ngay 31" in row:
            score += 5.0
        elif "so du" in row:
            score += 3.0
    if "tien gui" in q and "tctd" in q and "tctd" not in row and "to chuc tin dung" not in row and "tien gui" not in row:
        score -= 8.0
    if "von gop" in q and "von gop" not in row:
        score -= 8.0
    if "lai vay" in q and "lai vay" not in row and "lai" not in row:
        score -= 8.0
    if "hoi doai" in q and "hoi doai" not in row:
        score -= 10.0
    if "vay va no" in q and "vay va no" not in row and "vay" not in row:
        score -= 8.0
    # Synonym: "Quỹ Đầu tư Giá trị Bảo Việt" = BVIF
    if ("lai tien gui" in q or "tien gui" in q) and "lai tien gui" in row:
        score += 8.0
    return score


def _make_query(var_name: str, row_index: int, unit_factor: float, is_negative_abs: bool) -> str:
    """
    Sinh biểu thức Pandas chuẩn trên biến DataFrame (df1, df2, ...) từ evidence.
    Ví dụ: float(abs(df1.iloc[5]['Gia_tri']) / 1000000)
    """
    expr = f"float({var_name}.iloc[{int(row_index)}]['Gia_tri'])"
    if is_negative_abs:
        expr = f"abs({expr})"
    if unit_factor != 1.0:
        if unit_factor > 1.0:
            if unit_factor == int(unit_factor):
                expr = f"{expr} * {int(unit_factor)}"
            else:
                expr = f"{expr} * {unit_factor}"
        else:
            inv = int(round(1.0 / unit_factor))
            expr = f"{expr} / {inv}"
    return expr


def _get_unit_factor(value: float, source_unit: str, target_unit: str) -> float:
    """Tính toán hệ số nhân để quy đổi từ source_unit sang target_unit."""
    u = "" if pd.isna(source_unit) else normalize_text(source_unit)
    t = normalize_text(target_unit)
    if u == "" or u == "nan":
        if abs(value) >= 1_000_000_000:
            u = "vnd"
        else:
            u = t or "vnd"
    if "%" in u or "%" in t:
        return 1.0
    is_vnd = "vnd" in u or "dong" in u
    is_trieu = "trieu" in u
    is_ty = "ty" in u

    if "nghin ty" in t:
        if is_trieu:
            return 1.0 / 1_000_000
        if is_vnd and not is_trieu and not is_ty:
            return 1.0 / 1_000_000_000_000
        return 1.0
    if "ty" in t:
        if is_trieu:
            return 1.0 / 1_000
        if is_vnd and not is_trieu and not is_ty:
            return 1.0 / 1_000_000_000
        return 1.0
    if "trieu" in t:
        if is_ty:
            return 1000.0
        if is_vnd and not is_trieu and not is_ty:
            return 1.0 / 1_000_000
        return 1.0
    if "nghin" in t and "ty" not in t:
        if is_ty:
            return 1_000_000.0
        if is_trieu:
            return 1_000.0
        if is_vnd and not is_trieu and not is_ty:
            return 1.0 / 1_000
        return 1.0
    return 1.0


def try_multistep_rule_based_answer(question: str, csv_paths: list) -> Optional[FallbackResult]:
    """
    Nhận diện và giải quyết các câu hỏi tính toán đa bước phổ biến:
    1. Tổng nợ = Nợ ngắn hạn + Nợ dài hạn
    2. Biên lợi nhuận sau thuế = Lợi nhuận sau thuế / Doanh thu thuần * 100
    3. Biên lợi nhuận gộp = Lợi nhuận gộp / Doanh thu thuần * 100
    """
    qnorm = normalize_text(question)
    target_unit = detect_target_unit(question)
    path_to_var = {p: f"df{i+1}" for i, p in enumerate(csv_paths)}

    candidates = _candidate_paths(question, csv_paths)
    if not candidates:
        return None

    # TH1: Tổng nợ từ nợ ngắn hạn và nợ dài hạn
    if ("tong no" in qnorm or "no phai tra" in qnorm) and "ngan han" in qnorm and "dai han" in qnorm:
        for path in candidates:
            real_path = path if os.path.exists(path) else path.replace("data/", "", 1)
            if not os.path.exists(real_path):
                continue
            try:
                df = pd.read_csv(real_path)
            except Exception:
                continue
            if "Chi_tieu" not in df.columns or "Gia_tri" not in df.columns:
                continue
            
            s = df["Chi_tieu"].astype(str).str.lower()
            m_ngan = df[s.str.contains(r"nợ ngắn hạn|vay ngắn hạn", regex=True, na=False)]
            m_dai = df[s.str.contains(r"nợ dài hạn|vay dài hạn", regex=True, na=False)]
            
            if not m_ngan.empty and not m_dai.empty:
                idx1 = int(m_ngan.index[0])
                idx2 = int(m_dai.index[0])
                v1 = float(m_ngan.iloc[0]["Gia_tri"])
                v2 = float(m_dai.iloc[0]["Gia_tri"])
                u1 = m_ngan.iloc[0].get("Don_vi", "")
                factor = _get_unit_factor(v1 + v2, u1, target_unit)
                ans = (v1 + v2) * factor
                var = path_to_var.get(path, "df1")
                
                # Biểu thức tính tổng
                if factor == 1.0:
                    query = f"float({var}.iloc[{idx1}]['Gia_tri']) + float({var}.iloc[{idx2}]['Gia_tri'])"
                elif factor > 1.0:
                    query = f"(float({var}.iloc[{idx1}]['Gia_tri']) + float({var}.iloc[{idx2}]['Gia_tri'])) * {int(factor) if factor==int(factor) else factor}"
                else:
                    inv = int(round(1.0 / factor))
                    query = f"(float({var}.iloc[{idx1}]['Gia_tri']) + float({var}.iloc[{idx2}]['Gia_tri'])) / {inv}"
                
                return FallbackResult(float(ans), query, path, idx1, 99.0)

    # TH2: Biên lợi nhuận sau thuế / Biên lợi nhuận gộp
    if ("bien loi nhuan" in qnorm or "bien ln" in qnorm or "ty le" in qnorm) and ("sau thue" in qnorm or "gop" in qnorm):
        for path in candidates:
            real_path = path if os.path.exists(path) else path.replace("data/", "", 1)
            if not os.path.exists(real_path):
                continue
            try:
                df = pd.read_csv(real_path)
            except Exception:
                continue
            if "Chi_tieu" not in df.columns or "Gia_tri" not in df.columns:
                continue
            
            s = df["Chi_tieu"].astype(str).str.lower()
            is_gop = "gop" in qnorm
            pattern_ln = r"lợi nhuận gộp" if is_gop else r"lợi nhuận sau thuế"
            m_ln = df[s.str.contains(pattern_ln, regex=True, na=False)]
            m_dtt = df[s.str.contains(r"doanh thu thuần|doanh thu bán hàng", regex=True, na=False)]
            
            if not m_ln.empty and not m_dtt.empty:
                idx1 = int(m_ln.index[0])
                idx2 = int(m_dtt.index[0])
                v1 = float(m_ln.iloc[0]["Gia_tri"])
                v2 = float(m_dtt.iloc[0]["Gia_tri"])
                if v2 != 0:
                    ans = (v1 / v2) * 100.0
                    var = path_to_var.get(path, "df1")
                    query = f"(float({var}.iloc[{idx1}]['Gia_tri']) / float({var}.iloc[{idx2}]['Gia_tri'])) * 100"
                    return FallbackResult(float(ans), query, path, idx1, 99.0)

    return None


def try_rule_based_answer(question: str, csv_paths: list, min_score: float = 4.0) -> Optional[FallbackResult]:
    # 1. Thử giải câu hỏi đa bước trước
    multistep_res = try_multistep_rule_based_answer(question, csv_paths)
    if multistep_res is not None:
        return multistep_res

    # 2. Giải câu hỏi đơn
    target_unit = detect_target_unit(question)
    best = None
    
    # Map từng path sang tên biến df1, df2,...
    path_to_var = {p: f"df{i+1}" for i, p in enumerate(csv_paths)}

    for path in _candidate_paths(question, csv_paths):
        real_path = path if os.path.exists(path) else path.replace("data/", "", 1)
        if not os.path.exists(real_path):
            continue
        try:
            df = pd.read_csv(real_path)
        except Exception:
            continue
        if "Chi_tieu" not in df.columns or "Gia_tri" not in df.columns:
            continue
            
        var_name = path_to_var.get(path, "df1")

        for idx, row in df.iterrows():
            try:
                value = float(row.get("Gia_tri"))
            except Exception:
                continue
            score = _row_score(question, path, row.get("Chi_tieu", ""))
            if score <= 0:
                continue
                
            unit_factor = _get_unit_factor(value, row.get("Don_vi", ""), target_unit)
            raw_answer = value * unit_factor
            qnorm = normalize_text(question)
            is_cashflow_question = "luu chuyen" in qnorm or "dong tien" in qnorm
            is_neg_abs = False
            if raw_answer < 0 and not is_cashflow_question and any(k in qnorm for k in ["chi phi", "lai", "thu nhap", "doanh thu", "loi nhuan"]):
                answer = abs(raw_answer)
                is_neg_abs = True
            else:
                answer = raw_answer

            query = _make_query(var_name, idx, unit_factor, is_neg_abs)
            candidate = FallbackResult(float(answer), query, path, int(idx), score)
            if best is None or candidate.score > best.score:
                best = candidate
                
    if best is not None and best.score >= min_score:
        return best
    return None
