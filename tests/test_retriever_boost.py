"""Unit test cho retriever._path_bonus: kiểm tra boost/penalty đúng cho KQKD/LCTT/CĐKT."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.retriever import TableRetriever


def make_retriever():
    """Tạo retriever rỗng (không cần data dir)."""
    r = TableRetriever.__new__(TableRetriever)
    r.csv_dir = ""
    r.manifest = {}
    r.line_map = {}
    r.name_to_ticker = {}
    r.ticker_set = set()
    r._QUESTION_STOPWORDS = set()
    return r


def tok(text):
    """Tokenize giống retriever."""
    import re
    return re.findall(r"[\wÀ-ỹ]+", text.lower(), flags=re.UNICODE)


def test_kqkd_boost():
    r = make_retriever()
    qt = tok("doanh thu thuần")
    # KQKD file → boost
    b1 = r._path_bonus(qt, "data/processed_csv/AAA/AAA_2020_BaoCaoKetQuaKinhDoanh_consolidated.csv")
    assert b1 >= 5.0, f"KQKD boost too low: {b1}"
    # LCTT file → penalty
    b2 = r._path_bonus(qt, "data/processed_csv/AAA/AAA_2020_BaoCaoLuuChuyenTienTe_consolidated.csv")
    assert b2 < 0, f"KQKD question + LCTT file should be penalized: {b2}"
    print(f"  PASS test_kqkd_boost: KQKD={b1}, LCTT={b2}")


def test_lctt_boost():
    r = make_retriever()
    qt = tok("lưu chuyển tiền thuần từ hoạt động kinh doanh")
    b1 = r._path_bonus(qt, "data/processed_csv/AAA/AAA_2020_BaoCaoLuuChuyenTienTe_consolidated.csv")
    assert b1 >= 6.0, f"LCTT boost too low: {b1}"
    b2 = r._path_bonus(qt, "data/processed_csv/AAA/AAA_2020_BaoCaoKetQuaKinhDoanh_consolidated.csv")
    assert b2 < 0, f"LCTT question + KQKD file should be penalized: {b2}"
    print(f"  PASS test_lctt_boost: LCTT={b1}, KQKD={b2}")


def test_cdkt_boost():
    r = make_retriever()
    qt = tok("tổng tài sản")
    b1 = r._path_bonus(qt, "data/processed_csv/AAA/AAA_2020_BangCanDoiKeToan_consolidated.csv")
    assert b1 >= 5.0, f"CĐKT boost too low: {b1}"
    b2 = r._path_bonus(qt, "data/processed_csv/AAA/AAA_2020_BaoCaoLuuChuyenTienTe_consolidated.csv")
    assert b2 < 0, f"CĐKT question + LCTT file should be penalized: {b2}"
    print(f"  PASS test_cdkt_boost: CĐKT={b1}, LCTT={b2}")


def test_cdkt_no_phai_tra():
    r = make_retriever()
    qt = tok("nợ phải trả")
    b1 = r._path_bonus(qt, "data/processed_csv/AAA/AAA_2020_BangCanDoiKeToan_consolidated.csv")
    assert b1 >= 5.0, f"CĐKT nợ phải trả boost too low: {b1}"
    print(f"  PASS test_cdkt_no_phai_tra: {b1}")


def test_chi_phi_tai_chinh_kqkd():
    r = make_retriever()
    qt = tok("chi phí tài chính")
    b1 = r._path_bonus(qt, "data/processed_csv/AAA/AAA_2020_BaoCaoKetQuaKinhDoanh_consolidated.csv")
    assert b1 >= 5.0, f"Chi phí tài chính → KQKD boost too low: {b1}"
    print(f"  PASS test_chi_phi_tai_chinh_kqkd: {b1}")


def test_du_phong_thuyetminh():
    r = make_retriever()
    qt = tok("dự phòng rủi ro")
    b1 = r._path_bonus(qt, "data/processed_csv/AAA/AAA_2020_duphong_consolidated.csv")
    assert b1 >= 3.0, f"Dự phòng boost too low: {b1}"
    print(f"  PASS test_du_phong_thuyetminh: {b1}")


def test_cho_vay_thuyetminh():
    r = make_retriever()
    qt = tok("cho vay khách hàng")
    b1 = r._path_bonus(qt, "data/processed_csv/ACB/ACB_2020_chovay_consolidated.csv")
    assert b1 >= 3.0, f"Cho vay boost too low: {b1}"
    print(f"  PASS test_cho_vay_thuyetminh: {b1}")


def test_cam_ket_require_both_tokens():
    """cam kết cần cả 2 token, không chỉ 'cam' đơn lẻ."""
    r = make_retriever()
    qt = tok("cam kết ngoại bảng")
    b1 = r._path_bonus(qt, "data/processed_csv/ACB/ACB_2020_camket_consolidated.csv")
    assert b1 >= 3.0, f"Cam kết boost too low: {b1}"
    # Token 'cam' đơn lẻ không nên trigger
    qt2 = tok("lãi tiền gửi cam ngân hàng")
    b2 = r._path_bonus(qt2, "data/processed_csv/ACB/ACB_2020_camket_consolidated.csv")
    # Không có "kết" → không boost camket
    assert b2 < 3.0, f"Single 'cam' without 'kết' should not boost camket: {b2}"
    print(f"  PASS test_cam_ket_require_both_tokens: both={b1}, single_cam={b2}")


def test_vay_ngan_han_thuyetminh():
    r = make_retriever()
    qt = tok("vay ngắn hạn")
    b1 = r._path_bonus(qt, "data/processed_csv/AAA/AAA_2020_VayVaNoThueChinhNganHan_consolidated.csv")
    assert b1 >= 3.0, f"Vay ngắn hạn boost too low: {b1}"
    print(f"  PASS test_vay_ngan_han_thuyetminh: {b1}")


def test_dong_tien_lctt():
    r = make_retriever()
    qt = tok("dòng tiền từ hoạt động kinh doanh")
    b1 = r._path_bonus(qt, "data/processed_csv/AAA/AAA_2020_BaoCaoLuuChuyenTienTe_consolidated.csv")
    assert b1 >= 6.0, f"Dòng tiền LCTT boost too low: {b1}"
    print(f"  PASS test_dong_tien_lctt: {b1}")


def test_neutral_thuyetminh():
    """Câu hỏi thuyết minh chung không nên penalty bất kỳ file nào."""
    r = make_retriever()
    qt = tok("tiền gửi tại tổ chức tín dụng")
    b1 = r._path_bonus(qt, "data/processed_csv/ACB/ACB_2020_tiengui_consolidated.csv")
    assert b1 >= 3.0, f"Tiền gửi TCTD boost too low: {b1}"
    # Không penalty KQKD (vì không phải LCTT/CĐKT signal)
    b2 = r._path_bonus(qt, "data/processed_csv/ACB/ACB_2020_BaoCaoKetQuaKinhDoanh_consolidated.csv")
    assert b2 >= 0, f"Neutral question should not penalty KQKD: {b2}"
    print(f"  PASS test_neutral_thuyetminh: tiengui={b1}, kqkd={b2}")


if __name__ == "__main__":
    print("=== Testing retriever._path_bonus ===")
    test_kqkd_boost()
    test_lctt_boost()
    test_cdkt_boost()
    test_cdkt_no_phai_tra()
    test_chi_phi_tai_chinh_kqkd()
    test_du_phong_thuyetminh()
    test_cho_vay_thuyetminh()
    test_cam_ket_require_both_tokens()
    test_vay_ngan_han_thuyetminh()
    test_dong_tien_lctt()
    test_neutral_thuyetminh()
    print("\n=== ALL TESTS PASSED ===")
