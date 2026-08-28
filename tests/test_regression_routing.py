"""
Kiểm tra 39 câu bị regression (OLD chọn Core BCTC, NEW chọn nhầm Thuyết minh).
Mỗi case xác nhận retriever mới chọn file chứa chuỗi keyword mong đợi.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from retriever import TableRetriever

r = TableRetriever('data/processed_csv')

# (question, expected_keyword_in_filename, expected_scope)
# expected_keyword: substring nằm trong filename CSV kết quả
# expected_scope: 'separate', 'consolidated', 'aggregated', hoặc None (không kiểm tra)
CASES = [
    # CDKT cases
    ("Tiền và các khoản tương đương tiền của công ty mẹ SAB năm 2016 là bao nhiêu?",
     "bangcandoi", "separate"),
    ("Số dư vay ngắn hạn của công ty mẹ CEO cuối năm 2025 là bao nhiêu?",
     "bangcandoi", "separate"),
    ("Vốn chủ sở hữu của FIT là bao nhiêu tỷ đồng vào ngày 31/12/2015?",
     "bangcandoi", "separate"),
    ("Số dư phải thu theo tiến độ kế hoạch hợp đồng của FPT đến ngày 31/12/2025?",
     "bangcandoi", "consolidated"),
    ("Tổng nợ phải trả của công ty mẹ BAB đến ngày 31/12/2020 là bao nhiêu?",
     "bangcandoi", "separate"),
    ("Số dư tiền và các khoản tương đương tiền cuối năm 2019 của công ty mẹ VSC?",
     "bangcandoi", "separate"),
    ("Tiền và các khoản tương đương tiền của công ty mẹ DIG cuối 2024 là bao nhiêu?",
     "bangcandoi", "separate"),
    ("Tiền và các khoản tương đương tiền của công ty mẹ FIT năm 2018?",
     "bangcandoi", "separate"),
    ("Số dư vay ngắn hạn của HT1 cuối năm 2020 là bao nhiêu?",
     "bangcandoi", "separate"),
    ("Số dư tiền mặt của Công ty Cổ phần Tập đoàn Hoa Sen (HSG) vào ngày 31/12/2020?",
     "bangcandoi", "separate"),
    ("Số tiền đầu tư vào các công ty con của công ty mẹ HDB cuối năm 2021?",
     "bangcandoi", "separate"),
    ("Giá trị còn lại của tài sản vô hình của công ty mẹ MBB đến ngày 31/12/2021?",
     "bangcandoi", "separate"),
    ("Tổng số phải thu khác ngắn hạn của NVL cuối năm 2020 là bao nhiêu?",
     "bangcandoi", "consolidated"),
    ("Tổng dư nợ vay ngắn hạn từ ngân hàng của CTCP Đầu tư Nam Long (NLG) năm 2021?",
     "bangcandoi", "separate"),
    ("Tiền và các khoản tương đương tiền của SNZ năm 2017?",
     "bangcandoi", "consolidated"),
    ("Tổng chi phí xây dựng cơ bản đang dở dang của công ty mẹ DBC năm 2022?",
     "xaydung", "separate"),
    ("Tiền và các khoản tương đương tiền của DNH năm 2016?",
     "bangcandoi", "separate"),
    ("Giá trị thuần phải thu ngắn hạn của khách hàng của công ty mẹ HSG năm 2017?",
     "bangcandoi", "separate"),
    ("Tiền và các khoản tương đương tiền của ACV năm 2015?",
     "bangcandoi", "consolidated"),
    ("Giá trị tài sản tài chính chịu lãi suất cố định của HPG năm 2019?",
     "congcu", "consolidated"),
    ("Số dư tiền và các khoản tương đương tiền cuối năm 2019 của VPI?",
     "bangcandoi", "separate"),

    # KQKD cases
    ("Lợi nhuận sau thuế của CTCP Chứng khoán FPT (FTS) năm 2023 là bao nhiêu?",
     "baocaoketqua", None),
    ("Số dư tiền gửi tại các TCTD khác cuối năm 2016 của BID là bao nhiêu?",
     "bangcandoi", "consolidated"),
    ("Lợi nhuận thuần sau thuế của công ty mẹ DBC năm 2021 là bao nhiêu?",
     "baocaoketqua", None),
    ("Lợi nhuận trước thuế của công ty mẹ QNS năm 2023 là bao nhiêu?",
     "baocaoketqua", "separate"),
    ("Tổng lợi nhuận kế toán trước thuế của công ty mẹ DLG năm 2022?",
     "baocaoketqua", "separate"),
    ("Lợi nhuận thuần trong năm của HAG năm 2015 là bao nhiêu?",
     "baocaoketqua", None),
    ("Tổng lợi nhuận kế toán trước thuế của HAG năm 2023?",
     "baocaoketqua", None),
    ("Tổng doanh thu của GEG năm 2019 là bao nhiêu?",
     "baocaoketqua", None),
    ("Doanh thu thuần về bán hàng và cung cấp dịch vụ của HAG năm 2020?",
     "baocaoketqua", "separate"),
    ("Lợi nhuận sau thuế của công ty mẹ VIF năm 2021 là bao nhiêu?",
     "baocaoketqua", "separate"),

    # LCTT cases — chỉ test những câu CÓ file LCTT trong data
    ("Lưu chuyển tiền thuần từ hoạt động kinh doanh của công ty mẹ DXS năm 2021?",
     "luuchuyen", "separate"),
    ("Dòng tiền từ hoạt động kinh doanh của công ty mẹ PRT năm 2019?",
     "luuchuyen", "separate"),
    ("Lưu chuyển tiền thuần từ hoạt động tài chính của SNZ năm 2021?",
     "luuchuyen", "separate"),

    # STB CDKT known good
    ("Tổng tài sản của STB là bao nhiêu triệu đồng vào cuối năm 2016?",
     "bangcandoi", "separate"),

    # Scope test: câu hỏi hợp nhất
    ("Tổng vốn chủ sở hữu hợp nhất của MBB cuối năm 2020 là bao nhiêu?",
     "bangcandoi", "consolidated"),
    ("Lợi nhuận hợp nhất sau thuế của VIC năm 2022?",
     "baocaoketqua", "consolidated"),
]

passed = 0
failed = 0
fail_details = []

for q, kw, scope in CASES:
    paths = r.retrieve(q)
    if not paths:
        failed += 1
        fail_details.append((q, kw, scope, "NOT FOUND"))
        continue
    fname = paths[0].split('/')[-1].lower()
    kw_ok = kw.lower() in fname
    scope_ok = (scope is None) or (scope in fname)
    if kw_ok and scope_ok:
        passed += 1
    else:
        failed += 1
        fail_details.append((q[:65], kw, scope, fname))

print(f"\nRouting Regression Test: {passed}/{passed+failed} passed\n")
if fail_details:
    print("FAILED cases:")
    for q, kw, scope, got in fail_details:
        print(f"  Q: {q}")
        print(f"     Expected kw='{kw}' scope='{scope}' | Got: {got}")
        print()

if failed == 0:
    print("ALL REGRESSION TESTS PASSED!")
else:
    print(f"{failed} cases still failing.")
