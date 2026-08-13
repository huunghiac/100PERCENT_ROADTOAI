import csv
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data_extractor import extract_tables_from_txt, parse_number


class DataExtractorTests(unittest.TestCase):
    def test_parse_vietnamese_financial_numbers(self):
        self.assertEqual(parse_number("1.234.567.890"), 1234567890.0)
        self.assertEqual(parse_number("(1.234.567)"), -1234567.0)
        self.assertEqual(parse_number("1.234,56"), 1234.56)
        self.assertEqual(parse_number("-"), None)

    def test_extracts_canonical_income_statement_and_repairs_wrapped_label(self):
        sample = """
CÔNG TY CỔ PHẦN DEMO
BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH RIÊNG
Cho năm tài chính 2023
Đơn vị tính: VND
<table>
<tr><td>Chỉ tiêu</td><td>Mã số</td><td>Năm nay</td><td>Năm trước</td></tr>
<tr><td>1. Doanh thu thuần</td><td>10</td><td>2.500.000.000</td><td>2.000.000.000</td></tr>
<tr><td>2. Lãi/lỗ chênh lệch tỷ giá do đánh giá các khoản mục</td><td>20</td><td>1.200.000.000</td><td>900.000.000</td></tr>
<tr><td>tiền tệ có gốc ngoại tệ</td><td></td><td></td><td></td></tr>
<tr><td>3. Lãi cơ bản trên cổ phiếu</td><td>70</td><td>3.200</td><td>2.900</td></tr>
</table>
"""
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            raw = root / "financial_statements" / "DME" / "2023" / "DME_financial_statements_2023_separate"
            raw.mkdir(parents=True)
            txt = raw / "DME_financial_statements_2023_separate_extracted.txt"
            txt.write_text(sample, encoding="utf-8")
            out = root / "processed"

            written = extract_tables_from_txt(txt, out, raw_root=root)
            target = out / "DME_2023_BaoCaoKetQuaKinhDoanh.csv"
            self.assertIn(target, written)

            with target.open(encoding="utf-8", newline="") as fh:
                rows = list(csv.DictReader(fh))

            by_name = {row["Chi_tieu"]: row for row in rows}
            self.assertEqual(float(by_name["Doanh thu thuan"]["Gia_tri"]), 2.5)
            self.assertEqual(by_name["Doanh thu thuan"]["Don_vi"], "Ty dong")
            wrapped = "Lai/lo chenh lech ty gia do danh gia cac khoan muc tien te co goc ngoai te"
            self.assertEqual(float(by_name[wrapped]["Gia_tri"]), 1.2)
            self.assertEqual(by_name["Lai co ban tren co phieu"]["Don_vi"], "VND/co phieu")


if __name__ == "__main__":
    unittest.main()
