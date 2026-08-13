import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.data_extractor import (
    CSV_COLUMNS,
    OutputNameAllocator,
    ReportMetadata,
    ExtractionDiagnostics,
    RawTable,
    convert_candidate,
    build_header_paths,
    decide_value_column,
    detect_unit,
    expand_html_table,
    extract_html_tables,
    extract_tables_from_text,
    is_financial_candidate,
    is_continuation_title,
    merge_continuation_tables,
    parse_number,
    select_value_column,
)
from src.retriever import TableRetriever


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"


def fixture_text(name):
    return (FIXTURES / name).read_text(encoding="utf-8")


def folded(value):
    from src.data_extractor import fold_text

    return fold_text(value)


class DataExtractorTests(unittest.TestCase):
    def test_parse_normal_html_table(self):
        tables, detected, skipped = extract_tables_from_text(
            fixture_text("html_normal.txt"), 2023
        )
        self.assertEqual(detected, 1)
        self.assertEqual(skipped, 0)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].table_slug, "BaoCaoKetQuaKinhDoanh")
        self.assertEqual(tables[0].records[0]["Gia_tri"], 1234567)
        self.assertEqual(tables[0].records[1]["Gia_tri"], -12500)

    def test_rowspan_colspan_are_expanded(self):
        raw = extract_html_tables(fixture_text("rowspan_colspan.txt"))[0]
        self.assertTrue(all(len(row) == 3 for row in raw.rows))
        self.assertEqual(raw.rows[1], ["Chỉ tiêu", "31.12.2023", "31.12.2022"])
        tables, _, _ = extract_tables_from_text(fixture_text("rowspan_colspan.txt"), 2023)
        self.assertEqual(tables[0].records[0]["Gia_tri"], 6909300)

    def test_continuation_tables_are_merged(self):
        tables, detected, skipped = extract_tables_from_text(
            fixture_text("continuation.txt"), 2023
        )
        self.assertEqual(detected, 2)
        self.assertEqual(skipped, 0)
        self.assertEqual(len(tables), 1)
        self.assertEqual(len(tables[0].records), 4)
        self.assertIn("merged_continuation_table", tables[0].warnings)

    def test_wrapped_label_is_merged_only_into_value_row(self):
        tables, _, _ = extract_tables_from_text(fixture_text("wrapped_label.txt"), 2023)
        labels = [folded(row["Chi_tieu"]) for row in tables[0].records]
        self.assertIn("doanh thu thuan tu ban hang va cung cap dich vu", labels)
        self.assertEqual(tables[0].records[0]["Gia_tri"], 1250)

    def test_parenthesized_and_unicode_negative_numbers(self):
        self.assertEqual(parse_number("(1.234.567)"), -1234567)
        self.assertEqual(parse_number("− 2.500"), -2500)
        self.assertIsNone(parse_number("-"))

    def test_thousands_and_decimal_separators(self):
        self.assertEqual(parse_number("1.234.567"), 1234567)
        self.assertEqual(parse_number("1,234,567"), 1234567)
        self.assertEqual(parse_number("1,25"), 1.25)
        self.assertEqual(parse_number("1.234,50"), 1234.5)
        self.assertEqual(parse_number("1.234.567,89"), 1234567.89)
        self.assertEqual(parse_number("1,234,567.89"), 1234567.89)
        self.assertIsNone(parse_number("8996260000573083857692"))
        self.assertIsNone(parse_number("3.986.1146.908"))
        self.assertIsNone(parse_number("628.462(8.677)"))
        self.assertIsNone(parse_number("441.000.000-"))
        self.assertEqual(parse_number("182 435 184"), 182435184)

    def test_unit_detection(self):
        self.assertEqual(detect_unit("Đơn vị tính: Nghìn đồng"), "Nghin dong")
        self.assertEqual(detect_unit("Đơn vị: Triệu VND"), "Trieu VND")
        self.assertEqual(detect_unit("EPS (VND/cổ phiếu)"), "VND/co phieu")
        self.assertEqual(detect_unit("Tỷ lệ %"), "%")
        self.assertEqual(detect_unit("31/12/2015VND"), "VND")
        self.assertEqual(detect_unit("31/12/2024JPY"), "JPY")
        self.assertEqual(detect_unit("Nguyên tệ Yên Nhật"), "JPY")
        self.assertEqual(detect_unit("Số dư USD"), "USD")
        self.assertEqual(detect_unit("Giá trị EUR"), "EUR")

    def test_report_year_column_is_selected(self):
        rows = [
            ["Chỉ tiêu", "Mã số", "Năm 2022", "Năm 2023"],
            ["Doanh thu", "10", "100", "200"],
            ["Lợi nhuận", "20", "10", "25"],
        ]
        column, period, warnings = select_value_column(rows, 2023)
        self.assertEqual(column, 3)
        self.assertIn("2023", period)
        self.assertEqual(warnings, [])

        same_year_rows = [
            ["Chỉ tiêu", "31/12/2023", "01/01/2023"],
            ["Tài sản", "200", "100"],
            ["Nguồn vốn", "200", "100"],
        ]
        column, period, _ = select_value_column(same_year_rows, 2023)
        self.assertEqual(column, 1)
        self.assertIn("31/12/2023", period)

        textual_dates_reversed = [
            ["Chỉ tiêu", "Ngày 1 tháng 1 năm 2023", "Ngày 31 tháng 12 năm 2023"],
            ["Tài sản", "100", "200"],
            ["Nguồn vốn", "100", "200"],
        ]
        decision = decide_value_column(textual_dates_reversed, 2023)
        self.assertEqual(decision.column, 2)
        self.assertEqual(decision.confidence, "high")

    def test_scored_value_column_excludes_ratio_and_prior_period(self):
        rows = [
            ["Chỉ tiêu", "Năm 2023 Giá trị VND", "Năm 2023 Tỷ lệ %", "Năm trước"],
            ["Doanh thu", "1000", "12,5", "900"],
            ["Lợi nhuận", "100", "10", "80"],
        ]
        decision = decide_value_column(rows, 2023)
        self.assertEqual(decision.column, 1)
        self.assertEqual(decision.confidence, "high")
        ratio = next(item for item in decision.candidates if item["index"] == 2)
        self.assertIn("ratio_column", ratio["signals"])

    def test_current_period_wins_even_with_fewer_ocr_values(self):
        rows = [
            ["Chỉ tiêu", "Năm nay", "Năm trước"],
            ["Doanh thu", "100", "90"],
            ["Lợi nhuận", "", "8"],
            ["Chi phí", "", "82"],
        ]
        decision = decide_value_column(rows, 2023)
        self.assertEqual(decision.column, 1)
        self.assertIn("current_period_label", decision.candidates[0]["signals"])

    def test_unit_resolution_prefers_selected_header_over_nearby_percent(self):
        content = """
        Tỷ lệ nợ xấu là 3,2%.
        <h2>5.11 Phải trả ngắn hạn khác</h2>
        <table><tr><th>Chỉ tiêu</th><th>31/12/2023 VND</th><th>Tỷ lệ %</th></tr>
        <tr><td>Kinh phí công đoàn</td><td>123.949.400</td><td>2,5</td></tr>
        <tr><td>Các khoản khác</td><td>88.500.000</td><td>1,8</td></tr></table>
        """
        raw = extract_html_tables(content)[0]
        table, reason = convert_candidate(raw, 2023)
        self.assertEqual(reason, "")
        self.assertIsNotNone(table)
        self.assertEqual(table.unit, "VND")
        self.assertEqual(table.unit_source, "header")

    def test_global_header_unit_is_kept_when_selected_header_has_no_unit(self):
        content = """
        <h2>BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH</h2>
        <table><tr><th colspan="3">Đơn vị tính: VND</th></tr>
        <tr><th>Chỉ tiêu</th><th>Năm 2023</th><th>Năm 2022</th></tr>
        <tr><td>Doanh thu</td><td>1.000</td><td>900</td></tr>
        <tr><td>Lợi nhuận</td><td>100</td><td>80</td></tr></table>
        """
        raw = extract_html_tables(content)[0]
        table, reason = convert_candidate(raw, 2023)
        self.assertEqual(reason, "")
        self.assertEqual(table.unit, "VND")
        self.assertEqual(table.unit_source, "header")

    def test_explicit_preceding_unit_beats_incidental_percent_data_row(self):
        content = """
        <p>Đơn vị tính: triệu đồng</p>
        <h2>BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH</h2>
        <table>
        <tr><th>Chỉ tiêu</th><th>Năm 2023</th><th>Năm 2022</th></tr>
        <tr><td>Thuế suất thuế TNDN (%)</td><td>20</td><td>20</td></tr>
        <tr><td>Doanh thu thuần</td><td>1.000</td><td>900</td></tr>
        <tr><td>Lợi nhuận sau thuế</td><td>100</td><td>90</td></tr>
        </table>
        """
        raw = extract_html_tables(content)[0]
        self.assertEqual((raw.unit, raw.unit_source), ("Trieu dong", "preceding"))
        table, reason = convert_candidate(raw, 2023)
        self.assertEqual(reason, "")
        units = {row["Chi_tieu"]: row["Don_vi"] for row in table.records}
        self.assertEqual(units["Thuế suất thuế TNDN (%)"], "%")
        self.assertEqual(units["Doanh thu thuần"], "Trieu dong")
        self.assertEqual(units["Lợi nhuận sau thuế"], "Trieu dong")
        self.assertEqual(table.unit, "mixed")

    def test_direct_method_is_not_a_continuation_and_unsafe_parts_do_not_merge(self):
        direct = RawTable(
            title="BÁO CÁO LƯU CHUYỂN TIỀN TỆ (Theo phương pháp trực tiếp)",
            rows=[["Chỉ tiêu", "Năm 2023"], ["Thu tiền", "100"], ["Chi tiền", "(20)"]],
            parser="html",
            source_table_index=2,
        )
        self.assertFalse(is_continuation_title(direct.title))
        first, _ = convert_candidate(direct, 2023)
        incompatible = RawTable(
            title="BÁO CÁO LƯU CHUYỂN TIỀN TỆ (TIẾP THEO)",
            rows=[["Chỉ tiêu", "Năm 2022"], ["Thu tiền", "90"], ["Chi tiền", "(10)"]],
            parser="html",
            unit="VND",
            continued=True,
            source_table_index=3,
        )
        second, _ = convert_candidate(incompatible, 2023)
        self.assertIsNotNone(first)
        self.assertIsNone(second)  # prior-only/low-confidence part is quarantined

    def test_explicit_continuation_with_different_period_is_not_merged(self):
        from src.data_extractor import ExtractedTable, merge_continuation_tables

        def part(period, index, continued=False):
            return ExtractedTable(
                table_title="BÁO CÁO TÌNH HÌNH TÀI CHÍNH",
                table_slug="BaoCaoTinhHinhTaiChinh",
                unit="VND",
                value_period=period,
                parser="html",
                records=[
                    {"Chi_tieu": f"Tài sản {index}", "Gia_tri": index + 1, "Don_vi": "VND"},
                    {"Chi_tieu": f"Nợ {index}", "Gia_tri": index + 2, "Don_vi": "VND"},
                ],
                value_column_method="scored_explicit_period",
                value_column_header=period,
                value_column_confidence="high",
                source_table_indices=[index],
                header_signature=period,
                continued=continued,
            )

        merged = merge_continuation_tables(
            [part("31/12/2023", 0), part("31/12/2022", 1, continued=True)]
        )
        self.assertEqual(len(merged), 2)

    def test_middle_continuation_marker_and_special_eps_unit_merge_safely(self):
        title = (
            "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH HỢP NHẤT "
            "(tiếp theo) cho năm tài chính kết thúc ngày 31 tháng 12 năm 2022"
        )
        self.assertTrue(is_continuation_title(title))
        self.assertTrue(is_continuation_title("BẢNG CÂN ĐỐI KẾ TOÁN (TIẾP)"))
        self.assertFalse(
            is_continuation_title("Các chỉ tiêu tiếp theo cho năm tài chính 2022")
        )
        self.assertFalse(is_continuation_title("Báo cáo theo phương pháp trực tiếp"))

        def converted_part(index, continued, rows):
            raw = RawTable(
                title=(
                    title
                    if continued
                    else "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH HỢP NHẤT "
                    "cho năm tài chính kết thúc ngày 31 tháng 12 năm 2022"
                ),
                rows=rows,
                parser="html",
                unit="VND",
                unit_source="header",
                unit_confidence="high",
                continued=continued,
                source_table_index=index,
            )
            table, reason = convert_candidate(raw, 2022)
            self.assertEqual(reason, "")
            return table

        first = converted_part(
            5,
            False,
            [
                ["Chỉ tiêu", "Năm 2022", "Năm 2021"],
                ["Doanh thu thuần", "1.000", "900"],
                ["Lợi nhuận gộp", "200", "180"],
            ],
        )
        second = converted_part(
            6,
            True,
            [
                ["Chỉ tiêu", "Năm 2022", "Năm 2021"],
                ["Lợi nhuận sau thuế", "100", "90"],
                ["22. Lãi suy giảm trên cổ phiếu", "2.500", "2.100"],
            ],
        )
        self.assertEqual(second.unit, "mixed")
        eps = next(
            row for row in second.records if "suy giảm" in row["Chi_tieu"]
        )
        self.assertEqual(eps["Don_vi"], "VND/co phieu")
        merged = merge_continuation_tables([first, second])
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].source_table_indices, [5, 6])
        self.assertEqual(len(merged[0].records), 4)

    def test_page_context_continuation_marker_is_detected_without_direct_method_false_positive(self):
        content = """
        <h2>Báo cáo lưu chuyển tiền tệ riêng cho năm kết thúc</h2>
        <p>ngày 31 tháng 12 năm 2024 (Phương pháp trực tiếp)</p>
        <table><tr><th>Chỉ tiêu</th><th>2024 Triệu VND</th><th>2023 Triệu VND</th></tr>
        <tr><td>Thu nhập lãi</td><td>100</td><td>90</td></tr>
        <tr><td>Chi phí lãi</td><td>(50)</td><td>(45)</td></tr></table>

        <h2>Báo cáo lưu chuyển tiền tệ riêng cho năm kết thúc</h2>
        <p>ngày 31 tháng 12 năm 2024 (Phương pháp trực tiếp – tiếp theo)</p>
        <table><tr><th>Chỉ tiêu</th><th>2024 Triệu VND</th><th>2023 Triệu VND</th></tr>
        <tr><td>Mua tài sản cố định</td><td>(20)</td><td>(10)</td></tr>
        <tr><td>Tiền cuối năm</td><td>30</td><td>35</td></tr></table>
        """
        raw = extract_html_tables(content)
        self.assertFalse(raw[0].continued)
        self.assertTrue(raw[1].continued)
        tables, _, _ = extract_tables_from_text(content, 2024)
        self.assertEqual(len(tables), 1)
        self.assertEqual(tables[0].source_table_indices, [0, 1])

        unrelated_context = """
        <h2>THUYẾT MINH BÁO CÁO TÀI CHÍNH (TIẾP THEO)</h2>
        <p>Bảng cân đối kế toán tại ngày 31 tháng 12 năm 2023 như sau:</p>
        <table><tr><th>Chỉ tiêu</th><th>Năm 2024</th><th>Năm 2023</th></tr>
        <tr><td>Tài sản tham chiếu</td><td>100</td><td>90</td></tr>
        <tr><td>Nợ tham chiếu</td><td>50</td><td>45</td></tr></table>
        """
        self.assertFalse(extract_html_tables(unrelated_context)[0].continued)

    def test_explicit_non_core_continuation_merges_but_foreign_unit_boundary_does_not(self):
        from src.data_extractor import ExtractedTable

        def part(index, unit, continued=False):
            return ExtractedTable(
                table_title="18. Vay và nợ thuê tài chính",
                table_slug="18VayVaNoThueChinh",
                unit=unit,
                default_unit=unit,
                report_year=2024,
                value_period=f"31/12/2024{unit}",
                parser="html",
                records=[
                    {"Chi_tieu": f"Khoản vay {index}a", "Gia_tri": 100, "Don_vi": unit},
                    {"Chi_tieu": f"Khoản vay {index}b", "Gia_tri": 50, "Don_vi": unit},
                ],
                value_column_method="scored_explicit_period",
                value_column_header=f"31/12/2024{unit}",
                value_column_confidence="high",
                source_table_indices=[index],
                header_signature=f"31/12/2024|{unit}",
                continued=continued,
            )

        merged = merge_continuation_tables(
            [part(37, "VND"), part(38, "VND", True)]
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].source_table_indices, [37, 38])

        separated = merge_continuation_tables(
            [part(38, "VND"), part(39, "JPY", True), part(40, "VND", True)]
        )
        self.assertEqual(len(separated), 3)

    def test_period_semantics_distinguish_start_end_and_current_year_synonyms(self):
        start_only = [
            ["Chỉ tiêu", "01/01/2023"],
            ["Tài sản", "100"],
            ["Nợ phải trả", "50"],
        ]
        self.assertEqual(decide_value_column(start_only, 2023).confidence, "low")

        from src.data_extractor import ExtractedTable

        def part(period, index, continued=False):
            return ExtractedTable(
                table_title="BÁO CÁO TÌNH HÌNH TÀI CHÍNH",
                table_slug="BaoCaoTinhHinhTaiChinh",
                unit="VND",
                default_unit="VND",
                report_year=2023,
                value_period=period,
                parser="html",
                records=[
                    {"Chi_tieu": f"Tài sản {index}", "Gia_tri": index + 1, "Don_vi": "VND"},
                    {"Chi_tieu": f"Nợ {index}", "Gia_tri": index + 2, "Don_vi": "VND"},
                ],
                value_column_method="scored_explicit_period",
                value_column_header=period,
                value_column_confidence="high",
                source_table_indices=[index],
                header_signature=period,
                continued=continued,
            )

        self.assertEqual(
            len(
                merge_continuation_tables(
                    [part("31/12/2023", 0), part("01/01/2023", 1, True)]
                )
            ),
            2,
        )
        self.assertEqual(
            len(
                merge_continuation_tables(
                    [part("Năm nay", 0), part("Năm 2023", 1, True)]
                )
            ),
            1,
        )

    def test_continuation_unit_inheritance_uses_source_default_not_mixed_aggregate(self):
        from src.data_extractor import ExtractedTable, _inherit_continuation_units

        common = {
            "table_title": "BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH",
            "table_slug": "BaoCaoKetQuaKinhDoanh",
            "value_period": "Năm 2023",
            "report_year": 2023,
            "parser": "html",
            "value_column_confidence": "high",
            "header_signature": "nam 2023",
        }
        previous = ExtractedTable(
            **common,
            unit="mixed",
            default_unit="VND",
            records=[
                {"Chi_tieu": "Lợi nhuận", "Gia_tri": 100, "Don_vi": "VND"},
                {"Chi_tieu": "EPS", "Gia_tri": 5, "Don_vi": "VND/co phieu"},
            ],
            source_table_indices=[0],
        )
        current = ExtractedTable(
            **common,
            unit="",
            default_unit="",
            records=[
                {"Chi_tieu": "Doanh thu", "Gia_tri": 1000, "Don_vi": ""},
                {"Chi_tieu": "Chi phí", "Gia_tri": 900, "Don_vi": ""},
            ],
            source_table_indices=[1],
            continued=True,
        )
        _inherit_continuation_units([previous, current])
        self.assertEqual(current.unit, "VND")
        self.assertEqual(current.default_unit, "VND")
        self.assertTrue(all(row["Don_vi"] == "VND" for row in current.records))

        special_current = ExtractedTable(
            **common,
            unit="VND/co phieu",
            default_unit="",
            records=[
                {"Chi_tieu": "Lợi nhuận", "Gia_tri": 100, "Don_vi": ""},
                {"Chi_tieu": "EPS", "Gia_tri": 5, "Don_vi": "VND/co phieu"},
            ],
            source_table_indices=[1],
            continued=True,
        )
        _inherit_continuation_units([previous, special_current])
        self.assertEqual(special_current.default_unit, "VND")
        self.assertEqual(special_current.unit, "mixed")
        self.assertEqual(special_current.records[0]["Don_vi"], "VND")
        self.assertEqual(special_current.records[1]["Don_vi"], "VND/co phieu")

    def test_multilevel_header_uses_leaf_period_and_stops_before_body_labels(self):
        rows = [
            ["Chỉ tiêu", "Cho năm 2023 và 2022", "Cho năm 2023 và 2022"],
            ["", "2023", "2022"],
            ["Doanh thu", "100", "90"],
            ["Lợi nhuận", "10", "8"],
        ]
        decision = decide_value_column(rows, 2023)
        self.assertEqual(decision.column, 1)
        self.assertEqual(decision.confidence, "high")
        self.assertTrue(decision.header.endswith(" / 2023"))

        bank_rows = [
            ["", "Ngày 31 tháng 12 năm 2020 triệu đồng", "Ngày 31 tháng 12 năm 2019 triệu đồng"],
            ["Số dư tiền gửi bình quân tháng trước của:", "", ""],
            ["Khách hàng", "", ""],
            ["Không kỳ hạn", "100", "90"],
            ["Có kỳ hạn", "200", "180"],
        ]
        paths = build_header_paths(bank_rows)
        self.assertEqual(paths[1], "Ngày 31 tháng 12 năm 2020 triệu đồng")

    def test_parent_heading_is_not_merged_into_only_first_child(self):
        from src.data_extractor import merge_wrapped_label_rows

        rows = [
            ["Chỉ tiêu", "Năm nay"],
            ["Vốn đầu tư của Chủ sở hữu", ""],
            ["- Vốn góp đầu kỳ", "100"],
            ["- Vốn góp tăng trong kỳ", "20"],
        ]
        merged, warnings = merge_wrapped_label_rows(rows, 1)
        self.assertEqual(merged, rows)
        self.assertEqual(warnings, [])

    def test_unique_ocr_split_is_recovered_and_ambiguous_cell_is_traced(self):
        recoverable = RawTable(
            title="BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH",
            rows=[
                ["Chỉ tiêu", "Năm nay", "Năm trước"],
                ["Doanh thu", "1.000.000", "900.000"],
                [
                    "7. Chi phí tài chính- Trong đó: Chi phí lãi vay",
                    "89.962.600.00573.083.857.692",
                    "144.161.497.18888.792.729.468",
                ],
            ],
            parser="html",
            unit="VND",
            source_table_index=4,
        )
        diagnostics = ExtractionDiagnostics()
        table, reason = convert_candidate(recoverable, 2023, diagnostics=diagnostics)
        self.assertEqual(reason, "")
        values = {row["Chi_tieu"]: row["Gia_tri"] for row in table.records}
        self.assertEqual(values["7. Chi phí tài chính"], 89962600005)
        self.assertEqual(values["Trong đó: Chi phí lãi vay"], 73083857692)
        self.assertEqual(len(diagnostics.rejected_cells), 1)  # comparative cell remains traceable
        self.assertEqual(diagnostics.rejected_cells[0]["source_column"], 2)

        ambiguous = RawTable(
            title="BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH",
            rows=[
                ["Chỉ tiêu", "Năm nay"],
                ["Doanh thu", "1.000"],
                ["Chi phí", "123456789012345678901"],
            ],
            parser="html",
            source_table_index=5,
        )
        diagnostics = ExtractionDiagnostics()
        _, reason = convert_candidate(ambiguous, 2023, diagnostics=diagnostics)
        self.assertIn(
            reason,
            {
                "insufficient_numeric_rows",
                "fewer_than_two_normalized_rows",
                "no_reliable_value_column",
            },
        )
        self.assertEqual(len(diagnostics.rejected_cells), 1)
        self.assertEqual(diagnostics.rejected_cells[0]["confidence"], "low")

    def test_attached_parenthesized_ocr_value_is_traced_not_silently_lost(self):
        raw = RawTable(
            title="9. Dự phòng rủi ro cho vay khách hàng",
            rows=[
                ["", "2016Triệu VND", "2015Triệu VND"],
                ["Số dư đầu năm", "628.462(8.677)", "618.29410.168"],
                ["Chi phí bổ sung", "12.500", "10.250"],
                ["Số dư cuối năm", "619.785", "628.462"],
            ],
            parser="html",
            unit="Trieu VND",
            unit_source="header",
            unit_confidence="high",
            source_table_index=29,
        )
        diagnostics = ExtractionDiagnostics()
        table, reason = convert_candidate(raw, 2016, diagnostics=diagnostics)
        self.assertEqual(reason, "")
        self.assertEqual(table.value_column_header, "2016Triệu VND")
        self.assertFalse(any(row["Gia_tri"] == 6284628677 for row in table.records))
        rejected = {(item["source_row"], item["source_column"]) for item in diagnostics.rejected_cells}
        self.assertIn((1, 1), rejected)
        self.assertIn((1, 2), rejected)

    def test_toc_and_staff_tables_are_rejected(self):
        raw_tables = extract_html_tables(fixture_text("non_financial.txt"))
        self.assertEqual(len(raw_tables), 2)
        decisions = [is_financial_candidate(table) for table in raw_tables]
        self.assertEqual([item[0] for item in decisions], [False, False])
        tables, detected, skipped = extract_tables_from_text(
            fixture_text("non_financial.txt"), 2023
        )
        self.assertEqual(tables, [])
        self.assertEqual((detected, skipped), (2, 2))

    def test_categorical_debt_group_codes_are_not_financial_values(self):
        rows = [
            ["Nhóm", "Nhóm", "Tình hình quá hạn"],
            ["3", "Nợ dưới tiêu chuẩn", "Khoản nợ quá hạn dưới 30 ngày"],
            ["4", "Nợ nghi ngờ", "Khoản nợ quá hạn từ 181 đến 360 ngày"],
        ]
        decision = decide_value_column(rows, 2023)
        self.assertIsNone(decision.column)
        self.assertEqual(decision.confidence, "low")
        self.assertIn("categorical_code_column", decision.candidates[0]["signals"])
        raw = RawTable(title="Chính sách phân loại nợ", rows=rows, parser="html")
        self.assertEqual(is_financial_candidate(raw), (False, "categorical_code_table"))

    def test_plain_text_table_is_detected(self):
        tables, detected, skipped = extract_tables_from_text(
            fixture_text("plain_text.txt"), 2023
        )
        self.assertEqual((detected, skipped), (1, 0))
        self.assertEqual(tables[0].parser, "plain_text")
        self.assertEqual(tables[0].table_slug, "BaoCaoLuuChuyenTienTe")
        self.assertEqual(tables[0].records[1]["Gia_tri"], -25500)

    def test_html_and_plain_candidates_share_unique_source_indices(self):
        content = """
        <h2>BÁO CÁO KẾT QUẢ HOẠT ĐỘNG KINH DOANH</h2>
        <table>
        <tr><th>Chỉ tiêu</th><th>Năm 2023</th><th>Năm 2022</th></tr>
        <tr><td>Doanh thu</td><td>1.000</td><td>900</td></tr>
        <tr><td>Lợi nhuận</td><td>100</td><td>80</td></tr>
        </table>

        BẢNG CÂN ĐỐI KẾ TOÁN
        Chỉ tiêu | Năm 2023 | Năm 2022
        Tài sản | 2.000 | 1.800
        Nợ phải trả | 800 | 700
        """
        tables, detected, skipped = extract_tables_from_text(content, 2023)
        self.assertEqual((detected, skipped, len(tables)), (2, 0, 2))
        self.assertEqual(
            sorted(index for table in tables for index in table.source_table_indices),
            [0, 1],
        )

    def test_output_names_have_stable_collision_suffix(self):
        allocator = OutputNameAllocator()
        metadata = ReportMetadata(
            ticker="TST",
            company_name="Test Company",
            report_year=2023,
            report_type="consolidated",
            source_txt=Path("TST_2023_consolidated.txt"),
        )
        first = allocator.reserve(metadata, "BangCanDoiKeToan")
        second = allocator.reserve(metadata, "BangCanDoiKeToan")
        mixed_case = allocator.reserve(metadata, "BANGCANDOIKETOAN")
        self.assertEqual(first, "TST_2023_BangCanDoiKeToan_consolidated.csv")
        self.assertEqual(second, "TST_2023_BangCanDoiKeToan_consolidated_02.csv")
        self.assertEqual(mixed_case, "TST_2023_BANGCANDOIKETOAN_consolidated_03.csv")

    def test_mock_ground_truth_matches_csv_values(self):
        ground_truth_path = ROOT / "data" / "mock_ground_truth.jsonl"
        items = [
            json.loads(line)
            for line in ground_truth_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertGreaterEqual(len(items), 2)
        for item in items:
            evidence = item["expected_evidence"]
            self.assertTrue(evidence)
            dataframes = []
            for entry in evidence:
                csv_path = ROOT / Path(entry["csv_path"])
                self.assertTrue(csv_path.exists(), csv_path)
                frame = pd.read_csv(csv_path)
                self.assertEqual(tuple(frame.columns), CSV_COLUMNS)
                self.assertTrue(pd.api.types.is_numeric_dtype(frame["Gia_tri"]))
                dataframes.append(frame)

            calculation = item["expected_calculation"]
            values = []
            for indicator in calculation["indicators"]:
                matches = []
                for frame in dataframes:
                    mask = frame["Chi_tieu"].map(folded) == folded(indicator)
                    matches.extend(frame.loc[mask, "Gia_tri"].tolist())
                self.assertEqual(len(matches), 1, f"{item['id']}: {indicator}")
                values.append(float(matches[0]))

            operation = calculation["operation"]
            if operation == "value":
                actual = values[0]
            elif operation == "sum":
                actual = sum(values)
            elif operation == "subtract":
                actual = values[0] - sum(values[1:])
            elif operation == "divide_percent":
                actual = values[0] / values[1] * 100
            else:
                self.fail(f"Unsupported mock calculation: {operation}")
            self.assertAlmostEqual(actual, float(item["expected_answer"]), places=8)

    def test_retriever_finds_mock_by_ticker_and_year(self):
        retriever = TableRetriever(csv_dir="data/mock_csv")
        paths = retriever.retrieve("Doanh thu thuần của VNM năm 2023 là bao nhiêu?", top_k=5)
        self.assertTrue(paths)
        self.assertTrue(all(Path(path).exists() for path in paths))
        self.assertTrue(all(Path(path).name.startswith("VNM_2023_") for path in paths))


if __name__ == "__main__":
    unittest.main()
