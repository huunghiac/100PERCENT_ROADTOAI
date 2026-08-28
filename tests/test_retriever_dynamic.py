import sys
import os
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))
from retriever import TableRetriever


@pytest.fixture(scope="module")
def retriever():
    return TableRetriever(csv_dir="data/processed_csv")


def test_single_ticker_single_year_exact_indicator(retriever):
    q = "Lãi tiền gửi năm 2018 của công ty mẹ CTCP Hàng không Vietjet (VJC) là bao nhiêu triệu đồng?"
    paths = retriever.retrieve(q)
    assert len(paths) == 1, f"Expected 1 table for single query, got {len(paths)}"
    assert "VJC_2018" in paths[0]
    assert "separate" in paths[0]


def test_core_balance_sheet_priority(retriever):
    q = "Vốn cổ phần đã phát hành của công ty mẹ VGT là bao nhiêu nghìn tỷ đồng vào ngày 31/12/2024?"
    paths = retriever.retrieve(q)
    assert len(paths) == 1, f"Expected 1 table, got {len(paths)}"
    assert "VGT_2024" in paths[0]


def test_ratio_query_gets_two_tables(retriever):
    q = "Biên lợi nhuận gộp của HPG năm 2022 là bao nhiêu phần trăm?"
    paths = retriever.retrieve(q)
    assert len(paths) == 2, f"Ratio query must return 2 tables (KQKD + CDKT), got {len(paths)}"
    assert any("HPG_2022" in p for p in paths)


def test_multi_year_query_gets_per_year_table(retriever):
    q = "Doanh thu thuần của CTCP Tập đoàn Masan (MSN) năm 2020 và 2021 thay đổi như thế nào?"
    paths = retriever.retrieve(q)
    assert len(paths) == 2, f"Multi-year query must return 2 tables (1 for 2020, 1 for 2021), got {len(paths)}"
    assert any("2020" in p for p in paths)
    assert any("2021" in p for p in paths)


def test_multi_company_gets_per_ticker_table(retriever):
    q = "So sánh doanh thu thuần của HPG và HSG trong năm 2022?"
    paths = retriever.retrieve(q)
    assert len(paths) == 2, f"Multi-company query must return 2 tables (1 HPG, 1 HSG), got {len(paths)}"
    tickers_in_paths = [os.path.basename(p).split("_")[0] for p in paths]
    assert "HPG" in tickers_in_paths
    assert "HSG" in tickers_in_paths


if __name__ == "__main__":
    r = TableRetriever(csv_dir="data/processed_csv")
    test_single_ticker_single_year_exact_indicator(r)
    test_core_balance_sheet_priority(r)
    test_ratio_query_gets_two_tables(r)
    test_multi_year_query_gets_per_year_table(r)
    test_multi_company_gets_per_ticker_table(r)
    print("ALL 5 RETRIEVER DYNAMIC TOP-K TESTS PASSED!")

