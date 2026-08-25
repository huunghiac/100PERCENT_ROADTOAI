import os
import glob
import json
import re
from typing import Dict, List, Tuple
from bs4 import BeautifulSoup


def _doc_id_from_path(txt_path: str) -> str:
    parts = txt_path.replace("\\", "/").split("/")
    for part in reversed(parts):
        if "_financial_statements_" in part and not part.endswith(".txt"):
            return part
    fname = parts[-1]
    return re.sub(r"_extracted\.txt$", "", fname)


def build_table_line_map(txt_dir: str = "data/raw_vifinqa/financial_statements",
                         output_json: str = "data/table_line_map.json") -> Dict[str, int]:
    """
    Quét toàn bộ file OCR .txt để ánh xạ (doc_id, source_table_index) -> line_number (1-based).
    Key: f"{doc_id}|{source_table_index}"
    Value: line_number trong file OCR .txt
    """
    txt_files = sorted(glob.glob(f"{txt_dir}/*/*/*/*.txt"))
    print(f"[LineIndexer] Quét {len(txt_files)} file báo cáo OCR...")

    line_map: Dict[str, int] = {}
    total_tables = 0

    for i, file_path in enumerate(txt_files):
        doc_id = _doc_id_from_path(file_path)
        try:
            with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception as e:
            print(f"  [Error reading] {file_path}: {e}")
            continue

        # Tìm các dòng chứa thẻ <table>
        # Mỗi thẻ <table> mở tương ứng với 1 bảng HTML được BeautifulSoup parse theo thứ tự
        table_line_indices: List[int] = []
        for line_num, line in enumerate(lines, start=1):
            # Đếm số thẻ <table xuất hiện trong dòng
            matches = re.findall(r"<table\b", line, flags=re.IGNORECASE)
            for _ in matches:
                table_line_indices.append(line_num)

        for table_idx, line_num in enumerate(table_line_indices):
            key = f"{doc_id}|{table_idx}"
            line_map[key] = line_num
            total_tables += 1

        if (i + 1) % 500 == 0 or (i + 1) == len(txt_files):
            print(f"  Processed {i + 1}/{len(txt_files)} files -> {total_tables} tables mapped.")

    # Lưu ra file json
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(line_map, f, ensure_ascii=False, indent=2)

    print(f"[LineIndexer] Đã lưu mapping vào {output_json} (Tổng: {len(line_map)} bảng)")
    return line_map


if __name__ == "__main__":
    build_table_line_map()
