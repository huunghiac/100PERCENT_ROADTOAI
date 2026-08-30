"""
So sánh chi tiết submission_test300(old).json vs submission.json (new).
Đo lường:
1. Số câu có fallback float(df1.iloc[0]['Gia_tri']) -> phải giảm
2. Tỷ lệ câu chọn file CĐKT/KQKD/LCTT thay vì bảng Thuyết minh con số tiết mục
3. Query Match Rate (query trả về giá trị khớp với answer)
4. Số câu có pandas_query hợp lệ (không lỗi runtime)
"""
import json, re, os, sys
import pandas as pd
import numpy as np

def analyze_submission(path, label=""):
    if not os.path.exists(path):
        print(f"[{label}] File {path} not found.")
        return

    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    total = len(data)
    fallback_count = 0
    core_bctc_count = 0
    thuyet_minh_num_count = 0
    valid_query_count = 0
    zero_ans_count = 0
    single_tbl_count = 0
    multi_tbl_count = 0

    core_keywords = ['bangcandoi', 'baocaoketqua', 'ketquakinhdoanh', 'luuchuyentiente', 'baocaoluuchuyen']
    # Thuyet minh số tiết mục: _4TienVa, _5PhaiThu, _8HangTonKho, v.v.
    num_tm_pattern = re.compile(r'_\d+[a-zA-Z]')

    for item in data:
        q = item.get('pandas_query', '')
        ans = item.get('answer', 0)
        ev = item.get('evidence', [])
        tbls = item.get('relevant_tables', [])

        if "iloc[0]['Gia_tri']" in q or "iloc[0][\"Gia_tri\"]" in q:
            fallback_count += 1

        if ans == 0 or ans == 0.0:
            zero_ans_count += 1

        if len(ev) == 1:
            single_tbl_count += 1
        elif len(ev) > 1:
            multi_tbl_count += 1

        # Check file types in evidence
        for e in ev:
            csv_name = e.get('csv_path', '').lower()
            if any(k in csv_name for k in core_keywords):
                core_bctc_count += 1
                break
            elif num_tm_pattern.search(csv_name):
                thuyet_minh_num_count += 1
                break

    print(f"\n{'='*50}")
    print(f"BÁO CÁO PHÂN TÍCH: {label} ({total} câu)")
    print(f"{'='*50}")
    print(f"Fallback rate (iloc[0]['Gia_tri']): {fallback_count}/{total} ({fallback_count/total*100:.1f}%)")
    print(f"Core BCTC chosen (CĐKT/KQKD/LCTT):  {core_bctc_count}/{total} ({core_bctc_count/total*100:.1f}%)")
    print(f"Thuyết minh số tiết mục chosen:    {thuyet_minh_num_count}/{total} ({thuyet_minh_num_count/total*100:.1f}%)")
    print(f"Answer = 0 / 0.0:                  {zero_ans_count}/{total} ({zero_ans_count/total*100:.1f}%)")
    print(f"Single-table queries:              {single_tbl_count}/{total} ({single_tbl_count/total*100:.1f}%)")
    print(f"Multi-table queries:               {multi_tbl_count}/{total} ({multi_tbl_count/total*100:.1f}%)")

if __name__ == '__main__':
    old_path = 'submission_test300(old).json'
    new_path = 'submission.json'

    analyze_submission(old_path, label="OLD Submission (trước fix routing)")
    analyze_submission(new_path, label="NEW Submission (hiện tại)")
