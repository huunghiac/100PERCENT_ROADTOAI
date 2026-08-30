"""
Quick routing quality eval on 1012 questions.
Kiểm tra retriever có pick đúng loại BCTC (CDKT/KQKD/LCTT) không,
bằng cách match intent từ câu hỏi với loại file được retrieve.
Không cần LLM - chạy nhanh ~30s.
"""
import sys, json, re, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from retriever import TableRetriever

r = TableRetriever('data/processed_csv')

qs = [json.loads(l) for l in open(
    'data/raw_vifinqa/questions/questions.jsonl', encoding='utf-8') if l.strip()]

def detect_intent(q):
    t = q.lower()
    if any(ph in t for ph in [
        'lưu chuyển tiền', 'dòng tiền', 'tiền thuần từ hoạt động',
        'tiền thu từ', 'tiền chi từ', 'tiền cuối kỳ', 'tiền đầu kỳ',
        'lưu chuyển thuần từ']):
        return 'lctt'
    if any(ph in t for ph in [
        'doanh thu', 'lợi nhuận', 'lãi gộp', 'lãi thuần',
        'chi phí tài chính', 'chi phí bán hàng', 'giá vốn',
        'thu nhập lãi thuần', 'lỗ thuần']):
        return 'kqkd'
    if any(ph in t for ph in [
        'tổng tài sản', 'vốn chủ sở hữu', 'nợ phải trả', 'tiền mặt',
        'tiền và các khoản tương đương', 'vay ngắn hạn', 'vay dài hạn',
        'hàng tồn kho', 'phải thu', 'phải trả', 'tổng nguồn vốn',
        'tài sản ngắn hạn', 'tài sản dài hạn', 'tài sản cố định',
        'tổng cộng tài sản']):
        return 'cdkt'
    return None

def file_type(path):
    p = path.lower().split('/')[-1].split('\\')[-1]
    if 'baocaoketqua' in p or 'ketquakinhdoanh' in p or 'ketquahoatdong' in p:
        return 'kqkd'
    if 'luuchuyentiente' in p or 'baocaoluuchuyen' in p or 'luuchuyentienvate' in p:
        return 'lctt'
    if 'bangcandoi' in p or 'candoiketoan' in p or 'tinhhinhtaichinh' in p:
        return 'cdkt'
    return 'other'

total = 0
correct = 0
wrong = 0
no_intent = 0
not_found = 0

wrong_cases = []

for item in qs:
    q = item['question']
    intent = detect_intent(q)
    if intent is None:
        no_intent += 1
        continue
    total += 1
    paths = r.retrieve(q)
    if not paths:
        not_found += 1
        total -= 1
        continue
    got = file_type(paths[0])
    if got == intent:
        correct += 1
    else:
        wrong += 1
        if len(wrong_cases) < 20:
            wrong_cases.append((q[:70], intent, paths[0].split('\\')[-1].split('/')[-1]))

pct = correct / total * 100 if total else 0
print(f"\nRouting Quality (intent-based):")
print(f"  Total with clear intent: {total}")
print(f"  Correct type routed:     {correct} ({pct:.1f}%)")
print(f"  Wrong type:              {wrong}")
print(f"  No intent detected:      {no_intent}")
print(f"  Not found:               {not_found}")

if wrong_cases:
    print(f"\nSample WRONG routing ({len(wrong_cases)} shown):")
    for q, exp, got in wrong_cases:
        print(f"  [{exp.upper():<4}] Q: {q}")
        print(f"         Got: {got}")
