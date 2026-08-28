import json, re, sys
sys.path.insert(0, 'src')
from retriever import TableRetriever
r = TableRetriever('data/processed_csv')
qs = [json.loads(l) for l in open('data/raw_vifinqa/questions/questions.jsonl', encoding='utf-8') if l.strip()]

s_1t1y = 0
s_1tNy = 0
s_Nt1y = 0
s_NtNy = 0
ratio = 0
for q in qs:
    tk, yr, tks, yrs = r.extract_all_entities(q['question'])
    if len(tks)<=1 and len(yrs)<=1: s_1t1y += 1
    elif len(tks)<=1 and len(yrs)>1: s_1tNy += 1
    elif len(tks)>1 and len(yrs)<=1: s_Nt1y += 1
    else: s_NtNy += 1
print(f'1 Ticker, 1 Year: {s_1t1y} ({s_1t1y/10.12:.1f}%)')
print(f'1 Ticker, Multi-Year: {s_1tNy} ({s_1tNy/10.12:.1f}%)')
print(f'Multi-Ticker, 1 Year: {s_Nt1y} ({s_Nt1y/10.12:.1f}%)')
print(f'Multi-Ticker, Multi-Year: {s_NtNy} ({s_NtNy/10.12:.1f}%)')
