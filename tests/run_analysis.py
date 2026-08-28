import json, re, sys
sys.path.insert(0, 'src')
from retriever import TableRetriever
r = TableRetriever('data/processed_csv')
qs = [json.loads(l) for l in open('data/raw_vifinqa/questions/questions.jsonl', encoding='utf-8') if l.strip()]
s1 = sum(1 for q in qs if len(r.extract_all_entities(q['question'])[2])<=1 and len(r.extract_all_entities(q['question'])[3])<=1)
print(f'Single: {s1}/{len(qs)} ({s1/len(qs)*100:.1f}%)')
