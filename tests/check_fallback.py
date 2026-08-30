import json
data = json.load(open('submission.json', encoding='utf-8'))
fb = [x for x in data if "Gia_tri" in x.get('pandas_query', '') and "iloc[0]" in x.get('pandas_query', '')]
print(f"Fallback: {len(fb)}/{len(data)} ({len(fb)/len(data)*100:.1f}%)")
print()
for x in fb[:8]:
    print("Q:", x['question'][:80])
    print("Q:", x.get('pandas_query', '')[:120])
    print("A:", x['answer'])
    ev = x.get('evidence', [])
    if ev:
        print("Table:", ev[0].get('csv_path','')[:60])
    print()
