import json, re, sys
sys.path.insert(0, 'src')
from retriever import TableRetriever

r = TableRetriever(csv_dir='data/processed_csv')
questions = [json.loads(line) for line in open('data/raw_vifinqa/questions/questions.jsonl', encoding='utf-8') if line.strip()]

single_y_single_t = 0
multi_year = 0
multi_ticker = 0
ratio_cnt = 0
compare_cnt = 0
growth_cnt = 0

ratio_kw = ['tỷ số', 'tỉ số', 'hệ số', 'biên lợi nhuận', 'biên ln', 'tỷ suất', 'đòn bẩy', 'roe', 'roa', 'ros', 'nim', 'cir', 'npl']
compare_kw = ['cao nhất', 'thấp nhất', 'lớn nhất', 'nhỏ nhất', 'nhiều nhất', 'ít nhất', 'chênh lệch']
growth_kw = ['tăng trưởng', 'tăng bao nhiêu', 'giảm bao nhiêu', 'tăng trưởng bình quân']

for q in questions:
    txt = q['question'].lower()
    ticker, year, tickers, years = r.extract_all_entities(q['question'])
    if len(tickers) <= 1 and len(years) <= 1:
        single_y_single_t += 1
    if len(years) > 1:
        multi_year += 1
    if len(tickers) > 1:
        multi_ticker += 1
    if any(k in txt for k in ratio_kw):
        ratio_cnt += 1
    if any(k in txt for k in compare_kw):
        compare_cnt += 1
    if any(k in txt for k in growth_kw):
        growth_cnt += 1

print('Total questions:', len(questions))
print(f'Single Ticker & Single Year: {single_y_single_t} ({single_y_single_t/len(questions)*100:.1f}%)')
print(f'Multi Year: {multi_year} ({multi_year/len(questions)*100:.1f}%)')
print(f'Multi Ticker: {multi_ticker} ({multi_ticker/len(questions)*100:.1f}%)')
print(f'Ratio questions (ROE/Biên...): {ratio_cnt} ({ratio_cnt/len(questions)*100:.1f}%)')
print(f'Compare/Min/Max: {compare_cnt} ({compare_cnt/len(questions)*100:.1f}%)')
print(f'Growth questions: {growth_cnt} ({growth_cnt/len(questions)*100:.1f}%)')
