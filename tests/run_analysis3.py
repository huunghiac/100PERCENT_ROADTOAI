import json, re
qs = [json.loads(l) for l in open('data/raw_vifinqa/questions/questions.jsonl', encoding='utf-8') if l.strip()]

kqkd = ['doanh thu', 'lợi nhuận', 'chi phí tài chính', 'chi phí bán hàng', 'giá vốn', 'lãi gộp', 'thu nhập thuần', 'thu nhập lãi']
cdkt = ['tài sản', 'vốn chủ sở hữu', 'nợ phải trả', 'tiền mặt', 'tiền gửi', 'phải thu', 'hàng tồn kho', 'vay ngắn hạn', 'vay dài hạn', 'vốn cổ phần']
lctt = ['lưu chuyển tiền', 'dòng tiền', 'tiền thuần từ hoạt động']
thuyet_minh = ['thù lao', 'hđqt', 'hội đồng quản trị', 'ban giám đốc', 'cổ đông', 'thuế tndn', 'khấu hao', 'đầu tư vào công ty con']
ratio = ['tỷ số', 'tỉ số', 'hệ số', 'biên lợi nhuận', 'biên ln', 'tỷ suất', 'roe', 'roa', 'ros', 'nim', 'cir', 'npl', 'đòn bẩy']

c_kqkd, c_cdkt, c_lctt, c_tm, c_ratio = 0, 0, 0, 0, 0
for q in qs:
    t = q['question'].lower()
    if any(k in t for k in ratio): c_ratio += 1
    if any(k in t for k in kqkd): c_kqkd += 1
    if any(k in t for k in cdkt): c_cdkt += 1
    if any(k in t for k in lctt): c_lctt += 1
    if any(k in t for k in thuyet_minh): c_tm += 1

print(f'KQKD intent: {c_kqkd} ({c_kqkd/10.12:.1f}%)')
print(f'CDKT intent: {c_cdkt} ({c_cdkt/10.12:.1f}%)')
print(f'LCTT intent: {c_lctt} ({c_lctt/10.12:.1f}%)')
print(f'ThuyetMinh intent: {c_tm} ({c_tm/10.12:.1f}%)')
print(f'Ratio intent (can 2 bao cao): {c_ratio} ({c_ratio/10.12:.1f}%)')
