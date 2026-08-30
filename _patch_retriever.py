# -*- coding: utf-8 -*-
content = open('src/retriever.py', encoding='utf-8').read()
original_len = len(content)

# ============ PATCH 2: Add npl/bank_activity/bank_deposit signals ============
anchor2 = (
    "        import re as _re\n"
    "        is_numbered_note = bool(_re.search(r'_\\d+[a-z]', p.lower()))"
)
addition2 = "\n\n        # Tin hieu Phan tich chat luong no (NPL)\n        npl_signal = False\n        _npl_phrases = [\n            \"ph\u00e2n t\u00edch ch\u1ea5t l\u01b0\u1ee3ng n\u1ee3\", \"n\u1ee3 \u0111\u1ee7 ti\u00eau chu\u1ea9n\", \"n\u1ee3 c\u1ea7n ch\u00fa \u00fd\",\n            \"n\u1ee3 d\u01b0\u1edbi ti\u00eau chu\u1ea9n\", \"n\u1ee3 nghi ng\u1edd\", \"n\u1ee3 x\u1ea5u\", \"ph\u00e2n lo\u1ea1i n\u1ee3\",\n            \"n\u1ee3 c\u00f3 kh\u1ea3 n\u0103ng m\u1ea5t v\u1ed1n\",\n        ]\n        if any(ph in q_raw for ph in _npl_phrases):\n            npl_signal = True\n\n        # Tin hieu hoat dong thu nhap ngan hang chuyen biet\n        bank_activity_signal = False\n        _bank_activity_phrases = [\n            \"l\u00e3i thu\u1ea7n t\u1eeb ho\u1ea1t \u0111\u1ed9ng d\u1ecbch v\u1ee5\",\n            \"l\u00e3i thu\u1ea7n t\u1eeb ho\u1ea1t \u0111\u1ed9ng kinh doanh ngo\u1ea1i t\u1ec7\",\n            \"l\u00e3i thu\u1ea7n t\u1eeb ho\u1ea1t \u0111\u1ed9ng mua b\u00e1n ch\u1ee9ng kho\u00e1n\",\n            \"l\u00e3i thu\u1ea7n t\u1eeb ho\u1ea1t \u0111\u1ed9ng kh\u00e1c\",\n            \"thu nh\u1eadp t\u1eeb ho\u1ea1t \u0111\u1ed9ng d\u1ecbch v\u1ee5\",\n            \"l\u00e3i thu\u1ea7n t\u1eeb g\u00f3p v\u1ed1n\",\n        ]\n        if any(ph in q_raw for ph in _bank_activity_phrases):\n            bank_activity_signal = True\n\n        # Tin hieu tien gui NHNN / TCTD chuyen biet\n        bank_deposit_signal = False\n        _bank_deposit_phrases = [\n            \"ti\u1ec1n g\u1eedi t\u1ea1i nhnn\", \"ti\u1ec1n g\u1eedi t\u1ea1i ng\u00e2n h\u00e0ng nh\u00e0 n\u01b0\u1edbc\",\n            \"ti\u1ec1n g\u1eedi t\u1ea1i tctd\", \"s\u1ed1 d\u01b0 k\u1ef3 phi\u1ebfu\", \"tr\u00e1i phi\u1ebfu trung h\u1ea1n\",\n            \"ch\u1ee9ng ch\u1ec9 ti\u1ec1n g\u1eedi\",\n        ]\n        if any(ph in q_raw for ph in _bank_deposit_phrases):\n            bank_deposit_signal = True"

assert anchor2 in content, 'PATCH2 anchor not found'
content = content.replace(anchor2, anchor2 + addition2)
print('PATCH 2 ok: npl/bank_activity/bank_deposit signals added')

old1 = (
    '            "cho vay kh\u00e1ch h\u00e0ng", "t\u1ed5ng d\u01b0 n\u1ee3",\n'
    '            "quy\u1ec1n s\u1eed d\u1ee5ng \u0111\u1ea5t",\n'
    '        ]'
)
new1 = (
    '            "cho vay kh\u00e1ch h\u00e0ng", "t\u1ed5ng d\u01b0 n\u1ee3",\n'
    '            "quy\u1ec1n s\u1eed d\u1ee5ng \u0111\u1ea5t",\n'
    '            "l\u00e3i vay ph\u1ea3i tr\u1ea3", "l\u00e3i ph\u1ea3i tr\u1ea3", "chi ph\u00ed ph\u1ea3i tr\u1ea3",\n'
    '            "thu\u1ebf thu nh\u1eadp doanh nghi\u1ec7p ph\u1ea3i tr\u1ea3", "thu\u1ebf ph\u1ea3i tr\u1ea3",\n'
    '            "s\u1ed1 d\u01b0 cho vay", "d\u01b0 n\u1ee3 cho vay", "cho vay \u0111\u1ed1i v\u1edbi",\n'
    '            "tr\u00e1i phi\u1ebfu ph\u00e1t h\u00e0nh", "k\u1ef3 phi\u1ebfu", "ch\u1ee9ng ch\u1ec9 ti\u1ec1n g\u1eedi",\n'
    '            "b\u1ea5t \u0111\u1ed9ng s\u1ea3n \u0111\u1ea7u t\u01b0", "gi\u00e1 tr\u1ecb h\u1ee3p l\u00fd",\n'
    '        ]'
)
assert old1 in content, 'PATCH1 anchor not found'
content = content.replace(old1, new1)
print('PATCH 1 ok: _cdkt_phrases expanded')

# ============ PATCH 3: Add NPL/bank bonuses + page penalty before return bonus ============
old3 = (
    '        # Chi ph\u00ed qu\u1ea3n l\u00fd doanh nghi\u1ec7p (thuy\u1ebft minh)\n'
    '        if {"qu\u1ea3n", "l\u00fd"} <= qt or {"qu\u1ea3n", "l\u00fd", "doanh", "nghi\u1ec7p"} <= qt:\n'
    '            if "chiphiquanly" in p or "chiphiquanlydoanhnghiep" in p:\n'
    '                bonus += 3.0\n'
    '\n'
    '        return bonus'
)
new3 = (
    '        # Chi ph\u00ed qu\u1ea3n l\u00fd doanh nghi\u1ec7p (thuy\u1ebft minh)\n'
    '        if {"qu\u1ea3n", "l\u00fd"} <= qt or {"qu\u1ea3n", "l\u00fd", "doanh", "nghi\u1ec7p"} <= qt:\n'
    '            if "chiphiquanly" in p or "chiphiquanlydoanhnghiep" in p:\n'
    '                bonus += 3.0\n'
    '\n'
    '        # --- Ph\u00e2n t\u00edch ch\u1ea5t l\u01b0\u1ee3ng n\u1ee3 (NPL) ---\n'
    '        if npl_signal:\n'
    '            p_slug = p.split("/")[-1].lower()\n'
    '            if any(kw in p_slug for kw in ["phantich", "chatluong", "phanloai", "nochova"]):\n'
    '                bonus += 12.0\n'
    '            elif is_cdkt or is_kqkd:\n'
    '                bonus -= 3.0\n'
    '\n'
    '        # --- L\u00e3i thu\u1ea7n t\u1eeb ho\u1ea1t \u0111\u1ed9ng chuy\u00ean bi\u1ec7t (ng\u00e2n h\u00e0ng) ---\n'
    '        if bank_activity_signal:\n'
    '            p_slug = p.split("/")[-1].lower()\n'
    '            if any(kw in p_slug for kw in ["laithuan", "hoatdong", "dichvu", "ngoaite", "chungkhoan", "gopvon"]):\n'
    '                bonus += 10.0\n'
    '            if is_cdkt or is_lctt:\n'
    '                bonus -= 3.0\n'
    '\n'
    '        # --- Ti\u1ec1n g\u1eedi NHNN / TCTD chuy\u00ean bi\u1ec7t ---\n'
    '        if bank_deposit_signal:\n'
    '            p_slug = p.split("/")[-1].lower()\n'
    '            if any(kw in p_slug for kw in ["tiengui", "nhnn", "tctd", "tuongduongtien"]):\n'
    '                bonus += 12.0\n'
    '\n'
    '        # --- Penalty nh\u1eb9 cho trang ph\u1ee5 (_02, _03...) \u0111\u1ec3 \u01b0u ti\u00ean trang ch\u00ednh ---\n'
    '        if _re.search(r\'_0[2-9]\\.csv$\', p.lower()):\n'
    '            bonus -= 2.0\n'
    '\n'
    '        return bonus'
)
assert old3 in content, 'PATCH3 anchor not found'
content = content.replace(old3, new3)
print('PATCH 3 ok: NPL/bank bonuses + page penalty added')

# ============ Save ============
open('src/retriever.py', 'w', encoding='utf-8').write(content)
print(f'SAVED: {len(content)} chars (was {original_len}), delta={len(content)-original_len}')


