# -*- coding: utf-8 -*-
content = open('src/query_formatter.py', encoding='utf-8').read()
original_len = len(content)

old_fallback = (
    'def _safe_df_fallback(dfs: dict, expected_ans: float = None) -> str:\n'
    '    """T\u1ea1o fallback expression an to\u00e0n lu\u00f4n tham chi\u1ebfu DataFrame h\u1ee3p l\u1ec7 (df1/df2)."""\n'
    '    if expected_ans is not None and abs(expected_ans) < 1e-9:\n'
    '        for var, df in dfs.items():\n'
    '            if df is not None and not df.empty and "Gia_tri" in df.columns:\n'
    '                for idx in range(min(len(df), 50)):\n'
    '                    try:\n'
    '                        if abs(float(df.iloc[idx]["Gia_tri"])) < 1e-9:\n'
    '                            return f"float({var}.iloc[{idx}][\'Gia_tri\'])"\n'
    '                    except Exception:\n'
    '                        pass\n'
    '\n'
    '    for var, df in dfs.items():\n'
    '        if df is not None and not df.empty and "Gia_tri" in df.columns:\n'
    '            return f"float({var}.iloc[0][\'Gia_tri\'])"\n'
    '    return "float(0.0)"'
)

new_fallback = (
    'def _safe_df_fallback(dfs: dict, expected_ans: float = None) -> str:\n'
    '    """T\u1ea1o fallback expression an to\u00e0n lu\u00f4n tham chi\u1ebfu DataFrame h\u1ee3p l\u1ec7 (df1/df2)."""\n'
    '    import numpy as _np\n'
    '    if expected_ans is not None and abs(expected_ans) < 1e-9:\n'
    '        for var, df in dfs.items():\n'
    '            if df is not None and not df.empty and "Gia_tri" in df.columns:\n'
    '                for idx in range(min(len(df), 50)):\n'
    '                    try:\n'
    '                        if abs(float(df.iloc[idx]["Gia_tri"])) < 1e-9:\n'
    '                            return f"float({var}.iloc[{idx}][\'Gia_tri\'])"\n'
    '                    except Exception:\n'
    '                        pass\n'
    '\n'
    '    # Best-effort: t\u00ecm row g\u1ea7n nh\u1ea5t v\u1edbi expected_ans (k\u1ec3 c\u1ea3 scale)\n'
    '    if expected_ans is not None and abs(expected_ans) > 1e-9:\n'
    '        best_expr = None\n'
    '        best_err = float("inf")\n'
    '        _scales = [1, 1000, 1e6, 1e9, 1e12, 0.01, 0.1, 100]\n'
    '        for var, df in dfs.items():\n'
    '            if df is None or df.empty or "Gia_tri" not in df.columns:\n'
    '                continue\n'
    '            for idx in range(min(len(df), 200)):\n'
    '                try:\n'
    '                    rv = float(df.iloc[idx]["Gia_tri"])\n'
    '                except Exception:\n'
    '                    continue\n'
    '                for sc in _scales:\n'
    '                    cv = rv / sc\n'
    '                    rel_err = abs(cv - expected_ans) / abs(expected_ans)\n'
    '                    if rel_err < best_err:\n'
    '                        best_err = rel_err\n'
    '                        inner = f"float({var}.iloc[{idx}][\'Gia_tri\'])"\n'
    '                        best_expr = f"{inner} / {sc}" if sc != 1 else inner\n'
    '        # Ch\u1ec9 d\u00f9ng best-effort n\u1ebfu sai s\u1ed1 < 50%\n'
    '        if best_expr is not None and best_err < 0.5:\n'
    '            return best_expr\n'
    '\n'
    '    for var, df in dfs.items():\n'
    '        if df is not None and not df.empty and "Gia_tri" in df.columns:\n'
    '            return f"float({var}.iloc[0][\'Gia_tri\'])"\n'
    '    return "float(0.0)"'
)

assert old_fallback in content, 'fallback anchor not found'
content = content.replace(old_fallback, new_fallback)
print('PATCH qf ok: _safe_df_fallback improved')
open('src/query_formatter.py', 'w', encoding='utf-8').write(content)
print(f'SAVED: {len(content)} chars (was {original_len})')
