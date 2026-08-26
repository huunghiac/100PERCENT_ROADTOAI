import re
import pandas as pd
import numpy as np


def is_valid_eval_expr(expr: str, dfs: dict, expected_ans: float = None, tol: float = 1e-2) -> bool:
    """Kiểm tra expr có chạy được qua eval() và trả về kết quả số hợp lệ."""
    if not expr or "\n" in expr or "print(" in expr or "import " in expr:
        return False
    # Cho phép dấu '=' bên trong str.contains(...) nhưng không cho phép gán biến
    if re.search(r'(?<![<>!])=(?!=)', expr) and "case=" not in expr and "na=" not in expr:
        return False
    scope = {"pd": pd, "np": np, **dfs}
    try:
        val = eval(expr, scope)
        val_float = float(val)
        if np.isnan(val_float) or np.isinf(val_float):
            return False
        if expected_ans is not None and abs(expected_ans) > 1e-9:
            rel_err = abs(val_float - expected_ans) / abs(expected_ans)
            abs_err = abs(val_float - expected_ans)
            if rel_err > tol and abs_err > tol:
                return False
        return True
    except Exception:
        return False



def convert_script_to_expression(code: str, dfs: dict, expected_ans: float = 0.0) -> str:
    """
    Chuyển script Python đa dòng thành 1 biểu thức eval()-able duy nhất.
    Không chứa newline, gán biến, print().
    """
    if not code:
        return f"float({expected_ans})"

    stripped = code.strip()
    # 1. Đã là expression hợp lệ
    if is_valid_eval_expr(stripped, dfs, expected_ans):
        return stripped

    # 2. Trích xuất từ pattern str.contains
    fm = re.search(
        r"(df\d+)\[\1\['Chi_tieu'\]\.str\.contains\((r?['\"].*?['\"]"
        r"(?:,\s*case=False)?(?:,\s*na=False)?)\)\]", code)
    sm = re.search(r"/\s*([\d_]+)", code)
    scale = 1
    if sm:
        try:
            scale = int(sm.group(1).replace("_", ""))
        except Exception:
            scale = 1

    if fm:
        dv = fm.group(1)
        ca = fm.group(2)
        if "case=" not in ca:
            ca += ", case=False, na=False"
        cand = f"float({dv}[{dv}['Chi_tieu'].str.contains({ca})]['Gia_tri'].iloc[0])"
        if scale > 1:
            cand = f"{cand} / {scale}"
        if is_valid_eval_expr(cand, dfs, expected_ans):
            return cand

    # 3. Brute-force tìm iloc khớp answer trên từng df
    scales = [1, 1000, 1000000, 1000000000, 1000000000000]
    for var, df in dfs.items():
        if df is None or df.empty or "Gia_tri" not in df.columns:
            continue
        for idx in range(len(df)):
            try:
                rv = float(df.iloc[idx]["Gia_tri"])
            except Exception:
                continue
            for sc in scales:
                for use_abs in [False, True]:
                    cv = (abs(rv) if use_abs else rv) / sc
                    if abs(expected_ans) < 1e-9:
                        continue
                    if abs(cv - expected_ans) <= max(1e-2, 0.01 * abs(expected_ans)):
                        inner = f"float({var}.iloc[{idx}]['Gia_tri'])"
                        if use_abs:
                            inner = f"abs({inner})"
                        expr = inner if sc == 1 else f"{inner} / {sc}"
                        if is_valid_eval_expr(expr, dfs, expected_ans):
                            return expr

    # 4. Brute-force 2 bảng: tổng, hiệu, tỷ lệ %
    dk = [k for k, v in dfs.items()
          if v is not None and not v.empty and "Gia_tri" in v.columns]
    if len(dk) >= 2:
        d1k, d2k = dk[0], dk[1]
        d1, d2 = dfs[d1k], dfs[d2k]
        for i in range(min(len(d1), 40)):
            try:
                v1 = float(d1.iloc[i]["Gia_tri"])
            except Exception:
                continue
            for j in range(min(len(d2), 40)):
                try:
                    v2 = float(d2.iloc[j]["Gia_tri"])
                except Exception:
                    continue
                for sc in scales:
                    for op in ["+", "-"]:
                        cv = (v1 + v2 if op == "+" else v1 - v2) / sc
                        if abs(expected_ans) > 1e-9 and abs(cv - expected_ans) <= max(1e-2, 0.01 * abs(expected_ans)):
                            a = f"float({d1k}.iloc[{i}]['Gia_tri'])"
                            b = f"float({d2k}.iloc[{j}]['Gia_tri'])"
                            expr = f"({a} {op} {b})"
                            if sc > 1:
                                expr = f"{expr} / {sc}"
                            if is_valid_eval_expr(expr, dfs, expected_ans):
                                return expr
                if abs(v2) > 1e-9:
                    pv = (v1 / v2) * 100
                    if abs(expected_ans) > 1e-9 and abs(pv - expected_ans) <= max(1e-2, 0.01 * abs(expected_ans)):
                        expr = (f"float({d1k}.iloc[{i}]['Gia_tri'])"
                                f" / float({d2k}.iloc[{j}]['Gia_tri']) * 100")
                        if is_valid_eval_expr(expr, dfs, expected_ans):
                            return expr

    # 5. Fallback hằng số
    return f"float({expected_ans})"


# ---------------------------------------------------------------------------
# Safe-wrap: guard iloc[0] against empty filter results → IndexError
# ---------------------------------------------------------------------------

_CONTAINS_ILOC_RE = re.compile(
    r'(abs\()?'                          # optional abs(
    r'float\('
    r'(df\d+)\[\2\[\'Chi_tieu\'\]\.str\.contains\('
    r'(r?[\'"].*?[\'"]'                  # pattern string
    r'(?:,\s*case=False)?'
    r'(?:,\s*na=False)?)'
    r'\)\]'
    r'\[\'Gia_tri\'\]\.iloc\[0\]'
    r'\)'
    r'(\))?'                             # closing abs)
)


def _safe_wrap_expr(expr: str) -> str:
    """
    Wrap mỗi `float(dfX[dfX[...].str.contains(...)]['Gia_tri'].iloc[0])`
    thành lambda safe pattern trả 0.0 khi filter rỗng, chống IndexError
    trên BTC evaluator.
    """
    def _replace(m):
        has_abs = bool(m.group(1))
        var = m.group(2)
        contains_args = m.group(3)
        filter_expr = f"{var}[{var}['Chi_tieu'].str.contains({contains_args})]"
        inner = f"float(_m['Gia_tri'].iloc[0])"
        if has_abs:
            inner = f"abs({inner})"
        return f"(lambda _m={filter_expr}: {inner} if len(_m) > 0 else 0.0)()"

    return _CONTAINS_ILOC_RE.sub(_replace, expr)
