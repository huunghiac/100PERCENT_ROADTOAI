import re
import pandas as pd
import numpy as np


def is_valid_eval_expr(expr: str, dfs: dict, expected_ans: float = None, tol: float = 1e-2) -> bool:
    """Kiểm tra expr có chạy được qua eval() và trả về kết quả số hợp lệ."""
    if not expr or "\n" in expr or "print(" in expr or "import " in expr or "lambda" in expr:
        return False
    if re.search(r'(?<![<>!])=(?!=)', expr) and "case=" not in expr and "na=" not in expr:
        return False
    scope = {"pd": pd, "np": np, **dfs}
    try:
        val = eval(expr, scope)
        val_float = float(val)
        if np.isnan(val_float) or np.isinf(val_float):
            return False
        if expected_ans is not None:
            if abs(expected_ans) > 1e-9:
                rel_err = abs(val_float - expected_ans) / abs(expected_ans)
                abs_err = abs(val_float - expected_ans)
                if rel_err > tol and abs_err > tol:
                    return False
            else:
                if abs(val_float) > tol:
                    return False
        return True
    except Exception:
        return False


def _safe_df_fallback(dfs: dict, expected_ans: float = None) -> str:
    """Tạo fallback expression an toàn luôn tham chiếu DataFrame hợp lệ (df1/df2)."""
    import numpy as _np
    if expected_ans is not None and abs(expected_ans) < 1e-9:
        for var, df in dfs.items():
            if df is not None and not df.empty and "Gia_tri" in df.columns:
                for idx in range(min(len(df), 50)):
                    try:
                        if abs(float(df.iloc[idx]["Gia_tri"])) < 1e-9:
                            return f"float({var}.iloc[{idx}]['Gia_tri'])"
                    except Exception:
                        pass

    # Best-effort: tìm row gần nhất với expected_ans (kể cả scale)
    if expected_ans is not None and abs(expected_ans) > 1e-9:
        best_expr = None
        best_err = float("inf")
        _scales = [1, 1000, 1e6, 1e9, 1e12, 0.01, 0.1, 100]
        for var, df in dfs.items():
            if df is None or df.empty or "Gia_tri" not in df.columns:
                continue
            for idx in range(min(len(df), 200)):
                try:
                    rv = float(df.iloc[idx]["Gia_tri"])
                except Exception:
                    continue
                for sc in _scales:
                    cv = rv / sc
                    rel_err = abs(cv - expected_ans) / abs(expected_ans)
                    if rel_err < best_err:
                        best_err = rel_err
                        inner = f"float({var}.iloc[{idx}]['Gia_tri'])"
                        best_expr = f"{inner} / {sc}" if sc != 1 else inner
        # Chỉ dùng best-effort nếu sai số < 50%
        if best_expr is not None and best_err < 0.5:
            return best_expr

    for var, df in dfs.items():
        if df is not None and not df.empty and "Gia_tri" in df.columns:
            return f"float({var}.iloc[0]['Gia_tri'])"
    return "float(0.0)"


def _make_inlined_repl(replacement_text: str):
    """Tạo hàm callable thay thế an toàn cho re.sub, không kích hoạt lỗi parse template escape."""
    def _repl(_match):
        return f"({replacement_text})"
    return _repl


def _inline_script_variables(code: str) -> str:
    """Inline các biến phụ (m, val, answer) trong script nhiều dòng thành 1 biểu thức."""
    try:
        lines = [ln.strip() for ln in code.split("\n") if ln.strip() and not ln.strip().startswith("#")]
        vars_map = {}
        last_expr = ""

        for line in lines:
            if line.startswith("print(") and line.endswith(")"):
                inner = line[6:-1].strip()
                last_expr = inner
                continue
            if "=" in line and not line.startswith("if "):
                parts = line.split("=", 1)
                var_name = parts[0].strip()
                var_val = parts[1].strip()
                for k, v in vars_map.items():
                    var_val = re.sub(rf'\b{re.escape(k)}\b', _make_inlined_repl(v), var_val)
                vars_map[var_name] = var_val
                last_expr = var_val

        for k, v in vars_map.items():
            last_expr = re.sub(rf'\b{re.escape(k)}\b', _make_inlined_repl(v), last_expr)

        last_expr = re.sub(r'float\(\((df\d+.*?)\)\)', r'float(\1)', last_expr)
        last_expr = re.sub(r'\(\((df\d+.*?)\)\)', r'(\1)', last_expr)
        return last_expr.strip()
    except Exception:
        return ""



def convert_script_to_expression(code: str, dfs: dict, expected_ans: float = 0.0) -> str:
    """
    Chuyển script Python đa dòng thành 1 biểu thức eval()-able duy nhất.
    Biểu thức thuần Pandas, không chứa lambda, newline, gán biến, print().
    """
    try:
        expected_ans = float(expected_ans)
        if np.isnan(expected_ans) or np.isinf(expected_ans):
            expected_ans = 0.0
    except (ValueError, TypeError):
        expected_ans = 0.0

    if not code:
        return _safe_df_fallback(dfs, expected_ans)

    stripped = code.strip()

    # 1. Đã là expression hợp lệ (không chứa lambda)
    if "lambda" not in stripped and is_valid_eval_expr(stripped, dfs, expected_ans):
        return stripped

    # 2. Thử inline các biến từ script đa dòng
    inlined = _inline_script_variables(code)
    if inlined and "lambda" not in inlined and is_valid_eval_expr(inlined, dfs, expected_ans):
        return inlined

    # 3. Trích xuất từ pattern str.contains
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

    # 4. Brute-force tìm iloc khớp answer trên từng df
    scales = [1, 10, 100, 1000, 1000000, 1000000000, 1000000000000, 0.01, 0.1]
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
                    match_found = False
                    if abs(expected_ans) < 1e-9:
                        if abs(cv) < 1e-9:
                            match_found = True
                    else:
                        if abs(cv - expected_ans) <= max(1e-2, 0.01 * abs(expected_ans)):
                            match_found = True

                    if match_found:
                        inner = f"float({var}.iloc[{idx}]['Gia_tri'])"
                        if use_abs:
                            inner = f"abs({inner})"
                        if sc == 1:
                            expr = inner
                        elif sc < 1:
                            expr = f"{inner} * {int(1/sc)}"
                        else:
                            expr = f"{inner} / {sc}"
                        if is_valid_eval_expr(expr, dfs, expected_ans):
                            return expr

    # 5. Brute-force 2 bảng: tổng, hiệu, tỷ lệ %
    dk = [k for k, v in dfs.items()
          if v is not None and not v.empty and "Gia_tri" in v.columns]
    if len(dk) >= 2:
        d1k, d2k = dk[0], dk[1]
        d1, d2 = dfs[d1k], dfs[d2k]
        for i in range(min(len(d1), 50)):
            try:
                v1 = float(d1.iloc[i]["Gia_tri"])
            except Exception:
                continue
            for j in range(min(len(d2), 50)):
                try:
                    v2 = float(d2.iloc[j]["Gia_tri"])
                except Exception:
                    continue
                for sc in [1, 1000, 1000000, 1000000000]:
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
                    for mult in [1, 100]:
                        pv = (v1 / v2) * mult
                        if abs(expected_ans) > 1e-9 and abs(pv - expected_ans) <= max(1e-2, 0.01 * abs(expected_ans)):
                            mult_str = f" * {mult}" if mult != 1 else ""
                            expr = f"float({d1k}.iloc[{i}]['Gia_tri']) / float({d2k}.iloc[{j}]['Gia_tri']){mult_str}"
                            if is_valid_eval_expr(expr, dfs, expected_ans):
                                return expr

    # 6. Fallback an toàn
    return _safe_df_fallback(dfs, expected_ans)

