"""Safe conversion and execution of ViFinQA pandas queries.

The final query is an auditable consequence of the generated/deterministic
program. It is never searched or modified to resemble a pre-existing answer.
"""

from __future__ import annotations

import ast
import copy
import math
import re
from typing import Any, Mapping

import numpy as np
import pandas as pd


class QueryFormatError(ValueError):
    """A script cannot be represented faithfully as one eval expression."""


class QueryExecutionError(ValueError):
    """A final expression is invalid, non-scalar, non-numeric, or non-finite."""


_ALLOWED_BUILTINS = {
    "abs": abs,
    "all": all,
    "any": any,
    "bool": bool,
    "float": float,
    "int": int,
    "len": len,
    "list": list,
    "max": max,
    "min": min,
    "round": round,
    "sorted": sorted,
    "sum": sum,
    "tuple": tuple,
}

_FORBIDDEN_EXPR_NODES = (
    ast.Await,
    ast.DictComp,
    ast.GeneratorExp,
    ast.Lambda,
    ast.ListComp,
    ast.NamedExpr,
    ast.SetComp,
    ast.Yield,
    ast.YieldFrom,
)


def referenced_variables(expr: str) -> set[str]:
    """Return the exact ``dfN`` variables referenced by *expr*."""

    if not expr:
        return set()
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError:
        return set()
    return {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and re.fullmatch(r"df[1-9]\d*", node.id)
    }


def _validate_expression_ast(expr: str) -> ast.Expression:
    if not expr or "\n" in expr or "\r" in expr:
        raise QueryFormatError("Final pandas query must be one non-empty line")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise QueryFormatError(f"Not a valid Python expression: {exc.msg}") from exc
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_EXPR_NODES):
            raise QueryFormatError(f"Forbidden expression construct: {type(node).__name__}")
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                raise QueryFormatError("Dunder names are forbidden")
            if node.id.startswith("df") and not re.fullmatch(r"df[1-9]\d*", node.id):
                raise QueryFormatError(f"Invalid dataframe variable: {node.id}")
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            raise QueryFormatError("Dunder attributes are forbidden")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in {"eval", "exec", "open", "compile", "globals", "locals", "__import__"}:
                raise QueryFormatError(f"Forbidden call: {node.func.id}")
    return tree


def _coerce_numeric_scalar(value: Any) -> float | int:
    if isinstance(value, pd.Series):
        if len(value) != 1:
            raise QueryExecutionError(f"Query returned a Series with {len(value)} values")
        value = value.iloc[0]
    elif isinstance(value, pd.DataFrame):
        if value.size != 1:
            raise QueryExecutionError(f"Query returned a DataFrame with {value.size} values")
        value = value.iloc[0, 0]
    elif isinstance(value, np.ndarray):
        if value.size != 1:
            raise QueryExecutionError(f"Query returned an array with {value.size} values")
        value = value.item()

    if isinstance(value, (bool, np.bool_)):
        value = int(value)
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise QueryExecutionError(f"Query did not return a numeric scalar: {type(value).__name__}") from exc
    if not math.isfinite(numeric):
        raise QueryExecutionError("Query returned NaN or infinity")
    if numeric.is_integer() and abs(numeric) < 1e15:
        return int(numeric)
    return numeric


def execute_expression(expr: str, dfs: Mapping[str, pd.DataFrame]) -> float | int:
    """Execute a final expression against exactly the supplied evidence."""

    _validate_expression_ast(expr)
    required = referenced_variables(expr)
    missing = sorted(required.difference(dfs))
    if missing:
        raise QueryExecutionError(f"Query references missing evidence variables: {missing}")
    scope = {"__builtins__": _ALLOWED_BUILTINS, "pd": pd, "np": np, **dict(dfs)}
    try:
        value = eval(compile(expr, "<pandas_query>", "eval"), scope, {})
    except Exception as exc:
        raise QueryExecutionError(f"Query execution failed: {type(exc).__name__}: {exc}") from exc
    return _coerce_numeric_scalar(value)


def is_valid_eval_expr(
    expr: str,
    dfs: Mapping[str, pd.DataFrame],
    expected_ans: float | None = None,
    tol: float = 1e-12,
) -> bool:
    """Return whether *expr* executes to a finite numeric scalar.

    ``expected_ans`` and ``tol`` remain for API compatibility but are ignored.
    """

    del expected_ans, tol
    try:
        execute_expression(expr, dfs)
        return True
    except (QueryFormatError, QueryExecutionError):
        return False


class _InlineNames(ast.NodeTransformer):
    def __init__(self, values: Mapping[str, ast.expr]):
        self.values = values

    def visit_Name(self, node: ast.Name):  # noqa: N802
        replacement = self.values.get(node.id)
        if replacement is None:
            return node
        return ast.copy_location(copy.deepcopy(replacement), node)


def _expanded_expression(expr: ast.expr, values: Mapping[str, ast.expr]) -> ast.expr:
    previous = None
    current = copy.deepcopy(expr)
    for _ in range(max(1, len(values) + 1)):
        current = _InlineNames(values).visit(current)
        ast.fix_missing_locations(current)
        rendered = ast.dump(current, include_attributes=False)
        if rendered == previous:
            break
        previous = rendered
    return current


def _script_to_ast_expression(code: str) -> ast.expr:
    try:
        module = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise QueryFormatError(f"Generated script has invalid syntax: {exc.msg}") from exc

    values: dict[str, ast.expr] = {}
    final_expr: ast.expr | None = None
    for statement in module.body:
        if isinstance(statement, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(statement, ast.Assign):
            if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
                raise QueryFormatError("Only simple local assignments can be inlined")
            name = statement.targets[0].id
            values[name] = _expanded_expression(statement.value, values)
            if name in {"answer", "result"}:
                final_expr = values[name]
            continue
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            if statement.value is None:
                raise QueryFormatError("Annotated assignment has no value")
            values[statement.target.id] = _expanded_expression(statement.value, values)
            if statement.target.id in {"answer", "result"}:
                final_expr = values[statement.target.id]
            continue
        if isinstance(statement, ast.Expr):
            expression = statement.value
            if isinstance(expression, ast.Call) and isinstance(expression.func, ast.Name) and expression.func.id == "print":
                if len(expression.args) != 1 or expression.keywords:
                    raise QueryFormatError("print() must contain exactly one result")
                final_expr = _expanded_expression(expression.args[0], values)
            else:
                final_expr = _expanded_expression(expression, values)
            continue
        raise QueryFormatError(
            f"Cannot faithfully inline statement type {type(statement).__name__}; "
            "use the deterministic complex solver"
        )

    if final_expr is None:
        final_expr = values.get("answer") or values.get("result")
    if final_expr is None:
        raise QueryFormatError("Generated script has no answer/result/print expression")
    return _expanded_expression(final_expr, values)


def convert_script_to_expression(
    code: str,
    dfs: Mapping[str, pd.DataFrame],
    expected_ans: float | None = None,
) -> str:
    """Convert straight-line code without fitting it to an expected answer."""

    del expected_ans
    if not code or not code.strip():
        raise QueryFormatError("Generated query is empty")
    stripped = code.strip()
    if "\n" not in stripped and "\r" not in stripped:
        _validate_expression_ast(stripped)
        execute_expression(stripped, dfs)
        return stripped

    expression = ast.unparse(_script_to_ast_expression(stripped)).strip()
    _validate_expression_ast(expression)
    execute_expression(expression, dfs)
    return expression


def _safe_df_fallback(dfs: Mapping[str, pd.DataFrame], expected_ans: float | None = None) -> str:
    """Removed answer-fitting fallback retained as an explicit error API."""

    del dfs, expected_ans
    raise QueryFormatError("Arbitrary dataframe-row fallback is disabled")


def _safe_wrap_expr(expr: str) -> str:
    """Compatibility shim: validate and return *expr* without lambda rewriting."""

    _validate_expression_ast(expr)
    return expr
