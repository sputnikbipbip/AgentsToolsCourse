from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Annotated

from pydantic import Field

from agent_tools.registry import tool

_BINARY_OPS: dict[type[ast.operator], Callable[[int | float, int | float], int | float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS: dict[type[ast.unaryop], Callable[[int | float], int | float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _check_number(value: int | float) -> int | float:
    if isinstance(value, int) and value.bit_length() > 4096:
        raise ValueError("Result is too large.")
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Result must be finite.")
    return value


def _eval_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        if isinstance(node.value, bool):
            raise ValueError("Boolean values are not allowed in expressions.")
        return _check_number(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("Exponent is too large.")
        if isinstance(node.op, ast.Pow) and left < 0 and not float(right).is_integer():
            raise ValueError("Complex numbers are not supported.")
        return _check_number(_BINARY_OPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        operand = _eval_node(node.operand)
        return _check_number(_UNARY_OPS[type(node.op)](operand))
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


@tool
def calculate(
    expression: Annotated[
        str,
        Field(description="A basic arithmetic expression, e.g. '(12 * 4) + 7.5'"),
    ],
) -> str:
    """Safely evaluate a basic arithmetic expression (+, -, *, /, //, %, **)."""
    if len(expression) > 1000:
        raise ValueError("Expression is too long.")
    tree = ast.parse(expression, mode="eval")
    result = _eval_node(tree)
    return str(result)
