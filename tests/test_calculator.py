"""Unit tests for the AST-validated financial calculator tool in src/filing_analyst/agent/agents.py."""
import ast
import pytest

from filing_analyst.agent.agents import _eval_ast, calculator


def test_basic_arithmetic_and_parentheses():
    res = calculator.invoke({"expression": "(416161000000 - 391035000000) / 391035000000 * 100"})
    assert "6.42551" in res


def test_currency_comma_and_percent_stripping():
    res = calculator.invoke({"expression": "($416,161,000,000 - $391,035,000,000)"})
    assert "exact: 25126000000.0" in res


def test_allowed_functions_and_unary_ops():
    assert "65.47" in calculator.invoke({"expression": "round((215938 - 130497) / 130497 * 100, 2)"})
    assert "exact: 39130000000.0" in calculator.invoke({"expression": "abs(72880000000 - 112010000000)"})
    assert "exact: 716.924" in calculator.invoke({"expression": "max(416.161, 716.924, 94.827)"})
    assert "exact: -5.0" in calculator.invoke({"expression": "-5"})


def test_rejects_unsafe_ast_nodes():
    unsafe_expressions = [
        "__import__('os').system('echo hacked')",
        "open('/etc/passwd').read()",
        "(1).__class__.__bases__",
        "[x for x in range(5)]",
        "lambda x: x + 1",
    ]
    for expr in unsafe_expressions:
        with pytest.raises(ValueError, match="Unsupported expression element"):
            tree = ast.parse(expr, mode="eval")
            _eval_ast(tree.body)
        tool_out = calculator.invoke({"expression": expr})
        assert tool_out.startswith("Calculator error")
