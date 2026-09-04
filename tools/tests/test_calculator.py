"""Tests for the calculator tool.

Two things are being checked, and the second is the one that matters. First,
that the whitelist actually holds — every route out of arithmetic and into
the interpreter is refused, and every expression that is cheap to write but
expensive to evaluate is refused too. Second, that the tool is *sufficient*:
for every math task in the pinned suite, the arithmetic that task requires
can be expressed and evaluated here to within the task's own tolerance. A
calculator that is safe but can't solve the suite would cap the achievable
reward for reasons the training curve would never reveal.
"""

import math

import pytest

from task_suite.registry import load_suite
from tools.calculator import (
    MAX_AST_NODES,
    MAX_EXPRESSION_CHARS,
    CalculatorError,
    calculate,
    evaluate,
    format_number,
)


# --------------------------------------------------------------------------
# Arithmetic
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "expression,expected",
    [
        ("1 + 1", 2),
        ("2 * 3 + 4", 10),
        ("2 * (3 + 4)", 14),
        ("-5 + 2", -3),
        ("+7", 7),
        ("7 / 2", 3.5),
        ("7 // 2", 3),
        ("7 % 3", 1),
        ("2 ** 10", 1024),
        ("2 ** -1", 0.5),
        ("1.5 * 4", 6.0),
        ("16.99 * 509 * (1 - 12 / 100)", 7610.1608),
    ],
)
def test_evaluates_basic_arithmetic(expression, expected):
    assert evaluate(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression,expected",
    [
        ("abs(-4)", 4),
        ("round(3.14159, 2)", 3.14),
        ("round(2.5)", 2),  # Python's banker's rounding, same as the suite uses
        ("min(3, 1, 2)", 1),
        ("max(3, 1, 2)", 3),
        ("sum(1, 2, 3)", 6),
        ("sqrt(16)", 4.0),
        ("floor(3.9)", 3),
        ("ceil(3.1)", 4),
        ("log10(1000)", 3.0),
        ("pow(2, 8)", 256),
    ],
)
def test_evaluates_whitelisted_functions(expression, expected):
    assert evaluate(expression) == pytest.approx(expected)


def test_operator_precedence_is_python_precedence():
    assert evaluate("2 + 3 * 4 ** 2") == 50
    assert evaluate("(2 + 3) * 4") == 20


def test_deep_precision_is_preserved():
    """The tool must not round on the agent's behalf.

    Math tasks state their own rounding ("to the nearest cent"), and a tool
    that silently rounded would either pre-empt that instruction or corrupt an
    intermediate value in a multi-step calculation.
    """
    assert evaluate("1 / 3") == pytest.approx(0.3333333333333333, abs=1e-15)


# --------------------------------------------------------------------------
# The whitelist: routes out of arithmetic
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "expression",
    [
        "__import__('os').system('echo hi')",
        "open('/etc/passwd')",
        "().__class__",
        "(1).__class__.__bases__",
        "[1, 2, 3][0]",
        "{'a': 1}['a']",
        "(1, 2)",
        "lambda: 1",
        "[i for i in range(10)]",
        "x + 1",
        "eval('1+1')",
        "exec('x=1')",
        "globals()",
        "1 if True else 2",
        "1 < 2",
        "'abc' * 3",
        "True + 1",
        "None",
        "1 & 2",
        "1 << 2",
        "~1",
    ],
)
def test_rejects_anything_that_is_not_arithmetic(expression):
    with pytest.raises(CalculatorError):
        evaluate(expression)


def test_rejects_statements_outright():
    """`mode="eval"` means a statement is a parse error, not a security check.

    Worth pinning anyway: it is the reason the evaluator never has to reason
    about assignment, imports or control flow at all.
    """
    for source in ("import os", "x = 1", "print('hi')\nprint('there')"):
        with pytest.raises(CalculatorError):
            evaluate(source)


def test_error_message_names_the_problem():
    """A policy has to be able to learn from the failure, so it must be legible."""
    result = calculate("__import__('os')")
    assert not result.ok
    assert "__import__" in result.error


# --------------------------------------------------------------------------
# Cost guards: expressions that are valid but hostile
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "expression",
    [
        "9 ** 9 ** 9",
        "2 ** 100000",
        "10 ** 5000",
        "pow(2, 999999)",
        "99999 ** 9999",
    ],
)
def test_rejects_power_bombs(expression):
    """These are the ones that would hang a training run rather than error.

    Each is a handful of characters and a valid arithmetic expression; none
    completes in bounded time or memory. An RL policy emitting millions of
    expressions will produce them by accident.
    """
    with pytest.raises(CalculatorError):
        evaluate(expression)


def test_allows_powers_that_are_merely_large():
    """The guard must not be so tight it refuses real work."""
    assert evaluate("99 ** 500") == 99**500
    assert evaluate("1.0475 ** 12") == pytest.approx(1.0475**12)


def test_rejects_overlong_expressions():
    with pytest.raises(CalculatorError):
        evaluate("1 + " * MAX_EXPRESSION_CHARS + "1")


def test_rejects_expressions_with_too_many_terms():
    expression = "+".join("1" for _ in range(MAX_AST_NODES))
    with pytest.raises(CalculatorError):
        evaluate(expression)


@pytest.mark.parametrize("expression", ["1 / 0", "1 // 0", "1 % 0", "1.0 / 0"])
def test_rejects_division_by_zero(expression):
    with pytest.raises(CalculatorError):
        evaluate(expression)


@pytest.mark.parametrize("expression", ["1e308 * 10", "exp(10000)", "sqrt(-1)", "log(0)"])
def test_rejects_non_finite_results(expression):
    """inf and nan are not answers; returning one would poison the arithmetic
    downstream and give the verifier something it cannot score."""
    with pytest.raises(CalculatorError):
        evaluate(expression)


@pytest.mark.parametrize("expression", ["", "   ", "1 +", "((1)", "2 ++* 3"])
def test_rejects_empty_and_malformed_expressions(expression):
    with pytest.raises(CalculatorError):
        evaluate(expression)


def test_deeply_nested_expression_does_not_crash_the_process():
    """Whatever this does, it must come back as an error, not a traceback."""
    result = calculate("(" * 200 + "1" + ")" * 200)
    assert isinstance(result.ok, bool)


# --------------------------------------------------------------------------
# The call boundary
# --------------------------------------------------------------------------

def test_calculate_never_raises():
    for expression in ["1 + 1", "__import__('os')", "", "9 ** 9 ** 9", "1/0", "@@@"]:
        result = calculate(expression)
        assert result.ok is (result.value is not None)
        assert result.ok != (result.error is not None)


def test_calculate_is_deterministic():
    first = calculate("16.99 * 509 * (1 - 12 / 100)")
    second = calculate("16.99 * 509 * (1 - 12 / 100)")
    assert first == second


@pytest.mark.parametrize(
    "value,expected",
    [
        (2, "2"),
        (4.0, "4"),
        (3.5, "3.5"),
        (0.1 + 0.2, "0.3"),
        (7610.16, "7610.16"),
        (-3.25, "-3.25"),
    ],
)
def test_format_number_drops_float_noise(value, expected):
    assert format_number(value) == expected


def test_render_is_readable():
    assert calculate("2 + 2").render() == "2 + 2 = 4"
    assert calculate("1/0").render().startswith("error: ")


# --------------------------------------------------------------------------
# Sufficiency: can this tool actually solve the suite's math category?
# --------------------------------------------------------------------------

def _expression_for(template: str, params: dict) -> str:
    """Rebuild each math template's arithmetic as an expression string.

    This mirrors what a policy would have to type into the tool. It is written
    from the prompt's wording rather than copied from `tasks_math.py`, so it
    checks the calculator against the task as stated, not against the
    generator's internals.
    """
    if template == "bulk_discount":
        subtotal = f"{params['unit_price']} * {params['quantity']}"
        if params["quantity"] > params["threshold"]:
            return f"round({subtotal} * (1 - {params['discount']} / 100), 2)"
        return f"round({subtotal}, 2)"

    if template == "compound_interest":
        return f"round({params['principal']} * (1 + {params['rate']} / 100) ** {params['years']}, 2)"

    if template == "distance_from_speed":
        return f"round({params['speed_kmh']} * {params['minutes']} / 60, 2)"

    if template == "percentage_change":
        before, after = params["before"], params["after"]
        return f"round(({after} - {before}) / {before} * 100, 2)"

    if template == "combined_work_rate":
        return f"round(1 / (1 / {params['hours_a']} + 1 / {params['hours_b']}), 3)"

    if template == "weighted_grade":
        terms = " + ".join(
            f"{w} * {s}" for w, s in zip(params["weights"], params["scores"])
        )
        return f"round(({terms}) / 100, 2)"

    if template == "fuel_cost":
        return (
            f"round({params['distance_km']} * {params['litres_per_100km']} / 100 "
            f"* {params['price_per_litre']}, 2)"
        )

    if template == "flooring_cost":
        return f"round({params['length_m']} * {params['width_m']} * {params['price_per_sqm']}, 2)"

    if template == "successive_percentages":
        return (
            f"round({params['start']} * (1 + {params['rise_pct']} / 100) "
            f"* (1 - {params['fall_pct']} / 100), 2)"
        )

    if template == "currency_conversion":
        return (
            f"round({params['amount_usd']} * {params['rate']} "
            f"* (1 - {params['fee_pct']} / 100), 2)"
        )

    raise AssertionError(f"no expression written for math template {template!r}")


def _math_tasks():
    splits = load_suite()
    return [task for tasks in splits.values() for task in tasks if task.category == "math"]


def test_calculator_solves_every_math_task_in_the_suite():
    tasks = _math_tasks()
    assert len(tasks) == 210, "suite changed; this test's coverage claim needs revisiting"

    for task in tasks:
        expression = _expression_for(task.metadata["template"], task.metadata["params"])
        result = calculate(expression)

        assert result.ok, f"{task.task_id}: calculator refused {expression!r} ({result.error})"

        expected = task.ground_truth["value"]
        tolerance = task.ground_truth["tolerance"]
        assert abs(result.value - expected) <= tolerance, (
            f"{task.task_id}: {expression} gave {result.value}, expected {expected}"
        )


def test_every_math_template_is_covered_by_the_sufficiency_test():
    """Guards against a new template slipping in with no coverage here."""
    templates = {task.metadata["template"] for task in _math_tasks()}
    assert len(templates) == 10
    for template in templates:
        params = next(
            t.metadata["params"] for t in _math_tasks() if t.metadata["template"] == template
        )
        assert _expression_for(template, params)


def test_expression_strings_stay_inside_the_length_limit():
    """The sufficiency test would be hollow if real expressions were near the cap."""
    longest = max(
        len(_expression_for(t.metadata["template"], t.metadata["params"])) for t in _math_tasks()
    )
    assert longest < MAX_EXPRESSION_CHARS / 2, f"longest real expression is {longest} chars"


def test_math_is_not_trivially_doable_without_the_tool():
    """Sanity-check module 2's claim that these sums are awkward on purpose.

    If the answers were round numbers a model could produce mentally, a reward
    improvement on this category would say nothing about tool use.
    """
    values = [t.ground_truth["value"] for t in _math_tasks()]
    non_integer = sum(1 for v in values if not float(v).is_integer())
    assert non_integer / len(values) > 0.9
