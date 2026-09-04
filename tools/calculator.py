"""Arithmetic evaluator the agent calls instead of doing sums in its head.

Parsed with `ast` and walked node by node against a whitelist — never
`eval()`. The obvious shortcut here is `eval(expr, {"__builtins__": {}})`,
and it is not safe: expression syntax alone reaches attribute access and
subscripting, which is enough to walk from a harmless-looking literal back
to the interpreter internals. A whitelist walk has no such surface, because
a node type that isn't explicitly allowed is a hard error before anything
runs.

The other half of the job is refusing expressions that are *valid* but
hostile to evaluate. `9**9**9` is four characters of arithmetic that pins a
core and exhausts memory; nothing about it is a syntax problem. Since an RL
policy will emit millions of expressions during training and is under no
obligation to emit sensible ones, cost guards are a correctness requirement
here, not defensive padding — a single un-guarded power would hang a
training run with no error and no obvious cause.

The tool never raises at the call boundary: `calculate()` returns a result
object carrying either a value or an error string. A malformed expression is
ordinary agent behaviour that the policy should see, learn from and retry —
not an exception that tears down the rollout.
"""

import ast
import math
from dataclasses import dataclass

# Cost guards. All three are deliberately generous relative to what the math
# and multi_tool categories actually need (the largest legitimate exponent in
# the suite is a 12-year compound-interest term) and still nowhere near
# expensive enough to notice.
MAX_EXPRESSION_CHARS = 512
MAX_AST_NODES = 256
MAX_POW_EXPONENT = 1024
# Cap on the estimated size of an integer power's result. 4096 bits is a
# ~1233-digit number: far past any real answer, far short of anything slow.
MAX_POW_RESULT_BITS = 4096


class CalculatorError(Exception):
    """Raised for any expression the calculator declines to evaluate."""


@dataclass(frozen=True)
class CalcResult:
    """What the agent gets back. `ok` decides which of the other two is set."""

    ok: bool
    value: float | int | None
    error: str | None
    expression: str

    def render(self) -> str:
        """One line for the agent transcript."""
        if self.ok:
            return f"{self.expression} = {format_number(self.value)}"
        return f"error: {self.error}"


def format_number(value: float | int) -> str:
    """Format a result without scientific notation or float noise.

    `repr(0.1 + 0.2)` is `0.30000000000000004`; handing that to a policy that
    then has to state an answer "to the nearest cent" invites it to copy the
    noise through. Twelve significant decimals is well past the precision any
    task in the suite asks for and short enough to read.
    """
    if isinstance(value, int):
        return str(value)
    if value == int(value) and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.12f}".rstrip("0").rstrip(".")


# Binary operators, each mapped to the function that applies it.
_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}

_UNARY_OPS = {
    ast.UAdd: lambda a: +a,
    ast.USub: lambda a: -a,
}

# Functions a policy plausibly reaches for on these tasks. Deliberately
# excludes `factorial`, which is a one-argument way to burn a CPU forever.
_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": lambda *args: sum(args),
    "sqrt": math.sqrt,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "pow": lambda a, b: _guarded_pow(a, b),
}


def _guarded_pow(base: float | int, exponent: float | int) -> float | int:
    """Apply `**` only when the result is cheap to compute.

    Two separate rejections, because they fail differently. A huge exponent
    on a float overflows fast and harmlessly; a huge exponent on an int
    allocates until the machine gives up. The bit estimate catches the second
    without penalising the first.
    """
    if abs(exponent) > MAX_POW_EXPONENT:
        raise CalculatorError(
            f"exponent {format_number(exponent)} exceeds the limit of {MAX_POW_EXPONENT}"
        )

    if isinstance(base, int) and isinstance(exponent, int) and exponent > 0 and base not in (0, 1, -1):
        estimated_bits = exponent * math.log2(abs(base))
        if estimated_bits > MAX_POW_RESULT_BITS:
            raise CalculatorError("result is too large to compute")

    try:
        return base**exponent
    except (OverflowError, ZeroDivisionError) as exc:
        raise CalculatorError(str(exc)) from exc


def _check_number(value: object) -> float | int:
    """Reject anything that isn't a finite real number.

    `bool` is excluded on purpose even though it is an `int` subclass: it can
    only arrive here through a whitelisted call, and letting `True` flow on
    as `1` would make the tool's output depend on a coincidence of Python's
    type hierarchy rather than on the arithmetic.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalculatorError(f"unsupported value of type {type(value).__name__}")
    if isinstance(value, float) and not math.isfinite(value):
        raise CalculatorError("result is not a finite number")
    return value


def _eval_node(node: ast.AST) -> float | int:
    if isinstance(node, ast.Constant):
        return _check_number(node.value)

    if isinstance(node, ast.BinOp):
        handler = _BIN_OPS.get(type(node.op))
        if handler is None:
            raise CalculatorError(f"operator {type(node.op).__name__} is not allowed")

        left = _eval_node(node.left)
        right = _eval_node(node.right)

        if isinstance(node.op, ast.Pow):
            return _check_number(_guarded_pow(left, right))
        try:
            return _check_number(handler(left, right))
        except ZeroDivisionError:
            raise CalculatorError("division by zero") from None
        except OverflowError as exc:
            raise CalculatorError(str(exc)) from exc

    if isinstance(node, ast.UnaryOp):
        handler = _UNARY_OPS.get(type(node.op))
        if handler is None:
            raise CalculatorError(f"operator {type(node.op).__name__} is not allowed")
        return _check_number(handler(_eval_node(node.operand)))

    if isinstance(node, ast.Call):
        # `node.func` must be a bare Name. Anything else is an attribute or a
        # computed callable, which is exactly the route this evaluator exists
        # to close off.
        if not isinstance(node.func, ast.Name):
            raise CalculatorError("only direct calls to allowed functions are permitted")
        if node.keywords:
            raise CalculatorError("keyword arguments are not supported")

        function = _FUNCTIONS.get(node.func.id)
        if function is None:
            raise CalculatorError(f"unknown function {node.func.id!r}")

        args = [_eval_node(arg) for arg in node.args]
        try:
            return _check_number(function(*args))
        except CalculatorError:
            raise
        except (TypeError, ValueError) as exc:
            raise CalculatorError(f"{node.func.id}: {exc}") from exc
        except (OverflowError, ZeroDivisionError) as exc:
            raise CalculatorError(str(exc)) from exc

    raise CalculatorError(f"{type(node).__name__} is not allowed in an expression")


def evaluate(expression: str) -> float | int:
    """Evaluate `expression`, raising `CalculatorError` if it isn't allowed."""
    if not expression or not expression.strip():
        raise CalculatorError("empty expression")
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise CalculatorError(
            f"expression is longer than {MAX_EXPRESSION_CHARS} characters"
        )

    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"could not parse expression: {exc.msg}") from exc

    # Counted after parsing but before evaluating, so a deeply nested
    # expression is refused rather than recursed into.
    node_count = sum(1 for _ in ast.walk(tree))
    if node_count > MAX_AST_NODES:
        raise CalculatorError(f"expression has too many terms ({node_count})")

    return _eval_node(tree.body)


def calculate(expression: str) -> CalcResult:
    """Evaluate `expression`, reporting failure as data rather than raising.

    This is the entry point the agent loop uses; `evaluate` is the one tests
    and internal callers use when a failure should be loud.
    """
    try:
        return CalcResult(ok=True, value=evaluate(expression), error=None, expression=expression)
    except CalculatorError as exc:
        return CalcResult(ok=False, value=None, error=str(exc), expression=expression)
    except RecursionError:
        return CalcResult(
            ok=False, value=None, error="expression is nested too deeply", expression=expression
        )
