"""Reward for coding tasks: run the generated function against its tests.

Ground truth shape:
    {"entry_point": str, "tests": [{"args": [...], "kwargs": {...},
                                    "expected": <json value>}, ...]}

Test arguments and expected values are restricted to JSON-native types
(int, float, str, bool, list, dict, None) because they are serialized into
the harness as JSON rather than as `repr`, and a tuple round-tripping into a
list would silently change what the test asserts.

This module never executes anything itself. It builds a self-checking
program and hands it to an injected `runner`, which module 3 supplies as a
real sandbox (subprocess with resource and time limits). That inversion is
deliberate: the verifier stays pure and unit-testable with a fake runner,
and there is exactly one place in the repo where untrusted generated code
actually runs, so exactly one place to harden and test for safety.

Scoring is the fraction of tests passed, not all-or-nothing. Coding is the
category where a small policy most often gets the shape right and one edge
case wrong, and a dense signal there is worth more to GRPO than a cliff.
"""

import json
import re
from dataclasses import dataclass
from typing import Protocol

from .protocol import extract_answer

RESULT_SENTINEL = "__FORGE_RESULT__"
DEFAULT_TIMEOUT_SECONDS = 5.0

_FENCE_PATTERN = re.compile(r"```(?:[Pp]ython)?\s*\n(.*?)```", re.DOTALL)


@dataclass(frozen=True)
class CodeRunResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool


class CodeRunner(Protocol):
    """What module 3's sandbox has to provide: run source, return output."""

    def __call__(self, source: str, timeout: float) -> CodeRunResult: ...


def extract_code(answer: str) -> str:
    """Pull the submitted source out of an answer block.

    Takes the last fenced block if the answer is fenced, on the same
    reasoning as `extract_answer` taking the last answer block; falls back to
    the raw text, since a model that emits bare code inside <answer> has
    still answered.
    """
    blocks = _FENCE_PATTERN.findall(answer)
    if blocks:
        return blocks[-1]
    return answer


def build_harness(code: str, entry_point: str, tests: list[dict]) -> str:
    """Wrap submitted code in a program that reports per-test pass/fail.

    The prologue moves the real stdout to a private file descriptor and
    points fd 1 at the null device *before* the submitted code runs. Two
    reasons, one of them the important one: submissions print debugging
    noise that would otherwise have to be parsed around, and — since this is
    a reward function an RL policy is optimizing against — a submission
    cannot reach the channel the results are reported on just by calling
    `print`. This is a speed bump, not a security boundary; the sandbox is
    the security boundary.
    """
    payload = json.dumps(tests)
    return f'''\
import os as _forge_os, sys as _forge_sys

_forge_out = _forge_os.dup(1)
_forge_os.dup2(_forge_os.open(_forge_os.devnull, _forge_os.O_WRONLY), 1)

# ---- submitted code ----
{code}
# ---- end submitted code ----

import json as _forge_json


def _forge_eq(actual, expected):
    if isinstance(actual, bool) != isinstance(expected, bool):
        return False
    if isinstance(actual, float) or isinstance(expected, float):
        try:
            return abs(float(actual) - float(expected)) <= 1e-9 * max(1.0, abs(float(expected)))
        except (TypeError, ValueError):
            return False
    return actual == expected


_forge_tests = _forge_json.loads({payload!r})
_forge_results = []
for _forge_case in _forge_tests:
    try:
        _forge_value = {entry_point}(*_forge_case["args"], **_forge_case.get("kwargs", {{}}))
        _forge_results.append(bool(_forge_eq(_forge_value, _forge_case["expected"])))
    except Exception:
        _forge_results.append(False)

_forge_os.write(
    _forge_out,
    ("\\n{RESULT_SENTINEL}" + _forge_json.dumps(_forge_results) + "\\n").encode(),
)
'''


def parse_results(stdout: str, expected_count: int) -> list[bool] | None:
    """Read the harness's result line, or None if it never got there."""
    marker_at = stdout.rfind(RESULT_SENTINEL)
    if marker_at == -1:
        return None

    line = stdout[marker_at + len(RESULT_SENTINEL):].split("\n", 1)[0].strip()
    try:
        results = json.loads(line)
    except json.JSONDecodeError:
        return None

    if not isinstance(results, list) or len(results) != expected_count:
        return None
    if not all(isinstance(r, bool) for r in results):
        return None
    return results


def verify_code(
    response: str,
    ground_truth: dict,
    runner: CodeRunner,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> float:
    answer = extract_answer(response)
    if answer is None:
        return 0.0

    code = extract_code(answer)
    if not code.strip():
        return 0.0

    tests = ground_truth["tests"]
    if not tests:
        raise ValueError("code task has no test cases; it is not verifiable")

    harness = build_harness(code, ground_truth["entry_point"], tests)
    outcome = runner(harness, timeout)

    # A timeout or a crash before the result line means nothing was proven.
    results = parse_results(outcome.stdout, len(tests))
    if results is None:
        return 0.0

    return sum(results) / len(results)
