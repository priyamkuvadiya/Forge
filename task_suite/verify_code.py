"""Reward for coding tasks: run the generated function and grade it here.

Ground truth shape:
    {"entry_point": str, "tests": [{"args": [...], "kwargs": {...},
                                    "expected": <json value>}, ...]}

Test arguments and expected values are restricted to JSON-native types
(int, float, str, bool, list, dict, None) because they cross the process
boundary as JSON.

**The expected values never leave this process.** The harness sent to the
sandbox carries only the call arguments; it invokes the function, serializes
whatever came back, and the comparison against `expected` happens here, in
trusted code. This is the design decision that matters, and it was made after
the earlier one failed: a first version had the harness grade itself and
report booleans, which a submission could forge outright — poisoning
`sys.modules["json"]` so `dumps` returned "[true, true, true]" scored 1.0 on
a function that returned the string "definitely not the answer". Since this
is a reward function an RL policy optimizes against, that is not a
hypothetical. Withholding the answers collapses the attack: a submission that
wants a passing line has to print the correct return values, and it has no
way to learn what they are — printing the correct values *is* solving the
task. Everything the sandbox can still lie about, it gains nothing by lying
about.

This module never executes anything itself. It builds the harness and hands
it to an injected `runner`, which module 3 supplies as a real sandbox
(subprocess with resource and time limits). That inversion is deliberate: the
verifier stays pure and unit-testable with a fake runner, and there is
exactly one place in the repo where untrusted generated code actually runs,
so exactly one place to harden and test for safety.

Scoring is the fraction of tests passed, not all-or-nothing. Coding is the
category where a small policy most often gets the shape right and one edge
case wrong, and a dense signal there is worth more to GRPO than a cliff.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Protocol

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


def build_harness(code: str, entry_point: str, calls: list[dict]) -> str:
    """Wrap submitted code in a program that reports what the function returned.

    `calls` carries arguments only — never the expected values; see the module
    docstring. The prologue does two things before the submitted code runs:
    it imports what the epilogue needs, so a submission cannot swap those
    modules out from under it, and it moves the real stdout to a private file
    descriptor with fd 1 pointed at the null device, so a submission's own
    `print` output neither has to be parsed around nor reaches the channel
    results are reported on.

    The submitted code runs in its own module namespace rather than inline in
    this one. An earlier version inlined it, which meant a submission shared
    globals with the reporting code and could rebind the `_forge_*` names it
    depends on. That was left open deliberately - with the answers withheld, a
    submission that hijacks the report line still has to put correct values on
    it - and it is closed here because "it gains nothing" is a weaker
    guarantee than "it cannot".

    The submission is compiled into a fresh module object built by
    `types.ModuleType`, so its globals are its own. The `compile`/`exec` pair
    doing that is not the thing `CLAUDE.md` prohibits: the prohibition is on
    executing generated code in *this* process, and everything in this string
    already runs inside module 3's sandbox. There is no way to define a
    function from source text without executing that text somewhere; what
    matters is where.
    """
    payload = json.dumps(calls)
    return f'''\
import os as _forge_os, sys as _forge_sys, json as _forge_json, types as _forge_types

_forge_out = _forge_os.dup(1)
_forge_os.dup2(_forge_os.open(_forge_os.devnull, _forge_os.O_WRONLY), 1)

_forge_source = {code!r}
_forge_module = _forge_types.ModuleType("submission")
_forge_module.__dict__["__name__"] = "submission"

_forge_outputs = []
try:
    exec(compile(_forge_source, "submission.py", "exec"), _forge_module.__dict__)
    _forge_entry = _forge_module.__dict__[{entry_point!r}]
except BaseException:
    # The submission did not import, or defines no such function. Every case
    # fails, and it fails here rather than once per call.
    _forge_entry = None

for _forge_case in _forge_json.loads({payload!r}):
    if _forge_entry is None:
        _forge_outputs.append({{"ok": False, "value": None}})
        continue
    try:
        _forge_value = _forge_entry(*_forge_case["args"], **_forge_case.get("kwargs", {{}}))
        _forge_json.dumps(_forge_value)  # a value we cannot report faithfully is a failure
        _forge_outputs.append({{"ok": True, "value": _forge_value}})
    except Exception:
        _forge_outputs.append({{"ok": False, "value": None}})

_forge_os.write(
    _forge_out,
    ("\\n{RESULT_SENTINEL}" + _forge_json.dumps(_forge_outputs) + "\\n").encode(),
)
'''


def parse_outputs(stdout: str, expected_count: int) -> list[dict] | None:
    """Read the harness's report line, or None if it never got there."""
    marker_at = stdout.rfind(RESULT_SENTINEL)
    if marker_at == -1:
        return None

    line = stdout[marker_at + len(RESULT_SENTINEL):].split("\n", 1)[0].strip()
    try:
        outputs = json.loads(line)
    except json.JSONDecodeError:
        return None

    if not isinstance(outputs, list) or len(outputs) != expected_count:
        return None
    if not all(isinstance(o, dict) and isinstance(o.get("ok"), bool) for o in outputs):
        return None
    return outputs


def values_equal(actual: Any, expected: Any) -> bool:
    """Compare a returned value to its expected value, here in trusted code.

    Tuples do not survive JSON and arrive as lists, so a function returning
    `(1, 2)` where `[1, 2]` is expected scores as correct. That leniency is
    the price of grading outside the sandbox, and it is a fair trade: the
    problems ask for a value, not for a particular sequence type.
    """
    if isinstance(actual, bool) != isinstance(expected, bool):
        return False
    if isinstance(actual, float) or isinstance(expected, float):
        try:
            return abs(float(actual) - float(expected)) <= 1e-9 * max(1.0, abs(float(expected)))
        except (TypeError, ValueError):
            return False
    return actual == expected


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

    calls = [{"args": t["args"], "kwargs": t.get("kwargs", {})} for t in tests]
    outcome = runner(build_harness(code, ground_truth["entry_point"], calls), timeout)

    # A timeout or a crash before the report line means nothing was proven.
    outputs = parse_outputs(outcome.stdout, len(tests))
    if outputs is None:
        return 0.0

    passed = sum(
        1
        for output, test in zip(outputs, tests)
        if output["ok"] and values_equal(output.get("value"), test["expected"])
    )
    return passed / len(tests)
