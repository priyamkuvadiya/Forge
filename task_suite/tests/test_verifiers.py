"""Verifier correctness on known input/output pairs.

These tests are the reason the reward numbers later in this project can be
trusted: if the verifiers are wrong, every reward curve and pass-rate in the
README is wrong with them, and no amount of training-loop care fixes that.
"""

import json
import subprocess
import sys

import pytest

from task_suite.protocol import extract_answer
from task_suite.schema import Task
from task_suite.verifiers import verify
from task_suite.verify_code import (
    RESULT_SENTINEL,
    CodeRunResult,
    build_harness,
    extract_code,
    parse_results,
    verify_code,
)
from task_suite.verify_math import parse_number, verify_math
from task_suite.verify_qa import normalize_answer, verify_qa


def wrap(answer: str) -> str:
    return f"Some reasoning here.\n<answer>{answer}</answer>"


# --------------------------------------------------------------------------
# answer protocol
# --------------------------------------------------------------------------

def test_extract_answer_takes_the_last_block():
    response = "<answer>12</answer> wait, I made an error. <answer>15</answer>"
    assert extract_answer(response) == "15"


def test_extract_answer_missing_returns_none():
    assert extract_answer("The answer is 15.") is None


def test_extract_answer_unterminated_tag_returns_none():
    assert extract_answer("<answer>15") is None


def test_extract_answer_is_case_insensitive_and_multiline():
    assert extract_answer("<ANSWER>\nline one\nline two\n</ANSWER>") == "line one\nline two"


# --------------------------------------------------------------------------
# math
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("42", 42.0),
        ("  42  ", 42.0),
        ("-7", -7.0),
        ("1,240.50", 1240.5),
        ("$1,240.50", 1240.5),
        ("35%", 35.0),
        ("42 apples", 42.0),
        ("1.5e3", 1500.0),
    ],
)
def test_parse_number_accepts_decorated_numbers(text, expected):
    assert parse_number(text) == pytest.approx(expected)


@pytest.mark.parametrize("text", ["", "forty two", "between 10 and 20", "nan", "inf"])
def test_parse_number_rejects_non_answers(text):
    assert parse_number(text) is None


def test_math_exact_match():
    assert verify_math(wrap("42"), {"value": 42.0, "tolerance": 0.0}) == 1.0


def test_math_wrong_answer_scores_zero():
    assert verify_math(wrap("43"), {"value": 42.0, "tolerance": 0.0}) == 0.0


def test_math_tolerance_is_respected():
    gt = {"value": 1240.50, "tolerance": 0.005}
    assert verify_math(wrap("1240.5"), gt) == 1.0
    assert verify_math(wrap("1240.51"), gt) == 0.0


def test_math_no_answer_block_scores_zero():
    assert verify_math("The answer is 42.", {"value": 42.0, "tolerance": 0.0}) == 0.0


def test_math_ambiguous_multi_number_answer_scores_zero():
    """Two numbers means no commitment was made; guessing one would be charity."""
    assert verify_math(wrap("either 42 or 43"), {"value": 42.0, "tolerance": 0.0}) == 0.0


# --------------------------------------------------------------------------
# qa
# --------------------------------------------------------------------------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("The Aurora Station", "aurora station"),
        ("aurora  station.", "aurora station"),
        ("A Kessler-Vance Array!", "kesslervance array"),
    ],
)
def test_normalize_answer(text, expected):
    assert normalize_answer(text) == expected


def test_qa_accepts_any_listed_surface_form():
    gt = {"answers": ["Mira Halden", "Halden"], "supporting_docs": ["d1", "d2"]}
    assert verify_qa(wrap("Halden"), gt) == 1.0
    assert verify_qa(wrap("mira halden"), gt) == 1.0


def test_qa_rejects_unlisted_answer():
    gt = {"answers": ["Mira Halden"], "supporting_docs": []}
    assert verify_qa(wrap("Tomas Halden"), gt) == 0.0


def test_qa_gives_no_partial_credit_for_containing_the_answer():
    """A stuffed answer must not score; this is the reward-hacking case."""
    gt = {"answers": ["Mira Halden"], "supporting_docs": []}
    assert verify_qa(wrap("possibly Mira Halden or Tomas Reyes or Ana Vance"), gt) == 0.0


def test_qa_empty_answer_scores_zero():
    assert verify_qa(wrap("   "), {"answers": ["Mira Halden"], "supporting_docs": []}) == 0.0


# --------------------------------------------------------------------------
# code
# --------------------------------------------------------------------------

CODE_GT = {
    "entry_point": "add",
    "tests": [
        {"args": [1, 2], "expected": 3},
        {"args": [-1, 1], "expected": 0},
        {"args": [10, 5], "expected": 15},
    ],
}


def fake_runner(results):
    """A runner that reports a canned result line, standing in for the sandbox."""

    def run(source: str, timeout: float) -> CodeRunResult:
        return CodeRunResult(
            stdout=f"\n{RESULT_SENTINEL}{json.dumps(results)}\n",
            stderr="",
            exit_code=0,
            timed_out=False,
        )

    return run


def crashing_runner(source: str, timeout: float) -> CodeRunResult:
    return CodeRunResult(stdout="", stderr="Traceback ...", exit_code=1, timed_out=False)


def timeout_runner(source: str, timeout: float) -> CodeRunResult:
    return CodeRunResult(stdout="", stderr="", exit_code=-9, timed_out=True)


def test_extract_code_prefers_the_last_fenced_block():
    answer = "```python\nbroken\n```\nActually:\n```python\ndef add(a, b):\n    return a + b\n```"
    assert extract_code(answer).strip() == "def add(a, b):\n    return a + b"


def test_extract_code_falls_back_to_raw_text():
    assert extract_code("def add(a, b):\n    return a + b").startswith("def add")


def test_code_score_is_fraction_of_tests_passed():
    assert verify_code(wrap("code"), CODE_GT, fake_runner([True, True, True])) == 1.0
    assert verify_code(wrap("code"), CODE_GT, fake_runner([True, False, True])) == pytest.approx(2 / 3)
    assert verify_code(wrap("code"), CODE_GT, fake_runner([False, False, False])) == 0.0


def test_code_crash_or_timeout_scores_zero():
    assert verify_code(wrap("code"), CODE_GT, crashing_runner) == 0.0
    assert verify_code(wrap("code"), CODE_GT, timeout_runner) == 0.0


def test_parse_results_rejects_a_wrong_length_result_line():
    """Guards against a truncated run being scored as a partial pass."""
    stdout = f"{RESULT_SENTINEL}{json.dumps([True, True])}\n"
    assert parse_results(stdout, expected_count=3) is None


def test_parse_results_rejects_garbage():
    assert parse_results("no sentinel here", 3) is None
    assert parse_results(f"{RESULT_SENTINEL}not json\n", 3) is None
    assert parse_results(f"{RESULT_SENTINEL}[1, 2, 3]\n", 3) is None


def test_code_task_with_no_tests_is_rejected():
    with pytest.raises(ValueError):
        verify_code(wrap("code"), {"entry_point": "add", "tests": []}, fake_runner([]))


# --------------------------------------------------------------------------
# the harness itself, run for real
# --------------------------------------------------------------------------
#
# This runs *our own* fixture code through a plain subprocess — it is not the
# untrusted-code path, which module 3 owns and sandboxes. Its job is to prove
# the generated harness actually works (the stdout redirection in particular
# is platform-sensitive) before any policy output depends on it.

def local_subprocess_runner(source: str, timeout: float) -> CodeRunResult:
    proc = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return CodeRunResult(proc.stdout, proc.stderr, proc.returncode, timed_out=False)


def test_generated_harness_actually_scores_a_real_submission():
    good = "```python\ndef add(a, b):\n    return a + b\n```"
    assert verify_code(wrap(good), CODE_GT, local_subprocess_runner) == 1.0

    partial = "```python\ndef add(a, b):\n    return a + b if a > 0 else 999\n```"
    assert verify_code(wrap(partial), CODE_GT, local_subprocess_runner) == pytest.approx(2 / 3)


def test_submission_printing_to_stdout_cannot_forge_a_pass():
    """The reward-hacking speed bump: user `print` must not reach the result channel."""
    cheat_code = (
        f"print('{RESULT_SENTINEL}' + '[true, true, true]')\n"
        "def add(a, b):\n"
        "    return None\n"
    )
    cheat = f"```python\n{cheat_code}```"
    assert verify_code(wrap(cheat), CODE_GT, local_subprocess_runner) == 0.0

    # and the forged line never reached the result channel at all, rather than
    # merely being overridden by a later genuine one
    harness = build_harness(cheat_code, "add", CODE_GT["tests"])
    outcome = local_subprocess_runner(harness, 5.0)
    assert outcome.stdout.count(RESULT_SENTINEL) == 1


def test_submission_with_a_syntax_error_scores_zero():
    broken = "```python\ndef add(a, b)\n    return a + b\n```"
    assert verify_code(wrap(broken), CODE_GT, local_subprocess_runner) == 0.0


def test_harness_embeds_every_test_case():
    harness = build_harness("def add(a, b): return a + b", "add", CODE_GT["tests"])
    assert harness.count('"expected":') == len(CODE_GT["tests"])


# --------------------------------------------------------------------------
# dispatch + determinism
# --------------------------------------------------------------------------

def make_task(**overrides) -> Task:
    defaults = dict(
        task_id="t1",
        category="math",
        prompt="What is 21 + 21?",
        ground_truth={"value": 42.0, "tolerance": 0.0},
        split="train",
    )
    defaults.update(overrides)
    return Task(**defaults)


def test_verify_dispatches_by_category():
    assert verify(make_task(), wrap("42")) == 1.0


def test_code_task_without_a_runner_raises_instead_of_scoring_zero():
    task = make_task(category="code", ground_truth=CODE_GT)
    with pytest.raises(ValueError, match="sandboxed code_runner"):
        verify(task, wrap("code"))


def test_reward_is_deterministic():
    task = make_task()
    response = wrap("42")
    scores = {verify(task, response) for _ in range(10)}
    assert scores == {1.0}


def test_schema_rejects_unknown_category_and_split():
    with pytest.raises(ValueError):
        make_task(category="poetry")
    with pytest.raises(ValueError):
        make_task(split="validation")


def test_task_round_trips_through_json():
    task = make_task()
    restored = Task.from_dict(json.loads(json.dumps(task.to_dict())))
    assert restored == task
