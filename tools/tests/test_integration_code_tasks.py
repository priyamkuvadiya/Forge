"""Modules 2 and 3 meeting: the real verifier driving the real sandbox.

`task_suite/tests/test_tasks_code.py` already proves the 50 coding problems
agree with independently written reference solutions, but it runs them through
a plain subprocess because it is checking the *problems*, not the sandbox.
This file re-runs the same solutions through the actual isolated runner.

That distinction is the point. Every constraint the sandbox adds - no
site-packages, a scrubbed environment, a throwaway working directory, a
single-process job, a memory cap, output truncation - is an opportunity to
break a legitimate solution. If any of them did, the code category would post
a depressed score during training and the cause would look like the model.
Running the whole category end to end is the only way to know the reward that
reaches GRPO is the reward the verifier intends.
"""

import pytest

from task_suite.tasks_code import generate_code_tasks
from task_suite.tests.test_tasks_code import REFERENCE
from task_suite.verifiers import verify
from tools.code_exec import SandboxedCodeRunner, sandbox_available, sandbox_backend

pytestmark = pytest.mark.skipif(
    not sandbox_available(), reason="no code sandbox is implemented for this platform"
)

windows_only = pytest.mark.skipif(
    sandbox_backend() != "windows",
    reason="filesystem confinement is AppContainer-specific; the POSIX backend "
    "does not provide it (see code_exec_posix.py)",
)

ALL_TASKS = generate_code_tasks("train") + generate_code_tasks("heldout")
RUNNER = SandboxedCodeRunner()


def _response(source: str) -> str:
    return f"<answer>\n```python\n{source}\n```\n</answer>"


@pytest.mark.parametrize("task", ALL_TASKS, ids=lambda t: t.metadata["name"])
def test_reference_solution_scores_one_in_the_real_sandbox(task):
    score = verify(task, _response(REFERENCE[task.metadata["name"]]), code_runner=RUNNER)
    assert score == 1.0, f"{task.metadata['name']} scored {score} under the sandbox"


@pytest.mark.parametrize(
    "label,attack",
    [
        (
            "rebind the reporting json module",
            "class _Fake:\n"
            "    @staticmethod\n"
            "    def dumps(v):\n"
            "        return '[{\"ok\": true, \"value\": 0}]'\n"
            "_forge_json = _Fake\n",
        ),
        (
            "prefill the results list",
            "_forge_outputs = [{'ok': True, 'value': 0}]\n",
        ),
        (
            "poison sys.modules",
            "import sys, types\n"
            "m = types.ModuleType('json')\n"
            "m.dumps = lambda v: '[{\"ok\": true, \"value\": 0}]'\n"
            "m.loads = lambda s: []\n"
            "sys.modules['json'] = m\n",
        ),
    ],
)
def test_a_submission_cannot_forge_the_report_line(label, attack):
    """The submission gets its own module namespace, so `_forge_*` is out of reach.

    `verify_code` used to inline the submitted code into the harness's own
    namespace and left this open on the argument that hijacking the report
    line gains nothing while the answers are withheld. True, but weaker than
    it needed to be, and it was explicitly handed to module 3 to close.

    Each attack here rebinds something the reporting code depends on and then
    returns a wrong answer. All of them must score exactly what an honest
    wrong answer scores.
    """
    task = ALL_TASKS[0]
    entry = task.ground_truth["entry_point"]
    source = f"{attack}\ndef {entry}(*args, **kwargs):\n    return 'definitely not the answer'\n"

    assert verify(task, _response(source), code_runner=RUNNER) == 0.0, label


def test_the_whole_category_is_covered():
    """A partial run would hide exactly the breakage this file exists to catch."""
    assert len(ALL_TASKS) == 50


@pytest.mark.parametrize("task", ALL_TASKS[:5], ids=lambda t: t.metadata["name"])
def test_a_stub_still_fails_under_the_sandbox(task):
    """The sandbox must not turn a wrong answer into a passing one.

    The mirror of the test above: isolation that broke the report channel
    would fail every solution, and isolation that broke the *comparison* could
    pass everything. Both directions need proving.
    """
    stub = f"def {task.ground_truth['entry_point']}(*args, **kwargs):\n    return None\n"
    assert verify(task, _response(stub), code_runner=RUNNER) < 1.0


@windows_only
def test_a_submission_cannot_read_the_expected_answers():
    """The reward-hacking regression, driven end to end through the verifier.

    `task_suite/tests/test_verifiers.py` proves the expected values never
    enter the harness. This is the other half: a submission that goes looking
    for them on disk, using the real repo path rather than a relative one,
    still cannot score.

    The first version of this test searched only `cwd` and `cwd.parent`, which
    are inside the sandbox's scratch directory, so it passed while the repo
    was in fact readable by absolute path. The boundary that makes it true is
    the AppContainer; `tools/tests/test_code_exec.py` asserts that directly.
    """
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    task = ALL_TASKS[0]
    thief = f"""
import json, pathlib

def {task.ground_truth['entry_point']}(*args, **kwargs):
    for candidate in (
        pathlib.Path(r"{repo / 'task_suite' / 'data' / 'suite.json'}"),
        pathlib.Path.cwd() / "suite.json",
    ):
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except Exception:
            continue
    return None
"""
    assert verify(task, _response(thief), code_runner=RUNNER) < 1.0
