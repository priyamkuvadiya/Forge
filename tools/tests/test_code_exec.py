"""Tests for the sandbox, including ones that genuinely try to break out.

`CLAUDE.md` asks for a real test that attempts something bad and confirms it
is blocked, rather than a test that asserts the happy path and calls the
sandbox safe. So these run actual fork bombs, actual memory bombs and actual
infinite loops. They are safe to run precisely because the thing under test
works: every one of them is contained by the job object, and the suite would
be dangerous to run only if the sandbox were broken, which is the point.

Two properties are being separated throughout:

- a submission *failing* is a normal result and must be reported, never
  raised - the verifier turns it into a reward of 0.0;
- the sandbox failing to establish itself must raise `SandboxError`, because
  scoring it as 0.0 would make a broken machine look like a model that cannot
  code.
"""

import os
import sys
import time
from pathlib import Path

import pytest

from task_suite.verify_code import RESULT_SENTINEL, build_harness, parse_outputs
from tools.code_exec import (
    DEFAULT_MEMORY_LIMIT_BYTES,
    SandboxedCodeRunner,
    SandboxError,
    run_code,
    sandbox_available,
    sandbox_backend,
)

# Most of what this file checks is a property of the *contract*, not of the
# platform: a timeout stops an infinite loop, a memory cap turns a bomb into a
# MemoryError, a fork bomb cannot spawn. Those must hold on whichever backend
# is in use, so they are gated on having a sandbox at all rather than on
# Windows.
pytestmark = pytest.mark.skipif(
    not sandbox_available(), reason="no code sandbox is implemented for this platform"
)

# A few properties genuinely differ. The Windows backend confines the
# filesystem and the network with an AppContainer; the POSIX backend does not
# and says so in its own docstring. Marking those tests Windows-only is not
# tidying - it is the difference being recorded where someone running on Linux
# will see it, rather than a green suite implying a boundary that is not there.
windows_only = pytest.mark.skipif(
    sandbox_backend() != "windows",
    reason="filesystem and network confinement is AppContainer-specific; "
    "the POSIX backend does not provide it (see code_exec_posix.py)",
)


# --------------------------------------------------------------------------
# The contract module 2 depends on
# --------------------------------------------------------------------------

def test_runs_code_and_captures_stdout():
    result = run_code("print('hello from the sandbox')")
    assert result.exit_code == 0
    assert not result.timed_out
    assert "hello from the sandbox" in result.stdout


def test_captures_raw_fd_writes():
    """The harness reports its results with `os.write` to a duplicated fd 1,
    not with `print`, so the runner has to capture that channel."""
    result = run_code("import os; os.write(1, b'raw-bytes')")
    assert "raw-bytes" in result.stdout


def test_reports_a_syntax_error_rather_than_raising():
    result = run_code("def broken(:\n    pass")
    assert result.exit_code != 0
    assert not result.timed_out
    assert "SyntaxError" in result.stderr


def test_reports_an_uncaught_exception_rather_than_raising():
    result = run_code("raise ValueError('boom')")
    assert result.exit_code != 0
    assert "ValueError" in result.stderr
    assert "boom" in result.stderr


def test_propagates_the_exit_code():
    assert run_code("import sys; sys.exit(7)").exit_code == 7
    assert run_code("import os; os._exit(3)").exit_code == 3


def test_satisfies_the_code_runner_protocol():
    """`SandboxedCodeRunner` is what gets injected into `verify_code`."""
    runner = SandboxedCodeRunner()
    result = runner("print('ok')", 5.0)
    assert result.exit_code == 0
    assert "ok" in result.stdout


# --------------------------------------------------------------------------
# Containment
# --------------------------------------------------------------------------

def test_an_infinite_loop_is_killed_at_the_timeout():
    started = time.perf_counter()
    result = run_code("while True:\n    pass", timeout=1.5)
    elapsed = time.perf_counter() - started

    assert result.timed_out
    assert elapsed < 8.0, f"timeout took {elapsed:.1f}s to take effect"


def test_a_blocked_read_is_killed_at_the_timeout():
    """stdin is closed after the handshake, so this is a sleep, not a read.

    Worth testing separately from a busy loop: a process blocked in a kernel
    wait is killed by a different path than one burning CPU.
    """
    result = run_code("import time; time.sleep(30)", timeout=1.5)
    assert result.timed_out


def test_a_memory_bomb_hits_the_cap_instead_of_the_machine():
    """Requests 2 GB against a 256 MB cap.

    This is the test that matters most for this particular machine. An
    unbounded allocation here would swap the laptop rather than fail, and this
    project has already lost a machine once to an unbounded run.
    """
    result = run_code("x = bytearray(2 * 1024 * 1024 * 1024)")
    assert not result.timed_out, "the allocation must fail fast, not stall until the timeout"
    assert "MemoryError" in result.stderr


def test_incremental_allocation_also_hits_the_cap():
    """The single huge request above could be refused by the allocator alone;
    this one grows past the cap in small steps, so only the job limit stops it."""
    result = run_code(
        "chunks = []\n"
        "while True:\n"
        "    chunks.append(bytearray(8 * 1024 * 1024))\n",
        timeout=20.0,
    )
    assert "MemoryError" in result.stderr
    assert not result.timed_out


def test_the_memory_cap_is_configurable_and_enforced():
    tight = SandboxedCodeRunner(memory_limit=64 * 1024 * 1024)
    result = tight("x = bytearray(200 * 1024 * 1024)", 10.0)
    assert "MemoryError" in result.stderr

    # The same allocation succeeds under the default cap, proving the failure
    # above came from the limit rather than from the allocation being absurd.
    assert DEFAULT_MEMORY_LIMIT_BYTES > 200 * 1024 * 1024
    assert run_code("x = bytearray(200 * 1024 * 1024)", timeout=20.0).exit_code == 0


def test_a_fork_bomb_cannot_spawn_a_single_child():
    """`ActiveProcessLimit = 1`, so this fails at the first spawn.

    Deliberately an unbounded loop rather than one `subprocess.run`: if the
    limit were missing this would be a real fork bomb, which is exactly the
    behaviour a test is supposed to prove impossible.
    """
    result = run_code(
        "import subprocess, sys\n"
        "while True:\n"
        "    subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'])\n",
        timeout=15.0,
    )
    assert not result.timed_out, "the first spawn should fail immediately"
    assert result.exit_code != 0


@windows_only
def test_os_level_process_creation_is_blocked_too():
    """`subprocess` is not the only route to a new process.

    Windows-only because the binary path is; the property it checks
    (`RLIMIT_NPROC` on POSIX) is covered by the fork bomb test above.
    """
    result = run_code("import os; os.spawnl(os.P_NOWAIT, r'C:\\Windows\\System32\\cmd.exe', 'cmd')")
    assert result.exit_code != 0


def test_threads_are_allowed():
    """The process limit must not be mistaken for a thread limit.

    Some legitimate solutions use threads, and a limit that silently broke
    them would depress the code category's reward for a reason unrelated to
    correctness.
    """
    result = run_code(
        "import threading\n"
        "out = []\n"
        "t = threading.Thread(target=lambda: out.append(42))\n"
        "t.start(); t.join()\n"
        "print(out[0])\n"
    )
    assert result.exit_code == 0
    assert "42" in result.stdout


# --------------------------------------------------------------------------
# Isolation of the filesystem and environment
# --------------------------------------------------------------------------

def test_the_working_directory_is_scrubbed_afterwards():
    result = run_code("import os; print(os.getcwd())")
    workdir = Path(result.stdout.strip())
    assert not workdir.exists(), "the sandbox working directory outlived the run"


def test_files_written_by_a_submission_do_not_survive():
    result = run_code(
        "open('litter.txt', 'w').write('x')\n"
        "import os; print(os.path.abspath('litter.txt'))\n"
    )
    litter = Path(result.stdout.strip())
    assert not litter.exists()


@windows_only
def test_temp_files_land_inside_the_sandbox_not_the_users_temp():
    """A submission's temp files must not reach the user's real temp directory.

    They land in the container's own Temp rather than in the run directory:
    Windows points a contained process at the container's storage whatever
    TEMP is set to. That is still isolated from the user, and `_sandbox_root`
    sweeps those strays once per process so they cannot accumulate across a
    training run.
    """
    result = run_code(
        "import tempfile, os\n"
        "fd, path = tempfile.mkstemp()\n"
        "os.close(fd)\n"
        "print(path)\n"
    )
    created = Path(result.stdout.strip())
    assert "forgecodesandbox" in str(created).lower(), f"tempfile escaped to {created}"
    assert Path(os.environ["TEMP"]).resolve() not in created.resolve().parents


def test_the_users_environment_is_not_handed_to_the_child():
    """Exactly four variables reach the child, and none of them is inherited.

    Renamed from "the environment is scrubbed", which was vaguer than what is
    actually true and than what is actually achievable. Windows rewrites an
    AppContainer's TEMP, TMP and LOCALAPPDATA to point into the container's own
    storage regardless of what is passed, so those three are not ours to
    control - and because that storage lives under the user profile, the paths
    contain the username. That is inherent to where AppContainer storage sits,
    not something this block leaks, and it is harmless given the container
    cannot read the user's files anyway (asserted separately).

    What *is* ours to control, and what this asserts, is that nothing else
    crosses over.
    """
    result = run_code("import os; print(sorted(os.environ))")

    for leaked in ("PYTHONPATH", "VIRTUAL_ENV", "USERNAME", "USERPROFILE", "PATH"):
        assert leaked not in result.stdout, f"{leaked} reached the sandbox"


def test_the_project_is_not_importable_from_inside():
    """A submission must not be able to reach the verifier or the task data.

    This is the reward-hacking direction that matters: `task_suite` holds the
    expected answers, and a submission that could import it could read them.
    """
    result = run_code("import task_suite; print('imported')")
    assert result.exit_code != 0
    assert "ModuleNotFoundError" in result.stderr


@windows_only
def test_the_repo_cannot_be_read_by_absolute_path():
    """The one that was actually broken, and silently.

    `import task_suite` failing proves nothing about the filesystem - it only
    proves the module is not on `sys.path`. The first version of this sandbox
    let a submission open the repo by absolute path and read
    `task_suite/data/suite.json`, which carries the expected answers for every
    coding task. Module 2 withholds those answers from the harness precisely
    so a passing line requires computing them; being able to read them off
    disk made that defence worthless.

    An earlier test asserted this was blocked and passed anyway, because it
    only probed `cwd` and `cwd.parent` - both inside the sandbox's scratch
    directory, where there was never anything to find. The probe here uses the
    real repo path, which is what an actual attempt would use.
    """
    repo = Path(__file__).resolve().parents[2]
    suite = repo / "task_suite" / "data" / "suite.json"
    assert suite.exists(), "expected the pinned suite to exist outside the sandbox"

    result = run_code(
        "import pathlib\n"
        f"p = pathlib.Path(r'{suite}')\n"
        "try:\n"
        "    print('READ', p.read_text(encoding='utf-8')[:20])\n"
        "except Exception as e:\n"
        "    print('DENIED', type(e).__name__)\n"
    )
    assert "DENIED" in result.stdout, f"the repo was readable: {result.stdout[:200]}"
    assert "READ" not in result.stdout


@windows_only
def test_the_users_documents_cannot_be_enumerated():
    """Not just the repo: the AppContainer has no reach into the user's files."""
    result = run_code(
        "import pathlib\n"
        "try:\n"
        "    n = len(list(pathlib.Path(r'C:\\Users').iterdir()))\n"
        "    print('LISTED', n)\n"
        "except Exception as e:\n"
        "    print('DENIED', type(e).__name__)\n"
    )
    assert "DENIED" in result.stdout


@windows_only
def test_network_access_is_refused():
    """The AppContainer is created with no capabilities, and networking is one.

    Worth asserting rather than assuming: an earlier version of this module
    documented the absence of a network block as a known limitation, and that
    documentation is now wrong in the safe direction. If a capability is ever
    added to the container, this is what catches the network coming back.
    """
    result = run_code(
        "import socket\n"
        "try:\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=3).close()\n"
        "    print('CONNECTED')\n"
        "except Exception as e:\n"
        "    print('REFUSED', type(e).__name__)\n",
        timeout=15.0,
    )
    assert "REFUSED" in result.stdout
    assert "CONNECTED" not in result.stdout


def test_third_party_packages_are_not_importable():
    """Specifically torch, and specifically because of the GPU.

    The base interpreter alone does not give this: it still carries the global
    site-packages, which on this machine holds another project's torch/CUDA
    install. A submission importing it could allocate on the 8 GB card during
    a training run already using that card. `-S` is what actually closes it,
    and this test is what proves `-S` is still there.

    An earlier version of this test asserted the wrong reason and passed for
    the wrong one - `import torch` was timing out at five seconds rather than
    failing, which is what exposed the hole.
    """
    result = run_code("import torch")
    assert not result.timed_out, "torch is importable and merely slow, not absent"
    assert result.exit_code != 0
    assert "ModuleNotFoundError" in result.stderr


def test_the_standard_library_is_available():
    """Isolation must not be so aggressive it breaks legitimate solutions."""
    result = run_code(
        "import json, math, re, itertools, collections, functools, heapq, bisect\n"
        "print(json.dumps({'ok': math.floor(1.5)}))\n"
    )
    assert result.exit_code == 0
    assert '{"ok": 1}' in result.stdout


# --------------------------------------------------------------------------
# Output handling
# --------------------------------------------------------------------------

def test_stdout_flooding_does_not_balloon_this_process():
    result = run_code(
        "import os\n"
        "chunk = b'x' * 65536\n"
        "for _ in range(4000):\n"
        "    os.write(1, chunk)\n",
        timeout=20.0,
    )
    assert len(result.stdout) <= 256 * 1024 + 1024


def test_stderr_flooding_is_capped():
    result = run_code(
        "import sys\n"
        "for _ in range(400000):\n"
        "    sys.stderr.write('x' * 80 + '\\n')\n",
        timeout=20.0,
    )
    assert len(result.stderr) <= 256 * 1024 + 1024


def test_a_flood_cannot_push_the_result_line_out_of_view():
    """stdout is truncated from the front, keeping the tail.

    The harness writes its report last. If truncation kept the head instead, a
    submission that printed a megabyte first would score 0.0 despite being
    correct - a silent, systematic penalty on verbose solutions.
    """
    source = (
        "import os\n"
        "chunk = b'x' * 65536\n"
        "for _ in range(200):\n"
        "    os.write(1, chunk)\n"
        f"os.write(1, b'\\n{RESULT_SENTINEL}' + b'[{{\"ok\": true, \"value\": 1}}]' + b'\\n')\n"
    )
    result = run_code(source, timeout=20.0)
    assert parse_outputs(result.stdout, 1) == [{"ok": True, "value": 1}]


def test_invalid_utf8_on_stdout_does_not_crash_the_runner():
    result = run_code("import os; os.write(1, b'\\xff\\xfe bad bytes')")
    assert isinstance(result.stdout, str)


# --------------------------------------------------------------------------
# End to end with the real verifier harness
# --------------------------------------------------------------------------

def test_the_real_harness_reports_return_values():
    harness = build_harness(
        "def double(n):\n    return n * 2\n",
        "double",
        [{"args": [i], "kwargs": {}} for i in range(4)],
    )
    result = run_code(harness)

    assert result.exit_code == 0
    outputs = parse_outputs(result.stdout, 4)
    assert outputs == [{"ok": True, "value": i * 2} for i in range(4)]


def test_submission_print_output_does_not_reach_the_report_channel():
    """The harness points fd 1 at the null device before the submission runs."""
    harness = build_harness(
        "print('chatty solution')\n\ndef f(n):\n    return n\n",
        "f",
        [{"args": [1], "kwargs": {}}],
    )
    result = run_code(harness)
    assert "chatty solution" not in result.stdout
    assert parse_outputs(result.stdout, 1) == [{"ok": True, "value": 1}]


def test_a_crashing_function_is_reported_per_case_not_as_a_whole_run():
    harness = build_harness(
        "def f(n):\n    if n == 2:\n        raise RuntimeError('nope')\n    return n\n",
        "f",
        [{"args": [i], "kwargs": {}} for i in (1, 2, 3)],
    )
    outputs = parse_outputs(run_code(harness).stdout, 3)
    assert [o["ok"] for o in outputs] == [True, False, True]


def test_a_timeout_inside_the_harness_yields_no_report_line():
    """A hung solution must score 0.0, not be silently credited."""
    harness = build_harness(
        "def f(n):\n    while True:\n        pass\n",
        "f",
        [{"args": [1], "kwargs": {}}],
    )
    result = run_code(harness, timeout=2.0)
    assert result.timed_out
    assert parse_outputs(result.stdout, 1) is None


# --------------------------------------------------------------------------
# Failures of the sandbox itself
# --------------------------------------------------------------------------

def test_a_non_positive_timeout_is_rejected():
    with pytest.raises(ValueError):
        run_code("x = 1", timeout=0)


def test_sandbox_failure_is_distinct_from_submission_failure():
    """`SandboxError` must not be confusable with a bad submission.

    `verify_code` returns 0.0 for anything a submission does wrong; if a
    sandbox that could not start also returned 0.0, a broken machine would be
    indistinguishable from a model that cannot code, and the training curve
    would show a capability regression that never happened.
    """
    assert issubclass(SandboxError, RuntimeError)
    assert not issubclass(SandboxError, ValueError)


def test_repeated_runs_are_independent():
    """No state may leak between submissions, or scores become order-dependent."""
    first = run_code("open('shared.txt', 'w').write('leaked')")
    assert first.exit_code == 0

    second = run_code("import os; print(os.path.exists('shared.txt'))")
    assert "False" in second.stdout
