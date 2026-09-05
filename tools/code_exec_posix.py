"""The POSIX half of the sandbox: `fork` plus `setrlimit`.

**This file has never been executed.** It was written on a Windows machine
with no WSL and no container runtime, so there was no way to run it, and the
tests for it skip on the only platform available here. That is stated at the
top rather than buried, because the rest of this project's claims rest on
having actually run things. Treat this as untested code that looks right, and
the first thing to do on a Linux machine is run `pytest tools/tests -q` and
fix whatever this got wrong.

It exists because the Windows sandbox made the coding category unreproducible
anywhere else. `CLAUDE.md` requires the eval numbers to be reproducible, and a
runner that only works on one laptop fails that in the most basic way: a
reviewer cloning the repo on Linux would get 90-odd skipped tests and no way
to check the headline result.

The design mirrors `code_exec.py` deliberately - same `CodeRunner` contract,
same output-to-files handling, same "a submission failing is a result, a
sandbox failing to start is an exception" split - so that the two are readable
side by side and neither drifts into being the special case.

What differs is the isolation primitive. Windows needed an AppContainer to
stop a submission reading the repo; POSIX gets the equivalent from
`setrlimit` plus a chroot-free approach that this file does *not* attempt:

**The filesystem is not confined here, and that is a real gap.** Proper
confinement needs a mount namespace (`unshare`), which needs either root or
user namespaces enabled, neither of which can be assumed. So on POSIX a
submission can still read the repo and therefore the expected answers - the
exact hole that was closed on Windows. `RLIMIT_*` bounds resource use, not
reach. Anyone running training on Linux should either enable user namespaces
and add `unshare`, or run the whole thing in a container, and until then
should treat the coding category's reward as trusted-but-verifiable rather
than tamper-proof. This is written down in the README too.
"""

import os
import resource
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from task_suite.verify_code import CodeRunResult

from ._sandbox_common import (
    BOOTSTRAP as _BOOTSTRAP,
    DEFAULT_MEMORY_LIMIT_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    SandboxError,
    read_capped as _read_capped,
    sandbox_interpreter as _sandbox_interpreter,
)

# A submission bounded only by wall clock can still write until the disk
# fills. 64 MB is far past anything the suite's problems produce.
MAX_WRITE_BYTES = 64 * 1024 * 1024

_IS_POSIX = os.name == "posix"


def _apply_limits(memory_limit: int, timeout: float):
    """Built in the parent, run in the child between `fork` and `exec`.

    Everything here has to be async-signal-safe, which is why it only calls
    `setrlimit` and `setsid` and does no allocation or logging.
    """

    def preexec() -> None:
        # New session, so the whole process group can be killed on timeout
        # without the kill racing anything the child spawned.
        os.setsid()

        # Address space rather than RSS: RLIMIT_AS is what makes an
        # over-large allocation fail as MemoryError inside the child instead
        # of the kernel OOM-killing something else on the machine.
        resource.setrlimit(resource.RLIMIT_AS, (memory_limit, memory_limit))

        # CPU seconds as a backstop to the wall-clock timeout. Rounded up so a
        # program that is merely slow is stopped by the wall clock, with its
        # clearer "timed out" signal, rather than by SIGXCPU.
        cpu_seconds = int(timeout) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))

        # No new processes: a fork bomb fails at its first spawn. This is the
        # POSIX equivalent of the job object's ActiveProcessLimit = 1.
        resource.setrlimit(resource.RLIMIT_NPROC, (1, 1))

        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_WRITE_BYTES, MAX_WRITE_BYTES))

        # No core dumps: a 256 MB crash writes a 256 MB file otherwise.
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return preexec


def _environment(workdir: Path) -> dict[str, str]:
    environment = {"TMPDIR": str(workdir), "HOME": str(workdir), "PATH": ""}
    if "LANG" in os.environ:
        environment["LANG"] = os.environ["LANG"]
    return environment


def run_code(
    source: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    memory_limit: int = DEFAULT_MEMORY_LIMIT_BYTES,
) -> CodeRunResult:
    """Run `source` under resource limits. Satisfies the `CodeRunner` contract."""
    if not _IS_POSIX:
        raise SandboxError(
            f"this runner needs POSIX resource limits; os.name is {os.name!r}"
        )
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    workdir = Path(tempfile.mkdtemp(prefix=f"forge-sandbox-{uuid.uuid4().hex[:8]}-"))
    process = None

    try:
        submission = workdir / "submission.py"
        submission.write_text(source, encoding="utf-8")
        bootstrap = workdir / "_bootstrap.py"
        bootstrap.write_text(_BOOTSTRAP, encoding="utf-8")

        stdout_path = workdir / "_stdout"
        stderr_path = workdir / "_stderr"

        with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
            process = subprocess.Popen(
                [
                    _sandbox_interpreter(),
                    "-I",  # isolated: ignore env vars, user site-packages, cwd
                    "-S",  # no site directory: stdlib only
                    "-B",  # no .pyc litter
                    "-X", "utf8",
                    str(bootstrap),
                    str(submission),
                ],
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=err,
                cwd=str(workdir),
                env=_environment(workdir),
                preexec_fn=_apply_limits(memory_limit, timeout),
            )

            timed_out = False
            try:
                exit_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                # The process group, not the process: `setsid` above put the
                # child in its own group precisely so anything it started dies
                # with it.
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    process.kill()
                try:
                    exit_code = process.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    exit_code = -1

        return CodeRunResult(
            stdout=_read_capped(stdout_path, keep_tail=True),
            stderr=_read_capped(stderr_path, keep_tail=False),
            exit_code=exit_code,
            timed_out=timed_out,
        )

    finally:
        if process is not None and process.poll() is None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                process.kill()
        shutil.rmtree(workdir, ignore_errors=True)
