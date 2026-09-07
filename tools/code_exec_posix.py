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
import shutil
import signal
import subprocess
import tempfile
import uuid
from pathlib import Path

from task_suite.verify_code import CodeRunResult

from ._sandbox_common import (
    BOOTSTRAP as _BOOTSTRAP,
    DEFAULT_MEMORY_LIMIT_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    Confinement,
    SandboxError,
    read_capped as _read_capped,
    sandbox_interpreter as _sandbox_interpreter,
)

# A submission bounded only by wall clock can still write until the disk
# fills. 64 MB is far past anything the suite's problems produce.
MAX_WRITE_BYTES = 64 * 1024 * 1024

# How many extra tasks a submission may create beyond what the machine is
# already running. Generous enough for a solution that uses a thread pool,
# small enough that an unbounded fork loop stops within a few spawns.
NPROC_HEADROOM = 24

_IS_POSIX = os.name == "posix"

CONFINEMENT = Confinement(
    level="rlimit-only",
    # The honest entry. `setrlimit` bounds what a process consumes, not what
    # it can reach: a submission here can still read the repo, and therefore
    # the expected answers for its own task. Closing this needs a mount
    # namespace or Landlock; until then a coding score collected on this
    # backend carries this note with it.
    filesystem=False,
    network=False,
    resources=True,
    note=(
        "setrlimit caps memory, CPU, processes and file size, but confines "
        "neither the filesystem nor the network: a submission can read the "
        "repo and therefore the expected answers. Needs a mount namespace or "
        "Landlock. This backend has also never been executed - see the module "
        "docstring."
    ),
)


def prepare() -> None:
    """Nothing to set up: `setrlimit` needs no profile and no grants."""


# `resource` is POSIX-only, so it is imported inside the function that uses
# it rather than at module scope. That keeps this module importable on
# Windows, which matters for two things that are otherwise impossible from
# the development machine: reading `CONFINEMENT` to report what this backend
# would and would not confine, and letting linters and CI see the file at all.
def _resource():
    import resource

    return resource


def _apply_limits(memory_limit: int, timeout: float):
    """Built in the parent, run in the child between `fork` and `exec`.

    Everything here has to be async-signal-safe, which is why it only calls
    `setrlimit` and `setsid` and does no allocation or logging.
    """

    resource = _resource()

    # Counted in the parent, because everything inside `preexec` runs between
    # fork and exec and has to stay async-signal-safe.
    nproc_ceiling = _current_task_count() + NPROC_HEADROOM

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

        # Cap runaway process creation. Deliberately *not* 1, which is the
        # obvious mirror of the job object's ActiveProcessLimit and is wrong
        # on Linux: RLIMIT_NPROC counts tasks, and a thread is a task, so a
        # limit of 1 makes `threading.Thread().start()` raise "can't start new
        # thread". CI caught that - legitimate solutions do use threads, and
        # breaking them would depress the code category's reward for a reason
        # unrelated to correctness.
        #
        # It is also per-real-UID rather than per-process, so an absolute
        # value would either sit below what the machine is already using (and
        # fail immediately) or above anything worth capping. Headroom over
        # current usage avoids both: a handful of threads fit, an unbounded
        # fork loop hits the ceiling within a few spawns.
        nproc = min(nproc_ceiling, resource.getrlimit(resource.RLIMIT_NPROC)[1])
        resource.setrlimit(resource.RLIMIT_NPROC, (nproc, nproc))

        resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_WRITE_BYTES, MAX_WRITE_BYTES))

        # No core dumps: a 256 MB crash writes a 256 MB file otherwise.
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    return preexec


def _current_task_count() -> int:
    """Roughly how many tasks are running, for sizing `RLIMIT_NPROC`.

    Counts numeric entries in /proc, which is processes rather than threads
    and covers all users rather than this one - both errors push the estimate
    up, which is the safe direction: the limit ends up looser than intended
    rather than tight enough to break the child at startup. Falls back to a
    generous constant where /proc is not mounted.
    """
    try:
        return sum(1 for entry in os.listdir("/proc") if entry.isdigit())
    except OSError:
        return 512


def _environment(workdir: Path) -> dict[str, str]:
    # PATH is omitted entirely rather than set empty, matching the Windows
    # backend. `PATH=""` still defines the variable, which CI caught as a
    # difference between the two backends in what reaches the child. Nothing
    # needs it: the interpreter is invoked by absolute path.
    environment = {"TMPDIR": str(workdir), "HOME": str(workdir)}
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
