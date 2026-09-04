"""The sandbox: the one place in this repo where model-written code runs.

Everything else in `tools/` is safe by construction. This module is not, so
the isolation is the design rather than an implementation detail, and the
reasoning is written down here rather than left in a commit message.

**Why a subprocess under a Windows Job Object, and not a container.** This
machine has no Docker and no WSL, and Python on Windows has no `resource`
module, so the usual Linux recipe (`fork` + `setrlimit`) does not exist here.
That leaves installing a container runtime or using the isolation primitive
Windows already has.

Measured on this machine, since the speed argument for a subprocess turned
out to be much weaker than assumed. A bare `python -I -B -c pass` costs a
median of **249 ms** here; a full sandboxed run is **285 ms**. So the
isolation machinery in this module is only ~40 ms - essentially all of the
cost is Windows process creation plus CPython startup, which a container
would pay *in addition to* its own. The honest version of the argument is
therefore that a container is somewhat slower and needs a multi-gigabyte
install that becomes a dependency for anyone reproducing this project's eval
numbers, not that it is six times slower. `-S` was tried and saves nothing
(252 ms); interpreter startup here is not import cost.

What the threat model does not need is the rest: the code being run is
written by a 0.5B model attempting coding exercises, not by an attacker.

**What the Job Object actually buys.** A bare subprocess with a timeout stops
nothing except an infinite loop. The job adds the limits that protect the
machine, and this project has already lost a machine once to an unbounded GPU
run, so "it probably won't allocate that much" is not a control:

- `ProcessMemoryLimit` / `JobMemoryLimit` - an allocation past the cap fails
  inside the child as a normal `MemoryError`. The machine never swaps.
- `ActiveProcessLimit = 1` - the child cannot create a second process. A fork
  bomb fails at its first spawn rather than being raced against.
- `KILL_ON_JOB_CLOSE` - when the job handle closes, everything inside it dies.
  Nothing survives this function, including on an unhandled exception in the
  parent.
- `DIE_ON_UNHANDLED_EXCEPTION` - suppresses the Windows error dialog. Without
  it a crashing child pops a modal box and blocks until someone clicks it,
  which in an unattended training run means a hang, not a failure.

**The assignment race, and why there isn't one.** A process created and
*then* assigned to a job has a window in which it is unconstrained. The
child here spends that window blocked on `stdin`, waiting for a token the
parent only sends after assignment succeeds. If assignment fails, the token
is never sent and the submitted code never runs at all.

**Output goes to files, not pipes.** The child's memory is capped but the
bytes it can push through a pipe are not, and those accumulate in the
*parent*, which is outside the job. Files bound that to the timeout and the
disk, and remove any chance of a pipe-buffer deadlock.

**What this does not do, stated plainly.** There is no OS-level network
block; that needs a network namespace or a container, and Windows offers no
per-process equivalent worth the complexity here. Submitted code could open a
socket. It also runs as the calling user, so it can read files that user can
read - it is confined to a scratch working directory by `cwd` and a scrubbed
environment, not prevented from walking out of it by path. Both are real
limits of this approach and are the reason the sandbox is a subprocess rather
than a security boundary. If this project ever ran genuinely untrusted code,
this would need to be a container.
"""

import ctypes
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from ctypes import wintypes
from pathlib import Path

from task_suite.verify_code import CodeRunResult

DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_MEMORY_LIMIT_BYTES = 256 * 1024 * 1024

# Enough for the harness's report line even on the largest problem in the
# suite, and small enough that a runaway writer cannot bloat this process.
MAX_CAPTURED_BYTES = 256 * 1024

# Exit code the bootstrap uses when the go-token doesn't match. Distinct from
# anything CPython returns on its own, so it is unambiguous in a test.
BAD_TOKEN_EXIT_CODE = 97

_IS_WINDOWS = sys.platform == "win32"


def _sandbox_interpreter() -> str:
    """The interpreter the child runs, which is not the one running this code.

    Inside a virtualenv on Windows, `sys.executable` is a launcher stub that
    re-executes the *real* interpreter as a second process. Under
    `ActiveProcessLimit = 1` that spawn is refused and the child dies before
    it starts, reporting only "Unable to create process" - which is the fork
    bomb limit doing exactly its job, on the wrong process.

    `sys._base_executable` is the real interpreter, so it starts as one
    process and needs no spawn.

    It is not, on its own, more isolated. The base interpreter still has the
    *global* site-packages on `sys.path`, and on this machine that directory
    holds a torch/CUDA install belonging to another project - so a submission
    could `import torch` and allocate on an 8 GB card in the middle of a
    training run that is already using it. That is why the child also runs
    with `-S`: no site directory at all, stdlib only. It was measured to cost
    nothing (252 ms against 249 ms), and the coding tasks are stdlib-only by
    construction, so nothing legitimate is lost.
    """
    return getattr(sys, "_base_executable", None) or sys.executable

# --- Win32 constants -------------------------------------------------------

JobObjectExtendedLimitInformation = 9

JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

CREATE_NO_WINDOW = 0x08000000
CREATE_SUSPENDED_NONE = 0


class _IO_COUNTERS(ctypes.Structure):
    _fields_ = [
        ("ReadOperationCount", ctypes.c_ulonglong),
        ("WriteOperationCount", ctypes.c_ulonglong),
        ("OtherOperationCount", ctypes.c_ulonglong),
        ("ReadTransferCount", ctypes.c_ulonglong),
        ("WriteTransferCount", ctypes.c_ulonglong),
        ("OtherTransferCount", ctypes.c_ulonglong),
    ]


class _JOBOBJECT_BASIC_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("PerProcessUserTimeLimit", ctypes.c_int64),
        ("PerJobUserTimeLimit", ctypes.c_int64),
        ("LimitFlags", wintypes.DWORD),
        ("MinimumWorkingSetSize", ctypes.c_size_t),
        ("MaximumWorkingSetSize", ctypes.c_size_t),
        ("ActiveProcessLimit", wintypes.DWORD),
        ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
        ("PriorityClass", wintypes.DWORD),
        ("SchedulingClass", wintypes.DWORD),
    ]


class _JOBOBJECT_EXTENDED_LIMIT_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("BasicLimitInformation", _JOBOBJECT_BASIC_LIMIT_INFORMATION),
        ("IoInfo", _IO_COUNTERS),
        ("ProcessMemoryLimit", ctypes.c_size_t),
        ("JobMemoryLimit", ctypes.c_size_t),
        ("PeakProcessMemoryUsed", ctypes.c_size_t),
        ("PeakJobMemoryUsed", ctypes.c_size_t),
    ]


class SandboxError(RuntimeError):
    """The sandbox could not be established.

    Deliberately distinct from a submission failing: a submission that
    crashes is a score of 0.0, but a sandbox that could not be set up must
    never be scored at all, or a broken machine would read as a model that
    cannot code. `task_suite.verify_code` takes the same position when no
    runner is supplied.
    """


# The bootstrap the child actually starts in. It exists for two reasons: to
# block until the parent confirms the job assignment landed, and to run the
# submitted file through `runpy` in a fresh namespace, so the submission
# cannot see or shadow the bootstrap's own names.
_BOOTSTRAP = '''\
import sys
import runpy

token, target = sys.argv[1], sys.argv[2]

# Blocks here until the parent has assigned this process to the job object.
# If that assignment fails the parent never writes the token, and the
# submitted code below is never reached.
if sys.stdin.readline().strip() != token:
    raise SystemExit({bad_token})

try:
    sys.stdin.close()
except Exception:
    pass

runpy.run_path(target, run_name="__main__")
'''.format(bad_token=BAD_TOKEN_EXIT_CODE)


def _create_job(memory_limit: int) -> wintypes.HANDLE:
    """Create the job object the child will be confined to."""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.CreateJobObjectW.argtypes = [wintypes.LPVOID, wintypes.LPCWSTR]

    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise SandboxError(f"CreateJobObject failed: {ctypes.get_last_error()}")

    info = _JOBOBJECT_EXTENDED_LIMIT_INFORMATION()
    info.BasicLimitInformation.LimitFlags = (
        JOB_OBJECT_LIMIT_ACTIVE_PROCESS
        | JOB_OBJECT_LIMIT_PROCESS_MEMORY
        | JOB_OBJECT_LIMIT_JOB_MEMORY
        | JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        | JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION
    )
    # One process: the child itself. It cannot spawn a second.
    info.BasicLimitInformation.ActiveProcessLimit = 1
    info.ProcessMemoryLimit = memory_limit
    info.JobMemoryLimit = memory_limit

    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD
    ]
    ok = kernel32.SetInformationJobObject(
        job,
        JobObjectExtendedLimitInformation,
        ctypes.byref(info),
        ctypes.sizeof(info),
    )
    if not ok:
        error = ctypes.get_last_error()
        kernel32.CloseHandle(job)
        raise SandboxError(f"SetInformationJobObject failed: {error}")

    return job


def _assign_to_job(job: wintypes.HANDLE, process_handle: int) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    if not kernel32.AssignProcessToJobObject(job, wintypes.HANDLE(process_handle)):
        raise SandboxError(
            f"AssignProcessToJobObject failed: {ctypes.get_last_error()}"
        )


def _terminate_job(job: wintypes.HANDLE) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel32.TerminateJobObject(job, 1)


def _close_handle(job: wintypes.HANDLE) -> None:
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle(job)


def _sandbox_environment(workdir: Path) -> dict[str, str]:
    """A minimal environment, with temp files pointed inside the scratch dir.

    Redirecting TEMP/TMP matters more than it looks: a submission calling
    `tempfile.mkstemp()` would otherwise scatter files through the user's real
    temp directory, and those would outlive the run.
    """
    environment = {
        "TEMP": str(workdir),
        "TMP": str(workdir),
        "PATH": "",
    }
    # Windows needs SYSTEMROOT for socket and crypto initialisation; without it
    # the interpreter itself can fail to start.
    if "SYSTEMROOT" in os.environ:
        environment["SYSTEMROOT"] = os.environ["SYSTEMROOT"]
    return environment


def _read_capped(path: Path, keep_tail: bool) -> str:
    """Read at most `MAX_CAPTURED_BYTES` from `path`.

    `keep_tail` decides which end survives truncation, and the two streams
    want opposite answers. The harness writes its report line last, so stdout
    must keep its tail or a chatty submission would push the result out of
    view and score 0.0 for the wrong reason. A traceback starts at the top, so
    stderr keeps its head.
    """
    if not path.exists():
        return ""

    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > MAX_CAPTURED_BYTES and keep_tail:
            handle.seek(size - MAX_CAPTURED_BYTES)
        data = handle.read(MAX_CAPTURED_BYTES)

    return data.decode("utf-8", errors="replace")


def run_code(
    source: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    memory_limit: int = DEFAULT_MEMORY_LIMIT_BYTES,
) -> CodeRunResult:
    """Run `source` in a constrained child process and report what it did.

    Satisfies the `CodeRunner` contract in `task_suite.verify_code`. Never
    raises for anything the submitted code does - a crash, a timeout and an
    infinite loop are all ordinary results. It raises `SandboxError` only when
    the sandbox itself could not be established, because that must not be
    scored as a model failure.
    """
    if not _IS_WINDOWS:
        raise SandboxError(
            "this sandbox is implemented with Windows Job Objects; "
            f"platform {sys.platform!r} is not supported"
        )
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    workdir = Path(tempfile.mkdtemp(prefix="forge-sandbox-"))
    job = None
    process = None

    try:
        submission = workdir / "submission.py"
        submission.write_text(source, encoding="utf-8")
        bootstrap = workdir / "_bootstrap.py"
        bootstrap.write_text(_BOOTSTRAP, encoding="utf-8")

        token = uuid.uuid4().hex
        stdout_path = workdir / "_stdout"
        stderr_path = workdir / "_stderr"

        job = _create_job(memory_limit)

        with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
            process = subprocess.Popen(
                [
                    _sandbox_interpreter(),
                    "-I",  # isolated: ignore env vars, user site-packages, cwd
                    "-S",  # no site directory at all: stdlib only, no torch
                    "-B",  # no .pyc litter in the scratch directory
                    "-X", "utf8",
                    str(bootstrap),
                    token,
                    str(submission),
                ],
                stdin=subprocess.PIPE,
                stdout=out,
                stderr=err,
                cwd=str(workdir),
                env=_sandbox_environment(workdir),
                creationflags=CREATE_NO_WINDOW,
            )

            # `_handle` is private but is the only route to the process HANDLE
            # that `AssignProcessToJobObject` needs. The child is parked on
            # stdin until the token below is written, so this cannot lose a
            # race with the submitted code.
            _assign_to_job(job, int(process._handle))

            timed_out = False
            try:
                process.stdin.write(f"{token}\n".encode())
                process.stdin.flush()
                process.stdin.close()
            except OSError:
                # The child died between assignment and the handshake - a
                # legitimate outcome, not a sandbox failure. Fall through and
                # report whatever it managed to leave behind.
                pass

            try:
                exit_code = process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                # Terminate the job, not the process: killing the process
                # alone would leave anything it started running, and the whole
                # point of the job is that there is one thing to kill.
                _terminate_job(job)
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
        if job is not None:
            # KILL_ON_JOB_CLOSE means closing this handle kills anything still
            # inside, including on an unhandled exception above.
            _close_handle(job)
        if process is not None and process.stdin is not None:
            try:
                process.stdin.close()
            except OSError:
                pass
        shutil.rmtree(workdir, ignore_errors=True)


class SandboxedCodeRunner:
    """A `CodeRunner` with its limits bound, for injection into the verifier.

    The verifier takes a runner rather than importing one, so tests can pass a
    fake and this class can carry non-default limits without threading them
    through every call site.
    """

    def __init__(
        self,
        memory_limit: int = DEFAULT_MEMORY_LIMIT_BYTES,
        default_timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.memory_limit = memory_limit
        self.default_timeout = default_timeout

    def __call__(self, source: str, timeout: float | None = None) -> CodeRunResult:
        return run_code(
            source,
            timeout=self.default_timeout if timeout is None else timeout,
            memory_limit=self.memory_limit,
        )
