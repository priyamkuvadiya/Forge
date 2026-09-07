"""The sandbox: the one place in this repo where model-written code runs.

Everything else in `tools/` is safe by construction. This module is not, so
the isolation is the design rather than an implementation detail, and the
reasoning is written down here rather than left in a commit message.

Three mechanisms, each closing something the others do not.

**An AppContainer** confines what the process can *reach*. This is the part
that matters for reward hacking, and it was added after a probe showed the
first version did not have it: a submission that hardcoded the repo path could
open `task_suite/data/suite.json` and read the expected answers for its own
task. Module 2 withholds those answers from the harness precisely so that a
submission wanting a passing line has to compute the right values - a defence
that is worth nothing if the same values are readable from disk. Inside an
AppContainer the read fails with `PermissionError`, because an app container
SID has no access to the user's files unless it is granted explicitly.

**A Job Object** confines what the process can *consume*: per-process and
per-job memory caps so an allocation fails as `MemoryError` rather than
swapping the machine, `ActiveProcessLimit = 1` so a fork bomb fails at its
first spawn, `KILL_ON_JOB_CLOSE` so nothing outlives the call, and
`DIE_ON_UNHANDLED_EXCEPTION` so a crash cannot raise a modal dialog and hang
an unattended training run.

**`CREATE_SUSPENDED`** removes the assignment race outright. The process is
created suspended, assigned to the job, and only then resumed, so there is no
window in which submitted code runs unconstrained. An earlier version parked
the child on stdin waiting for a token to get the same guarantee; suspending
is simpler and does not depend on the child cooperating.

**Why not a container.** This machine has no Docker and no WSL, and Python on
Windows has no `resource` module, so `fork` + `setrlimit` does not exist here.
Measured, a bare `python -I -B -c pass` costs a median of 249 ms on this
machine and a full sandboxed run 285 ms, so the isolation machinery is only
~40 ms - essentially all of the cost is Windows process creation plus CPython
startup, which a container would pay in addition to its own. The honest
argument against a container is therefore its multi-gigabyte install becoming
a dependency for anyone reproducing this project's eval numbers, not a
six-fold speed gap. `-S` was measured and saves nothing (252 ms).

**What this does and does not cover.** The container is created with no
capabilities, so network access is refused (`WinError 10013`) and the user's
files are unreachable - both asserted in `tools/tests/test_code_exec.py`
rather than assumed. What it *is* granted, it can read freely: the interpreter
directory and its own scratch space. The process still runs as the calling
user, so this is a strong boundary against a model that wanders into the repo
and a weak one against a determined attacker who already has local code
execution. For the threat this project actually faces - a small model
optimising a reward it can see - it closes the route that matters.

**One persistent side effect, stated plainly.** Setting this up grants the
AppContainer SID read+execute on the Python installation directory, once, the
first time the profile is created. That is an additive ACL change to a
directory outside this repo. It is required - without it the interpreter
cannot start inside the container - and it is scoped to one generated SID, but
it is a change to the machine and not only to this project.
"""

import ctypes
import msvcrt
import os
import shutil
import subprocess
import sys
import uuid
from ctypes import wintypes
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

APP_CONTAINER_NAME = "ForgeCodeSandbox"

CONFINEMENT = Confinement(
    level="appcontainer+job",
    filesystem=True,
    network=True,
    resources=True,
    note=(
        "AppContainer denies the repo and the user's files and refuses network "
        "access; a Job Object caps memory and forbids child processes. Asserted "
        "in tools/tests/test_code_exec.py."
    ),
)


def prepare() -> None:
    """Do the one-time setup now rather than inside someone's first test run.

    Creating the AppContainer profile and granting it read access to the
    interpreter is an `icacls` pass over roughly 50,000 files - about 90
    seconds. It is idempotent and marker-guarded, so it happens once per
    machine, but as a lazy side effect of the first sandboxed call it looks
    exactly like a hang. `python -m tools.code_exec --setup` makes it a
    deliberate step with output.
    """
    _sandbox_root()

_IS_WINDOWS = sys.platform == "win32"

# --- Win32 constants -------------------------------------------------------

JobObjectExtendedLimitInformation = 9

JOB_OBJECT_LIMIT_ACTIVE_PROCESS = 0x00000008
JOB_OBJECT_LIMIT_DIE_ON_UNHANDLED_EXCEPTION = 0x00000400
JOB_OBJECT_LIMIT_PROCESS_MEMORY = 0x00000100
JOB_OBJECT_LIMIT_JOB_MEMORY = 0x00000200
JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x00002000

CREATE_NO_WINDOW = 0x08000000
CREATE_SUSPENDED = 0x00000004
CREATE_UNICODE_ENVIRONMENT = 0x00000400
EXTENDED_STARTUPINFO_PRESENT = 0x00080000

STARTF_USESTDHANDLES = 0x00000100

PROC_THREAD_ATTRIBUTE_HANDLE_LIST = 0x00020002
PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES = 0x00020009

WAIT_TIMEOUT = 0x00000102
INFINITE = 0xFFFFFFFF

ERROR_ALREADY_EXISTS_HRESULT = 0x800700B7


# --- structures ------------------------------------------------------------

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


class _SID_AND_ATTRIBUTES(ctypes.Structure):
    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]


class _SECURITY_CAPABILITIES(ctypes.Structure):
    _fields_ = [
        ("AppContainerSid", ctypes.c_void_p),
        ("Capabilities", ctypes.POINTER(_SID_AND_ATTRIBUTES)),
        ("CapabilityCount", wintypes.DWORD),
        ("Reserved", wintypes.DWORD),
    ]


class _STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD), ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR), ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD), ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD), ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD), ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD), ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD), ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.c_void_p),
        ("hStdInput", wintypes.HANDLE), ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class _STARTUPINFOEXW(ctypes.Structure):
    _fields_ = [("StartupInfo", _STARTUPINFOW), ("lpAttributeList", ctypes.c_void_p)]


class _PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE), ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD), ("dwThreadId", wintypes.DWORD),
    ]


def _kernel32():
    return ctypes.WinDLL("kernel32", use_last_error=True)


# --- AppContainer ----------------------------------------------------------

_app_container: tuple[ctypes.c_void_p, str] | None = None


def _sid_to_string(sid: ctypes.c_void_p) -> str:
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    text = wintypes.LPWSTR()
    if not advapi32.ConvertSidToStringSidW(sid, ctypes.byref(text)):
        raise SandboxError(f"ConvertSidToStringSid failed: {ctypes.get_last_error()}")
    return text.value


def _grant_app_container_access(sid_text: str, target: Path, permissions: str) -> None:
    """Grant the app container SID access to a path, via icacls.

    icacls rather than the ACL APIs because this runs once per machine and the
    ctypes for a full DACL rewrite are several hundred lines that would need
    their own tests to be trustworthy. The grant is additive and scoped to one
    generated SID.
    """
    result = subprocess.run(
        ["icacls", str(target), "/grant", f"*{sid_text}:{permissions}", "/T", "/Q"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise SandboxError(
            f"could not grant the sandbox access to {target}: {result.stderr.strip()}"
        )


def _app_container_sid() -> tuple[ctypes.c_void_p, str]:
    """The AppContainer SID, creating the profile and its grants once.

    The interpreter directory has to be granted read+execute or the child
    cannot start at all. That happens only when the profile is newly created,
    so the cost is paid once per machine rather than once per run.
    """
    global _app_container
    if _app_container is not None:
        return _app_container

    userenv = ctypes.WinDLL("userenv", use_last_error=True)
    userenv.CreateAppContainerProfile.restype = ctypes.c_long
    userenv.DeriveAppContainerSidFromAppContainerName.restype = ctypes.c_long

    sid = ctypes.c_void_p()
    created = userenv.CreateAppContainerProfile(
        APP_CONTAINER_NAME,
        APP_CONTAINER_NAME,
        "Forge sandbox for model-written code",
        None,
        0,
        ctypes.byref(sid),
    )

    if created == 0:
        sid_text = _sid_to_string(sid)
    else:
        hresult = created & 0xFFFFFFFF
        if hresult != ERROR_ALREADY_EXISTS_HRESULT:
            raise SandboxError(
                f"CreateAppContainerProfile failed: 0x{hresult:08x}"
            )
        derived = userenv.DeriveAppContainerSidFromAppContainerName(
            APP_CONTAINER_NAME, ctypes.byref(sid)
        )
        if derived != 0:
            raise SandboxError(
                f"DeriveAppContainerSidFromAppContainerName failed: "
                f"0x{derived & 0xFFFFFFFF:08x}"
            )
        sid_text = _sid_to_string(sid)

    _app_container = (sid, sid_text)
    return _app_container


_sandbox_root_path: Path | None = None


def _app_container_folder(sid_text: str) -> Path:
    """The container's own redirected storage directory."""
    userenv = ctypes.WinDLL("userenv", use_last_error=True)
    userenv.GetAppContainerFolderPath.restype = ctypes.c_long
    path = wintypes.LPWSTR()
    result = userenv.GetAppContainerFolderPath(sid_text, ctypes.byref(path))
    if result != 0:
        raise SandboxError(
            f"GetAppContainerFolderPath failed: 0x{result & 0xFFFFFFFF:08x}"
        )
    return Path(path.value)


def _sandbox_root() -> Path:
    """The directory every run's scratch space sits inside.

    Inside the container's own folder, not the user's temp directory, and that
    is not a stylistic choice. An AppContainer cannot reach `%TEMP%` even with
    an explicit inheritable ACE granting it full control on the directory:
    access is refused walking the ancestor chain, so the grant never gets a
    chance to apply. Measured directly - reading a file there fails with
    `PermissionError` while the container's own folder both lists and writes
    fine, which is what it is designed for.

    The upside is that this needs no ACL manipulation at all. The only grant
    left anywhere is read+execute on the interpreter directory, without which
    the container cannot start Python.
    """
    global _sandbox_root_path
    if _sandbox_root_path is not None:
        return _sandbox_root_path

    _, sid_text = _app_container_sid()
    container_temp = _app_container_folder(sid_text) / "Temp"
    root = container_temp / "forge-runs"
    root.mkdir(parents=True, exist_ok=True)

    # Windows points a contained process's temp directory at the container's
    # own Temp, whatever TEMP is set to, so a submission calling
    # `tempfile.mkstemp()` writes a sibling of `forge-runs` rather than
    # something inside the run directory that gets cleaned up with it. Swept
    # once per process: a GRPO run makes tens of thousands of calls, and
    # anything a submission leaves behind would otherwise accumulate for the
    # life of the machine.
    for stray in container_temp.iterdir():
        if stray == root:
            continue
        if stray.is_dir():
            shutil.rmtree(stray, ignore_errors=True)
        else:
            stray.unlink(missing_ok=True)

    # Written only after the grant succeeds, so a run that dies partway
    # through re-does it next time. Tying this to "was the profile newly
    # created?" instead, as a first version did, meant any failure after
    # profile creation left the grant permanently unapplied and every later
    # run failed to start the interpreter.
    # Inside `root`, so the sweep above never deletes it. Outside it, the
    # sweep would remove the marker every process and re-run an `icacls` pass
    # over ~50k interpreter files each time.
    #
    # Keyed on the interpreter as well as the SID. A Python upgrade installs
    # into a new directory (or replaces the old one's ACLs) and the grant is
    # silently lost; keyed on the SID alone, the marker would still be there
    # and every run would fail to start with a permission error that says
    # nothing about why.
    marker = root / ".forge-interpreter-granted"
    interpreter = Path(_sandbox_interpreter())
    expected = f"{sid_text}\n{interpreter}\n{sys.version}"

    if not marker.exists() or marker.read_text(encoding="utf-8") != expected:
        _grant_app_container_access(sid_text, interpreter.parent, "(OI)(CI)(RX)")
        marker.write_text(expected, encoding="utf-8")

    _sandbox_root_path = root
    return root


# --- Job object ------------------------------------------------------------

def _create_job(memory_limit: int) -> wintypes.HANDLE:
    kernel32 = _kernel32()
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
    info.BasicLimitInformation.ActiveProcessLimit = 1
    info.ProcessMemoryLimit = memory_limit
    info.JobMemoryLimit = memory_limit

    kernel32.SetInformationJobObject.argtypes = [
        wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD
    ]
    if not kernel32.SetInformationJobObject(
        job, JobObjectExtendedLimitInformation, ctypes.byref(info), ctypes.sizeof(info)
    ):
        error = ctypes.get_last_error()
        kernel32.CloseHandle(job)
        raise SandboxError(f"SetInformationJobObject failed: {error}")

    return job


# --- launching -------------------------------------------------------------


def _environment_block(workdir: Path) -> ctypes.Array:
    """A minimal environment for the child.

    Note what this does *not* control. Windows rewrites an AppContainer's
    environment on the way in: TEMP, TMP and LOCALAPPDATA all arrive at the
    child pointing into the container's own storage
    (`...\\Packages\\<name>\\AC`), whatever is passed here. Measured, not
    assumed - the child prints TEMP as the container's Temp directory rather
    than the run directory set below, which is also why a submission calling
    `tempfile.mkstemp()` writes a sibling of the run directory rather than
    something inside it (see `_sandbox_root`'s sweep).

    So the values below are advisory for TEMP/TMP and load-bearing only in
    that the variables must be *present*. What this block genuinely does is
    withhold everything else: the user's real environment never reaches the
    child.
    """
    values = {"TEMP": str(workdir), "TMP": str(workdir)}

    # These two are not optional and were found by bisection, not by reading
    # documentation. Without LOCALAPPDATA, `CreateProcess` fails outright with
    # ERROR_ENVVAR_NOT_FOUND (203): an AppContainer redirects its per-container
    # storage to a path underneath it, so the container cannot be constructed
    # without it. Its *value* turns out not to matter - a nonexistent path
    # works, because Windows substitutes the redirected one - but the variable
    # has to exist. SYSTEMROOT is needed for the interpreter's own socket and
    # crypto initialisation.
    for name in ("SYSTEMROOT", "LOCALAPPDATA"):
        if name in os.environ:
            values[name] = os.environ[name]

    # No PATH at all rather than an empty one. A `NAME=` entry with an empty
    # value makes the block malformed. Omitting it costs nothing here, and
    # `ActiveProcessLimit = 1` already means the child cannot launch anything
    # it might have found on a PATH.
    block = "".join(f"{k}={v}\0" for k, v in values.items()) + "\0"
    return ctypes.create_unicode_buffer(block)


def _spawn(
    command: str,
    workdir: Path,
    out_handle: int,
    err_handle: int,
    job: wintypes.HANDLE,
) -> _PROCESS_INFORMATION:
    """Create the child suspended inside the AppContainer, then job it, then run it."""
    kernel32 = _kernel32()
    sid, _ = _app_container_sid()

    # Two attributes: the container to run in, and an explicit list of the only
    # handles the child may inherit. Without the handle list, `bInheritHandles`
    # would hand the child every inheritable handle this process happens to
    # hold.
    size = ctypes.c_size_t(0)
    kernel32.InitializeProcThreadAttributeList(None, 2, 0, ctypes.byref(size))
    attributes = (ctypes.c_byte * size.value)()
    if not kernel32.InitializeProcThreadAttributeList(
        attributes, 2, 0, ctypes.byref(size)
    ):
        raise SandboxError(
            f"InitializeProcThreadAttributeList failed: {ctypes.get_last_error()}"
        )

    capabilities = _SECURITY_CAPABILITIES()
    capabilities.AppContainerSid = sid
    capabilities.Capabilities = None
    capabilities.CapabilityCount = 0
    capabilities.Reserved = 0

    if not kernel32.UpdateProcThreadAttribute(
        attributes, 0, ctypes.c_size_t(PROC_THREAD_ATTRIBUTE_SECURITY_CAPABILITIES),
        ctypes.byref(capabilities), ctypes.sizeof(capabilities), None, None,
    ):
        raise SandboxError(
            f"UpdateProcThreadAttribute(security) failed: {ctypes.get_last_error()}"
        )

    handles = (wintypes.HANDLE * 2)(
        wintypes.HANDLE(out_handle), wintypes.HANDLE(err_handle)
    )
    if not kernel32.UpdateProcThreadAttribute(
        attributes, 0, ctypes.c_size_t(PROC_THREAD_ATTRIBUTE_HANDLE_LIST),
        ctypes.byref(handles), ctypes.sizeof(handles), None, None,
    ):
        raise SandboxError(
            f"UpdateProcThreadAttribute(handles) failed: {ctypes.get_last_error()}"
        )

    startup = _STARTUPINFOEXW()
    startup.StartupInfo.cb = ctypes.sizeof(_STARTUPINFOEXW)
    startup.lpAttributeList = ctypes.cast(attributes, ctypes.c_void_p)
    startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES
    startup.StartupInfo.hStdInput = None
    startup.StartupInfo.hStdOutput = wintypes.HANDLE(out_handle)
    startup.StartupInfo.hStdError = wintypes.HANDLE(err_handle)

    # Declared explicitly: without argtypes, ctypes narrows pointer-sized
    # arguments to 32 bits on a 64-bit build, which corrupts the attribute
    # list and environment pointers in ways that fail far from here.
    kernel32.CreateProcessW.argtypes = [
        wintypes.LPCWSTR, wintypes.LPWSTR, ctypes.c_void_p, ctypes.c_void_p,
        wintypes.BOOL, wintypes.DWORD, ctypes.c_void_p, wintypes.LPCWSTR,
        ctypes.c_void_p, ctypes.c_void_p,
    ]
    kernel32.CreateProcessW.restype = wintypes.BOOL

    process = _PROCESS_INFORMATION()
    created = kernel32.CreateProcessW(
        None,
        ctypes.create_unicode_buffer(command),
        None,
        None,
        True,
        CREATE_SUSPENDED | CREATE_NO_WINDOW | EXTENDED_STARTUPINFO_PRESENT
        | CREATE_UNICODE_ENVIRONMENT,
        _environment_block(workdir),
        str(workdir),
        ctypes.byref(startup.StartupInfo),
        ctypes.byref(process),
    )
    error = ctypes.get_last_error()
    kernel32.DeleteProcThreadAttributeList(attributes)
    if not created:
        raise SandboxError(f"CreateProcess failed: {error}")

    # Suspended, so this cannot lose a race with the submitted code.
    kernel32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    if not kernel32.AssignProcessToJobObject(job, process.hProcess):
        error = ctypes.get_last_error()
        kernel32.TerminateProcess(process.hProcess, 1)
        kernel32.CloseHandle(process.hThread)
        kernel32.CloseHandle(process.hProcess)
        raise SandboxError(f"AssignProcessToJobObject failed: {error}")

    if kernel32.ResumeThread(process.hThread) == 0xFFFFFFFF:
        error = ctypes.get_last_error()
        kernel32.TerminateProcess(process.hProcess, 1)
        raise SandboxError(f"ResumeThread failed: {error}")

    return process


def run_code(
    source: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    memory_limit: int = DEFAULT_MEMORY_LIMIT_BYTES,
) -> CodeRunResult:
    """Run `source` in a confined child process and report what it did.

    Satisfies the `CodeRunner` contract in `task_suite.verify_code`. Never
    raises for anything the submitted code does - a crash, a timeout and an
    infinite loop are all ordinary results. It raises `SandboxError` only when
    the sandbox itself could not be established, because that must not be
    scored as a model failure.
    """
    if not _IS_WINDOWS:
        raise SandboxError(
            "this sandbox is implemented with Windows Job Objects and AppContainers; "
            f"platform {sys.platform!r} is not supported"
        )
    if timeout <= 0:
        raise ValueError("timeout must be positive")

    kernel32 = _kernel32()

    # `Path.mkdir()` at its default mode, deliberately not `tempfile.mkdtemp`.
    # mkdtemp creates with mode 0o700, and since Python 3.13 Windows honours
    # that by writing an *explicit* DACL granting only the current user - which
    # disables ACE inheritance on the directory. The run directory would then
    # opt out of the container's inherited access to its own folder, and the
    # child would fail to open its own bootstrap with a bare
    # "[Errno 13] Permission denied" pointing at a path that looks perfectly
    # reasonable. Nothing about the message suggests the mode argument.
    workdir = _sandbox_root() / f"run-{uuid.uuid4().hex}"
    workdir.mkdir()
    job = None
    process = None
    streams: list = []

    try:
        submission = workdir / "submission.py"
        submission.write_text(source, encoding="utf-8")
        bootstrap = workdir / "_bootstrap.py"
        bootstrap.write_text(_BOOTSTRAP, encoding="utf-8")
        stdout_path = workdir / "_stdout"
        stderr_path = workdir / "_stderr"

        job = _create_job(memory_limit)

        handles = []
        for path in (stdout_path, stderr_path):
            stream = open(path, "wb")
            streams.append(stream)
            handle = msvcrt.get_osfhandle(stream.fileno())
            os.set_handle_inheritable(handle, True)
            handles.append(handle)

        # -I isolated (no env vars, no user site-packages, no cwd on sys.path)
        # -S no site directory at all, so the *global* site-packages are gone
        #    too. On this machine those hold another project's torch/CUDA
        #    install, and a submission importing it could allocate on the 8 GB
        #    card mid-training. Measured to cost nothing: 252 ms against 249 ms.
        # -B no .pyc litter in the scratch directory
        command = (
            f'"{_sandbox_interpreter()}" -I -S -B -X utf8 "{bootstrap}" "{submission}"'
        )
        process = _spawn(command, workdir, handles[0], handles[1], job)

        timed_out = False
        waited = kernel32.WaitForSingleObject(process.hProcess, int(timeout * 1000))
        if waited == WAIT_TIMEOUT:
            timed_out = True
            # Terminate the job, not the process: killing the process alone
            # would leave anything it started running, and the whole point of
            # the job is that there is one thing to kill.
            kernel32.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
            kernel32.TerminateJobObject(job, 1)
            kernel32.WaitForSingleObject(process.hProcess, 5000)

        code = wintypes.DWORD()
        kernel32.GetExitCodeProcess(process.hProcess, ctypes.byref(code))
        exit_code = ctypes.c_int32(code.value).value

        for stream in streams:
            stream.close()
        streams = []

        return CodeRunResult(
            stdout=_read_capped(stdout_path, keep_tail=True),
            stderr=_read_capped(stderr_path, keep_tail=False),
            exit_code=exit_code,
            timed_out=timed_out,
        )

    finally:
        for stream in streams:
            stream.close()
        if process is not None:
            kernel32.CloseHandle(process.hThread)
            kernel32.CloseHandle(process.hProcess)
        if job is not None:
            # KILL_ON_JOB_CLOSE means closing this handle kills anything still
            # inside, including on an unhandled exception above.
            kernel32.CloseHandle(job)
        shutil.rmtree(workdir, ignore_errors=True)


# `SandboxedCodeRunner` deliberately does not live here. It only binds limits
# and delegates, so it belongs in `code_exec.py` where it is written once for
# both platforms rather than copied into each.
