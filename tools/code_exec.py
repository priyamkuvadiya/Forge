"""The sandbox: the one place in this repo where model-written code runs.

This module picks an implementation and presents one interface. The isolation
primitives have nothing in common between platforms, but the contract does -
`task_suite/verify_code.py` asks for `CodeRunner(source, timeout) ->
CodeRunResult`, and everything above this line should neither know nor care
which one it got.

- **Windows** (`code_exec_windows.py`): an AppContainer for what the process
  can reach, a Job Object for what it can consume, and a suspended start so
  there is no window in which submitted code runs unconstrained. Fully
  implemented and adversarially tested on the machine this project is built
  on.
- **POSIX** (`code_exec_posix.py`): `fork` plus `setrlimit`. **Written but
  never executed** - there is no Linux, WSL or container runtime on the
  development machine - and its filesystem confinement is a known gap. Read
  that module's docstring before trusting it.

Anywhere else, `sandbox_available()` is False and constructing a runner raises
`SandboxError` rather than returning something that quietly fails every call.
That distinction matters: `verify()` refuses to score a coding task without a
runner precisely so a missing sandbox can never be mistaken for a model that
cannot code.
"""

import os
import sys

from task_suite.verify_code import CodeRunResult

from ._sandbox_common import (
    DEFAULT_MEMORY_LIMIT_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_CAPTURED_BYTES,
    SandboxError,
)

__all__ = [
    "CodeRunResult",
    "DEFAULT_MEMORY_LIMIT_BYTES",
    "DEFAULT_TIMEOUT_SECONDS",
    "MAX_CAPTURED_BYTES",
    "SandboxError",
    "SandboxedCodeRunner",
    "run_code",
    "sandbox_available",
    "sandbox_backend",
]


def sandbox_backend() -> str | None:
    """Which implementation this platform gets, or None if there is none."""
    if sys.platform == "win32":
        return "windows"
    if os.name == "posix":
        return "posix"
    return None


def sandbox_available() -> bool:
    return sandbox_backend() is not None


def _implementation():
    backend = sandbox_backend()
    if backend == "windows":
        from . import code_exec_windows

        return code_exec_windows
    if backend == "posix":
        from . import code_exec_posix

        return code_exec_posix
    raise SandboxError(
        f"no code sandbox is implemented for platform {sys.platform!r}; "
        "coding tasks cannot be scored here"
    )


def run_code(
    source: str,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    memory_limit: int = DEFAULT_MEMORY_LIMIT_BYTES,
) -> CodeRunResult:
    """Run `source` in whatever sandbox this platform provides.

    Never raises for anything the submitted code does - a crash, a timeout and
    an infinite loop are all ordinary results. Raises `SandboxError` only when
    the sandbox itself could not be established.
    """
    return _implementation().run_code(source, timeout=timeout, memory_limit=memory_limit)


class SandboxedCodeRunner:
    """A `CodeRunner` with its limits bound, for injection into the verifier.

    The verifier takes a runner rather than importing one, so tests can pass a
    fake and this class can carry non-default limits without threading them
    through every call site. Platform-agnostic on purpose: it holds limits and
    delegates, so neither implementation needs its own copy.
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
