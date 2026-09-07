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

import argparse
import os
import sys

from task_suite.verify_code import CodeRunResult

from ._sandbox_common import (
    DEFAULT_MEMORY_LIMIT_BYTES,
    DEFAULT_TIMEOUT_SECONDS,
    MAX_CAPTURED_BYTES,
    Confinement,
    SandboxError,
)

__all__ = [
    "CodeRunResult",
    "Confinement",
    "DEFAULT_MEMORY_LIMIT_BYTES",
    "DEFAULT_TIMEOUT_SECONDS",
    "MAX_CAPTURED_BYTES",
    "SandboxError",
    "SandboxedCodeRunner",
    "confinement",
    "prepare",
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


def confinement() -> Confinement:
    """What this platform's backend actually confines.

    Recorded by the eval harness beside every coding score. The two backends
    are not equally strong - POSIX confines resources but not the filesystem -
    and a number that does not say which one produced it is a number nobody
    can interpret later.
    """
    return _implementation().CONFINEMENT


def prepare() -> None:
    """Do any one-time, machine-level setup now instead of lazily.

    On Windows this is the ~90-second `icacls` pass that grants the
    AppContainer read access to the interpreter. It is idempotent and only
    happens once per machine, but as a lazy side effect of someone's first
    test run it is indistinguishable from a hang. On POSIX there is nothing
    to do.
    """
    _implementation().prepare()


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.code_exec",
        description="Inspect or prepare the code sandbox.",
    )
    parser.add_argument(
        "--setup",
        action="store_true",
        help="do the one-time machine setup now (Windows: ~90s of icacls)",
    )
    arguments = parser.parse_args(argv)

    backend = sandbox_backend()
    if backend is None:
        print(f"backend:     none ({sys.platform!r} is unsupported)")
        print("coding tasks cannot be scored on this platform.")
        return 1

    print(f"backend:     {backend}")

    if arguments.setup:
        print("setup:       running (this is the slow one-time step)...")
        prepare()
        print("setup:       done")

    detail = confinement()
    print(f"confinement: {detail.summary()}")
    print(f"             {detail.note}")

    if not detail.filesystem:
        # Loud, because this is the difference that matters for reward
        # hacking and it is easy to read a green test run as meaning it is
        # handled.
        print()
        print("WARNING: this backend does not confine the filesystem, so a")
        print("         submission can read the expected answers off disk.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
