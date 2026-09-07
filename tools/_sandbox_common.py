"""What both sandbox implementations share.

The Windows and POSIX runners isolate code with completely different
primitives, but everything around that - the limits, the child bootstrap, how
output is truncated, what counts as a sandbox failure rather than a submission
failure - is the same, and duplicating it invites the two to drift into
disagreeing about how a submission is scored. Only the isolation lives in the
platform files.
"""

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_MEMORY_LIMIT_BYTES = 256 * 1024 * 1024

# Enough for the harness's report line even on the largest problem in the
# suite, and small enough that a runaway writer cannot bloat the parent.
MAX_CAPTURED_BYTES = 256 * 1024


@dataclass(frozen=True)
class Confinement:
    """What a backend actually confines, as data rather than as prose.

    The two backends are not equally strong, and the weaker one is weaker in
    the dimension that matters most for this project: on POSIX a submission
    can still read the repo, and therefore the expected answers module 2 works
    so hard to withhold. Leaving that difference in a docstring means a coding
    score collected on Linux looks identical to one collected on Windows.

    So it is a value the eval harness records beside every coding number. A
    result then says which boundary it was collected under, and nobody has to
    remember to ask.
    """

    level: str
    filesystem: bool
    network: bool
    resources: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def summary(self) -> str:
        confined = [
            name
            for name, on in (
                ("filesystem", self.filesystem),
                ("network", self.network),
                ("resources", self.resources),
            )
            if on
        ]
        return f"{self.level} (confines: {', '.join(confined) or 'nothing'})"


class SandboxError(RuntimeError):
    """The sandbox could not be established.

    Deliberately distinct from a submission failing: a submission that crashes
    is a score of 0.0, but a sandbox that could not be set up must never be
    scored at all, or a broken machine would read as a model that cannot code.
    `task_suite.verify_code` takes the same position when no runner is
    supplied.
    """


# The child starts here. `runpy` gives the submitted file a fresh module
# namespace, so it cannot see or shadow anything the bootstrap defines.
BOOTSTRAP = """\
import sys
import runpy

runpy.run_path(sys.argv[1], run_name="__main__")
"""


def read_capped(path: Path, keep_tail: bool) -> str:
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


def sandbox_interpreter() -> str:
    """The real interpreter, never a virtualenv launcher.

    On Windows a venv's `python.exe` is a stub that re-executes the base
    interpreter as a *second* process, which the single-process limit
    correctly refuses; on POSIX `RLIMIT_NPROC` does the same. The base
    interpreter also carries no venv site-packages, so a submission cannot
    reach this project's dependencies.
    """
    import sys

    return getattr(sys, "_base_executable", None) or sys.executable
