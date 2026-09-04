"""Tools the agent can call.

Three of them, and they are not equally dangerous. The calculator parses
arithmetic against an AST whitelist; the search tool reads a fixed local
corpus; the code executor runs text a language model wrote, which is the only
one of the three that can do damage and therefore the only one whose isolation
mechanism is a design decision rather than an implementation detail.

Keeping them behind one package matters for module 7: the Go middle layer owns
tool-call routing at serving time, and it should be routing to one boundary
with one contract, not to three ad-hoc entry points.

`code_exec` is Windows-only, because its isolation is built on Job Objects.
It is imported lazily below rather than at module import, so that the
calculator and the search tool stay usable on any platform - the eval harness
and the Go layer have reasons to touch those two without needing a sandbox.
"""

from .calculator import CalcResult, CalculatorError, calculate, evaluate
from .search import Document, SearchHit, SearchIndex, load_index, render_hits

__all__ = [
    "CalcResult",
    "CalculatorError",
    "calculate",
    "evaluate",
    "Document",
    "SearchHit",
    "SearchIndex",
    "load_index",
    "render_hits",
    "SandboxError",
    "SandboxedCodeRunner",
    "run_code",
]


def __getattr__(name: str):
    """Import the sandbox on first use.

    A plain top-level import would make `import tools` fail on any platform
    where the sandbox cannot exist, taking the two portable tools down with
    it for no reason.
    """
    if name in ("SandboxError", "SandboxedCodeRunner", "run_code"):
        from . import code_exec

        return getattr(code_exec, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
