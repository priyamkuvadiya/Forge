"""Tools the agent can call.

Three of them, and they are not equally dangerous. The calculator parses
arithmetic against an AST whitelist; the search tool reads a fixed local
corpus; the code executor runs text a language model wrote, which is the only
one of the three that can do damage and therefore the only one whose isolation
mechanism is a design decision rather than an implementation detail.

Keeping them behind one package matters for module 7: the Go middle layer owns
tool-call routing at serving time, and it should be routing to one boundary
with one contract, not to three ad-hoc entry points.

The code executor (`code_exec`) is not implemented yet — its isolation
mechanism is still open, see `CLAUDE.md`. It is deliberately not imported here
so that the calculator and search tool are usable without it.
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
]
