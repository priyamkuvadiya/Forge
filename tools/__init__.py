"""Tools the agent can call.

Three of them, and they are not equally dangerous. The calculator parses
arithmetic against an AST whitelist; the search tool reads a fixed local
corpus; the code executor runs text a language model wrote, which is the only
one of the three that can do damage and therefore the only one whose isolation
mechanism is a design decision rather than an implementation detail.

That one contract is `toolbox.py`: a call format the policy emits, a uniform
`ToolResult`, a per-attempt budget, and a serialisable trace. Module 4 emits
calls in it, module 5 trains against it, module 7's Go layer re-serves it, and
module 9 reads the traces - so it is defined once, here, rather than invented
separately by each.

`code_exec` dispatches on platform: an AppContainer plus a Job Object on
Windows, `fork` plus `setrlimit` on POSIX. It is imported lazily below rather
than at module import, so that the calculator and the search tool stay usable
even where no sandbox can be built at all.
"""

from .adapters import CalculatorTool, CodeTool, SearchTool, build_registry
from .calculator import CalcResult, CalculatorError, calculate, evaluate
from .search import Document, SearchHit, SearchIndex, load_index, render_hits
from .toolbox import (
    DEFAULT_MAX_CALLS,
    TOOL_INSTRUCTIONS,
    Tool,
    ToolCall,
    ToolCallSpan,
    ToolRegistry,
    ToolResult,
    ToolSession,
    find_tool_calls,
    parse_tool_calls,
)

__all__ = [
    # the contract
    "Tool",
    "ToolCall",
    "ToolCallSpan",
    "ToolResult",
    "ToolRegistry",
    "ToolSession",
    "find_tool_calls",
    "parse_tool_calls",
    "TOOL_INSTRUCTIONS",
    "DEFAULT_MAX_CALLS",
    "build_registry",
    "CalculatorTool",
    "SearchTool",
    "CodeTool",
    # the underlying tools
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
    "sandbox_available",
]


def __getattr__(name: str):
    """Import the sandbox on first use.

    A plain top-level import would make `import tools` fail on any platform
    where the sandbox cannot exist, taking the two portable tools down with
    it for no reason.
    """
    if name in ("SandboxError", "SandboxedCodeRunner", "run_code", "sandbox_available"):
        from . import code_exec

        return getattr(code_exec, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
