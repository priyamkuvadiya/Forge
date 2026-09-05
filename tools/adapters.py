"""The three tools, behind the one contract in `toolbox.py`.

Thin on purpose. Every one of these wraps a module that already exists and
already has its own tests; the adapter's whole job is to turn that module's
particular return type into a `ToolResult` and to write the description the
policy reads. Any logic that appears here rather than in the underlying module
would be logic the module's own tests do not cover.

The descriptions are part of the system prompt, so they are written for a
0.5B model rather than for a reader of this file: short, one concrete example
each, and explicit about the single argument the tool takes.
"""

from pathlib import Path

from .calculator import calculate
from .search import DEFAULT_TOP_K, SearchIndex, load_index, render_hits
from .toolbox import Tool, ToolRegistry, ToolResult


class CalculatorTool:
    name = "calculator"
    description = (
        "Evaluate one arithmetic expression and return its value. "
        "Supports + - * / // % ** and the functions abs, round, min, max, sum, "
        "sqrt, floor, ceil, log, log10, log2, exp, pow. "
        'Example: <tool name="calculator">round(16.99 * 509 * (1 - 12 / 100), 2)</tool>'
    )

    def invoke(self, arguments: str) -> ToolResult:
        result = calculate(arguments)
        if result.ok:
            return ToolResult(tool=self.name, ok=True, output=result.render())
        # The calculator's own error strings name the problem ("division by
        # zero", "unknown function 'foo'"), which is what a policy needs in
        # order to retry differently rather than retry identically.
        return ToolResult(
            tool=self.name,
            ok=False,
            output=f"error: {result.error}",
            error=result.error,
        )


class SearchTool:
    name = "search"
    description = (
        "Search the document corpus and return the most relevant documents in full. "
        "Takes a plain-text query. Facts are spread across documents, so a question "
        "may need more than one search: read what comes back, then search again for "
        "a name you found in it. A query too broad to distinguish between documents "
        "returns nothing, which means you should search for something more specific. "
        'Example: <tool name="search">Ashen Bay Survey</tool>'
    )

    def __init__(self, index: SearchIndex | None = None, k: int = DEFAULT_TOP_K) -> None:
        # The index is built once and reused. Rebuilding it per call would put
        # corpus parsing on the hot path of every rollout.
        self._index = index if index is not None else load_index()
        self._k = k

    def invoke(self, arguments: str) -> ToolResult:
        query = arguments.strip()
        if not query:
            return ToolResult(
                tool=self.name,
                ok=False,
                output="error: empty search query",
                error="empty query",
            )

        hits = self._index.search(query, k=self._k)
        # An empty result is a successful search that matched nothing, not a
        # failure. Marking it `ok=False` would teach the policy that its query
        # was malformed when in fact it was merely too broad.
        return ToolResult(tool=self.name, ok=True, output=render_hits(hits))


class CodeTool:
    """Runs code in the sandbox, as a scratchpad for the policy.

    Distinct from how coding *tasks* are scored: `task_suite/verify_code.py`
    builds its own harness and injects the same runner. Both paths go through
    one sandbox, which is the point - there is one place where model-written
    code executes, and it is hardened and tested once.
    """

    name = "python"
    description = (
        "Run a short Python program and return what it printed. "
        "Standard library only, no network, no file access outside its own "
        "scratch directory, and it is stopped after a few seconds. "
        "Print what you want to see; nothing is returned automatically. "
        'Example: <tool name="python">print(sum(range(101)))</tool>'
    )

    def __init__(self, runner=None) -> None:
        # Imported here rather than at module scope: the sandbox is
        # platform-specific and this module must stay importable without it,
        # so that the calculator and search tools work anywhere.
        if runner is None:
            from .code_exec import SandboxedCodeRunner

            runner = SandboxedCodeRunner()
        self._runner = runner

    def invoke(self, arguments: str) -> ToolResult:
        source = arguments.strip()
        if not source:
            return ToolResult(
                tool=self.name, ok=False, output="error: no code given", error="empty program"
            )

        outcome = self._runner(source)

        if outcome.timed_out:
            return ToolResult(
                tool=self.name,
                ok=False,
                output="error: the program was stopped for taking too long",
                error="timed out",
            )

        if outcome.exit_code != 0:
            # The traceback is the useful part, and it is at the end of stderr.
            tail = "\n".join(outcome.stderr.strip().splitlines()[-6:])
            return ToolResult(
                tool=self.name,
                ok=False,
                output=f"the program failed:\n{tail}" if tail else "the program failed",
                error=f"exit code {outcome.exit_code}",
            )

        printed = outcome.stdout.strip()
        return ToolResult(
            tool=self.name,
            ok=True,
            output=printed if printed else "(the program printed nothing)",
        )


def build_registry(
    corpus_path: Path | str | None = None,
    include_code: bool = True,
    code_runner=None,
) -> ToolRegistry:
    """The registry the agent, the eval harness and the serving path all use.

    `include_code` is not a convenience flag. The sandbox is Windows-only
    today, and a registry that advertised a `python` tool which then failed on
    every call would be worse than one that does not offer it - the policy
    would burn its whole budget discovering that. Callers on a platform
    without a sandbox get a registry of two tools and a prompt that describes
    exactly those two.
    """
    tools: list[Tool] = [
        CalculatorTool(),
        SearchTool(load_index(corpus_path) if corpus_path else None),
    ]
    if include_code:
        tools.append(CodeTool(code_runner))
    return ToolRegistry(tools)
