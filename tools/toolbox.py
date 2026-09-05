"""One contract for calling tools, and one record of what was called.

`tools/__init__.py` has claimed since it was written that the point of this
package is "routing to one boundary with one contract, not three ad-hoc entry
points". Until this module existed that was aspirational: the calculator
returned a `CalcResult`, search returned a `list[SearchHit]`, and the sandbox
returned a `CodeRunResult`, with three different rendering conventions and no
way for a model to name any of them. Anything that wanted to *use* the tools -
the baseline agent, the RL rollout loop, the Go middle layer - would have had
to invent that layer, and each would have invented it slightly differently.

Four things live here, and each is needed by more than one later module:

**A call format the policy emits.** `<tool name="calculator">2 + 2</tool>`,
deliberately shaped like the `<answer>` contract in `task_suite/protocol.py`
so the policy learns one syntax rather than two. It is parsed strictly, for
the same reason answers are: a format the verifier is generous about is a
format RL never has to learn.

**A uniform result.** Every tool returns a `ToolResult`, and every failure -
bad syntax, unknown tool, a calculator refusing an expression, a sandbox
timing out - comes back as a result with `ok=False`, never as an exception.
A tool that raises would abort a rollout, and one bad tool call is ordinary
policy behaviour that should cost reward, not crash training.

**A budget.** A policy with no cap can loop on `search` forever; at 8 rollouts
a task across thousands of steps that is not a hypothetical cost. The budget
lives here rather than in the agent loop so that the eval harness and the
serving path cannot disagree about it.

**A trace.** Every call is recorded with its arguments, result and duration,
serialisable to plain dicts. Module 7 stores these in SQLite for the live
reasoning view; module 9 needs the call *counts* to answer the question the
`no_tool` category exists to ask - whether the policy learned when not to
reach for a tool.
"""

import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Protocol

# How many tool calls one task attempt may make before the budget refuses.
# Generous next to what these tasks need: the longest legitimate chain in the
# suite is a three-hop QA question, which is three searches and possibly one
# calculation. The cap exists to stop an unbounded loop, not to make the
# policy ration itself.
DEFAULT_MAX_CALLS = 8

_TOOL_PATTERN = re.compile(
    r"""<tool\s+name\s*=\s*["']?(?P<name>[a-zA-Z_][a-zA-Z0-9_]*)["']?\s*>"""
    r"""(?P<arguments>.*?)</tool>""",
    re.DOTALL | re.IGNORECASE,
)

TOOL_INSTRUCTIONS = """\
You may call a tool by writing:

<tool name="TOOL_NAME">ARGUMENTS</tool>

Write nothing after a tool call; stop and wait for the result, which will be \
given to you before you continue. You may call tools several times. When you \
have the final answer, give it inside <answer></answer> tags instead."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: str


@dataclass(frozen=True)
class ToolResult:
    """What every tool returns, whatever happened.

    `output` is what the policy sees. `error` is for the trace and for humans;
    a failing result still puts something readable in `output`, because a
    policy that is told only "error" has nothing to correct.
    """

    tool: str
    ok: bool
    output: str
    error: str | None = None

    def render(self) -> str:
        """The block appended to the transcript before the policy continues."""
        return f"<tool_result name=\"{self.tool}\">\n{self.output}\n</tool_result>"


class Tool(Protocol):
    """What the registry needs from anything callable.

    `invoke` takes the raw argument text rather than parsed parameters. These
    tools each take exactly one free-text argument - an expression, a query,
    a program - and a JSON envelope would be a second format for a 0.5B model
    to get wrong for no benefit.
    """

    name: str
    description: str

    def invoke(self, arguments: str) -> ToolResult: ...


def parse_tool_calls(text: str) -> list[ToolCall]:
    """Every well-formed tool call in `text`, in order.

    Unterminated calls do not match, deliberately - a truncated generation is
    a failed generation, the same position `extract_answer` takes on an
    unclosed `<answer>`.
    """
    return [
        ToolCall(name=match.group("name").lower().strip(), arguments=match.group("arguments").strip())
        for match in _TOOL_PATTERN.finditer(text)
    ]


class BudgetExceeded(Exception):
    """Raised by `ToolSession` only if a caller ignores `calls_remaining`."""


@dataclass
class TraceEntry:
    index: int
    tool: str
    arguments: str
    ok: bool
    output: str
    error: str | None
    duration_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ToolRegistry:
    """The set of tools available, and the text describing them to the policy."""

    def __init__(self, tools: list[Tool]) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"two tools are both named {tool.name!r}")
            self._tools[tool.name] = tool

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return sorted(self._tools)

    def describe(self) -> str:
        """The tool section of the system prompt.

        Built from the tools actually registered rather than written out in a
        prompt template, so a prompt can never advertise a tool that is not
        present - which on this project is a live risk, since the code tool is
        unavailable on platforms without the sandbox.
        """
        lines = [TOOL_INSTRUCTIONS, "", "Available tools:"]
        for name in self.names:
            lines.append(f"\n- {name}: {self._tools[name].description}")
        return "\n".join(lines)

    def invoke(self, call: ToolCall) -> ToolResult:
        """Run one call. Never raises for anything the policy did wrong."""
        tool = self._tools.get(call.name)
        if tool is None:
            available = ", ".join(self.names)
            return ToolResult(
                tool=call.name,
                ok=False,
                output=f"There is no tool named '{call.name}'. Available tools: {available}.",
                error="unknown tool",
            )

        try:
            return tool.invoke(call.arguments)
        except Exception as exc:  # noqa: BLE001 - a tool bug must not kill a rollout
            # A tool raising is a bug in *our* code, not the policy's. It still
            # cannot be allowed to abort a training run, so it becomes a failed
            # result and is visible in the trace.
            return ToolResult(
                tool=call.name,
                ok=False,
                output=f"The {call.name} tool failed unexpectedly.",
                error=f"{type(exc).__name__}: {exc}",
            )


class ToolSession:
    """One task attempt: a registry, a budget, and the trace of what happened.

    Stateful and single-use by design. The trace is the artefact modules 7 and
    9 consume, and it only means anything if it covers exactly one attempt.
    """

    def __init__(self, registry: ToolRegistry, max_calls: int = DEFAULT_MAX_CALLS) -> None:
        if max_calls < 1:
            raise ValueError("max_calls must be at least 1")
        self.registry = registry
        self.max_calls = max_calls
        self.trace: list[TraceEntry] = []

    @property
    def calls_made(self) -> int:
        return len(self.trace)

    @property
    def calls_remaining(self) -> int:
        return self.max_calls - self.calls_made

    def invoke(self, call: ToolCall) -> ToolResult:
        """Run a call against the budget, recording it either way.

        Exhausting the budget returns a result rather than raising, and that
        result is deliberately explicit about what happened: a policy that is
        simply ignored when it calls a tool learns nothing, whereas one told
        it has run out can still commit to an answer with what it has.
        """
        if self.calls_remaining <= 0:
            return ToolResult(
                tool=call.name,
                ok=False,
                output=(
                    f"Tool call budget exhausted ({self.max_calls} calls). "
                    "Give your final answer now inside <answer></answer> tags."
                ),
                error="budget exceeded",
            )

        started = time.perf_counter()
        result = self.registry.invoke(call)
        duration_ms = (time.perf_counter() - started) * 1000

        self.trace.append(
            TraceEntry(
                index=len(self.trace),
                tool=call.name,
                arguments=call.arguments,
                ok=result.ok,
                output=result.output,
                error=result.error,
                duration_ms=round(duration_ms, 3),
            )
        )
        return result

    def trace_as_dicts(self) -> list[dict[str, Any]]:
        """Plain data, ready for JSON or a SQLite row."""
        return [entry.to_dict() for entry in self.trace]
