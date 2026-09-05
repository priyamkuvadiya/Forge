"""Tests for the tool contract: parsing, dispatch, budget and trace.

The contract matters more than any single tool behind it. Module 4 emits calls
in this format, module 5 trains against whatever it returns, module 7 re-serves
it from Go, and module 9 reads the traces. A change here that goes unnoticed
desynchronises all four.
"""

import sys

import pytest

from tools.adapters import CalculatorTool, CodeTool, SearchTool, build_registry
from tools.toolbox import (
    DEFAULT_MAX_CALLS,
    ToolCall,
    ToolRegistry,
    ToolResult,
    ToolSession,
    parse_tool_calls,
)


from tools.code_exec import SandboxedCodeRunner, sandbox_available, sandbox_backend


class _Echo:
    name = "echo"
    description = "Echo the argument back."

    def invoke(self, arguments: str) -> ToolResult:
        return ToolResult(tool=self.name, ok=True, output=arguments)


class _Exploding:
    name = "boom"
    description = "Always raises."

    def invoke(self, arguments: str) -> ToolResult:
        raise RuntimeError("this tool is broken")


# --------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------

def test_parses_a_single_call():
    calls = parse_tool_calls('<tool name="calculator">2 + 2</tool>')
    assert calls == [ToolCall(name="calculator", arguments="2 + 2")]


def test_parses_several_calls_in_order():
    text = (
        'first <tool name="search">Ashen Bay</tool> then '
        '<tool name="calculator">1 + 1</tool>'
    )
    assert [c.name for c in parse_tool_calls(text)] == ["search", "calculator"]


@pytest.mark.parametrize(
    "text",
    [
        "<tool name='calculator'>2 + 2</tool>",
        "<tool name=calculator>2 + 2</tool>",
        '<TOOL NAME="Calculator">2 + 2</TOOL>',
        '<tool   name = "calculator" >2 + 2</tool>',
    ],
)
def test_tolerates_quoting_casing_and_spacing(text):
    """A 0.5B model's formatting is not what this project is measuring.

    Strict about structure, forgiving about whitespace and quotes: the policy
    still has to produce a well-formed, terminated call naming a real tool.
    """
    assert parse_tool_calls(text) == [ToolCall(name="calculator", arguments="2 + 2")]


def test_multiline_arguments_survive():
    call = parse_tool_calls('<tool name="python">a = 1\nprint(a)</tool>')[0]
    assert call.arguments == "a = 1\nprint(a)"


def test_an_unterminated_call_does_not_match():
    """A truncated generation is a failed generation.

    Same position `extract_answer` takes on an unclosed `<answer>`; being
    generous here would hand the policy a tool call it never finished writing.
    """
    assert parse_tool_calls('<tool name="calculator">2 + 2') == []


def test_text_without_calls_parses_to_nothing():
    assert parse_tool_calls("I think the answer is 4.") == []
    assert parse_tool_calls("<answer>4</answer>") == []


# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

def test_duplicate_tool_names_are_refused():
    with pytest.raises(ValueError):
        ToolRegistry([_Echo(), _Echo()])


def test_unknown_tool_returns_a_result_not_an_exception():
    registry = ToolRegistry([_Echo()])
    result = registry.invoke(ToolCall(name="nope", arguments=""))

    assert not result.ok
    # The policy has to be able to recover, so the message names what exists.
    assert "echo" in result.output


def test_a_tool_that_raises_becomes_a_failed_result():
    """A bug in our tool code must not abort a rollout.

    Losing a whole GRPO step to an unhandled exception in a tool would be a
    training failure that looks like a policy failure.
    """
    registry = ToolRegistry([_Exploding()])
    result = registry.invoke(ToolCall(name="boom", arguments=""))

    assert not result.ok
    assert "RuntimeError" in result.error


def test_describe_lists_exactly_the_registered_tools():
    """The prompt is built from the registry, so it cannot advertise a tool
    that is not there - which is a live risk, since the sandbox is
    platform-specific."""
    described = ToolRegistry([_Echo()]).describe()

    assert "echo" in described
    assert "calculator" not in described
    assert "<tool name=" in described


# --------------------------------------------------------------------------
# Session: budget and trace
# --------------------------------------------------------------------------

def test_calls_are_recorded_in_the_trace():
    session = ToolSession(ToolRegistry([_Echo()]))
    session.invoke(ToolCall(name="echo", arguments="hello"))

    entry = session.trace_as_dicts()[0]
    assert entry["tool"] == "echo"
    assert entry["arguments"] == "hello"
    assert entry["ok"] is True
    assert entry["duration_ms"] >= 0


def test_failed_calls_are_recorded_too():
    """A trace that only holds successes cannot explain a low reward."""
    session = ToolSession(ToolRegistry([_Echo()]))
    session.invoke(ToolCall(name="nope", arguments=""))

    assert session.calls_made == 1
    assert session.trace_as_dicts()[0]["ok"] is False


def test_the_budget_stops_further_calls():
    session = ToolSession(ToolRegistry([_Echo()]), max_calls=2)
    for _ in range(2):
        assert session.invoke(ToolCall(name="echo", arguments="x")).ok

    blocked = session.invoke(ToolCall(name="echo", arguments="x"))
    assert not blocked.ok
    assert blocked.error == "budget exceeded"


def test_the_budget_message_tells_the_policy_what_to_do_instead():
    """Silently dropping the call would teach the policy nothing."""
    session = ToolSession(ToolRegistry([_Echo()]), max_calls=1)
    session.invoke(ToolCall(name="echo", arguments="x"))
    blocked = session.invoke(ToolCall(name="echo", arguments="x"))

    assert "<answer>" in blocked.output


def test_a_blocked_call_does_not_consume_more_budget():
    session = ToolSession(ToolRegistry([_Echo()]), max_calls=1)
    session.invoke(ToolCall(name="echo", arguments="x"))
    session.invoke(ToolCall(name="echo", arguments="x"))

    assert session.calls_made == 1
    assert session.calls_remaining == 0


def test_a_zero_budget_is_refused():
    with pytest.raises(ValueError):
        ToolSession(ToolRegistry([_Echo()]), max_calls=0)


def test_the_default_budget_covers_the_longest_legitimate_chain():
    """Three-hop QA is three searches; cross-tool adds a calculation.

    The cap exists to stop an unbounded loop, not to make the policy ration
    itself, so it has to sit comfortably above what the suite actually needs.
    """
    assert DEFAULT_MAX_CALLS >= 5
    assert ToolSession(ToolRegistry([_Echo()])).calls_remaining == DEFAULT_MAX_CALLS


def test_the_trace_is_plain_data():
    """Module 7 puts these in SQLite and module 9 counts them."""
    import json

    session = ToolSession(ToolRegistry([_Echo()]))
    session.invoke(ToolCall(name="echo", arguments="hello"))

    assert json.loads(json.dumps(session.trace_as_dicts()))[0]["tool"] == "echo"


def test_sessions_do_not_share_state():
    """Traces only mean anything if each covers exactly one task attempt."""
    registry = ToolRegistry([_Echo()])
    first = ToolSession(registry)
    ToolSession(registry).invoke(ToolCall(name="echo", arguments="x"))

    assert first.calls_made == 0


# --------------------------------------------------------------------------
# The real tools behind the contract
# --------------------------------------------------------------------------

def test_calculator_tool_returns_the_value():
    result = CalculatorTool().invoke("round(16.99 * 509 * (1 - 12 / 100), 2)")
    assert result.ok
    assert "7610.16" in result.output


def test_calculator_tool_reports_a_refusal_usefully():
    result = CalculatorTool().invoke("9 ** 9 ** 9")
    assert not result.ok
    assert "exponent" in result.output


def test_search_tool_returns_documents():
    result = SearchTool().invoke("Ashen Bay Survey")
    assert result.ok
    assert "exp-10" in result.output


def test_search_tool_treats_no_matches_as_success():
    """An empty result means the query was too broad, not malformed.

    Marking it a failure would teach the policy to rewrite a well-formed
    query rather than a more specific one.
    """
    result = SearchTool().invoke("vessel")
    assert result.ok
    assert "No documents matched" in result.output


def test_search_tool_rejects_an_empty_query():
    assert not SearchTool().invoke("   ").ok


def test_registry_prompt_names_every_tool_it_holds():
    registry = build_registry(include_code=False)
    described = registry.describe()

    assert registry.names == ["calculator", "search"]
    for name in registry.names:
        assert name in described


def test_a_registry_without_the_sandbox_does_not_advertise_python():
    """Otherwise a policy on a platform without the sandbox burns its whole
    budget discovering the tool does not work."""
    assert "python" not in build_registry(include_code=False)


# --------------------------------------------------------------------------
# The code tool, against the real sandbox
# --------------------------------------------------------------------------

sandboxed = pytest.mark.skipif(
    not sandbox_available(), reason="no code sandbox is implemented for this platform"
)


@sandboxed
def test_code_tool_returns_what_the_program_printed():
    result = CodeTool().invoke("print(sum(range(101)))")
    assert result.ok
    assert result.output == "5050"


@sandboxed
def test_code_tool_explains_a_program_that_printed_nothing():
    """Silence and success are indistinguishable to a policy otherwise."""
    result = CodeTool().invoke("x = 1 + 1")
    assert result.ok
    assert "printed nothing" in result.output


@sandboxed
def test_code_tool_returns_the_traceback_tail_on_failure():
    result = CodeTool().invoke("raise ValueError('boom')")
    assert not result.ok
    assert "ValueError" in result.output


@sandboxed
def test_code_tool_reports_a_timeout_in_words():
    result = CodeTool(runner=SandboxedCodeRunner(default_timeout=1.5)).invoke(
        "while True:\n    pass"
    )
    assert not result.ok
    assert "too long" in result.output


@pytest.mark.skipif(
    sandbox_backend() != "windows",
    reason="filesystem confinement is AppContainer-specific (see code_exec_posix.py)",
)
def test_code_tool_runs_through_the_same_sandbox_as_the_verifier():
    """One place where model-written code executes, not two.

    If the tool ever grew its own runner, the hardening and the adversarial
    tests would cover only half the paths that actually execute code.
    """
    result = CodeTool().invoke(
        "import pathlib\n"
        "try:\n"
        "    pathlib.Path(r'C:\\Users').iterdir().__next__()\n"
        "    print('READABLE')\n"
        "except Exception:\n"
        "    print('DENIED')\n"
    )
    assert "DENIED" in result.output
