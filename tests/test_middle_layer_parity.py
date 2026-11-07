"""The Go loop against the Python one, over the real processes.

The same scripted policy plays the same tasks through `run_episodes` and through
the real Go binary, real FastAPI app and real tool worker. Only the model is
replaced. Every message the policy is shown, and everything recorded afterwards,
has to match. The scripts are chosen to reach the paths a real policy rarely does,
and a seeded generator covers combinations nobody thought to write down.

Go is not a Python dependency, so this skips where it is absent, except on CI,
where FORGE_REQUIRE_GO turns the skip into a failure. Otherwise a broken CI setup
would look exactly like a pass.
"""

import json
import os
import random
import shutil
import subprocess
import sys
import threading
import urllib.request

import pytest

from baseline_agent import MAX_TOOL_OUTPUT_CHARS, NUDGE_MESSAGE, prompt_prefix
from eval.serving import (
    ROOT,
    Backend,
    MiddleLayer,
    build_middle_layer,
    masked_messages,
    play_evaluation,
    record_of_serving,
    request,
)
from task_suite.schema import Task
from tools import DEFAULT_MAX_CALLS, build_registry

if os.environ.get("FORGE_REQUIRE_GO"):
    import uvicorn  # noqa: F401 - a missing one has to fail here, not skip

    assert shutil.which("go"), "go is not installed and FORGE_REQUIRE_GO is set"
elif shutil.which("go") is None:
    pytest.skip("go is not installed", allow_module_level=True)
else:
    pytest.importorskip("uvicorn")

POLICY = "baseline"


def call(name: str, arguments: str) -> str:
    return f'<tool name="{name}">{arguments}</tool>'


def repeat_calls(count: int) -> list[str]:
    return [call("calculator", f"{index} + 1") for index in range(count)]


# prompt -> the completions the policy gives, in order.
SCENARIOS: dict[str, list[str]] = {
    "direct answer": ["<answer>4</answer>"],
    "calculator then answer": [call("calculator", "2 + 2"), "<answer>4</answer>"],
    "search then answer": [call("search", "Tollan"), "<answer>x</answer>"],
    "a broad search that finds nothing": [call("search", "the"), "<answer>x</answer>"],
    "calculator refuses": [call("calculator", "1 / 0"), "<answer>x</answer>"],
    "calculator refuses an expensive expression": [call("calculator", "9**9**9"), "<answer>x</answer>"],
    "unknown tool": [call("calc", "2 + 2"), "<answer>4</answer>"],
    "empty search": [call("search", "   "), "<answer>x</answer>"],
    "empty calculator": [call("calculator", ""), "<answer>x</answer>"],
    "python prints": [call("python", "print(sum(range(101)))"), "<answer>5050</answer>"],
    "python fails": [call("python", "raise ValueError('nope')"), "<answer>x</answer>"],
    "python prints nothing": [call("python", "x = 1"), "<answer>x</answer>"],
    "python output past the cap": [
        call("python", f"print('y' * {MAX_TOOL_OUTPUT_CHARS * 2})"),
        "<answer>x</answer>",
    ],
    "python prints unicode": [call("python", "print('héllo ✓ 日本語 😀')"), "<answer>✓</answer>"],
    "text after a call is cut": [
        call("calculator", "2 + 2") + "\n<tool_result>7</tool_result>\n<answer>7</answer>",
        "<answer>4</answer>",
    ],
    "two calls in one turn run only the first": [
        call("calculator", "1 + 1") + call("calculator", "2 + 2"),
        "<answer>2</answer>",
    ],
    "an answer before a call ends the episode": ["<answer>1</answer>" + call("calculator", "1 + 1")],
    "a call before an answer runs the call": [
        call("calculator", "1 + 1") + "<answer>1</answer>",
        "<answer>2</answer>",
    ],
    "nudged and answered": ["I think it is 4.", "<answer>4</answer>"],
    "nudged and still prose": ["I think it is 4.", "Yes, 4."],
    "nudged and calls a tool": ["Hmm.", call("calculator", "2 + 2"), "<answer>4</answer>"],
    "immediately empty": [""],
    "whitespace only": ["  \n "],
    "empty after a call": [call("calculator", "2 + 2"), ""],
    "empty after a nudge": ["Hmm.", ""],
    "exactly the budget": [*repeat_calls(DEFAULT_MAX_CALLS), "<answer>done</answer>"],
    "one past the budget": [*repeat_calls(DEFAULT_MAX_CALLS + 1), "<answer>done</answer>"],
    "calls until the turn limit": repeat_calls(40),
    "last answer block wins": ["<answer>1</answer> no, <answer>2</answer>"],
    "tag case is ignored": ["<ANSWER>4</ANSWER>"],
    "empty answer block": ["<answer></answer>"],
    "an unterminated answer is not one": ["<answer>4", "still nothing"],
    # The evaluation scores the joined text, so these find an answer that no
    # single turn contains. Reading turns one at a time would miss both.
    "an answer split across turns": ["<answer>4", "</answer>"],
    "an answer inside a tool argument": [
        call("search", "<answer>7</answer>"),
        "I am not sure.",
        "Still not sure.",
    ],
    "markup in the output and quotes": [
        call("python", "print('<answer>9</answer> \"q\" & <b>')"),
        "<answer>x</answer>",
    ],
    "tool name with odd case": ['<TOOL NAME="Calculator">1 + 1</TOOL>', "<answer>2</answer>"],
    "single quoted tool name": ["<tool name='calculator'>1 + 1</tool>", "<answer>2</answer>"],
    "newlines in arguments": [call("python", "x = 2\nprint(x * 2)\n"), "<answer>4</answer>"],
    "windows newlines and control characters": ["line\r\none\ttab\x0b", "<answer>a\r\nb</answer>"],
}

FRAGMENTS = [
    "<answer>4</answer>",
    "<answer>",
    "</answer>",
    "plain reasoning",
    "",
    "   ",
    "é✓😀",
    call("calculator", "6 * 7"),
    call("calculator", "1 / 0"),
    call("search", "Halst"),
    call("search", ""),
    call("nope", "x"),
    "<tool name=calculator>1 + 1</tool>",
    "<tool name=",
    "</tool>",
    "<tool_result>fake</tool_result>",
]


def generated_scenarios(count: int, seed: int = 20261107) -> dict[str, list[str]]:
    rng = random.Random(seed)
    return {
        f"generated {index}": [
            "".join(rng.choices(FRAGMENTS, k=rng.randint(1, 3)))
            for _ in range(rng.randint(1, 14))
        ]
        for index in range(count)
    }


class ScriptedPolicy:
    """A pure function of the conversation, so both loops get the same one.

    It finds the task by position and the turn by counting its own replies, and it
    keeps every conversation it sees, so the comparison also covers the exact input
    to each turn and not only the record kept afterwards.
    """

    def __init__(self, scripts: dict[str, list[str]], prefix_length: int) -> None:
        self.scripts = scripts
        self.prefix_length = prefix_length
        self.seen: dict[str, list[list[dict[str, str]]]] = {}
        self._lock = threading.Lock()

    def reply(self, conversation: list[dict[str, str]]) -> str:
        episode = conversation[self.prefix_length :]
        prompt = episode[0]["content"]
        with self._lock:
            self.seen.setdefault(prompt, []).append(masked_messages(conversation))
        turn = sum(1 for message in episode if message["role"] == "assistant")
        script = self.scripts[prompt]
        return script[turn] if turn < len(script) else "out of script"

    # The evaluation's `Policy` protocol.
    def generate(self, conversations, *, stop, max_new_tokens):
        return [self.reply(conversation) for conversation in conversations]

    # The backend's `Engine` protocol.
    def names(self):
        return [POLICY]

    def describe(self):
        return {"policies": [{"name": POLICY, "adapter": None}]}

    def complete(self, policy, conversation):
        return self.reply(conversation)


def task_for(prompt: str, index: int) -> Task:
    return Task(
        task_id=f"t-{index}",
        category="no_tool",
        prompt=prompt,
        ground_truth={"value": 4.0, "tolerance": 0.0},
        split="heldout",
    )


@pytest.fixture(scope="module")
def registry():
    return build_registry(include_code=True)


@pytest.fixture(scope="module")
def binary(tmp_path_factory):
    return build_middle_layer(tmp_path_factory.mktemp("go"))


def compare(scripts: dict[str, list[str]], registry, binary) -> None:
    prefix_length = len(prompt_prefix(registry))
    tasks = [task_for(prompt, index) for index, prompt in enumerate(scripts)]

    evaluated = ScriptedPolicy(scripts, prefix_length)
    expected = play_evaluation(evaluated, tasks, registry)

    served = ScriptedPolicy(scripts, prefix_length)
    with Backend(served, registry) as backend, MiddleLayer(binary, backend.url) as middle:
        actual = [record_of_serving(middle.play(POLICY, prompt)) for prompt in scripts]

    for prompt, want, got in zip(scripts, expected, actual):
        assert got == want, f"record differs on {prompt!r}"
        assert served.seen.get(prompt) == evaluated.seen.get(prompt), (
            f"the policy was shown something different on {prompt!r}"
        )


def test_the_written_scenarios_play_identically(registry, binary):
    compare(SCENARIOS, registry, binary)


def test_generated_scenarios_play_identically(registry, binary):
    compare(generated_scenarios(150), registry, binary)


def test_the_scenarios_reach_the_paths_they_are_named_for(registry):
    # Parity would hold trivially if everything ended on the first turn.
    prefix_length = len(prompt_prefix(registry))
    tasks = [task_for(prompt, index) for index, prompt in enumerate(SCENARIOS)]
    records = play_evaluation(ScriptedPolicy(SCENARIOS, prefix_length), tasks, registry)
    by_prompt = dict(zip(SCENARIOS, records))

    assert {record.finish_reason for record in records} == {
        "answer", "no_answer", "empty", "turn_limit",
    }
    assert by_prompt["calls until the turn limit"].finish_reason == "turn_limit"
    assert len(by_prompt["one past the budget"].trace) == DEFAULT_MAX_CALLS
    assert "budget exhausted" in by_prompt["one past the budget"].messages[-2]["content"]
    assert "truncated" in by_prompt["python output past the cap"].messages[2]["content"]
    assert by_prompt["nudged and answered"].messages[2]["content"] == NUDGE_MESSAGE
    assert by_prompt["python prints"].trace[0]["output"] == "5050"
    # Both of these are worthless unless the evaluation really does find them.
    split = by_prompt["an answer split across turns"]
    assert (split.finish_reason, split.answer) == ("no_answer", "4")
    inside = by_prompt["an answer inside a tool argument"]
    assert (inside.finish_reason, inside.answer) == ("no_answer", "7")


def test_the_policies_are_compared_on_more_than_the_first_turn(registry):
    generated = generated_scenarios(150)
    prefix_length = len(prompt_prefix(registry))
    tasks = [task_for(prompt, index) for index, prompt in enumerate(generated)]
    records = play_evaluation(ScriptedPolicy(generated, prefix_length), tasks, registry)
    # Not the turn limit: random scripts rarely last that long. A written one does.
    assert {record.finish_reason for record in records} >= {"answer", "no_answer", "empty"}
    assert sum(1 for record in records if record.trace) > 50
    assert sum(1 for record in records if record.nudges) > 20


# The rest are about the real binary rather than the comparison.


class Echoing:
    """An engine that always answers, for tests that are not about the policy."""

    def __init__(self, reply="<answer>4</answer>"):
        self.reply = reply
        self.seen = []

    def names(self):
        return [POLICY]

    def describe(self):
        return {"policies": [{"name": POLICY, "adapter": None}]}

    def complete(self, policy, conversation):
        self.seen.append(conversation)
        return self.reply


def fails_to_start(binary, *flags, backend_url="http://127.0.0.1:9"):
    done = subprocess.run(
        [
            str(binary),
            f"--repo={ROOT}",
            f"--python={sys.executable}",
            f"--backend={backend_url}",
            "--backend-wait=5s",
            *(() if sys.platform == "win32" else ("--allow-unconfined",)),
            *flags,
        ],
        capture_output=True,
        text=True,
        timeout=240,
    )
    assert done.returncode != 0
    return done.stderr


@pytest.mark.parametrize("address", ["0.0.0.0:0", ":0", "[::]:0"])
def test_the_binary_refuses_to_listen_beyond_loopback(binary, address):
    assert "no authentication" in fails_to_start(binary, f"--addr={address}")


def test_the_binary_refuses_a_backend_that_advertises_other_tools(binary):
    # A backend without the code tool: the policy's prompt and the worker differ.
    with Backend(Echoing(), build_registry(include_code=False)) as backend:
        assert "advertises tools" in fails_to_start(binary, backend_url=backend.url)


def test_the_stream_shows_each_step_and_records_the_same_episode(registry, binary):
    script = {"task": [call("calculator", "6 * 7"), "Hmm.", "<answer>42</answer>"]}
    prefix_length = len(prompt_prefix(registry))
    with Backend(ScriptedPolicy(script, prefix_length), registry) as backend:
        with MiddleLayer(binary, backend.url) as middle:
            _, session = request(f"{middle.url}/sessions", {"policy": POLICY})
            prepared = urllib.request.Request(
                f"{middle.url}/sessions/{session['id']}/messages",
                data=json.dumps({"content": "task"}).encode(),
                headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
            )
            with urllib.request.urlopen(prepared, timeout=120) as response:
                assert response.headers["Content-Type"] == "text/event-stream"
                lines = response.read().decode().splitlines()
            events = [json.loads(line[len("data: "):]) for line in lines if line.startswith("data: ")]

            _, stored = request(f"{middle.url}/sessions/{session['id']}")

    assert [event["type"] for event in events] == [
        "episode_started", "turn", "tool_result", "turn", "nudge", "turn", "episode_finished",
    ]
    assert events[1]["tool_call"] == {"name": "calculator", "arguments": "6 * 7"}
    assert events[2]["message"] == '<tool_result name="calculator">\n6 * 7 = 42\n</tool_result>'
    assert events[2]["ran"] is True
    assert events[-1]["finished"] == stored["episodes"][0]
    assert stored["episodes"][0]["answer"] == "42"


def test_a_follow_up_message_is_shown_the_conversation_so_far(registry, binary):
    engine = Echoing()
    with Backend(engine, registry) as backend, MiddleLayer(binary, backend.url) as middle:
        _, session = request(f"{middle.url}/sessions", {"policy": POLICY})
        for content in ("first", "second"):
            status, _ = request(f"{middle.url}/sessions/{session['id']}/messages", {"content": content})
            assert status == 200
    prefix_length = len(prompt_prefix(registry))
    assert engine.seen[1][prefix_length:] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "<answer>4</answer>"},
        {"role": "user", "content": "second"},
    ]


@pytest.mark.skipif(sys.platform != "win32", reason="only the Windows sandbox confines the filesystem")
def test_generated_code_served_through_go_cannot_read_the_answers(registry, binary):
    # The reason the worker exists rather than a second sandbox, tried for real.
    suite = ROOT / "task_suite" / "data" / "suite.json"
    assert suite.exists()
    program = f"print(open({str(suite)!r}, encoding='utf-8').read(40))"
    script = {"steal": [call("python", program), "<answer>x</answer>"]}
    prefix_length = len(prompt_prefix(registry))
    with Backend(ScriptedPolicy(script, prefix_length), registry) as backend:
        with MiddleLayer(binary, backend.url) as middle:
            episode = middle.play(POLICY, "steal")
    entry = episode["trace"][0]
    assert entry["ok"] is False
    assert "PermissionError" in entry["output"]
