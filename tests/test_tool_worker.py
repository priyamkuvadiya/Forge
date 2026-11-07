"""The tool worker, in a real process, over real pipes.

The protocol's failure modes are about processes: stray output on stdout, a child
reading stdin, a parent that disappears. An in-process test would miss all three,
so most of these start a subprocess. The tools in it are mostly stand-ins, because
the real ones have their own tests in `tools/tests`.
"""

import json
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from baseline_agent import MAX_TOOL_OUTPUT_CHARS, render_for_policy
from tool_worker import Worker
from tools import DEFAULT_MAX_CALLS, ToolCall, ToolRegistry, ToolResult, ToolSession

ROOT = Path(__file__).resolve().parent.parent

# Starts `tool_worker.serve` with deliberately badly behaved tools, using the
# real `claim_protocol_streams`.
HOSTILE_WORKER = r"""
import os, subprocess, sys, threading
from tool_worker import claim_protocol_streams, serve
from tools import ToolRegistry, ToolResult

reader, writer = claim_protocol_streams()
release = threading.Event()

class Noisy:
    name = "noisy"
    description = "prints to stdout three ways"
    def invoke(self, arguments):
        print("python print")
        os.write(1, b"raw write to fd 1\n")
        subprocess.run([sys.executable, "-c", "print('child print')"], check=True)
        return ToolResult(tool=self.name, ok=True, output="clean")

class StdinReader:
    name = "stdin_reader"
    description = "a child that reads its inherited stdin"
    def invoke(self, arguments):
        done = subprocess.run(
            [sys.executable, "-c", "import sys; print(repr(sys.stdin.read()))"],
            capture_output=True, text=True, timeout=30, stdin=None,
        )
        return ToolResult(tool=self.name, ok=True, output=done.stdout.strip())

class Slow:
    name = "slow"
    description = "blocks until another call releases it"
    def invoke(self, arguments):
        release.wait(30)
        return ToolResult(tool=self.name, ok=True, output="slow done")

class Release:
    name = "release"
    description = "releases the slow tool"
    def invoke(self, arguments):
        release.set()
        return ToolResult(tool=self.name, ok=True, output="released")

serve(ToolRegistry([Noisy(), StdinReader(), Slow(), Release()]), reader, writer, {"tools": []})
"""


class Pipe:
    def __init__(self, *command: str) -> None:
        self.process = subprocess.Popen(
            command, cwd=ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

    def send(self, **request) -> None:
        self.process.stdin.write(json.dumps(request).encode() + b"\n")
        self.process.stdin.flush()

    def read(self) -> dict:
        # A read that hangs would hang the suite, so it runs under a deadline.
        box: list[bytes] = []
        thread = threading.Thread(target=lambda: box.append(self.process.stdout.readline()))
        thread.daemon = True
        thread.start()
        thread.join(60)
        assert box, "no line from the worker within 60 seconds"
        return json.loads(box[0])

    def call(self, **request) -> dict:
        self.send(**request)
        return self.read()

    def close(self) -> int:
        self.process.stdin.close()
        return self.process.wait(30)


@pytest.fixture
def hostile():
    pipe = Pipe(sys.executable, "-u", "-c", HOSTILE_WORKER)
    assert pipe.read()["ready"] is True
    assert pipe.call(id=0, op="open", episode="e", max_calls=8)["ok"] is True
    yield pipe
    pipe.process.kill()
    pipe.process.wait(30)


def test_nothing_printed_by_a_tool_or_its_children_can_corrupt_the_protocol(hostile):
    # Every line on the protocol stream must parse, so a corrupted stream fails
    # in `read`. The next request also has to work.
    reply = hostile.call(id=1, op="invoke", episode="e", name="noisy", arguments="")
    assert reply["ok"] is True and reply["output"] == "clean"
    assert hostile.call(id=2, op="finish", episode="e", response="")["ok"] is True


def test_a_child_cannot_swallow_requests_from_the_protocol_stream(hostile):
    # With the real stdin inherited, this child would block forever or eat the
    # next request. It has to see an immediately empty stream instead.
    reply = hostile.call(id=1, op="invoke", episode="e", name="stdin_reader", arguments="")
    assert reply["ok"] is True and reply["output"] == "''"


def test_a_slow_call_does_not_block_another(hostile):
    # `slow` only returns once `release` has run, so if calls ran one at a time
    # this would deadlock until the tool's own timeout.
    hostile.send(id=1, op="invoke", episode="e", name="slow", arguments="")
    hostile.send(id=2, op="invoke", episode="e", name="release", arguments="")
    first, second = hostile.read(), hostile.read()
    assert [first["id"], second["id"]] == [2, 1]
    assert second["output"] == "slow done"


def test_closing_stdin_ends_the_process():
    # The worker's only lifetime guarantee: when its parent goes, so does it.
    pipe = Pipe(sys.executable, "-u", "-c", HOSTILE_WORKER)
    assert pipe.read()["ready"] is True
    assert pipe.close() == 0


def test_garbage_gets_an_error_and_the_process_survives(hostile):
    hostile.process.stdin.write(b"this is not json\n[1, 2]\n")
    hostile.process.stdin.flush()
    assert hostile.read()["ok"] is False
    assert hostile.read()["ok"] is False
    assert hostile.call(id=9, op="nonsense")["ok"] is False
    assert hostile.call(id=10, op=["unhashable"])["ok"] is False
    assert hostile.call(id=11, op="invoke", episode="e", name="release", arguments="")["ok"] is True


def test_the_real_entry_point_serves_the_real_tools():
    command = [sys.executable, "-u", "-m", "tool_worker"]
    if sys.platform != "win32":
        command.append("--allow-unconfined")
    pipe = Pipe(*command)
    try:
        ready = pipe.read()
        assert ready["ready"] is True
        assert ready["max_calls"] == DEFAULT_MAX_CALLS
        assert ready["max_tool_output_chars"] == MAX_TOOL_OUTPUT_CHARS
        assert ready["tools"] == ["calculator", "python", "search"]
        assert ready["sandbox"]["filesystem_confined"] is (sys.platform == "win32")

        assert pipe.call(id=1, op="open", episode="e", max_calls=8)["ok"] is True
        reply = pipe.call(
            id=2, op="invoke", episode="e", name="python", arguments="print('héllo ✓')"
        )
        assert reply["tool_ok"] is True
        assert reply["output"] == "héllo ✓"
        assert reply["message"] == '<tool_result name="python">\nhéllo ✓\n</tool_result>'
    finally:
        assert pipe.close() == 0


def test_an_unconfined_sandbox_is_refused_without_the_explicit_flag():
    # The backend's report is replaced rather than the test being Linux-only: this
    # machine is where the demo runs, and a skip here would never have noticed the
    # check going missing.
    program = (
        "import dataclasses, tool_worker\n"
        "from tools import code_exec\n"
        "real = code_exec.confinement()\n"
        "code_exec.confinement = lambda: dataclasses.replace(real, filesystem=False)\n"
        "raise SystemExit(tool_worker.main([]))\n"
    )
    done = subprocess.run(
        [sys.executable, "-c", program],
        cwd=ROOT, stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120,
    )
    assert done.returncode == 1
    assert done.stdout == ""
    assert "--allow-unconfined" in done.stderr


# The rest run in-process: they are about what the replies say, not the pipes.


class Echo:
    name = "echo"
    description = "returns its argument"

    def invoke(self, arguments):
        return ToolResult(tool=self.name, ok=True, output=arguments)


def request(op, **fields):
    return {"id": 1, "op": op, "episode": "e", **fields}


@pytest.fixture
def worker():
    worker = Worker(ToolRegistry([Echo()]))
    assert worker.handle(request("open", max_calls=2))["ok"] is True
    return worker


def test_the_message_is_exactly_what_the_evaluation_appends(worker):
    long = "x" * (MAX_TOOL_OUTPUT_CHARS + 500)
    reply = worker.handle(request("invoke", name="echo", arguments=long))
    expected = render_for_policy(ToolResult(tool="echo", ok=True, output=long))
    assert reply["message"] == expected
    assert "[truncated" in reply["message"]
    # The trace keeps all of it, as the evaluation's does.
    assert reply["trace"]["output"] == long


def test_the_budget_is_toolsessions_and_a_refused_call_leaves_no_trace(worker):
    registry = ToolRegistry([Echo()])
    reference = ToolSession(registry, max_calls=2)
    for _ in range(2):
        assert worker.handle(request("invoke", name="echo", arguments="a"))["trace"] is not None
        reference.invoke(ToolCall("echo", "a"))

    refused = worker.handle(request("invoke", name="echo", arguments="a"))
    expected = reference.invoke(ToolCall("echo", "a"))
    assert refused["tool_ok"] is False
    assert refused["output"] == expected.output
    assert refused["tool_error"] == "budget exceeded"
    assert refused["trace"] is None
    assert len(worker.handle(request("finish", response=""))["trace"]) == 2


def test_tool_failures_are_results_and_bad_requests_are_errors(worker):
    unknown = worker.handle(request("invoke", name="nope", arguments=""))
    assert unknown["ok"] is True and unknown["tool_ok"] is False
    assert "no tool named" in unknown["output"]

    stray = worker.handle({"id": 2, "op": "invoke", "episode": "other", "name": "echo", "arguments": ""})
    assert stray == {"id": 2, "ok": False, "error": "unknown episode 'other'"}

    missing = worker.handle({"id": 3, "op": "invoke", "episode": "e"})
    assert missing["ok"] is False and "KeyError" in missing["error"]


def test_opening_an_open_episode_cannot_refill_its_budget(worker):
    worker.handle(request("invoke", name="echo", arguments="a"))
    assert worker.handle(request("open", max_calls=2))["ok"] is False
    assert len(worker.handle(request("finish", response=""))["trace"]) == 1


def test_finish_reads_the_answer_across_turns_and_closes_the_episode(worker):
    # The joined text is what the evaluation scores, including an answer split
    # over two turns that neither turn contains on its own.
    reply = worker.handle(request("finish", response="<answer>4\n</answer>"))
    assert reply["answer"] == "4"
    assert worker.handle(request("finish", response=""))["ok"] is False
