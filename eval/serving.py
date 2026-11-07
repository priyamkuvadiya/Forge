"""Does the serving path play the policy the evaluation measured? Checked, not assumed.

The published numbers describe a policy inside `baseline_agent.run_episodes`. In
the demo it runs inside a different loop, written in Go, over two process
boundaries. This compares the two directly: the same policy, the same task, one
run through each loop, and the full transcripts have to match character for
character - every message, the nudge count, the finish reason, the answer, and
every tool call with its output.

The test suite does this with a scripted policy, chosen to reach the awkward
paths (tests/test_middle_layer_parity.py). This module adds the real thing, with
greedy decoding so the two runs are comparable, on held-out tasks:

    python -m eval.serving --adapter run1=artifacts/rl/run1/adapter-final \\
                           --adapter run2=artifacts/rl/run2/adapter-final

Needs the GPU. Do not run the test suite alongside it.
"""

import argparse
import json
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from baseline_agent import Episode, make_episodes, run_episodes, select_tasks
from task_suite import Task, extract_answer
from tools import ToolRegistry, build_registry, code_exec

ROOT = Path(__file__).resolve().parent.parent
MIDDLE_LAYER = ROOT / "middle-layer"


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def request(url: str, body: Any = None, *, timeout: float = 600) -> tuple[int, Any]:
    data = None if body is None else json.dumps(body).encode()
    prepared = urllib.request.Request(url, data=data, method="POST" if data else "GET")
    prepared.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(prepared, timeout=timeout) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as error:
        return error.code, json.load(error)


def wait_for(url: str, process: subprocess.Popen | None = None, timeout: float = 120) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            status, _ = request(url, timeout=5)
            if status == 200:
                return
        except (OSError, ValueError):
            pass
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"exited with {process.returncode} while waiting for {url}")
        if time.monotonic() > deadline:
            raise RuntimeError(f"{url} did not answer within {timeout} seconds")
        time.sleep(0.2)


def build_middle_layer(directory: Path) -> Path:
    binary = directory / ("middle-layer.exe" if sys.platform == "win32" else "middle-layer")
    subprocess.run(["go", "build", "-o", str(binary), "."], cwd=MIDDLE_LAYER, check=True)
    return binary


def scratch_database(binary: Path) -> Path:
    return binary.parent / f"scratch-{uuid.uuid4().hex}.db"


class Backend:
    """The real FastAPI app around any engine, in a thread of this process."""

    def __init__(self, engine, registry: ToolRegistry) -> None:
        import uvicorn

        from backend.app import create_app

        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.server = uvicorn.Server(
            uvicorn.Config(
                create_app(engine, registry), host="127.0.0.1", port=self.port, log_level="warning"
            )
        )
        self.thread = threading.Thread(target=self.server.run, daemon=True)

    def __enter__(self) -> "Backend":
        self.thread.start()
        wait_for(f"{self.url}/health")
        return self

    def __exit__(self, *_) -> None:
        self.server.should_exit = True
        self.thread.join(30)


class MiddleLayer:
    """The real Go binary, with its real Python tool worker."""

    def __init__(
        self, binary: Path, backend_url: str, *, db: Path | None = None, metrics: Path | None = None
    ) -> None:
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.command = [
            str(binary),
            f"--addr=127.0.0.1:{self.port}",
            f"--backend={backend_url}",
            f"--repo={ROOT}",
            f"--python={sys.executable}",
            "--backend-wait=30s",
            # A file of its own unless told otherwise, or a comparison would write
            # into the real database.
            f"--db={db or scratch_database(binary)}",
            f"--metrics={metrics or 'none'}",
        ]
        if sys.platform != "win32":
            # A test run on CI, with nothing valuable on the runner's disk.
            self.command.append("--allow-unconfined")

    def __enter__(self) -> "MiddleLayer":
        self.process = subprocess.Popen(self.command)
        wait_for(f"{self.url}/health", self.process, timeout=240)
        return self

    def __exit__(self, *_) -> None:
        self.process.kill()
        self.process.wait(30)

    def play(self, policy: str, prompt: str) -> dict[str, Any]:
        status, session = request(f"{self.url}/sessions", {"policy": policy})
        assert status == 201, session
        status, episode = request(f"{self.url}/sessions/{session['id']}/messages", {"content": prompt})
        assert status == 200, episode
        return episode


@dataclass
class Record:
    """The parts of an episode the two loops have to agree on. Timings differ."""

    messages: list[dict[str, str]]
    nudges: int
    finish_reason: str
    answer: str | None
    trace: list[dict[str, Any]]


# The sandbox gives every run a directory with a random name, and a traceback
# prints it, so two runs of one failing program differ with no Go involved. Only
# that name is masked, so anything else that differs still fails the comparison.
_SCRATCH_DIRECTORY = re.compile(r"run-[0-9a-f]{32}|forge-sandbox-[0-9a-f]{8}-\w+")


def mask_scratch(text: str) -> str:
    return _SCRATCH_DIRECTORY.sub("SCRATCH", text)


def masked_messages(messages: list[dict[str, str]]) -> list[dict[str, str]]:
    return [{**message, "content": mask_scratch(message["content"])} for message in messages]


def _trace(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            key: mask_scratch(value) if key == "output" else value
            for key, value in entry.items()
            if key != "duration_ms"
        }
        for entry in entries
    ]


def record_of_evaluation(episode: Episode, prefix_length: int) -> Record:
    return Record(
        # The prefix is the backend's, so the middle layer never holds it.
        messages=masked_messages(episode.messages[prefix_length:]),
        nudges=episode.nudges,
        finish_reason=episode.finish_reason or "",
        answer=extract_answer(episode.response),
        trace=_trace(episode.session.trace_as_dicts()),
    )


def record_of_serving(episode: dict[str, Any]) -> Record:
    return Record(
        messages=masked_messages(episode["messages"]),
        nudges=episode["nudges"],
        finish_reason=episode["finish_reason"],
        answer=episode["answer"],
        trace=_trace(episode["trace"]),
    )


def play_evaluation(policy, tasks: list[Task], registry: ToolRegistry) -> list[Record]:
    episodes = make_episodes(tasks, registry)
    # Everything in front of the first task is the prefix, on every episode.
    prefix_length = len(episodes[0].messages) - 1 if episodes else 0
    run_episodes(episodes, policy)
    return [record_of_evaluation(episode, prefix_length) for episode in episodes]


class _ServedAsPolicy:
    """One engine policy behind the evaluation's `Policy` protocol."""

    def __init__(self, engine, name: str) -> None:
        self.engine = engine
        self.name = name

    def generate(self, conversations, *, stop, max_new_tokens):
        # The engine fixes `stop` and `max_new_tokens` at the evaluated values.
        return [self.engine.complete(self.name, conversation) for conversation in conversations]


def main(argv: list[str] | None = None) -> int:
    from backend.__main__ import parse_adapter
    from backend.engine import ServedPolicies
    from task_suite.registry import load_suite

    parser = argparse.ArgumentParser(prog="python -m eval.serving")
    parser.add_argument(
        "--adapter", type=parse_adapter, action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument("--per-category", type=int, default=4)
    arguments = parser.parse_args(argv)

    registry = build_registry(include_code=code_exec.sandbox_available())
    tasks = select_tasks(load_suite()["heldout"], per_category=arguments.per_category)
    engine = ServedPolicies(dict(arguments.adapter), temperature=0.0)

    mismatches = 0
    with tempfile.TemporaryDirectory() as directory:
        binary = build_middle_layer(Path(directory))
        with Backend(engine, registry) as backend, MiddleLayer(binary, backend.url) as middle:
            for name in engine.names():
                expected = play_evaluation(_ServedAsPolicy(engine, name), tasks, registry)
                calls = same = 0
                for task, reference in zip(tasks, expected):
                    served = record_of_serving(middle.play(name, task.prompt))
                    calls += len(served.trace)
                    if served == reference:
                        same += 1
                    else:
                        mismatches += 1
                        print(f"  {task.task_id}\n    served:    {served}\n    reference: {reference}")
                print(f"{name}: {same}/{len(tasks)} episodes identical, {calls} tool calls")
    return 0 if mismatches == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
