"""The JSON the frontend is written against, captured from the real processes.

frontend/src/api/types.ts describes the middle layer's wire format by hand, and
the frontend's tests read this capture instead of trusting that description. A
scripted policy plays a few episodes through the real Go binary, the real
backend app and the real tool worker, so tool results are rendered exactly as a
real episode renders them. What comes back is normalised (IDs, timestamps and
timings vary) and compared with frontend/src/test/wire.json. If either side
changes the format, this test or the frontend's fails.

Regenerate after a deliberate change:

    FORGE_UPDATE_WIRE=1 python -m pytest tests/test_frontend_wire.py
"""

import json
import os
import re
import shutil
import threading
import urllib.request
from pathlib import Path

import pytest

from baseline_agent import prompt_prefix
from eval.serving import ROOT, Backend, MiddleLayer, build_middle_layer, request
from tools import DEFAULT_MAX_CALLS, build_registry

if os.environ.get("FORGE_REQUIRE_GO"):
    import uvicorn  # noqa: F401

    assert shutil.which("go"), "go is not installed and FORGE_REQUIRE_GO is set"
elif shutil.which("go") is None:
    pytest.skip("go is not installed", allow_module_level=True)
else:
    pytest.importorskip("uvicorn")

FIXTURE = ROOT / "frontend" / "src" / "test" / "wire.json"
POLICY = "baseline"


def call(name: str, arguments: str) -> str:
    return f'<tool name="{name}">{arguments}</tool>'


SCRIPTS = {
    # The evaluated setting: a call that works, one to a tool that does not
    # exist, a turn with no answer (nudged), then the answer.
    "What is 6 * 7?": [
        call("calculator", "6 * 7"),
        call("calc", "6 * 7"),
        "Let me think about it.",
        "<answer>42</answer>",
    ],
    # A follow-up that spends the budget and has one call refused.
    "Now look up the Ashen Bay Survey.": [call("search", "Ashen Bay Survey")] * (DEFAULT_MAX_CALLS + 1)
    + ["<answer>done</answer>"],
}
HOLD = "hold until cancelled"


class Engine:
    """Scripted by prompt; the HOLD prompt blocks until released."""

    def __init__(self, prefix_length: int) -> None:
        self.prefix_length = prefix_length
        self.entered = threading.Event()
        self.release = threading.Event()

    def names(self):
        return [POLICY]

    def describe(self):
        return {"policies": [{"name": POLICY, "adapter": None}]}

    def complete(self, policy, conversation):
        prompt = conversation[-1]["content"] if conversation[-1]["role"] == "user" else ""
        if prompt == HOLD:
            self.entered.set()
            self.release.wait(120)
            return "<answer>too late</answer>"
        # The episode is everything after the last user message that starts a script.
        for start in range(len(conversation) - 1, self.prefix_length - 1, -1):
            content = conversation[start]["content"]
            if conversation[start]["role"] == "user" and content in SCRIPTS:
                turn = sum(1 for m in conversation[start:] if m["role"] == "assistant")
                return SCRIPTS[content][turn]
        return "out of script"


def stream(url: str, method: str = "GET", body=None, stop_after=None):
    data = None if body is None else json.dumps(body).encode()
    prepared = urllib.request.Request(url, data=data, method=method, headers={"Accept": "text/event-stream"})
    if data is not None:
        prepared.add_header("Content-Type", "application/json")
    events = []
    with urllib.request.urlopen(prepared, timeout=120) as response:
        for raw in response:
            line = raw.decode("utf-8").rstrip("\r\n")
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: "):]))
                if stop_after and stop_after(events):
                    break
    return events


VOLATILE_NUMBERS = {"duration_ms", "latency_ms", "wall_clock_seconds"}


def normalise(value, ids):
    """IDs become stable placeholders, times and timings fixed values."""
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if key in VOLATILE_NUMBERS and isinstance(item, (int, float)):
                out[key] = 1.5
            elif key == "created_at":
                out[key] = "2026-01-01T00:00:00Z"
            else:
                out[key] = normalise(item, ids)
        return out
    if isinstance(value, list):
        return [normalise(item, ids) for item in value]
    if isinstance(value, str):
        def name(match):
            return ids.setdefault(match.group(0), f"{match.group(1)}-{len(ids) + 1}")

        # Ports are picked free per run, and an error message can name one.
        value = re.sub(r"127\.0\.0\.1:\d+", "127.0.0.1:PORT", value)
        return re.sub(r"\b([se])-[0-9a-f]{16}\b", name, value)
    return value


def run_log(directory: Path) -> Path:
    """An excerpt of run1's real log: config, two steps, one step with no update."""
    lines = (ROOT / "artifacts" / "rl" / "run1" / "steps.jsonl").read_text().splitlines()
    records = [json.loads(line) for line in lines if line.strip()]
    config = next(r for r in records if r["record"] == "config")
    steps = [r for r in records if r["record"] == "step"]
    idle = next(r for r in steps if r["kept_episodes"] == 0)
    chosen = [config, steps[0], steps[1], idle]
    (directory / "run1").mkdir(parents=True)
    (directory / "run1" / "steps.jsonl").write_text("".join(json.dumps(r) + "\n" for r in chosen))
    return directory


def capture(tmp_path) -> dict:
    registry = build_registry(include_code=True)
    engine = Engine(len(prompt_prefix(registry)))
    binary = build_middle_layer(tmp_path)
    metrics = run_log(tmp_path / "runs")
    with Backend(engine, registry) as backend, MiddleLayer(binary, backend.url, metrics=metrics) as middle:
        _, session = request(f"{middle.url}/sessions", {"policy": POLICY})
        sid = session["id"]
        streamed = [
            stream(f"{middle.url}/sessions/{sid}/messages", "POST", {"content": prompt}) for prompt in SCRIPTS
        ]
        _, stored = request(f"{middle.url}/sessions/{sid}")

        _, held = request(f"{middle.url}/sessions", {"policy": POLICY})
        hid = held["id"]
        started = []
        thread = threading.Thread(
            target=lambda: started.extend(
                stream(f"{middle.url}/sessions/{hid}/messages", "POST", {"content": HOLD})
            )
        )
        thread.start()
        assert engine.entered.wait(60)
        _, running = request(f"{middle.url}/sessions/{hid}")
        replay = stream(f"{middle.url}/sessions/{hid}/live", stop_after=lambda events: len(events) >= 1)
        _, cancelled = request(f"{middle.url}/sessions/{hid}/cancel", {})
        thread.join(60)
        engine.release.set()
        _, after = request(f"{middle.url}/sessions/{hid}")
        _, sessions = request(f"{middle.url}/sessions")
        _, runs = request(f"{middle.url}/runs")
        _, run = request(f"{middle.url}/runs/run1")

    return normalise(
        {
            "streamed": streamed,
            "session": stored,
            "running_session": running,
            "live_replay": replay,
            "cancel": cancelled,
            "cancelled_stream": started,
            "cancelled_session": after,
            "sessions": sessions["sessions"],
            "runs": runs["runs"],
            "run": run,
        },
        {},
    )


def test_the_frontend_fixture_is_what_the_middle_layer_sends(tmp_path):
    captured = capture(tmp_path)

    # The capture has to reach what it is for, or the frontend tests prove little.
    first, second = captured["streamed"]
    assert [e["type"] for e in first] == [
        "episode_started", "turn", "tool_result", "turn", "tool_result", "turn", "nudge", "turn", "episode_finished",
    ]
    finished = second[-1]["finished"]
    assert len(finished["trace"]) == DEFAULT_MAX_CALLS
    assert sum(1 for e in second if e["type"] == "tool_result" and e["ran"] is False) == 1
    assert captured["running_session"]["running"] is True
    assert captured["live_replay"][0]["prompt"] == HOLD
    assert captured["cancelled_session"]["episodes"][0]["finish_reason"] == "cancelled"
    assert any(r["kept_episodes"] == 0 for r in captured["run"]["records"])

    text = json.dumps(captured, indent=2, ensure_ascii=False) + "\n"
    if os.environ.get("FORGE_UPDATE_WIRE"):
        FIXTURE.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE.write_text(text, encoding="utf-8", newline="\n")
    assert FIXTURE.exists(), "no fixture; run with FORGE_UPDATE_WIRE=1"
    assert json.loads(FIXTURE.read_text(encoding="utf-8")) == captured, (
        "the middle layer's JSON changed; check the frontend, then regenerate with FORGE_UPDATE_WIRE=1"
    )
