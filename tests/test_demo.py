"""The one-command launcher, demo.py.

The rules that decide what to rebuild are checked directly. The launcher itself
is run for real against the real Go binary and tool worker, with a scripted
backend in place of the model: it has to serve the page and the API on one
port, and on Ctrl+C stop the middle layer politely, so that an episode still
running is recorded as cancelled rather than read back later as interrupted.
"""

import json
import os
import shutil
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import pytest

import demo
from eval.serving import Backend, build_middle_layer, free_port, request

GO = shutil.which("go") is not None


def touch(path: Path, when: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text("x")
    os.utime(path, (when, when))
    return path


class TestWhatIsRebuilt:
    def test_a_missing_build_is_stale(self, tmp_path):
        assert demo.stale(tmp_path / "dist" / "index.html", [])

    def test_a_newer_source_makes_it_stale_and_an_older_one_does_not(self, tmp_path):
        built = touch(tmp_path / "dist" / "index.html", 1000)
        source = touch(tmp_path / "src" / "App.tsx", 900)
        assert not demo.stale(built, [source])
        touch(source, 1100)
        assert demo.stale(built, [source])

    def test_tests_and_their_fixtures_are_not_page_sources(self, tmp_path):
        frontend = tmp_path / "frontend"
        touch(frontend / "src" / "App.tsx", 1)
        touch(frontend / "src" / "chat" / "steps.test.ts", 1)
        touch(frontend / "src" / "test" / "wire.json", 1)
        touch(frontend / "package.json", 1)
        names = {p.relative_to(frontend).as_posix() for p in demo.page_sources(frontend)}
        assert names == {"src/App.tsx", "package.json"}

    def test_go_tests_are_not_binary_sources(self, tmp_path):
        layer = tmp_path / "middle-layer"
        for name in ("main.go", "main_test.go", "internal/api/api.go", "go.mod", "go.sum"):
            touch(layer / name, 1)
        names = {p.relative_to(layer).as_posix() for p in demo.go_sources(layer)}
        assert names == {"main.go", "internal/api/api.go", "go.mod", "go.sum"}

    def test_the_real_tree_has_sources_for_both(self):
        assert any(p.name == "App.tsx" for p in demo.page_sources(demo.FRONTEND))
        assert any(p.name == "main.go" for p in demo.go_sources(demo.MIDDLE_LAYER))


class TestCommands:
    def test_the_backend_gets_every_adapter(self):
        command = demo.backend_command({"run1": Path("a"), "run2": Path("b")}, 8001)
        assert command[1:] == ["-u", "-m", "backend", "--port", "8001", "--adapter", "run1=a", "--adapter", "run2=b"]

    def test_the_middle_layer_serves_the_page_on_loopback(self):
        command = demo.middle_layer_command("http://127.0.0.1:8000", 9090, None)
        assert "--addr=127.0.0.1:9090" in command
        assert f"--frontend={demo.FRONTEND / 'dist'}" in command
        assert not any(arg.startswith("--db") for arg in command)

    def test_the_defaults_name_run1_and_run2(self):
        assert set(demo.DEFAULT_ADAPTERS) == {"run1", "run2"}

    def test_missing_adapters_are_refused_before_anything_starts(self, capsys):
        assert demo.main(["--adapter", "run9=does/not/exist", "--no-build", "--no-browser"]) == 1
        assert f"run9={Path('does/not/exist')}" in capsys.readouterr().out


class Holding:
    """A backend engine whose "hold" message blocks until released."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def names(self):
        return ["baseline"]

    def describe(self):
        return {"policies": [{"name": "baseline", "adapter": None}]}

    def complete(self, policy, conversation):
        if conversation[-1]["content"] == "hold":
            self.entered.set()
            self.release.wait(120)
        return "<answer>4</answer>"


def interrupt(process: subprocess.Popen) -> None:
    """What Ctrl+C in the launcher's terminal does."""
    if sys.platform == "win32":
        process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        process.send_signal(signal.SIGINT)


@pytest.mark.skipif(not GO and not os.environ.get("FORGE_REQUIRE_GO"), reason="go is not installed")
def test_the_launcher_serves_the_page_and_stops_cleanly(tmp_path):
    from tools import build_registry

    binary = build_middle_layer(tmp_path)
    page = tmp_path / "dist"
    (page / "assets").mkdir(parents=True)
    (page / "index.html").write_text('<!doctype html><script type="module" src="/assets/app.js"></script>')
    (page / "assets" / "app.js").write_text("// the page")
    db = tmp_path / "forge.db"
    ready = tmp_path / "ready.json"
    port = free_port()

    engine = Holding()
    extra = ["--extra=--metrics=none"]
    if sys.platform != "win32":
        extra.append("--extra=--allow-unconfined")  # A CI runner, nothing on its disk.
    with Backend(engine, build_registry(include_code=True)) as backend:
        command = [
            sys.executable, "-m", "demo", "--backend", backend.url, "--port", str(port), "--db", str(db),
            "--no-build", "--no-browser", "--binary", str(binary), "--page", str(page),
            "--ready-file", str(ready), *extra,
        ]
        flags = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if sys.platform == "win32" else {}
        launcher = subprocess.Popen(command, cwd=demo.ROOT, **flags)
        try:
            deadline = time.monotonic() + 240
            while not ready.exists():
                assert launcher.poll() is None, "the launcher exited before it was ready"
                assert time.monotonic() < deadline, "the launcher was never ready"
                time.sleep(0.5)
            url = json.loads(ready.read_text())["url"]

            # One port: the page at /, the API under /api/.
            with urllib.request.urlopen(f"{url}/", timeout=10) as response:
                assert "assets/app.js" in response.read().decode()
                assert "default-src 'self'" in response.headers["Content-Security-Policy"]
            status, health = request(f"{url}/api/health")
            assert status == 200 and health["backend"] == "ok"

            # An episode left running when the demo is stopped.
            _, session = request(f"{url}/api/sessions", {"policy": "baseline"})
            sent = threading.Thread(
                target=lambda: request(f"{url}/api/sessions/{session['id']}/messages", {"content": "hold"}, timeout=60),
                daemon=True,
            )
            sent.start()
            assert engine.entered.wait(60)

            interrupt(launcher)
            assert launcher.wait(90) == 0
        finally:
            if launcher.poll() is None:
                launcher.kill()
                launcher.wait(30)
            engine.release.set()

    # The lock is released and the episode's ending was recorded before exit.
    with sqlite3.connect(db) as connection:
        rows = connection.execute("SELECT finish_reason FROM episodes").fetchall()
    assert rows == [("cancelled",)]
