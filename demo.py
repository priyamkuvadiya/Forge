"""Start the whole demo with one command.

    python -m demo

Builds what is missing or stale (the page, the middle layer), starts the policy
backend on the GPU and the middle layer, which serves the page, and prints the
address. Ctrl+C stops both: the middle layer first, so an episode still running
is cancelled and recorded as such rather than read back later as interrupted.

By default the backend serves the baseline plus run1 and run2, and the page
calls run2 "trained". `--adapter` replaces that list, `--backend` uses a backend
that is already running instead of starting one, and `--no-build` uses what is
already built.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
MIDDLE_LAYER = ROOT / "middle-layer"
WINDOWS = sys.platform == "win32"
BINARY = MIDDLE_LAYER / ("middle-layer.exe" if WINDOWS else "middle-layer")

DEFAULT_ADAPTERS = {
    "run1": ROOT / "artifacts" / "rl" / "run1" / "adapter-final",
    "run2": ROOT / "artifacts" / "rl" / "run2" / "adapter-final",
}


def log(message: str) -> None:
    print(f"[demo] {message}", flush=True)


def page_sources(frontend: Path) -> list[Path]:
    """What the built page is made from. Tests and their fixtures are not."""
    sources = [
        f for f in (frontend / "src").rglob("*")
        if f.is_file() and ".test." not in f.name and "test" not in f.relative_to(frontend / "src").parts
    ]
    for name in ("index.html", "package.json", "package-lock.json", "vite.config.ts", "tsconfig.json"):
        if (frontend / name).exists():
            sources.append(frontend / name)
    return sources


def stale(built: Path, sources: list[Path]) -> bool:
    if not built.exists():
        return True
    return any(source.stat().st_mtime > built.stat().st_mtime for source in sources)


def go_sources(middle_layer: Path) -> list[Path]:
    sources = [f for f in middle_layer.rglob("*.go") if not f.name.endswith("_test.go")]
    return sources + [middle_layer / "go.mod", middle_layer / "go.sum"]


def run(command: list[str], cwd: Path) -> None:
    log(" ".join(command))
    subprocess.run(command, cwd=cwd, check=True, shell=False)


def build(force: bool) -> None:
    if force or stale(FRONTEND / "dist" / "index.html", page_sources(FRONTEND)):
        npm = "npm.cmd" if WINDOWS else "npm"
        if not (FRONTEND / "node_modules").exists():
            run([npm, "ci"], FRONTEND)
        run([npm, "run", "build"], FRONTEND)
    if force or stale(BINARY, go_sources(MIDDLE_LAYER)):
        run(["go", "build", "-o", str(BINARY), "."], MIDDLE_LAYER)


def parse_adapter(text: str) -> tuple[str, Path]:
    name, sep, path = text.partition("=")
    if not sep or not name or not path:
        raise argparse.ArgumentTypeError(f"expected NAME=PATH, got {text!r}")
    return name, Path(path)


def backend_command(adapters: dict[str, Path], port: int) -> list[str]:
    command = [sys.executable, "-u", "-m", "backend", "--port", str(port)]
    for name, path in adapters.items():
        command += ["--adapter", f"{name}={path}"]
    return command


def middle_layer_command(
    backend_url: str, port: int, db: Path | None, binary: Path = BINARY, page: Path = FRONTEND / "dist"
) -> list[str]:
    command = [
        str(binary),
        f"--addr=127.0.0.1:{port}",
        f"--backend={backend_url}",
        f"--repo={ROOT}",
        f"--python={sys.executable}",
        f"--frontend={page}",
    ]
    if db is not None:
        command.append(f"--db={db}")
    return command


def start(command: list[str]) -> subprocess.Popen:
    # Each child in its own process group, so a Ctrl+C in this terminal reaches
    # only this script, which then stops them in order.
    if WINDOWS:
        return subprocess.Popen(command, cwd=ROOT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    return subprocess.Popen(command, cwd=ROOT, start_new_session=True)


def interrupt(process: subprocess.Popen) -> None:
    """The polite stop: what Ctrl+C would send it. Go reads both as os.Interrupt."""
    if process.poll() is not None:
        return
    try:
        if WINDOWS:
            process.send_signal(signal.CTRL_BREAK_EVENT)
        else:
            process.send_signal(signal.SIGINT)
    except OSError:
        pass


def stop(process: subprocess.Popen | None, name: str, grace: float) -> None:
    if process is None or process.poll() is not None:
        return
    interrupt(process)
    try:
        process.wait(grace)
        log(f"{name} stopped")
    except subprocess.TimeoutExpired:
        log(f"{name} did not stop in {grace:.0f}s; killing it")
        process.kill()
        process.wait(10)


def healthy(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def wait_until_serving(url: str, processes: dict[str, subprocess.Popen], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while not healthy(url):
        for name, process in processes.items():
            if process.poll() is not None:
                raise RuntimeError(f"{name} exited with {process.returncode} before the demo was up")
        if time.monotonic() > deadline:
            raise RuntimeError(f"nothing answered at {url} within {timeout:.0f}s")
        time.sleep(1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m demo", description=__doc__.split("\n")[0])
    parser.add_argument("--adapter", type=parse_adapter, action="append", metavar="NAME=PATH",
                        help="an adapter to serve (default: run1 and run2's final adapters)")
    parser.add_argument("--backend", metavar="URL", help="use this running backend instead of starting one")
    parser.add_argument("--backend-port", type=int, default=8000)
    parser.add_argument("--port", type=int, default=8080, help="where the page and API are served")
    parser.add_argument("--db", type=Path, help="the SQLite file (default: middle-layer/forge.db)")
    parser.add_argument("--no-build", action="store_true", help="do not build the page or the middle layer")
    parser.add_argument("--rebuild", action="store_true", help="build both even if they look current")
    parser.add_argument("--no-browser", action="store_true", help="do not open the page")
    # For the tests: run a binary and page built elsewhere, and say when ready.
    parser.add_argument("--binary", type=Path, default=BINARY, help=argparse.SUPPRESS)
    parser.add_argument("--page", type=Path, default=FRONTEND / "dist", help=argparse.SUPPRESS)
    parser.add_argument("--ready-file", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--extra", action="append", default=[], help=argparse.SUPPRESS)
    arguments = parser.parse_args(argv)

    adapters = dict(arguments.adapter) if arguments.adapter else dict(DEFAULT_ADAPTERS)
    if arguments.backend is None:
        missing = [f"{name}={path}" for name, path in adapters.items() if not path.exists()]
        if missing:
            log("adapter weights not found (they are not in git; train them, or pass --adapter): " + ", ".join(missing))
            return 1

    if not arguments.no_build:
        try:
            build(arguments.rebuild)
        except (OSError, subprocess.CalledProcessError) as error:
            log(f"build failed: {error}")
            return 1
    if not arguments.binary.exists() or not (arguments.page / "index.html").exists():
        log("nothing built to run; drop --no-build")
        return 1

    # Ctrl+Break on Windows ends Python outright by default; make it a Ctrl+C.
    if WINDOWS:
        signal.signal(signal.SIGBREAK, signal.default_int_handler)

    processes: dict[str, subprocess.Popen] = {}
    url = f"http://127.0.0.1:{arguments.port}"
    try:
        backend_url = arguments.backend
        if backend_url is None:
            backend_url = f"http://127.0.0.1:{arguments.backend_port}"
            log(f"starting the backend (loads the model on the GPU): {', '.join(['baseline', *adapters])}")
            processes["backend"] = start(backend_command(adapters, arguments.backend_port))
        processes["middle layer"] = start(
            middle_layer_command(backend_url, arguments.port, arguments.db, arguments.binary, arguments.page)
            + arguments.extra
        )
        # The middle layer waits for the backend itself, so it is the one to watch.
        wait_until_serving(f"{url}/api/health", processes, timeout=600)
        log(f"ready: {url}")
        log("Ctrl+C to stop")
        if arguments.ready_file:
            arguments.ready_file.write_text(json.dumps({"url": url, "pid": os.getpid()}))
        if not arguments.no_browser:
            webbrowser.open(url)
        while True:
            for name, process in processes.items():
                if process.poll() is not None:
                    log(f"{name} exited with {process.returncode}")
                    return 1
            time.sleep(1)
    except KeyboardInterrupt:
        log("stopping")
        return 0
    except RuntimeError as error:
        log(str(error))
        return 1
    finally:
        # The middle layer first: it cancels what is running and records it,
        # which needs the backend only to have stopped answering, not to be gone.
        stop(processes.get("middle layer"), "middle layer", grace=30)
        stop(processes.get("backend"), "backend", grace=15)


if __name__ == "__main__":
    sys.exit(main())
