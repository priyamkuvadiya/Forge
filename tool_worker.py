"""Runs tools for the Go middle layer, over its stdin and stdout.

The middle layer owns tool execution, but the sandbox is Python: an AppContainer
and a Job Object on Windows, hardened over several rounds of finding real bugs.
A Go rewrite would be a second sandbox to harden and a second implementation of
the budget and result format, free to drift from what the evaluated policies
were measured against. So Go supervises this process and it does the work with
the same `ToolSession` the evaluation used.

**Why a pipe rather than HTTP.** There is no port for anything else on the
machine to reach, and the process cannot outlive its parent: when the middle
layer dies, stdin closes and this exits.

**The protocol** is one JSON object per line. The first line out is a handshake
saying what this process serves. After that each request carries an `id`, and
its reply echoes it, since replies can arrive out of order:

    {"id": 1, "op": "open", "episode": "e1", "max_calls": 8}
    {"id": 2, "op": "invoke", "episode": "e1", "name": "calculator", "arguments": "2+2"}
    {"id": 3, "op": "finish", "episode": "e1", "response": "...every assistant turn..."}

A reply is `{"id": n, "ok": true, ...}` or `{"id": n, "ok": false, "error": "..."}`.
`ok: false` means the request could not be served. A tool that failed is an
ordinary result with `ok: true` around it, because that is the policy's business
and it gets to read the error.

**What crosses the boundary is already final.** `invoke` returns the exact
message to append to the conversation, rendered and truncated here, and `finish`
returns the answer as `extract_answer` reads it. Go passes these along without
interpreting them, so none of those rules exist twice.

**Budgets do not survive a restart, on purpose.** Sessions live in this process.
If it dies and is restarted, an in-flight episode gets "unknown episode" rather
than a silently refilled budget.

    python -m tool_worker
"""

import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, BinaryIO

from baseline_agent import MAX_TOOL_OUTPUT_CHARS, render_for_policy
from task_suite import extract_answer
from tools import DEFAULT_MAX_CALLS, ToolCall, ToolRegistry, ToolSession, build_registry, code_exec

# A single-user demo rarely has two episodes at a tool at once. This only stops a
# slow sandbox run in one session from holding up a calculator call in another.
DEFAULT_WORKERS = 4


class RequestError(Exception):
    """The request could not be served. Reported to Go, never to the policy."""


class Worker:
    def __init__(self, registry: ToolRegistry) -> None:
        self.registry = registry
        self._sessions: dict[str, ToolSession] = {}
        self._lock = threading.Lock()

    def _session(self, request: dict[str, Any]) -> ToolSession:
        with self._lock:
            session = self._sessions.get(request["episode"])
        if session is None:
            raise RequestError(f"unknown episode {request['episode']!r}")
        return session

    def open(self, request: dict[str, Any]) -> dict[str, Any]:
        session = ToolSession(self.registry, max_calls=request["max_calls"])
        with self._lock:
            if request["episode"] in self._sessions:
                raise RequestError(f"episode {request['episode']!r} is already open")
            self._sessions[request["episode"]] = session
        return {}

    def invoke(self, request: dict[str, Any]) -> dict[str, Any]:
        session = self._session(request)
        before = session.calls_made
        result = session.invoke(ToolCall(name=request["name"], arguments=request["arguments"]))
        return {
            "tool": result.tool,
            "tool_ok": result.ok,
            "output": result.output,
            "tool_error": result.error,
            "message": render_for_policy(result),
            # Null when the budget refused the call: it never ran, so as in the
            # evaluation it is not in the trace.
            "trace": session.trace[-1].to_dict() if session.calls_made > before else None,
        }

    def finish(self, request: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.pop(request["episode"], None)
        if session is None:
            raise RequestError(f"unknown episode {request['episode']!r}")
        return {
            "answer": extract_answer(request["response"]),
            "trace": session.trace_as_dicts(),
        }

    def handle(self, request: dict[str, Any]) -> dict[str, Any]:
        operations = {"open": self.open, "invoke": self.invoke, "finish": self.finish}
        try:
            # Inside the `try`: an `op` that is a list or an object is unhashable.
            operation = operations.get(request.get("op"))
            if operation is None:
                raise RequestError(f"unknown op {request.get('op')!r}")
            return {"id": request.get("id"), "ok": True, **operation(request)}
        except RequestError as error:
            return {"id": request.get("id"), "ok": False, "error": str(error)}
        except Exception as error:  # noqa: BLE001 - one bad request must not take down the rest
            return {
                "id": request.get("id"),
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }


def serve(
    registry: ToolRegistry,
    reader: BinaryIO,
    writer: BinaryIO,
    handshake: dict[str, Any],
    *,
    workers: int = DEFAULT_WORKERS,
) -> None:
    """Answer requests until `reader` closes, then let running calls finish."""
    worker = Worker(registry)
    write_lock = threading.Lock()

    def send(message: dict[str, Any]) -> None:
        # ensure_ascii (the default): the pipe stays pure ASCII whatever a tool
        # printed, so no console codepage can get between the two processes.
        line = json.dumps(message).encode("ascii") + b"\n"
        with write_lock:
            writer.write(line)
            writer.flush()

    send({"ready": True, **handshake})

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for line in reader:
            try:
                request = json.loads(line)
                if not isinstance(request, dict):
                    raise ValueError("a request must be a JSON object")
            except ValueError as error:
                send({"id": None, "ok": False, "error": f"malformed request: {error}"})
                continue

            # Only a tool call can be slow. `open` and `finish` run here, so an
            # `open` is complete before Go can send anything that depends on it.
            if request.get("op") == "invoke":
                pool.submit(lambda r=request: send(worker.handle(r)))
            else:
                send(worker.handle(request))


def claim_protocol_streams() -> tuple[BinaryIO, BinaryIO]:
    """Keep stdin and stdout for the protocol, and hand everyone else nothing.

    A stray `print` in any library would corrupt the protocol, and a child that
    inherited stdin could swallow a request. The protocol moves to private
    descriptors, fd 1 becomes stderr and fd 0 becomes the null device, so that
    neither can happen in this process or anything it starts.
    """
    reader = os.fdopen(os.dup(0), "rb")
    writer = os.fdopen(os.dup(1), "wb")
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0)
    os.close(null)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return reader, writer


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tool_worker", description=__doc__.split("\n")[0])
    parser.add_argument(
        "--allow-unconfined",
        action="store_true",
        help=(
            "serve the python tool on a sandbox that does not confine the filesystem "
            "(POSIX). Generated code can then read any file this user can."
        ),
    )
    arguments = parser.parse_args(argv)

    reader, writer = claim_protocol_streams()

    # The rule the backend uses, so this holds exactly the tools its prompt
    # advertises. The middle layer compares the two before serving anything.
    available = code_exec.sandbox_available()
    runner = code_exec.SandboxedCodeRunner() if available else None
    if runner is not None:
        # `verify` refuses to score on an unconfined sandbox. Here the stakes are
        # higher than a score: the code is running on a machine someone uses.
        if not runner.confines_filesystem and not arguments.allow_unconfined:
            print(
                f"the {code_exec.sandbox_backend()} sandbox does not confine the filesystem; "
                "refusing to run generated code without --allow-unconfined",
                file=sys.stderr,
            )
            return 1
        # The first real use of the sandbox should not be a user's first request.
        probe = runner("print('forge')")
        if probe.exit_code != 0:
            print(f"sandbox preflight failed: {probe.stderr[-400:]}", file=sys.stderr)
            return 1

    registry = build_registry(include_code=available, code_runner=runner)
    confinement = code_exec.confinement() if available else None
    serve(
        registry,
        reader,
        writer,
        {
            "tools": registry.names,
            "max_calls": DEFAULT_MAX_CALLS,
            "max_tool_output_chars": MAX_TOOL_OUTPUT_CHARS,
            "sandbox": {
                "backend": code_exec.sandbox_backend(),
                "confinement": confinement.summary() if confinement else None,
                "filesystem_confined": confinement.filesystem if confinement else None,
                "allow_unconfined": arguments.allow_unconfined,
            },
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
