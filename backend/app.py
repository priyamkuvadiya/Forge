"""The HTTP surface: conversation in, the policy's next turn out.

The backend's whole job is to run a policy. It does not execute tools, keep
sessions or decide when an episode is over - those belong to the Go middle
layer (module 7). What it does own is everything that would quietly change the
policy being served if a caller got it wrong:

- **The prompt prefix.** The system prompt and the worked examples are
  prepended here, from the same `make_episodes` the evaluation used, so a
  caller sends only the task and what followed it. Both arms were measured
  under that exact prefix; a middle layer rebuilding it would be serving a
  different prompt from the one the numbers describe.
- **Reading the completion.** `baseline_agent.read_completion` decides whether
  a turn is a tool call, an answer or neither, and cuts a tool call at its
  closing tag, exactly as during evaluation. The caller gets the cut text and
  the parsed call; it never re-parses the raw generation.
- **Generation settings.** Temperature, top-p, the stop strings and the
  384-token cap are fixed at the evaluated values and are not per-request.

What a caller still has to match by hand - the nudge text, the call budget,
the tool-output cap - is published at `GET /contract` rather than duplicated.
"""

import time
from typing import Any, Literal, Protocol

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from baseline_agent import (
    DEFAULT_MAX_NUDGES,
    MAX_TOOL_OUTPUT_CHARS,
    NUDGE_MESSAGE,
    STOP_STRINGS,
    Message,
    prompt_prefix,
    read_completion,
)
from tools import DEFAULT_MAX_CALLS, ToolRegistry

BASELINE = "baseline"


class Engine(Protocol):
    """What the app needs from a model server, so tests can run without one."""

    def describe(self) -> dict[str, Any]: ...

    def names(self) -> list[str]: ...

    def complete(self, policy: str, conversation: list[Message]) -> str: ...


class ChatMessage(BaseModel):
    # No "system": the system prompt is the backend's, not the caller's.
    role: Literal["user", "assistant"]
    content: str


class TurnRequest(BaseModel):
    policy: str
    messages: list[ChatMessage] = Field(min_length=1)


class ToolCallOut(BaseModel):
    name: str
    arguments: str


class TurnResponse(BaseModel):
    policy: str
    kind: Literal["empty", "tool_call", "answer", "no_answer"]
    text: str
    tool_call: ToolCallOut | None
    answer: str | None
    raw: str
    latency_ms: float


def create_app(engine: Engine, registry: ToolRegistry) -> FastAPI:
    app = FastAPI(title="Forge backend")
    prefix = prompt_prefix(registry)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/policies")
    def policies() -> dict[str, Any]:
        return engine.describe()

    @app.get("/contract")
    def contract() -> dict[str, Any]:
        return {
            "tools": registry.names,
            "max_calls": DEFAULT_MAX_CALLS,
            "max_nudges": DEFAULT_MAX_NUDGES,
            "nudge_message": NUDGE_MESSAGE,
            "max_tool_output_chars": MAX_TOOL_OUTPUT_CHARS,
            "stop": list(STOP_STRINGS),
        }

    # A plain `def`, not `async def`: FastAPI runs it on a worker thread, so a
    # generation that takes seconds does not block /health. The engine
    # serialises the GPU itself.
    @app.post("/turn", response_model=TurnResponse)
    def turn(request: TurnRequest) -> TurnResponse:
        if request.policy not in engine.names():
            raise HTTPException(status_code=404, detail=f"no policy named {request.policy!r}")
        if request.messages[0].role != "user":
            raise HTTPException(status_code=422, detail="the conversation must open with the user")
        if request.messages[-1].role != "user":
            # The policy only ever generated in reply to a user turn (a task,
            # a tool result or a nudge); asking it to continue its own turn is
            # a state it was never trained or evaluated in.
            raise HTTPException(status_code=422, detail="the last message must be from the user")

        conversation = [dict(message) for message in prefix] + [
            {"role": m.role, "content": m.content} for m in request.messages
        ]
        started = time.perf_counter()
        raw = engine.complete(request.policy, conversation)
        latency_ms = (time.perf_counter() - started) * 1000

        completion = read_completion(raw)
        call = completion.call
        return TurnResponse(
            policy=request.policy,
            kind=completion.kind,
            text=completion.visible,
            tool_call=ToolCallOut(name=call.name, arguments=call.arguments) if call else None,
            answer=completion.answer,
            raw=raw,
            latency_ms=latency_ms,
        )

    return app
