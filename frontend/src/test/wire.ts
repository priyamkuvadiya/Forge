// The middle layer's real JSON, captured by tests/test_frontend_wire.py from the
// real Go binary, backend app and tool worker. That test fails if the capture no
// longer matches what the binary sends, so these types are checked from both sides.
import type { AgentEvent, Run, RunSummary, Session, SessionSummary } from "../api/types";
import raw from "./wire.json";

export interface Wire {
  /** Two episodes in one session, as streamed: the evaluated one, then a follow-up. */
  streamed: [AgentEvent[], AgentEvent[]];
  session: Session;
  /** A session while its episode is held mid-way. */
  running_session: Session;
  /** What /live replayed to a page that arrived mid-episode, before it left. */
  live_replay: AgentEvent[];
  cancel: { cancelling: string };
  cancelled_stream: AgentEvent[];
  cancelled_session: Session;
  sessions: SessionSummary[];
  runs: RunSummary[];
  run: Run;
}

export const wire = raw as unknown as Wire;

/** Serialises events the way the middle layer's startStream does. */
export function sse(events: AgentEvent[]): string {
  return events.map((e) => `event: ${e.type}\ndata: ${JSON.stringify(e)}\n\n`).join("");
}

/** A Response whose body arrives in the given pieces. */
export function streamResponse(pieces: (string | Uint8Array)[], init: ResponseInit = {}): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const piece of pieces) controller.enqueue(typeof piece === "string" ? encoder.encode(piece) : piece);
      controller.close();
    },
  });
  return new Response(body, { status: 200, headers: { "Content-Type": "text/event-stream" }, ...init });
}

export function json(value: unknown, status = 200): Response {
  return new Response(JSON.stringify(value), { status, headers: { "Content-Type": "application/json" } });
}
