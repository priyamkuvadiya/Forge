import { API_BASE, ApiError } from "./client";
import type { AgentEvent } from "./types";

/**
 * Reads a text/event-stream body. The middle layer writes one `event:` line and
 * one `data:` line per event, and JSON has no raw newlines, so each `data:` line
 * is one whole event (see startStream in internal/api). Anything else is ignored.
 */
export async function* parseEventStream(body: ReadableStream<Uint8Array>): AsyncGenerator<AgentEvent> {
  const reader = body.getReader();
  // stream: true keeps a character split across two chunks intact.
  const decoder = new TextDecoder();
  let buffered = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffered += decoder.decode(value, { stream: true });
      let newline: number;
      while ((newline = buffered.indexOf("\n")) >= 0) {
        const line = buffered.slice(0, newline).replace(/\r$/, "");
        buffered = buffered.slice(newline + 1);
        if (line.startsWith("data: ")) {
          yield JSON.parse(line.slice("data: ".length)) as AgentEvent;
        }
      }
    }
  } finally {
    reader.releaseLock();
  }
}

export type StreamOutcome =
  /** The stream delivered episode_finished. */
  | "finished"
  /** Nothing is running (204 from /live): reload the session instead. */
  | "idle"
  /** The stream ended without a final event: the connection dropped, or we left. */
  | "dropped";

async function follow(
  response: Response,
  onEvent: (event: AgentEvent) => void,
): Promise<StreamOutcome> {
  if (response.status === 204) return "idle";
  if (!response.ok || !response.body) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { error?: string };
      if (body.error) message = body.error;
    } catch {
      // Not JSON.
    }
    throw new ApiError(response.status, message);
  }
  let finished = false;
  try {
    for await (const event of parseEventStream(response.body)) {
      onEvent(event);
      if (event.type === "episode_finished") finished = true;
    }
  } catch (error) {
    // An aborted or broken connection. The episode itself carries on in the
    // middle layer; the caller can reattach with followLive.
    if (finished) return "finished";
    if (error instanceof DOMException && error.name === "AbortError") return "dropped";
    if (error instanceof TypeError) return "dropped";
    throw error;
  }
  return finished ? "finished" : "dropped";
}

/** Sends a message and follows the episode it starts. */
export async function streamMessage(
  sessionId: string,
  content: string,
  onEvent: (event: AgentEvent) => void,
  signal?: AbortSignal,
): Promise<StreamOutcome> {
  const response = await fetch(`${API_BASE}/sessions/${encodeURIComponent(sessionId)}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
    body: JSON.stringify({ content }),
    signal: signal ?? null,
  });
  return follow(response, onEvent);
}

/**
 * Reattaches to a session's running episode, from its first event. This is
 * what makes a refresh mid-episode lose nothing: the episode never stopped.
 */
export async function followLive(
  sessionId: string,
  onEvent: (event: AgentEvent) => void,
  signal?: AbortSignal,
): Promise<StreamOutcome> {
  const response = await fetch(`${API_BASE}/sessions/${encodeURIComponent(sessionId)}/live`, {
    headers: { Accept: "text/event-stream" },
    signal: signal ?? null,
  });
  return follow(response, onEvent);
}

/** Asks the middle layer to stop the running episode; it then ends as cancelled. */
export async function cancelEpisode(sessionId: string): Promise<void> {
  const response = await fetch(`${API_BASE}/sessions/${encodeURIComponent(sessionId)}/cancel`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: "{}",
  });
  // 409: it already finished, which is what was wanted.
  if (!response.ok && response.status !== 409) {
    throw new ApiError(response.status, `${response.status} ${response.statusText}`);
  }
}
