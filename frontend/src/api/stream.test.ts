import { afterEach, describe, expect, it, vi } from "vitest";
import { json, sse, streamResponse, wire } from "../test/wire";
import { ApiError } from "./client";
import { cancelEpisode, followLive, parseEventStream, streamMessage } from "./stream";
import type { AgentEvent } from "./types";

async function collect(response: Response): Promise<AgentEvent[]> {
  const events: AgentEvent[] = [];
  for await (const event of parseEventStream(response.body!)) events.push(event);
  return events;
}

afterEach(() => vi.unstubAllGlobals());

describe("parseEventStream", () => {
  const events = wire.streamed[1];
  const text = sse(events);

  it("reads every event whole, in order", async () => {
    expect(await collect(streamResponse([text]))).toEqual(events);
  });

  it("survives any split into chunks, including inside a multibyte character", async () => {
    const bytes = new TextEncoder().encode(sse(wire.streamed[0]) + sse([
      { type: "turn", episode: "e", turn: 0, kind: "answer", text: "✓ naïve — 42" },
    ]));
    for (const size of [1, 2, 3, 7, 64]) {
      const pieces: Uint8Array[] = [];
      for (let i = 0; i < bytes.length; i += size) pieces.push(bytes.slice(i, i + size));
      const got = await collect(streamResponse(pieces));
      expect(got.at(-1)).toMatchObject({ text: "✓ naïve — 42" });
      expect(got).toHaveLength(wire.streamed[0].length + 1);
    }
  });

  it("accepts CRLF line endings and ignores lines that are not data", async () => {
    const crlf = `: comment\r\nevent: x\r\ndata: ${JSON.stringify(events[0])}\r\n\r\n`;
    expect(await collect(streamResponse([crlf]))).toEqual([events[0]]);
  });
});

describe("following an episode", () => {
  it("finished when the final event arrives", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => streamResponse([sse(wire.streamed[0])])));
    const seen: AgentEvent[] = [];
    expect(await streamMessage("s", "hi", (e) => seen.push(e))).toBe("finished");
    expect(seen).toEqual(wire.streamed[0]);
  });

  it("dropped when the stream ends without one", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => streamResponse([sse(wire.live_replay)])));
    expect(await followLive("s", () => {})).toBe("dropped");
  });

  it("idle when /live says nothing is running", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response(null, { status: 204 })));
    expect(await followLive("s", () => {})).toBe("idle");
  });

  it("raises the server's own message for a refusal", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json({ error: "this session is already running an episode" }, 409)));
    const failure = streamMessage("s", "hi", () => {});
    await expect(failure).rejects.toBeInstanceOf(ApiError);
    await expect(failure).rejects.toMatchObject({ status: 409, message: "this session is already running an episode" });
  });

  it("sends JSON with the stream header, which the middle layer requires", async () => {
    const fetch = vi.fn(async () => streamResponse([sse(wire.streamed[0])]));
    vi.stubGlobal("fetch", fetch);
    await streamMessage("s/1", "hi", () => {});
    const [url, init] = fetch.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/sessions/s%2F1/messages");
    expect(init.headers).toMatchObject({ "Content-Type": "application/json", Accept: "text/event-stream" });
    expect(JSON.parse(init.body as string)).toEqual({ content: "hi" });
  });
});

describe("cancel", () => {
  it("posts JSON, and treats 'nothing running' as done", async () => {
    const fetch = vi.fn(async () => json(wire.cancel, 202));
    vi.stubGlobal("fetch", fetch);
    await cancelEpisode("s");
    const [url, init] = fetch.mock.calls[0] as unknown as [string, RequestInit];
    expect(url).toBe("/api/sessions/s/cancel");
    expect(init).toMatchObject({ method: "POST", body: "{}", headers: { "Content-Type": "application/json" } });

    vi.stubGlobal("fetch", vi.fn(async () => json({ error: "nothing is running in this session" }, 409)));
    await expect(cancelEpisode("s")).resolves.toBeUndefined();
    vi.stubGlobal("fetch", vi.fn(async () => json({ error: "no such session" }, 404)));
    await expect(cancelEpisode("s")).rejects.toMatchObject({ status: 404 });
  });
});
