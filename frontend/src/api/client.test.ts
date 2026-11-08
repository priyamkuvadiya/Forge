import { afterEach, describe, expect, it, vi } from "vitest";
import { json, wire } from "../test/wire";
import { ApiError, createSession, getHealth, getRun, getSession, listSessions } from "./client";

afterEach(() => vi.unstubAllGlobals());

describe("client", () => {
  it("reads the captured responses as the types say", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) => {
        if (url === "/api/sessions") return json({ sessions: wire.sessions });
        if (url.startsWith("/api/runs/")) return json(wire.run);
        return json(wire.session);
      }),
    );
    expect(await listSessions()).toEqual(wire.sessions);
    expect((await getSession("x")).episodes).toHaveLength(2);
    expect((await getRun("run1")).records.length).toBe(wire.run.steps);
  });

  it("creating a session posts JSON, since the middle layer refuses anything else", async () => {
    const fetch = vi.fn(async () => json(wire.session, 201));
    vi.stubGlobal("fetch", fetch);
    await createSession("run2");
    const [, init] = fetch.mock.calls[0] as unknown as [string, RequestInit];
    expect(init).toMatchObject({ method: "POST", headers: { "Content-Type": "application/json" } });
    expect(JSON.parse(init.body as string)).toEqual({ policy: "run2" });
  });

  it("turns an error body into the error's message", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json({ error: 'no policy named "run9"' }, 422)));
    await expect(createSession("run9")).rejects.toEqual(new ApiError(422, 'no policy named "run9"'));
  });

  it("falls back to the status line for a body that is not JSON", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => new Response("Bad Gateway", { status: 502, statusText: "Bad Gateway" })));
    await expect(listSessions()).rejects.toMatchObject({ status: 502, message: "502 Bad Gateway" });
  });

  it("a 503 from /health is a report that the backend is down, not an error", async () => {
    const report = { backend: "connection refused", tools: { worker_running: true, tools: [], sandbox: {} } };
    vi.stubGlobal("fetch", vi.fn(async () => json(report, 503)));
    expect((await getHealth()).backend).toBe("connection refused");
  });
});
