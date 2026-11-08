import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { afterEach, describe, expect, it, vi } from "vitest";
import type { Session } from "../api/types";
import { json, sse, streamResponse, wire } from "../test/wire";
import { ChatView } from "./ChatView";

// The middle layer, faked at fetch with its own captured responses.
type Route = (url: string, init?: RequestInit) => Response | Promise<Response>;

function serve(routes: Record<string, Route | Route[]>) {
  const calls: { url: string; init?: RequestInit }[] = [];
  const fetch = vi.fn(async (url: string, init?: RequestInit) => {
    calls.push({ url, init });
    const key = `${init?.method ?? "GET"} ${url}`;
    const route = routes[key];
    if (!route) throw new Error(`unexpected ${key}`);
    // A list answers in order, then repeats its last.
    const handler = Array.isArray(route) ? (route.length > 1 ? route.shift()! : route[0]!) : route;
    return handler(url, init);
  });
  vi.stubGlobal("fetch", fetch);
  return calls;
}

const policies = () =>
  json({ model: "m", sampling: {}, max_new_tokens: 384, policies: [{ name: "baseline", adapter: null }] });

const empty = (s: Session): Session => ({ ...s, episodes: [], running: false });

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe("opening a session with an episode running in it", () => {
  it("reattaches and shows what this page never sent, then the ending", async () => {
    const id = wire.running_session.id;
    // /live replays the started event and then the connection drops; the page
    // retries, finds nothing running, and reloads the finished session.
    const calls = serve({
      "GET /api/sessions": () => json({ sessions: wire.sessions }),
      "GET /api/policies": policies,
      [`GET /api/sessions/${id}`]: [() => json(wire.running_session), () => json(wire.cancelled_session)],
      [`GET /api/sessions/${id}/live`]: [
        () => streamResponse([sse(wire.live_replay)]),
        () => new Response(null, { status: 204 }),
      ],
    });
    render(<ChatView sessionId={id} onSelect={() => {}} />);

    expect(await screen.findByText("hold until cancelled")).toBeTruthy();
    expect(screen.getByText("running")).toBeTruthy();
    expect((screen.getByRole("textbox") as HTMLTextAreaElement).disabled).toBe(true);

    // After the retry: the record, as cancelled, and the composer free again.
    expect(await screen.findByText("cancelled", {}, { timeout: 3000 })).toBeTruthy();
    expect(screen.queryByText("running")).toBeNull();
    expect((screen.getByRole("textbox") as HTMLTextAreaElement).disabled).toBe(false);
    expect(calls.filter((c) => c.url.endsWith("/live"))).toHaveLength(2);
  });

  it("does not open /live for a session with nothing running", async () => {
    const id = wire.session.id;
    const calls = serve({
      "GET /api/sessions": () => json({ sessions: wire.sessions }),
      "GET /api/policies": policies,
      [`GET /api/sessions/${id}`]: () => json(wire.session),
    });
    render(<ChatView sessionId={id} onSelect={() => {}} />);
    expect(await screen.findByText("What is 6 * 7?")).toBeTruthy();
    expect(calls.some((c) => c.url.endsWith("/live"))).toBe(false);
  });
});

describe("sending a message", () => {
  // Sending refreshes the session list, which re-renders the parent. That once
  // re-ran the conversation's mount effect, whose cleanup aborted this stream.
  it("streams the episode, then shows the recorded one", async () => {
    const id = wire.session.id;
    const finished = { ...wire.session, episodes: wire.session.episodes.slice(0, 1) };
    let release!: () => void;
    const held = new Promise<void>((resolve) => (release = resolve));
    const calls = serve({
      "GET /api/sessions": () => json({ sessions: wire.sessions }),
      "GET /api/policies": policies,
      [`GET /api/sessions/${id}`]: [() => json(empty(wire.session)), () => json(finished)],
      [`POST /api/sessions/${id}/messages`]: async () => {
        // Everything but the end, then wait: the live view is on screen.
        const events = wire.streamed[0];
        const encoder = new TextEncoder();
        const body = new ReadableStream<Uint8Array>({
          async start(controller) {
            controller.enqueue(encoder.encode(sse(events.slice(0, -1))));
            await held;
            controller.enqueue(encoder.encode(sse(events.slice(-1))));
            controller.close();
          },
        });
        return new Response(body, { status: 200 });
      },
    });
    render(<ChatView sessionId={id} onSelect={() => {}} />);
    const box = (await screen.findByRole("textbox")) as HTMLTextAreaElement;
    fireEvent.change(box, { target: { value: "What is 6 * 7?" } });
    fireEvent.keyDown(box, { key: "Enter" });

    // Live: the tool cards and the nudge, before the episode has ended.
    await screen.findByText("calc");
    const names = () => [...document.querySelectorAll(".tool-name")].map((n) => n.textContent);
    expect(names()).toEqual(["calculator", "calc"]);
    expect(screen.getByText(/nudge/)).toBeTruthy();
    expect(screen.getByText("running")).toBeTruthy();
    expect(screen.getByText("evaluated setting")).toBeTruthy();

    await act(async () => release());
    expect(await screen.findByText("answered")).toBeTruthy();
    expect(screen.queryByText("running")).toBeNull();
    const post = calls.find((c) => c.init?.method === "POST")!;
    expect(JSON.parse(post.init!.body as string)).toEqual({ content: "What is 6 * 7?" });
  });

  it("labels a second message as outside what was evaluated", async () => {
    const id = wire.session.id;
    serve({
      "GET /api/sessions": () => json({ sessions: wire.sessions }),
      "GET /api/policies": policies,
      [`GET /api/sessions/${id}`]: () => json(wire.session),
    });
    render(<ChatView sessionId={id} onSelect={() => {}} />);
    await screen.findByText("What is 6 * 7?");
    const cards = document.querySelectorAll("article.episode");
    expect(within(cards[0] as HTMLElement).queryByText("evaluated setting")).toBeTruthy();
    expect(within(cards[1] as HTMLElement).queryByText(/follow-up: outside what was evaluated/)).toBeTruthy();
    // The refused ninth search is marked as never having run.
    expect(within(cards[1] as HTMLElement).getByText("not run: call budget spent")).toBeTruthy();
  });
});

describe("cancel", () => {
  it("asks the middle layer to stop the episode", async () => {
    const id = wire.running_session.id;
    let finish!: () => void;
    const ended = new Promise<void>((resolve) => (finish = resolve));
    const calls = serve({
      "GET /api/sessions": () => json({ sessions: wire.sessions }),
      "GET /api/policies": policies,
      [`GET /api/sessions/${id}`]: [() => json(wire.running_session), () => json(wire.cancelled_session)],
      [`GET /api/sessions/${id}/live`]: () => {
        const encoder = new TextEncoder();
        const body = new ReadableStream<Uint8Array>({
          async start(controller) {
            controller.enqueue(encoder.encode(sse(wire.live_replay)));
            await ended;
            controller.enqueue(encoder.encode(sse(wire.cancelled_stream.slice(-1))));
            controller.close();
          },
        });
        return new Response(body, { status: 200 });
      },
      [`POST /api/sessions/${id}/cancel`]: () => {
        finish();
        return json(wire.cancel, 202);
      },
    });
    render(<ChatView sessionId={id} onSelect={() => {}} />);
    fireEvent.click(await screen.findByRole("button", { name: "cancel" }));
    await waitFor(() => expect(calls.some((c) => c.url.endsWith("/cancel"))).toBe(true));
    expect(await screen.findByText("cancelled")).toBeTruthy();
    expect(screen.getByText(/Stopped before it finished/)).toBeTruthy();
  });
});
