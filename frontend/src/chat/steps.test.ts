import { describe, expect, it } from "vitest";
import type { AgentEvent, Episode } from "../api/types";
import { wire } from "../test/wire";
import { applyEvent, emptyView, fromEpisode, settingAt, type EpisodeView } from "./steps";

function fold(events: AgentEvent[]): EpisodeView {
  return events.reduce(applyEvent, emptyView(""));
}

/** Everything but the last event, which replaces the live view with the record. */
function live(events: AgentEvent[]): EpisodeView {
  return fold(events.slice(0, -1));
}

const [first, second] = wire.streamed;
const finishedOf = (events: AgentEvent[]) => {
  const last = events.at(-1)!;
  if (last.type !== "episode_finished") throw new Error("no finished event");
  return last.finished;
};

describe("one view for live events and recorded episodes", () => {
  it.each([
    ["evaluated", first],
    ["follow-up", second],
  ] as const)("the %s episode: live steps match the recorded ones", (_, events) => {
    const recorded = fromEpisode(finishedOf(events));
    const streaming = live(events);
    expect(streaming.prompt).toBe(recorded.prompt);
    expect(streaming.steps.map((s) => s.kind)).toEqual(recorded.steps.map((s) => s.kind));
    // What the policy was shown and whether each call ran agree exactly.
    const tools = (v: EpisodeView) =>
      v.steps.flatMap((s) => (s.kind === "tool" ? [{ tool: s.tool, shown: s.shown, ran: s.ran, trace: s.trace }] : []));
    expect(tools(streaming)).toEqual(tools(recorded));
    expect(streaming.nudges).toBe(recorded.nudges);
  });

  it("the recorded episode in the session is the one the stream finished with", () => {
    expect(wire.session.episodes).toEqual([finishedOf(first), finishedOf(second)]);
    expect(fold(first)).toEqual(fromEpisode(wire.session.episodes[0]!));
  });

  it("an empty turn adds nothing, as it adds nothing to the conversation in agent.Run", () => {
    const started = wire.streamed[0][0]!;
    const view = fold([started, { type: "turn", episode: started.episode, turn: 0, kind: "empty" }]);
    expect(view.steps).toEqual([]);
  });

  it("a live view has no ending until the final event", () => {
    expect(live(first).finish).toBeNull();
    expect(fold(first).finish).toBe("answer");
    expect(fold(first).answer).toBe("42");
  });
});

describe("reading a recorded episode", () => {
  const evaluated = fromEpisode(wire.session.episodes[0]!);
  const followUp = fromEpisode(wire.session.episodes[1]!);

  it("separates a working call, a call to a tool that does not exist, and a nudge", () => {
    expect(evaluated.steps.map((s) => s.kind)).toEqual([
      "assistant", "tool", "assistant", "tool", "assistant", "nudge", "assistant",
    ]);
    const [calculator, invented] = evaluated.steps.filter((s) => s.kind === "tool");
    expect(calculator).toMatchObject({ tool: "calculator", ran: true, trace: { ok: true, arguments: "6 * 7" } });
    // It ran (and counted against the budget); it just failed.
    expect(invented).toMatchObject({ tool: "calc", ran: true, trace: { ok: false } });
    expect(evaluated.steps[0]).toMatchObject({ toolCall: { name: "calculator", arguments: "6 * 7" } });
  });

  it("pairs results with trace entries, and marks the call past the budget as never run", () => {
    const tools = followUp.steps.filter((s) => s.kind === "tool");
    expect(tools).toHaveLength(9);
    expect(tools.slice(0, 8).every((s) => s.ran && s.trace !== null)).toBe(true);
    expect(tools.map((s) => s.trace?.index ?? null)).toEqual([0, 1, 2, 3, 4, 5, 6, 7, null]);
    expect(tools[8]).toMatchObject({ ran: false, trace: null });
    expect(tools[8]!.shown).toContain("budget exhausted");
  });

  it("does not take a refused call's place in the trace when the record is missing an entry", () => {
    // A trace one short, as from a worker that died: the last result that ran
    // reads as refused rather than borrowing a neighbour's entry.
    const episode: Episode = { ...wire.session.episodes[1]!, trace: wire.session.episodes[1]!.trace.slice(0, 7) };
    const tools = fromEpisode(episode).steps.filter((s) => s.kind === "tool");
    expect(tools.map((s) => s.ran)).toEqual([true, true, true, true, true, true, true, false, false]);
  });
});

describe("reattaching mid-episode", () => {
  it("learns the prompt from the started event", () => {
    const view = fold(wire.live_replay);
    expect(view.prompt).toBe("hold until cancelled");
    expect(view.finish).toBeNull();
  });

  it("a cancelled episode ends as a system finish with its error", () => {
    const view = fold(wire.cancelled_stream);
    expect(view.finish).toBe("cancelled");
    expect(view.error).toContain("context canceled");
  });
});

describe("evaluated versus follow-up", () => {
  const answer = { finish_reason: "answer" as const };
  const failed = { finish_reason: "backend_error" as const };
  const cancelled = { finish_reason: "cancelled" as const };

  it("only an episode with no earlier conversation is evaluated", () => {
    expect(settingAt([], 0)).toBe("evaluated");
    expect(settingAt([answer], 1)).toBe("follow-up");
    expect(settingAt([answer, answer], 0)).toBe("evaluated");
  });

  it("system-failed episodes are not history, so they do not count", () => {
    expect(settingAt([failed, cancelled], 2)).toBe("evaluated");
    expect(settingAt([failed, answer], 2)).toBe("follow-up");
  });

  it("matches the captured session", () => {
    const episodes = wire.session.episodes;
    expect([0, 1].map((i) => settingAt(episodes, i))).toEqual(["evaluated", "follow-up"]);
  });
});
