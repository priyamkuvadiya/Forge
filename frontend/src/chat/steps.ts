import { systemFailed, type AgentEvent, type Episode, type FinishReason, type ToolCall, type TraceEntry } from "../api/types";

// One view of an episode, whether it is arriving live as events or was loaded
// finished from the session. Both describe the same appends agent.Run makes.

export type Step =
  | { kind: "assistant"; text: string; toolCall: ToolCall | null; latencyMs: number | null }
  | {
      kind: "tool";
      tool: string;
      /** Exactly what the policy was shown, rendered by the tool worker. */
      shown: string;
      /** False when the call budget refused it, so it never ran. */
      ran: boolean;
      trace: TraceEntry | null;
    }
  | { kind: "nudge"; text: string };

export interface EpisodeView {
  id: string;
  prompt: string;
  steps: Step[];
  /** Null while it is still running. */
  finish: FinishReason | null;
  answer: string | null;
  error: string | null;
  nudges: number;
}

/** Folds one event into a live view. Returns a new object; never mutates. */
export function applyEvent(view: EpisodeView, event: AgentEvent): EpisodeView {
  switch (event.type) {
    case "episode_started":
      // A page that reattaches mid-episode learns the prompt from here.
      return { ...view, id: event.episode, prompt: event.prompt ?? view.prompt };
    case "turn":
      if (event.kind === "empty") return view;
      return {
        ...view,
        steps: [
          ...view.steps,
          {
            kind: "assistant",
            text: event.text ?? "",
            toolCall: event.tool_call ?? null,
            latencyMs: event.latency_ms ?? null,
          },
        ],
      };
    case "tool_result":
      return {
        ...view,
        steps: [
          ...view.steps,
          {
            kind: "tool",
            tool: event.tool ?? "?",
            shown: event.message ?? "",
            ran: event.ran ?? false,
            trace: event.trace ?? null,
          },
        ],
      };
    case "nudge":
      return {
        ...view,
        nudges: view.nudges + 1,
        steps: [...view.steps, { kind: "nudge", text: event.message ?? "" }],
      };
    case "episode_finished":
      // The finished record is authoritative: the answer and trace are the
      // worker's reading, not anything inferred from the events.
      return fromEpisode(event.finished);
  }
}

export function emptyView(prompt: string): EpisodeView {
  return { id: "", prompt, steps: [], finish: null, answer: null, error: null, nudges: 0 };
}

const RESULT_NAME = /^<tool_result name="([^"]*)">/;
const TOOL_CALL = /<tool name="([^"]*)">([\s\S]*?)<\/tool>/;

/**
 * Rebuilds the steps of a recorded episode from its messages.
 *
 * After the prompt, every user message follows an assistant one and is either a
 * tool result (the worker renders all of them as `<tool_result name="...">`) or
 * a nudge. Results pair with trace entries in order, exactly: the budget refuses
 * a call only once the budget's worth have run, and every call that ran is
 * traced, so the first `trace.length` results ran and any after them were
 * refused. The tool call on an assistant turn is read here for display only;
 * the backend's reading of it is what the loop acted on.
 */
export function fromEpisode(episode: Episode): EpisodeView {
  const [first, ...rest] = episode.messages;
  const steps: Step[] = [];
  let traced = 0;
  for (const message of rest) {
    if (message.role === "assistant") {
      const match = TOOL_CALL.exec(message.content);
      steps.push({
        kind: "assistant",
        text: message.content,
        toolCall: match ? { name: match[1] ?? "", arguments: match[2] ?? "" } : null,
        latencyMs: null,
      });
      continue;
    }
    const result = RESULT_NAME.exec(message.content);
    if (result) {
      const trace = episode.trace[traced] ?? null;
      if (trace) traced++;
      steps.push({ kind: "tool", tool: result[1] ?? "?", shown: message.content, ran: trace !== null, trace });
    } else {
      steps.push({ kind: "nudge", text: message.content });
    }
  }
  return {
    id: episode.id,
    prompt: first?.content ?? "",
    steps,
    finish: episode.finish_reason,
    answer: episode.answer,
    error: episode.error ?? null,
    nudges: episode.nudges,
  };
}

/** Where an episode stands against what the evaluation measured. */
export type Setting = "evaluated" | "follow-up";

/**
 * The setting of the episode at `index` (which may be one past the end, for the
 * one running now). Evaluated means the policy saw no earlier conversation, as
 * on every held-out task. System-failed episodes are left out of the history the
 * policy is shown (session.Store.Begin), so they do not count.
 */
export function settingAt(episodes: Pick<Episode, "finish_reason">[], index: number): Setting {
  return episodes.slice(0, index).every((e) => systemFailed(e.finish_reason)) ? "evaluated" : "follow-up";
}

/** What each ending means, in the words the evaluation uses. */
export function describeFinish(reason: FinishReason): string {
  switch (reason) {
    case "answer":
      return "answered";
    case "no_answer":
      return "gave no <answer> after the nudge";
    case "empty":
      return "produced nothing";
    case "turn_limit":
      return "ran out of turns";
    case "backend_error":
      return "backend failed";
    case "tool_error":
      return "tool worker failed";
    case "cancelled":
      return "cancelled";
    case "interrupted":
      return "interrupted: the middle layer stopped mid-episode";
  }
}
