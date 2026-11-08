// The middle layer's JSON, as its Go types define it (middle-layer/internal).
// Field names are the wire names, so these can be checked against the Go structs
// line by line.

// agent.Message
export interface Message {
  role: "user" | "assistant";
  content: string;
}

// agent.ToolCall
export interface ToolCall {
  name: string;
  arguments: string;
}

// agent.TraceEntry, itself a mirror of tools.toolbox.TraceEntry.
export interface TraceEntry {
  index: number;
  tool: string;
  arguments: string;
  ok: boolean;
  output: string;
  error: string | null;
  duration_ms: number;
}

// How an episode ended. The first four are the evaluation's own; the rest mean
// the system failed rather than the policy (agent.Episode.Failed).
export type PolicyFinish = "answer" | "no_answer" | "empty" | "turn_limit";
export type SystemFinish = "backend_error" | "tool_error" | "cancelled" | "interrupted";
export type FinishReason = PolicyFinish | SystemFinish;

const systemFinishes: ReadonlySet<string> = new Set<SystemFinish>([
  "backend_error",
  "tool_error",
  "cancelled",
  "interrupted",
]);

/** Mirrors agent.Episode.Failed: the system, not the policy, ended it. */
export function systemFailed(reason: FinishReason): boolean {
  return systemFinishes.has(reason);
}

// agent.Episode
export interface Episode {
  id: string;
  policy: string;
  messages: Message[];
  nudges: number;
  finish_reason: FinishReason;
  answer: string | null;
  trace: TraceEntry[];
  error?: string;
}

// agent.Event. Go omits unused fields, so each type carries only what it means.
export type TurnKind = "empty" | "tool_call" | "answer" | "no_answer";

interface EventBase {
  episode: string;
  turn: number;
}

export interface StartedEvent extends EventBase {
  type: "episode_started";
  /** The user message that started it. */
  prompt?: string;
}

export interface TurnEvent extends EventBase {
  type: "turn";
  kind: TurnKind;
  text?: string;
  tool_call?: ToolCall;
  latency_ms?: number;
}

export interface ToolResultEvent extends EventBase {
  type: "tool_result";
  tool?: string;
  ok?: boolean;
  /** What the policy was shown. */
  message?: string;
  tool_error?: string;
  /** False when the call budget refused the call, so it never ran. */
  ran?: boolean;
  duration_ms?: number;
  /** The call's trace entry; absent when it never ran. */
  trace?: TraceEntry;
}

export interface NudgeEvent extends EventBase {
  type: "nudge";
  message?: string;
}

export interface FinishedEvent extends EventBase {
  type: "episode_finished";
  finished: Episode;
}

export type AgentEvent = StartedEvent | TurnEvent | ToolResultEvent | NudgeEvent | FinishedEvent;

// session.Session and session.Summary
export interface Session {
  id: string;
  policy: string;
  created_at: string;
  running: boolean;
  episodes: Episode[];
}

export interface SessionSummary {
  id: string;
  policy: string;
  created_at: string;
  running: boolean;
  episodes: number;
}

// GET /policies, passed through from the backend (backend/engine.py describe).
export interface AdapterInfo {
  path?: string;
  sha256?: string;
  [key: string]: unknown;
}

export interface PolicyInfo {
  name: string;
  adapter: AdapterInfo | null;
}

export interface PoliciesResponse {
  model: string;
  sampling: Record<string, unknown>;
  max_new_tokens: number;
  policies: PolicyInfo[];
}

// GET /health (api.health and toolworker.Info)
export interface Health {
  backend: string;
  tools: {
    worker_running: boolean;
    tools: string[];
    sandbox: {
      backend: string | null;
      confinement: string | null;
      filesystem_confined: boolean | null;
      allow_unconfined: boolean;
    };
  };
}

// metrics.Summary and metrics.Run. A record is a step log line verbatim, so its
// fields are whatever that run logged: run1 has no "penalized", for example.
export interface RunSummary {
  name: string;
  finished: boolean;
  wall_clock_seconds: number | null;
  steps: number;
}

export interface StepRecord {
  record: "step";
  step: number;
  [field: string]: unknown;
}

export interface Run extends RunSummary {
  config: Record<string, unknown>;
  records: StepRecord[];
}
