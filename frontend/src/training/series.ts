import type { StepRecord } from "../api/types";

// Reads step records into chart series. Nothing here invents a value: a field a
// run never logged, or logged only as a placeholder, is null, and a chart draws
// a gap rather than a zero.

export type Point = { step: number; value: number | null };

export const CATEGORIES = ["math", "code", "qa", "multi_tool", "no_tool"] as const;
export const TOOLS = new Set(["calculator", "search", "python"]);

function num(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

/**
 * Whether the step ran a backward pass. When every group was degenerate nothing
 * is kept, and train_grpo logs loss, KL and memory as 0.0 placeholders rather
 * than measurements (rl_training/train_grpo.py, `if not transcripts`).
 */
export function trained(record: StepRecord): boolean {
  return (num(record.kept_episodes) ?? 0) > 0;
}

export type Metric = (record: StepRecord) => number | null;

export const metrics = {
  meanReward: (r: StepRecord) => num(r.mean_reward),
  category: (name: string): Metric => (r: StepRecord) => {
    const per = r.per_category_reward;
    return per && typeof per === "object" ? num((per as Record<string, unknown>)[name]) : null;
  },
  /** Only from steps that trained; see `trained`. */
  kl: (r: StepRecord) => (trained(r) ? num(r.mean_kl) : null),
  usableGroups: (r: StepRecord) => {
    const groups = num(r.groups);
    const degenerate = num(r.degenerate_groups);
    return groups === null || degenerate === null ? null : groups - degenerate;
  },
  toolCallsPerEpisode: (r: StepRecord) => {
    const episodes = num(r.episodes);
    if (!episodes || !r.tool_calls || typeof r.tool_calls !== "object") return null;
    const total = Object.values(r.tool_calls as Record<string, unknown>).reduce<number>(
      (sum, n) => sum + (num(n) ?? 0),
      0,
    );
    return total / episodes;
  },
  /** Calls to a tool that does not exist (`calc`, `calculus`...), per episode. */
  inventedPerEpisode: (r: StepRecord) => {
    const episodes = num(r.episodes);
    if (!episodes || !r.tool_calls || typeof r.tool_calls !== "object") return null;
    let invented = 0;
    for (const [name, n] of Object.entries(r.tool_calls as Record<string, unknown>)) {
      if (!TOOLS.has(name)) invented += num(n) ?? 0;
    }
    return invented / episodes;
  },
  toolCalls: (name: string): Metric => (r: StepRecord) => {
    const episodes = num(r.episodes);
    if (!episodes || !r.tool_calls || typeof r.tool_calls !== "object") return null;
    return (num((r.tool_calls as Record<string, unknown>)[name]) ?? 0) / episodes;
  },
  /** run2 onwards. Absent from run1's log, so null there, never 0. */
  penalizedFraction: (r: StepRecord) => {
    const episodes = num(r.episodes);
    const penalized = num(r.penalized);
    return episodes && penalized !== null && "penalized" in r ? penalized / episodes : null;
  },
  answeredFraction: (r: StepRecord) => {
    const episodes = num(r.episodes);
    const answered = num(r.answered);
    return episodes && answered !== null && "answered" in r ? answered / episodes : null;
  },
  secondsPerStep: (r: StepRecord) => {
    const rollout = num(r.seconds_rollout);
    const backward = num(r.seconds_backward);
    return rollout === null ? null : rollout + (backward ?? 0);
  },
  reservedGib: (r: StepRecord) => (trained(r) ? num(r.reserved_gib) : null),
} satisfies Record<string, Metric | ((name: string) => Metric)>;

export function series(records: StepRecord[], metric: Metric): Point[] {
  return records.map((r: StepRecord) => ({ step: r.step, value: metric(r) }));
}

/**
 * Trailing mean over the last `window` steps that have a value. Labelled as such
 * wherever it is drawn: per-step training reward is 8 prompts and very noisy.
 */
export function smooth(points: Point[], window: number): Point[] {
  const out: Point[] = [];
  const recent: number[] = [];
  for (const point of points) {
    if (point.value !== null) {
      recent.push(point.value);
      if (recent.length > window) recent.shift();
    }
    out.push({ step: point.step, value: point.value === null || recent.length === 0 ? null : mean(recent) });
  }
  return out;
}

function mean(values: number[]): number {
  return values.reduce((a, b) => a + b, 0) / values.length;
}

/** Mean of the values in steps [from, to], or null when there are none. */
export function windowMean(points: Point[], from: number, to: number): number | null {
  const values = points.filter((p) => p.step >= from && p.step <= to && p.value !== null).map((p) => p.value!);
  return values.length ? mean(values) : null;
}
