import { describe, expect, it } from "vitest";
import type { StepRecord } from "../api/types";
import { wire } from "../test/wire";
import { metrics, series, smooth, trained, windowMean } from "./series";

// An excerpt of run1's real log: two steps and one where every group was
// degenerate, so nothing was updated and the loop logged 0.0 placeholders.
const records = wire.run.records;
const idle = records.find((r) => r.kept_episodes === 0)!;
const busy = records.find((r) => (r.kept_episodes as number) > 0)!;

describe("what is a measurement", () => {
  it("the excerpt has an update-free step to test against", () => {
    expect(idle).toBeDefined();
    expect(idle.mean_kl).toBe(0);
    expect(trained(idle)).toBe(false);
    expect(trained(busy)).toBe(true);
  });

  it("KL and memory from a step with no update are gaps, not zeros", () => {
    expect(metrics.kl(idle)).toBeNull();
    expect(metrics.reservedGib(idle)).toBeNull();
    expect(metrics.kl(busy)).toBe(busy.mean_kl);
  });

  it("reward is a measurement either way: the rollouts happened", () => {
    expect(metrics.meanReward(idle)).toBe(idle.mean_reward);
  });

  it("fields run1 never logged are gaps, not zeros", () => {
    expect("penalized" in busy).toBe(false);
    expect(metrics.penalizedFraction(busy)).toBeNull();
    expect(metrics.answeredFraction(busy)).toBeNull();
  });

  it("a run2-style record reads its fields", () => {
    const record = { ...busy, penalized: 16, answered: 48 } as StepRecord;
    expect(metrics.penalizedFraction(record)).toBe(16 / 64);
    expect(metrics.answeredFraction(record)).toBe(48 / 64);
  });
});

describe("tool calls", () => {
  const record = {
    record: "step",
    step: 1,
    episodes: 64,
    tool_calls: { calc: 1, calculator: 3, calculus: 2, search: 6 },
  } as StepRecord;

  it("per episode, over every episode of the step", () => {
    expect(metrics.toolCallsPerEpisode(record)).toBe(12 / 64);
    expect(metrics.toolCalls("search")(record)).toBe(6 / 64);
    expect(metrics.toolCalls("python")(record)).toBe(0);
  });

  it("invented names are those outside the three real tools", () => {
    expect(metrics.inventedPerEpisode(record)).toBe(3 / 64);
  });
});

describe("per category", () => {
  it("a category the step did not sample is a gap", () => {
    const record = { record: "step", step: 1, per_category_reward: { math: 0.5 } } as StepRecord;
    expect(metrics.category("math")(record)).toBe(0.5);
    expect(metrics.category("qa")(record)).toBeNull();
  });
});

describe("smoothing", () => {
  const points = [1, 2, null, 4, 5].map((value, i) => ({ step: i + 1, value }));

  it("is a trailing mean over values present, and keeps gaps as gaps", () => {
    expect(smooth(points, 2).map((p) => p.value)).toEqual([1, 1.5, null, 3, 4.5]);
  });

  it("a window of one is the series itself", () => {
    expect(smooth(points, 1)).toEqual(points);
  });

  it("windowMean ignores gaps and is null with nothing in range", () => {
    expect(windowMean(points, 1, 3)).toBe(1.5);
    expect(windowMean(points, 3, 3)).toBeNull();
  });

  it("series keeps the step numbers", () => {
    expect(series(records, metrics.meanReward).map((p) => p.step)).toEqual(records.map((r) => r.step));
  });
});
