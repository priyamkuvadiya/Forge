import { useEffect, useMemo, useState } from "react";
import { getRun, listRuns } from "../api/client";
import type { Run, RunSummary } from "../api/types";
import { LineChart, type Line } from "./LineChart";
import { CATEGORIES, metrics, series, smooth, trained, windowMean, type Metric } from "./series";

// What each known run was, beyond what its config says. Anything else shows its
// config alone.
const RUN_NOTES: Record<string, string> = {
  run1: "Correctness-only reward. Served as run1.",
  run2: "No-search penalty on qa and multi_tool. Served as trained. Its reward is still verify(), so it is comparable with run1's.",
  "pilot-lr1e-5": "30-step pilot that chose the learning rate.",
  "pilot-lr5e-5": "30-step pilot, rejected: tool calls fell and math stayed at 0.",
  "run2-trial": "10-step trial on qa and multi_tool only. Its rewards are not comparable with the others.",
};

const COLORS: Record<string, string> = {
  run1: "#4c78d8",
  run2: "#e8803a",
  "pilot-lr1e-5": "#3aa676",
  "pilot-lr5e-5": "#a660c8",
  "run2-trial": "#8a8a85",
};
const FALLBACK = ["#c2484d", "#2fa3b5", "#b59a2f"];

const DEFAULT_RUNS = ["run1", "run2"];
const CARD_GIB = 8.0;

const pct = (v: number) => `${(v * 100).toFixed(0)}%`;

interface Arguments {
  steps?: number;
  learning_rate?: number;
  prompts_per_step?: number;
  group_size?: number;
  no_search_penalty?: number;
  categories?: string[] | null;
}

function argumentsOf(run: Run): Arguments {
  const args = (run.config as { arguments?: Arguments }).arguments;
  return args ?? {};
}

export function TrainingView() {
  const [summaries, setSummaries] = useState<RunSummary[]>([]);
  const [runs, setRuns] = useState<Record<string, Run>>({});
  const [selected, setSelected] = useState<string[]>(DEFAULT_RUNS);
  const [span, setSpan] = useState(20);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    listRuns()
      .then(setSummaries)
      .catch((e: unknown) => setError(e instanceof Error ? e.message : String(e)));
  }, []);

  useEffect(() => {
    for (const name of selected) {
      if (runs[name]) continue;
      getRun(name)
        .then((run) => setRuns((all) => ({ ...all, [name]: run })))
        .catch((e: unknown) => setError(`${name}: ${e instanceof Error ? e.message : String(e)}`));
    }
  }, [selected, runs]);

  const shown = useMemo(
    () => selected.map((name) => runs[name]).filter((run): run is Run => run !== undefined),
    [selected, runs],
  );
  const color = (name: string) =>
    COLORS[name] ?? FALLBACK[summaries.findIndex((s) => s.name === name) % FALLBACK.length]!;

  const lines = (metric: Metric, smoothed = true): Line[] =>
    shown.map((run) => {
      const raw = series(run.records, metric);
      return {
        label: run.name,
        color: color(run.name),
        points: smoothed && span > 1 ? smooth(raw, span) : raw,
        ...(smoothed && span > 1 ? { raw } : {}),
      };
    });

  const toggle = (name: string) =>
    setSelected((current) => (current.includes(name) ? current.filter((n) => n !== name) : [...current, name]));

  const smoothNote = span > 1 ? `Line: trailing mean of ${span} steps; faint: each step.` : "Each step.";

  return (
    <div className="training">
      <header className="training-head">
        <h2>Training runs</h2>
        <p className="muted">
          Rewards here are on <strong>training</strong> prompts: 8 per step, each rolled out 8 times, scored by
          verify(). They show what the policy was learning, not how well it does. The result is the held-out
          evaluation in the README: macro reward 0.183 for the baseline, 0.467 for run1 and 0.503 for run2. These
          records are the training loop's step log, copied verbatim into the middle layer's SQLite file. Nothing on
          this page computes a reward.
        </p>
      </header>

      {error && <div className="error-banner">{error}</div>}

      <div className="run-picker">
        {summaries.map((summary) => (
          <label key={summary.name} className={selected.includes(summary.name) ? "chosen" : ""}>
            <input type="checkbox" checked={selected.includes(summary.name)} onChange={() => toggle(summary.name)} />
            <i style={{ background: color(summary.name) }} />
            <span>
              <strong>{summary.name}</strong>
              <span className="muted small-text">
                {summary.steps} steps
                {summary.wall_clock_seconds !== null && `, ${(summary.wall_clock_seconds / 3600).toFixed(2)} h`}
                {!summary.finished && ", unfinished"}
              </span>
            </span>
          </label>
        ))}
        <label className="smoothing">
          <span className="muted">smoothing</span>
          <select value={span} onChange={(e) => setSpan(Number(e.target.value))}>
            <option value={1}>none</option>
            <option value={10}>10 steps</option>
            <option value={20}>20 steps</option>
          </select>
        </label>
      </div>

      {shown.length > 0 && <RunTable runs={shown} color={color} />}

      <section className="chart-grid">
        <LineChart
          title="Mean reward, training prompts"
          lines={lines(metrics.meanReward)}
          yDomain={[0, 1]}
          note={`verify() reward over all 64 episodes of the step. ${smoothNote}`}
        />
        <LineChart
          title="Usable groups per step"
          lines={lines(metrics.usableGroups)}
          format={(v) => v.toFixed(1)}
          note="Groups of 8 rollouts whose rewards differ. A group where all 8 score the same has no advantage and teaches nothing; a step with none skips the update."
        />
      </section>

      <h3>Reward by category</h3>
      <section className="chart-grid small">
        {CATEGORIES.map((category) => (
          <LineChart
            key={category}
            title={category}
            lines={lines(metrics.category(category))}
            yDomain={[0, 1]}
            height={150}
          />
        ))}
      </section>
      <p className="muted chart-note">
        A step samples 8 prompts round-robin over the categories, so each category's point is one or two prompts.
        qa stays at 0 in both runs: neither policy chains searches, which multi-hop questions need.
      </p>

      <h3>Tool use</h3>
      <section className="chart-grid">
        <LineChart
          title="Calculator calls per episode"
          lines={lines(metrics.toolCalls("calculator"))}
          note={smoothNote}
        />
        <LineChart title="Search calls per episode" lines={lines(metrics.toolCalls("search"))} note={smoothNote} />
        <LineChart
          title="Python calls per episode"
          lines={lines(metrics.toolCalls("python"))}
          note="The code tool. Almost never called in training, and no held-out code episode called it in either arm."
        />
        <LineChart
          title="Calls to tools that do not exist"
          lines={lines(metrics.inventedPerEpisode)}
          format={(v) => v.toFixed(3)}
          note={'Per episode: names like "calc" and "calculus". Each one fails and costs a call.'}
        />
      </section>

      <h3>The run2 penalty</h3>
      <section className="chart-grid">
        <LineChart
          title="Penalised episodes"
          lines={lines(metrics.penalizedFraction)}
          yDomain={[0, 0.5]}
          format={pct}
          note="Share of the step's episodes that answered a qa or multi_tool task wrongly without a successful search, and scored −0.25. Only run2 and its trial log it."
        />
        <LineChart
          title="Episodes that gave an answer"
          lines={lines(metrics.answeredFraction)}
          yDomain={[0, 1]}
          format={pct}
          note="Logged from run2 on, to tell 'learned to search' from 'learned to stop answering'. Both would lower the penalised share."
        />
      </section>

      <h3>Staying close to the base model, and the machine</h3>
      <section className="chart-grid">
        <LineChart
          title="KL from the base model"
          lines={lines(metrics.kl)}
          format={(v) => v.toFixed(4)}
          note="Per token, on the step's own rollouts. Steps with no update log 0.0 as a placeholder and are left out. Drift on general knowledge is measured separately (measure_drift, README)."
        />
        <LineChart
          title="Seconds per step"
          lines={lines(metrics.secondsPerStep)}
          format={(v) => v.toFixed(0)}
          note="Rollout plus backward. Generation is most of it."
        />
        <LineChart
          title="GPU memory reserved"
          lines={[
            ...lines(metrics.reservedGib, false),
            {
              label: `card ${CARD_GIB} GiB`,
              color: "#b3261e",
              points: shown[0]?.records.map((r) => ({ step: r.step, value: CARD_GIB })) ?? [],
            },
          ]}
          yDomain={[0, 9]}
          format={(v) => `${v.toFixed(1)}G`}
          note="What the CUDA allocator held at the end of the step's update. Past the card's size, Windows silently spills to system RAM rather than failing, so the loop checks this every step. Steps with no update are left out."
        />
      </section>
    </div>
  );
}

function RunTable({ runs, color }: { runs: Run[]; color: (name: string) => string }) {
  return (
    <div className="table-wrap">
      <table className="run-table">
        <thead>
          <tr>
            <th>run</th>
            <th>steps</th>
            <th>lr</th>
            <th>penalty</th>
            <th>categories</th>
            <th title="Steps where every group was degenerate, so nothing was updated">no-update steps</th>
            <th title="Steps whose memory reservation went past the card">spilled</th>
            <th title="Mean training-prompt reward over the first and last 20 steps">reward, first → last 20</th>
            <th>note</th>
          </tr>
        </thead>
        <tbody>
          {runs.map((run) => {
            const args = argumentsOf(run);
            const reward = series(run.records, metrics.meanReward);
            const last = run.records.at(-1)?.step ?? 0;
            const first = windowMean(reward, 1, 20);
            const end = windowMean(reward, Math.max(1, last - 19), last);
            return (
              <tr key={run.name}>
                <td>
                  <i className="swatch" style={{ background: color(run.name) }} />
                  {run.name}
                </td>
                <td>{run.steps}</td>
                <td>{args.learning_rate ?? "–"}</td>
                <td>{args.no_search_penalty ? args.no_search_penalty : "none"}</td>
                <td>{args.categories?.join(", ") ?? "all five"}</td>
                <td>{run.records.filter((r) => !trained(r)).length}</td>
                <td>{run.records.filter((r) => r.spilled === true).length}</td>
                <td>
                  {first === null ? "–" : first.toFixed(3)} → {end === null ? "–" : end.toFixed(3)}
                </td>
                <td className="muted">{RUN_NOTES[run.name] ?? ""}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
