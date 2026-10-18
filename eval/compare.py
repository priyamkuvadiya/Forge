"""Module 9: baseline vs RL-trained policy, paired, on the held-out split.

Both arms are produced by the same command - `python -m baseline_agent`, the
trained one with `--adapter` - so they share the agent loop, the prompt, the
tool contract, the verifiers and the sampler, and differ only in weights. This
module does not generate anything. It reads the two results files, refuses to
compare them if anything other than the weights differs, and reports.

The design was fixed before either number existed, and each choice is there
because the held-out set is small (148 tasks, 20 of them coding):

- **Paired.** Every statistic about the difference is computed per task and
  then aggregated, so the 20 coding problems are compared against themselves
  rather than as two independent proportions.
- **Intervals resample tasks**, through `baseline_agent.bootstrap_ci`, the same
  function that put intervals on the baseline. Rollouts of one task are not
  independent observations.
- **Macro-averaged**, never pooled: math is 41% of the held-out pool.
- **Dense coding score**, which `verify` already returns as the fraction of
  tests passed.
- **Tool use is reported beside reward**, including on `no_tool`, because a
  policy that gained reward by calling tools everywhere has not learned *when*
  to call one, and the reward alone cannot show that.
- **The sandbox confinement is carried beside every coding number.**

    python -m eval.compare --baseline artifacts/eval/baseline_bs16.json \\
        --trained artifacts/eval/trained_run1.json --drift artifacts/rl/drift_run1.json \\
        --out artifacts/eval/heldout_comparison.json
"""

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from baseline_agent import (
    BOOTSTRAP_RESAMPLES,
    BOOTSTRAP_SEED,
    EpisodeResult,
    bootstrap_ci,
    summarize,
)
from task_suite.schema import CATEGORIES

# Settings that change the task the policy is being measured on. A difference
# in any of these means the two numbers answer different questions.
_POLICY_MUST_MATCH = ("model", "dtype", "temperature", "top_p")

# The categories where a correct answer structurally needs more than one call.
_MULTI_HOP = ("qa", "multi_tool")


class Incomparable(ValueError):
    """The two results files differ in something other than the weights."""


def load(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def episodes_of(record: dict[str, Any]) -> list[EpisodeResult]:
    return [EpisodeResult(**episode) for episode in record["episodes"]]


def _rollouts_by_task(record: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for episode in record["episodes"]:
        counts[episode["task_id"]] += 1
    return dict(counts)


def check_comparable(baseline: dict[str, Any], trained: dict[str, Any]) -> list[str]:
    """Raise if the arms are not a fair fight; return the caveats if they are.

    Refusal, not a warning, for anything that changes the task: a comparison
    across different prompts or sampling temperatures would be a number with a
    confident interval around the wrong question. Batch size is a caveat, not a
    refusal - it changes nothing about the distribution being sampled, only the
    padding and bf16 accumulation order - but it is stated in the output.
    """
    problems = []
    if baseline.get("arm") != "baseline-prompted":
        problems.append(f"baseline file is arm {baseline.get('arm')!r}")
    if trained.get("arm") != "rl-trained":
        problems.append(f"trained file is arm {trained.get('arm')!r}")
    if baseline["split"] != trained["split"]:
        problems.append(f"splits differ: {baseline['split']} vs {trained['split']}")
    if baseline["split"] != "heldout" or trained["split"] != "heldout":
        problems.append("module 9 reports the held-out split only")
    if baseline["agent"] != trained["agent"]:
        problems.append(f"agent settings differ: {baseline['agent']} vs {trained['agent']}")
    for key in _POLICY_MUST_MATCH:
        if baseline["policy"].get(key) != trained["policy"].get(key):
            problems.append(
                f"policy {key} differs: {baseline['policy'].get(key)!r} "
                f"vs {trained['policy'].get(key)!r}"
            )
    if _rollouts_by_task(baseline) != _rollouts_by_task(trained):
        problems.append("the arms did not run the same tasks the same number of times")
    for name, record in (("baseline", baseline), ("trained", trained)):
        if record["sandbox"].get("allow_unconfined"):
            problems.append(f"{name} scored coding through an unconfined sandbox")
    if problems:
        raise Incomparable("; ".join(problems))

    caveats = []
    for key in ("batch_size", "seed"):
        if baseline["policy"].get(key) != trained["policy"].get(key):
            caveats.append(
                f"{key} differs: baseline {baseline['policy'].get(key)}, "
                f"trained {trained['policy'].get(key)}"
            )
    if baseline["sandbox"]["backend"] != trained["sandbox"]["backend"]:
        caveats.append("coding was scored by different sandbox backends")
    return caveats


def task_means(results: list[EpisodeResult]) -> dict[str, float]:
    by_task: dict[str, list[float]] = defaultdict(list)
    for result in results:
        by_task[result.task_id].append(result.reward)
    return {task: sum(rewards) / len(rewards) for task, rewards in by_task.items()}


def paired_rows(
    baseline: list[EpisodeResult], trained: list[EpisodeResult]
) -> list[EpisodeResult]:
    """Per-rollout reward differences, shaped so `bootstrap_ci` can resample them.

    The two arms' rollouts are independent draws, so rollout i of one is not
    paired with rollout i of the other. That does not matter here: averaged over
    a task's rollouts, the differences give exactly that task's difference in
    means, and `bootstrap_ci` averages over rollouts before resampling tasks.
    The pairing that matters - same task, both arms - is exact.
    """
    rewards: dict[str, list[float]] = defaultdict(list)
    for result in sorted(baseline, key=lambda r: (r.task_id, r.rollout)):
        rewards[result.task_id].append(result.reward)
    rows = []
    for result in sorted(trained, key=lambda r: (r.task_id, r.rollout)):
        base = rewards[result.task_id].pop(0)
        rows.append(
            EpisodeResult(
                **{**result.to_dict(), "reward": result.reward - base}
            )
        )
    return rows


def macro_delta_ci(
    deltas_by_category: dict[str, list[float]],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """An interval for the macro-average of per-task deltas.

    Stratified: each resample draws tasks with replacement *within* each
    category, then averages the category means. `bootstrap_ci` resamples one
    pool, which here would let the category mix vary from draw to draw - the
    very weighting the macro-average exists to hold fixed. Same seed and
    resample count as `bootstrap_ci`, so the two kinds of interval are
    reproducible in the same way.
    """
    pools = [values for values in deltas_by_category.values() if values]
    if not pools:
        return (0.0, 0.0)
    rng = random.Random(seed)
    draws = sorted(
        sum(sum(rng.choices(pool, k=len(pool))) / len(pool) for pool in pools) / len(pools)
        for _ in range(resamples)
    )
    lower = (1.0 - confidence) / 2.0
    return (
        draws[int(lower * resamples)],
        draws[min(int((1.0 - lower) * resamples), resamples - 1)],
    )


def tool_use(results: list[EpisodeResult], registered: list[str]) -> dict[str, Any]:
    """How a policy used the tools, beyond what `summarize` already reports."""
    calls = [call for result in results for call in result.tool_calls]
    invented = [call["tool"] for call in calls if call["tool"] not in registered]
    report: dict[str, Any] = {
        "episodes": len(results),
        "calls": len(calls),
        # A name outside the registry. The trace records these as failed
        # calls; counted separately because "uses tools more" and "makes up
        # tool names more" are opposite findings that both raise call counts.
        "invented_tool_calls": len(invented),
        "invented_tool_names": sorted(set(invented)),
        "failed_calls": sum(1 for call in calls if not call["ok"]),
    }
    multi_hop = [r for r in results if r.category in _MULTI_HOP]
    if multi_hop:
        # The baseline never chained: no qa or multi_tool episode called twice.
        report["multi_hop_chain_rate"] = sum(
            1 for r in multi_hop if r.n_tool_calls >= 2
        ) / len(multi_hop)

    math = [r for r in results if r.category == "math"]
    with_calc = [r for r in math if "calculator" in r.tools_used]
    without = [r for r in math if "calculator" not in r.tools_used]
    if math:
        # Whether a math gain came through the calculator or around it. Train
        # tasks were seen about twice each in training, so gains that do not
        # route through the tool are the ones to suspect of memorisation -
        # though on held-out tasks there is nothing to have memorised.
        report["math"] = {
            "calculator_rate": len(with_calc) / len(math),
            "reward_with_calculator": (
                sum(r.reward for r in with_calc) / len(with_calc) if with_calc else None
            ),
            "reward_without_calculator": (
                sum(r.reward for r in without) / len(without) if without else None
            ),
            "episodes_with_calculator": len(with_calc),
        }
    return report


def _sha256(path: Path) -> str | None:
    weights = path / "adapter_model.safetensors"
    return hashlib.sha256(weights.read_bytes()).hexdigest() if weights.exists() else None


def drift_beside(drift: dict[str, Any], adapter: dict[str, Any] | None) -> dict[str, Any]:
    """The drift probe's report for the evaluated adapter, matched by content.

    `adapter-final` was never probed under that name - the run's drift pass
    covered `adapter-step10` to `adapter-step200` - so the report is matched by
    hashing each probed checkpoint against the evaluated adapter's recorded
    sha256. A match by name alone would vouch for weights nobody checked.
    """
    wanted = (adapter or {}).get("sha256")
    for report in drift["reports"]:
        source = Path(report["source"])
        if wanted and source.exists() and _sha256(source) == wanted:
            return {
                "matched_by": "sha256",
                "source": source.name,
                "lost": report["lost"],
                "gained": report["gained"],
                "by_kind": {
                    kind: {key: stats[key] for key in ("probes", "base", "adapter")}
                    for kind, stats in report["by_kind"].items()
                },
                "mean_kl_per_token": report["drift"]["mean_kl_per_token"],
            }
    return {"matched_by": None, "note": "no probed checkpoint has the evaluated weights"}


def compare(
    baseline: dict[str, Any],
    trained: dict[str, Any],
    *,
    drift: dict[str, Any] | None = None,
) -> dict[str, Any]:
    caveats = check_comparable(baseline, trained)
    base_rows, trained_rows = episodes_of(baseline), episodes_of(trained)
    base_summary, trained_summary = summarize(base_rows), summarize(trained_rows)

    by_category: dict[str, Any] = {}
    deltas_by_category: dict[str, list[float]] = {}
    for category in CATEGORIES:
        base_cat = [r for r in base_rows if r.category == category]
        trained_cat = [r for r in trained_rows if r.category == category]
        if not base_cat:
            continue
        base_means, trained_means = task_means(base_cat), task_means(trained_cat)
        deltas = [trained_means[task] - base_means[task] for task in sorted(base_means)]
        deltas_by_category[category] = deltas
        low, high = bootstrap_ci(paired_rows(base_cat, trained_cat))
        by_category[category] = {
            "tasks": len(deltas),
            "baseline": base_summary["by_category"][category],
            "trained": trained_summary["by_category"][category],
            "delta": sum(deltas) / len(deltas),
            "delta_ci95": [low, high],
            "tasks_better": sum(1 for d in deltas if d > 1e-9),
            "tasks_worse": sum(1 for d in deltas if d < -1e-9),
            "tasks_same": sum(1 for d in deltas if abs(d) <= 1e-9),
        }

    macro_delta = sum(c["delta"] for c in by_category.values()) / len(by_category)
    registered = baseline["agent"]["tools"]
    return {
        "generated_by": "eval.compare",
        "split": baseline["split"],
        "rollouts_per_task": max(_rollouts_by_task(baseline).values()),
        "caveats": caveats,
        "arms": {
            "baseline": {"policy": baseline["policy"], "sandbox": baseline["sandbox"]},
            "trained": {"policy": trained["policy"], "sandbox": trained["sandbox"]},
        },
        "macro": {
            "baseline": base_summary["macro_mean_reward"],
            "trained": trained_summary["macro_mean_reward"],
            "delta": macro_delta,
            "delta_ci95": list(macro_delta_ci(deltas_by_category)),
        },
        "by_category": by_category,
        "tool_use": {
            "baseline": tool_use(base_rows, registered),
            "trained": tool_use(trained_rows, registered),
        },
        "drift": drift_beside(drift, trained["policy"].get("adapter")) if drift else None,
    }


def format_comparison(report: dict[str, Any]) -> str:
    def ci(pair):
        return f"[{pair[0]:+.3f}, {pair[1]:+.3f}]"

    lines = [
        f"{'category':<11}{'tasks':>6}{'baseline':>10}{'trained':>9}{'delta':>9}"
        f"{'delta ci95':>19}{'+/-/=':>11}{'tool rate b->t':>17}"
    ]
    for category, row in report["by_category"].items():
        b, t = row["baseline"], row["trained"]
        lines.append(
            f"{category:<11}{row['tasks']:>6}{b['mean_reward']:>10.3f}"
            f"{t['mean_reward']:>9.3f}{row['delta']:>+9.3f}{ci(row['delta_ci95']):>19}"
            f"{row['tasks_better']:>4}/{row['tasks_worse']}/{row['tasks_same']:<3}"
            f"{b['any_tool_rate']:>9.3f} -> {t['any_tool_rate']:.3f}"
        )
    macro = report["macro"]
    lines += [
        "",
        f"macro-average  baseline {macro['baseline']:.4f}  trained {macro['trained']:.4f}  "
        f"delta {macro['delta']:+.4f} {ci(macro['delta_ci95'])}",
    ]
    for arm in ("baseline", "trained"):
        use = report["tool_use"][arm]
        math = use.get("math", {})
        lines.append(
            f"{arm:<9} calls {use['calls']}, invented {use['invented_tool_calls']} "
            f"{use['invented_tool_names']}, multi-hop chain rate "
            f"{use.get('multi_hop_chain_rate', 0):.3f}, math calculator rate "
            f"{math.get('calculator_rate', 0):.3f}"
        )
    for caveat in report["caveats"]:
        lines.append(f"CAVEAT: {caveat}")
    drift = report.get("drift")
    if drift and drift.get("matched_by"):
        lines.append(
            f"drift ({drift['source']}): {drift['lost']} probe answers lost, "
            f"{drift['gained']} gained, KL/token {drift['mean_kl_per_token']:.4f}"
        )
    elif drift:
        lines.append(f"drift: {drift['note']}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.compare", description=__doc__.split("\n")[0])
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--trained", type=Path, required=True)
    parser.add_argument("--drift", type=Path, help="a measure_drift report to set beside it")
    parser.add_argument("--out", type=Path)
    arguments = parser.parse_args(argv)

    try:
        report = compare(
            load(arguments.baseline),
            load(arguments.trained),
            drift=load(arguments.drift) if arguments.drift else None,
        )
    except Incomparable as error:
        print(f"refusing to compare: {error}", file=sys.stderr)
        return 1

    print(format_comparison(report))
    if arguments.out:
        arguments.out.parent.mkdir(parents=True, exist_ok=True)
        arguments.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {arguments.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
