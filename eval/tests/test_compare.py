"""The comparison's statistics and, above all, its refusals.

A comparison harness fails silently in one way that matters: it reports a
confident interval around two runs that were not the same experiment. Most of
these tests are about refusing that.
"""

import copy

import pytest

from eval.compare import (
    Incomparable,
    check_comparable,
    compare,
    macro_delta_ci,
    paired_rows,
    task_means,
    tool_use,
)
from baseline_agent import EpisodeResult


def _episode(task_id, category, rollout, reward, calls=()):
    return {
        "task_id": task_id,
        "category": category,
        "split": "heldout",
        "rollout": rollout,
        "reward": reward,
        "answer": "x",
        "finish_reason": "answer",
        "nudges": 0,
        "n_tool_calls": len(calls),
        "tools_used": sorted({c for c in calls}),
        "tool_calls": [
            {"index": i, "tool": c, "ok": c in ("calculator", "search"), "arguments": "",
             "output_chars": 0, "error": None, "duration_ms": 0.0}
            for i, c in enumerate(calls)
        ],
    }


def _record(rewards, *, arm="baseline-prompted", calls=None):
    """rewards: {(task_id, category): [reward per rollout]}"""
    calls = calls or {}
    episodes = [
        _episode(task, category, i, r, calls.get((task, i), ()))
        for (task, category), rs in rewards.items()
        for i, r in enumerate(rs)
    ]
    policy = {"model": "m", "dtype": "bfloat16", "temperature": 0.7, "top_p": 0.9,
              "batch_size": 16, "seed": 0}
    if arm == "rl-trained":
        policy["adapter"] = {"path": "a", "sha256": "f00"}
    return {
        "arm": arm,
        "split": "heldout",
        "policy": policy,
        "agent": {"max_calls": 8, "max_nudges": 1, "max_new_tokens": 384,
                  "few_shot": True, "tools": ["calculator", "search"]},
        "sandbox": {"backend": "windows", "allow_unconfined": False},
        "episodes": episodes,
    }


BASE = {("m0", "math"): [0, 0], ("m1", "math"): [1, 0], ("c0", "code"): [0.5, 0.5]}
TRAINED = {("m0", "math"): [1, 1], ("m1", "math"): [1, 0], ("c0", "code"): [0.25, 0.25]}


@pytest.fixture
def arms():
    return _record(BASE), _record(TRAINED, arm="rl-trained")


def test_deltas_are_per_task_and_counted_by_sign(arms):
    report = compare(*arms)
    math, code = report["by_category"]["math"], report["by_category"]["code"]

    assert math["delta"] == pytest.approx(0.5)  # m0 +1, m1 0
    assert (math["tasks_better"], math["tasks_worse"], math["tasks_same"]) == (1, 0, 1)
    assert code["delta"] == pytest.approx(-0.25)
    assert code["tasks_worse"] == 1


def test_the_macro_delta_weights_categories_equally(arms):
    report = compare(*arms)
    assert report["macro"]["delta"] == pytest.approx((0.5 - 0.25) / 2)
    assert report["macro"]["trained"] - report["macro"]["baseline"] == pytest.approx(
        report["macro"]["delta"]
    )


def test_paired_rows_average_to_the_difference_in_task_means(arms):
    base, trained = [EpisodeResult(**e) for e in arms[0]["episodes"]], [
        EpisodeResult(**e) for e in arms[1]["episodes"]
    ]
    deltas = task_means(paired_rows(base, trained))
    b, t = task_means(base), task_means(trained)
    assert deltas == {task: pytest.approx(t[task] - b[task]) for task in b}


def test_identical_arms_give_a_zero_width_interval_at_zero():
    report = compare(_record(BASE), _record(BASE, arm="rl-trained"))
    assert report["macro"]["delta_ci95"] == [0.0, 0.0]
    for row in report["by_category"].values():
        assert row["delta_ci95"] == [0.0, 0.0]


def test_the_macro_interval_holds_the_category_mix_fixed():
    """A constant delta inside each category leaves no room to move.

    A pooled resample would vary how many tasks come from each category and so
    produce a non-degenerate interval here; stratified, it cannot.
    """
    low, high = macro_delta_ci({"a": [1.0] * 3, "b": [0.0] * 30})
    assert low == high == pytest.approx(0.5)


def test_the_macro_interval_brackets_the_point_estimate():
    deltas = {"a": [0.0, 1.0, 0.5, 0.25], "b": [-0.5, 0.0, 0.1]}
    point = sum(sum(v) / len(v) for v in deltas.values()) / 2
    low, high = macro_delta_ci(deltas)
    assert low <= point <= high and low < high


@pytest.mark.parametrize(
    "mutate, message",
    [
        (lambda b, t: t["policy"].update(temperature=1.0), "temperature"),
        (lambda b, t: t["policy"].update(model="other"), "model"),
        (lambda b, t: t["agent"].update(few_shot=False), "agent settings"),
        (lambda b, t: t.update(split="train"), "split"),
        (lambda b, t: t.update(arm="baseline-prompted"), "arm"),
        (lambda b, t: t["episodes"].pop(), "same tasks"),
        (lambda b, t: t["sandbox"].update(allow_unconfined=True), "unconfined"),
    ],
)
def test_runs_that_differ_in_more_than_weights_are_refused(arms, mutate, message):
    baseline, trained = copy.deepcopy(arms)
    mutate(baseline, trained)
    with pytest.raises(Incomparable, match=message):
        check_comparable(baseline, trained)


def test_a_batch_size_difference_is_a_stated_caveat_not_a_refusal(arms):
    baseline, trained = copy.deepcopy(arms)
    baseline["policy"]["batch_size"] = 32
    caveats = check_comparable(baseline, trained)
    assert any("batch_size" in c for c in caveats)
    assert compare(baseline, trained)["caveats"] == caveats


def test_tool_use_separates_invented_names_and_chaining():
    record = _record(
        {("q0", "qa"): [0, 0], ("m0", "math"): [1, 0]},
        calls={
            ("q0", 0): ("search", "search"),
            ("q0", 1): ("calc",),
            ("m0", 0): ("calculator",),
        },
    )
    use = tool_use([EpisodeResult(**e) for e in record["episodes"]], ["calculator", "search"])

    assert use["invented_tool_calls"] == 1 and use["invented_tool_names"] == ["calc"]
    assert use["multi_hop_chain_rate"] == 0.5
    assert use["math"]["calculator_rate"] == 0.5
    assert use["math"]["reward_with_calculator"] == 1.0
    assert use["math"]["reward_without_calculator"] == 0.0


def test_unmatched_drift_is_reported_as_unmatched(arms):
    drift = {"reports": [{"source": "no/such/dir", "lost": 0, "gained": 0,
                          "by_kind": {}, "drift": {"mean_kl_per_token": 0.0}}]}
    report = compare(*arms, drift=drift)
    assert report["drift"]["matched_by"] is None
