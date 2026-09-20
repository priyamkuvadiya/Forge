"""Group-relative advantages, and the accounting for groups that give nothing.

No torch here on purpose - this is the arithmetic that decides how large a step
every rollout takes, it runs in CI on a machine with no torch, and it is small
enough to check against numbers worked out by hand rather than against another
implementation of itself.
"""

import math

import pytest

from rl_training.advantages import (
    STD_EPSILON,
    expected_signal_rate,
    group_advantages,
)


def test_a_group_that_agrees_produces_no_gradient():
    """The failure mode the module exists to count."""
    advantages, stats = group_advantages([0.0, 0.0, 0.0, 0.0], ["t1"] * 4)

    assert advantages == [0.0, 0.0, 0.0, 0.0]
    assert stats.n_degenerate == 1
    assert stats.degenerate_fraction == 1.0


def test_a_group_at_full_marks_is_equally_dead():
    """All-correct is as useless to GRPO as all-wrong, and easy to forget."""
    advantages, stats = group_advantages([1.0] * 8, ["t1"] * 8)

    assert advantages == [0.0] * 8
    assert stats.n_degenerate == 1
    assert stats.mean_reward == 1.0


def test_advantages_are_centred_within_their_group():
    rewards = [0.0, 1.0, 0.0, 1.0]
    advantages, _ = group_advantages(rewards, ["t1", "t1", "t1", "t1"])

    assert math.isclose(sum(advantages), 0.0, abs_tol=1e-9)
    assert advantages[1] > 0 > advantages[0]


def test_groups_do_not_leak_into_each_other():
    """t2 is uniformly excellent and must not lift t1's losing rollout."""
    rewards = [0.0, 1.0, 1.0, 1.0]
    groups = ["t1", "t1", "t2", "t2"]
    advantages, stats = group_advantages(rewards, groups)

    assert advantages[2] == 0.0 and advantages[3] == 0.0, "t2 agrees with itself"
    assert advantages[0] < 0 < advantages[1]
    assert stats.n_groups == 2
    assert stats.n_degenerate == 1


def test_scaling_uses_the_population_standard_deviation():
    """Worked by hand: rewards 0 and 1 have population sd 0.5, not 0.707."""
    advantages, _ = group_advantages([0.0, 1.0], ["t1", "t1"])

    expected = 0.5 / (0.5 + STD_EPSILON)
    assert math.isclose(advantages[1], expected, rel_tol=1e-9)
    assert math.isclose(advantages[0], -expected, rel_tol=1e-9)


def test_unnormalized_variant_centres_without_scaling():
    advantages, _ = group_advantages(
        [0.0, 1.0], ["t1", "t1"], normalize_by_std=False
    )

    assert advantages == [-0.5, 0.5]


def test_epsilon_stops_a_near_tie_from_exploding():
    """Seven zeros and one 0.05 must not produce a unit-scale advantage."""
    rewards = [0.0] * 7 + [0.05]
    advantages, _ = group_advantages(rewards, ["t1"] * 8)

    assert max(abs(a) for a in advantages) < 3.0


def test_a_single_rollout_group_is_degenerate_not_a_division_by_zero():
    advantages, stats = group_advantages([0.7], ["t1"])

    assert advantages == [0.0]
    assert stats.n_degenerate == 1


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError, match="rewards for"):
        group_advantages([0.0, 1.0], ["t1"])


def test_empty_input_is_not_a_crash():
    advantages, stats = group_advantages([], [])
    assert advantages == []
    assert stats.n_groups == 0
    assert stats.degenerate_fraction == 0.0


def test_dense_code_rewards_are_not_flattened_to_binary():
    """Coding scores fractions of tests passed; partial credit must survive."""
    rewards = [0.25, 0.5, 0.75, 1.0]
    advantages, stats = group_advantages(rewards, ["code-1"] * 4)

    assert stats.n_degenerate == 0
    assert advantages[0] < advantages[1] < advantages[2] < advantages[3]


def episodes(*specs):
    out = []
    for task_id, category, rewards in specs:
        for rollout, reward in enumerate(rewards):
            out.append(
                {
                    "task_id": task_id,
                    "category": category,
                    "rollout": rollout,
                    "reward": reward,
                }
            )
    return out


def test_signal_estimate_separates_never_solved_from_always_solved():
    estimate = expected_signal_rate(
        episodes(
            ("qa-1", "qa", [0.0, 0.0, 0.0, 0.0]),
            ("code-1", "code", [1.0, 1.0, 1.0, 1.0]),
            ("math-1", "math", [0.0, 1.0, 0.0, 0.0]),
        )
    )

    assert estimate.n_groups == 3
    assert estimate.n_degenerate == 2
    assert estimate.n_all_zero == 1
    assert estimate.n_all_max == 1
    assert math.isclose(estimate.signal_fraction, 1 / 3)


def test_signal_estimate_reports_per_category():
    estimate = expected_signal_rate(
        episodes(
            ("qa-1", "qa", [0.0, 0.0]),
            ("qa-2", "qa", [0.0, 0.0]),
            ("math-1", "math", [0.0, 1.0]),
        )
    )

    assert estimate.per_category["qa"]["signal_fraction"] == 0.0
    assert estimate.per_category["math"]["signal_fraction"] == 1.0


def test_a_partially_solved_group_is_not_counted_as_all_zero():
    """Guards the branch order: degenerate is checked before all_zero."""
    estimate = expected_signal_rate(episodes(("math-1", "math", [0.0, 0.2])))

    assert estimate.n_degenerate == 0
    assert estimate.n_all_zero == 0
    assert estimate.signal_fraction == 1.0
