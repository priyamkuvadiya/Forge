"""Group-relative advantages, and the thing they cannot do.

GRPO's one idea is that you do not need a value network to know whether a
rollout was good: sample a *group* of rollouts from the same prompt, and score
each one against its own group's mean. The baseline is the group, which is why
there is no critic to train and why this fits on an 8GB card at all.

The consequence nobody advertises is that **a group whose rollouts all scored
the same carries no gradient whatsoever**. Every advantage in it is zero, the
prompt contributes nothing to the update, and the reward curve cannot tell you
it happened - a run where 90% of groups are degenerate looks exactly like a run
that is learning slowly. On a suite where module 4 measured held-out math at
0.021 and QA at 0.000, that is not a corner case, it is the expected case: a
group of eight rollouts that all score zero is eight forward passes bought and
thrown away.

So this module counts them. `GroupStats.degenerate_fraction` is meant to be
logged next to the reward every step, and `expected_signal_rate()` exists to be
run *before* training against module 4's committed per-episode rewards, so the
question "will this suite produce a gradient at all" is answered by measurement
rather than after a wasted afternoon of GPU.

Kept deliberately free of torch. It is arithmetic over small lists, it is the
part most likely to be quietly wrong, and it runs in CI on a machine with no
torch installed - which is the only reason it is checked on every push rather
than only on this laptop.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

# Guards the division when a group's rollouts differ, but only barely. Small
# enough not to shrink real advantages, large enough that a group with rewards
# differing in the 8th decimal does not produce an advantage of 1e7.
STD_EPSILON = 1e-4


@dataclass(frozen=True)
class GroupStats:
    """What a batch of groups looked like, for the step log."""

    n_groups: int
    n_degenerate: int
    mean_reward: float
    mean_abs_advantage: float

    @property
    def degenerate_fraction(self) -> float:
        return self.n_degenerate / self.n_groups if self.n_groups else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_groups": self.n_groups,
            "n_degenerate": self.n_degenerate,
            "degenerate_fraction": round(self.degenerate_fraction, 4),
            "mean_reward": round(self.mean_reward, 4),
            "mean_abs_advantage": round(self.mean_abs_advantage, 4),
        }


def _standard_deviation(values: Sequence[float], mean: float) -> float:
    """Population standard deviation.

    Population, not sample: the group *is* the population here - it is the
    complete set of rollouts drawn for this prompt, not a sample from some
    larger set of them - and dividing by `n - 1` would inflate advantages by
    ~7% at the group size of 8 this project uses, for no reason beyond habit.
    """
    if len(values) < 2:
        return 0.0
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def group_advantages(
    rewards: Sequence[float],
    group_ids: Sequence[Any],
    *,
    normalize_by_std: bool = True,
    std_epsilon: float = STD_EPSILON,
) -> tuple[list[float], GroupStats]:
    """Centre each reward on its group, and say how many groups were dead.

    `normalize_by_std=False` gives the Dr. GRPO variant, which centres but does
    not scale. The scaling is not free: dividing by a near-zero standard
    deviation turns a group where seven rollouts scored 0.0 and one scored 0.05
    into advantages of the same magnitude as a group that genuinely split, so
    the noisiest groups shout loudest. It is left on by default because it is
    what GRPO specifies and what the baseline comparison should be against, and
    exposed because on this reward distribution it is a real suspect.
    """
    if len(rewards) != len(group_ids):
        raise ValueError(
            f"{len(rewards)} rewards for {len(group_ids)} group ids"
        )

    members: dict[Any, list[int]] = defaultdict(list)
    for index, group in enumerate(group_ids):
        members[group].append(index)

    advantages = [0.0] * len(rewards)
    degenerate = 0

    for indices in members.values():
        group_rewards = [rewards[i] for i in indices]
        mean = sum(group_rewards) / len(group_rewards)
        deviation = _standard_deviation(group_rewards, mean)

        # A group is degenerate when every rollout scored identically - which
        # includes the single-rollout case, where the group mean *is* the
        # rollout and centring annihilates it. Counting it as degenerate is
        # the honest reading: group size 1 gives GRPO nothing to compare.
        if deviation == 0.0:
            degenerate += 1
            continue

        scale = (deviation + std_epsilon) if normalize_by_std else 1.0
        for index in indices:
            advantages[index] = (rewards[index] - mean) / scale

    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    mean_abs = (
        sum(abs(value) for value in advantages) / len(advantages) if advantages else 0.0
    )
    stats = GroupStats(
        n_groups=len(members),
        n_degenerate=degenerate,
        mean_reward=mean_reward,
        mean_abs_advantage=mean_abs,
    )
    return advantages, stats


@dataclass(frozen=True)
class SignalEstimate:
    """How much of a reward table would survive being turned into advantages."""

    n_groups: int
    n_degenerate: int
    n_all_zero: int
    n_all_max: int
    mean_reward: float
    per_category: dict[str, dict[str, Any]]

    @property
    def signal_fraction(self) -> float:
        return 1.0 - (self.n_degenerate / self.n_groups if self.n_groups else 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_groups": self.n_groups,
            "n_degenerate": self.n_degenerate,
            "n_all_zero": self.n_all_zero,
            "n_all_max": self.n_all_max,
            "signal_fraction": round(self.signal_fraction, 4),
            "mean_reward": round(self.mean_reward, 4),
            "per_category": self.per_category,
        }


def expected_signal_rate(
    episodes: Iterable[dict[str, Any]],
    *,
    reward_key: str = "reward",
    group_key: str = "task_id",
    category_key: str = "category",
) -> SignalEstimate:
    """Given per-episode rewards, how many prompts would produce a gradient?

    Takes the per-episode dicts module 4 writes, so this can be pointed
    straight at `artifacts/baseline/heldout.json` and answer the question
    before a single training step is run. A prompt whose rollouts all score
    zero and a prompt whose rollouts all score full marks are both dead, and
    they are counted separately because they mean opposite things: the first is
    a task the policy cannot start, the second is one it has already finished.
    """
    grouped: dict[Any, list[float]] = defaultdict(list)
    categories: dict[Any, str] = {}
    for episode in episodes:
        key = episode[group_key]
        grouped[key].append(float(episode[reward_key]))
        categories[key] = episode.get(category_key, "unknown")

    per_category: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"groups": 0, "degenerate": 0, "all_zero": 0, "all_max": 0}
    )
    degenerate = all_zero = all_max = 0
    total_reward = 0.0
    total_episodes = 0

    for key, group_rewards in grouped.items():
        category = categories[key]
        bucket = per_category[category]
        bucket["groups"] += 1
        total_reward += sum(group_rewards)
        total_episodes += len(group_rewards)

        mean = sum(group_rewards) / len(group_rewards)
        if _standard_deviation(group_rewards, mean) == 0.0:
            degenerate += 1
            bucket["degenerate"] += 1
            if mean == 0.0:
                all_zero += 1
                bucket["all_zero"] += 1
            elif mean == 1.0:
                all_max += 1
                bucket["all_max"] += 1

    for bucket in per_category.values():
        bucket["signal_fraction"] = round(
            1.0 - bucket["degenerate"] / bucket["groups"], 4
        )

    return SignalEstimate(
        n_groups=len(grouped),
        n_degenerate=degenerate,
        n_all_zero=all_zero,
        n_all_max=all_max,
        mean_reward=total_reward / total_episodes if total_episodes else 0.0,
        per_category=dict(per_category),
    )
