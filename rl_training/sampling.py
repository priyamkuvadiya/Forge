"""Which prompts a training step spends its rollouts on.

The train split is not balanced and cannot be made balanced without changing
the suite, which is frozen: 340 tasks as math 150, qa 85, no_tool 40,
multi_tool 35, code 30. Sampling prompts uniformly would make math 44% of every
batch and code 9%, so the policy would spend nearly half its gradient on the
category module 4 measured at 0.021 and a twelfth of it on the category it is
already best at. Module 2 settled this in advance: balance by **sampling**, not
by trimming the pool, so no task is thrown away and the imbalance costs nothing
permanent.

So a batch is built by choosing categories round-robin and drawing a task from
each category's own shuffled queue. Each queue is refilled when it empties,
which means the small categories are seen far more often than the large ones -
code's 30 tasks cycle roughly five times for each pass through math's 150 - and
that is the intended trade, not an accident: a category with fewer tasks needs
to be revisited more often to contribute equally to the gradient.

One thing this module deliberately does *not* do. Module 5's signal measurement
found that at the baseline's skill most groups are degenerate - every rollout
in them scores identically, so they produce no gradient at all - and it is
tempting to bias sampling towards prompts that have produced signal before.
That is a real technique and it is also a way to quietly train on an easier
distribution than the one being reported. If it gets used it should be a stated
experiment with its own number, not a default hidden in the sampler, so the
hook for it is `weights=` and the default is uniform across categories.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


@dataclass
class CategorySampler:
    """Draws category-balanced batches of tasks, reproducibly.

    Deterministic given `seed`: the same seed replays the same task order, so a
    training run is reproducible and two runs being compared can be given the
    same prompt stream. That matters more here than usual - with most groups
    degenerate, which prompts a run happened to draw is a large part of how
    much signal it got, and a comparison across two different prompt streams
    would be measuring the draw as much as the change.
    """

    tasks: Sequence[Any]
    seed: int = 0
    weights: dict[str, float] | None = None
    category_of: Any = None

    _queues: dict[str, list[Any]] = field(default_factory=dict, init=False)
    _order: list[str] = field(default_factory=list, init=False)
    _cursor: int = field(default=0, init=False)
    _drawn: Counter = field(default_factory=Counter, init=False)

    def __post_init__(self) -> None:
        if not self.tasks:
            raise ValueError("no tasks to sample from")

        self._random = random.Random(self.seed)
        self._by_category: dict[str, list[Any]] = defaultdict(list)
        for task in self.tasks:
            self._by_category[self._category(task)].append(task)

        self._order = sorted(self._by_category)
        if self.weights:
            unknown = set(self.weights) - set(self._order)
            if unknown:
                raise ValueError(f"weights name categories not present: {sorted(unknown)}")
            # An integer repeat count per category, so a weight of 2.0 simply
            # means the category's slot comes up twice per round. Cruder than
            # a probability, and reproducible without a second RNG stream.
            expanded: list[str] = []
            for category in self._order:
                repeats = max(1, round(self.weights.get(category, 1.0)))
                expanded.extend([category] * repeats)
            self._order = expanded

        for category in set(self._order):
            self._refill(category)

    def _category(self, task: Any) -> str:
        if self.category_of is not None:
            return self.category_of(task)
        return getattr(task, "category", None) or task["category"]

    def _refill(self, category: str) -> None:
        pool = list(self._by_category[category])
        self._random.shuffle(pool)
        self._queues[category] = pool

    @property
    def categories(self) -> list[str]:
        return sorted(self._by_category)

    @property
    def drawn(self) -> dict[str, int]:
        return dict(self._drawn)

    def next_task(self) -> Any:
        category = self._order[self._cursor % len(self._order)]
        self._cursor += 1
        if not self._queues[category]:
            self._refill(category)
        task = self._queues[category].pop()
        self._drawn[category] += 1
        return task

    def batch(self, size: int) -> list[Any]:
        """`size` tasks, spread across categories as evenly as `size` allows.

        A batch smaller than the number of categories cannot contain them all;
        successive batches continue the round-robin rather than restarting it,
        so the balance holds across the run even when it cannot hold within a
        single batch.
        """
        if size <= 0:
            raise ValueError(f"batch size must be positive, got {size}")
        return [self.next_task() for _ in range(size)]


def category_counts(tasks: Iterable[Any], category_of: Any = None) -> dict[str, int]:
    """What a task list looks like per category, for a step log or a header."""
    counts: Counter = Counter()
    for task in tasks:
        if category_of is not None:
            counts[category_of(task)] += 1
        else:
            counts[getattr(task, "category", None) or task["category"]] += 1
    return dict(sorted(counts.items()))
