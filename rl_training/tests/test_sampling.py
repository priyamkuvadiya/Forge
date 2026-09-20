"""Category balance, reproducibility, and that no task is quietly dropped.

Driven partly by hand-built stubs and partly by the real frozen suite, because
the numbers that matter here - math is 44% of the train split, code is 9% - are
properties of that suite and a stub cannot notice if they change.
"""

import pytest

from rl_training.sampling import CategorySampler, category_counts


class Stub:
    def __init__(self, task_id, category):
        self.task_id = task_id
        self.category = category

    def __repr__(self):
        return f"Stub({self.task_id})"


def suite(**counts):
    return [
        Stub(f"{category}-{index}", category)
        for category, total in counts.items()
        for index in range(total)
    ]


def test_an_imbalanced_pool_produces_a_balanced_draw():
    """The whole point: 100 math and 10 code must not draw 10:1."""
    sampler = CategorySampler(suite(math=100, code=10), seed=0)

    drawn = category_counts(sampler.batch(200))

    assert drawn["math"] == drawn["code"] == 100


def test_small_categories_are_revisited_more_often():
    """The intended trade, stated as a test so it cannot drift silently."""
    sampler = CategorySampler(suite(math=100, code=10), seed=0)
    sampler.batch(200)

    # 100 draws from a 10-task pool means every code task was seen ~10 times.
    assert sampler.drawn["code"] == 100


def test_every_task_is_reachable():
    """A shuffled queue that never refills would silently train on a subset."""
    sampler = CategorySampler(suite(math=5, code=3), seed=1)

    seen = {task.task_id for task in sampler.batch(200)}

    assert len(seen) == 8


def test_a_queue_is_exhausted_before_it_repeats():
    """Sampling without replacement within a pass, so no task is drawn twice
    while another has not been drawn at all."""
    sampler = CategorySampler(suite(code=4), seed=2)

    first_pass = [task.task_id for task in sampler.batch(4)]

    assert len(set(first_pass)) == 4


def test_the_same_seed_replays_the_same_stream():
    tasks = suite(math=10, code=10, qa=10)
    first = [task.task_id for task in CategorySampler(tasks, seed=7).batch(30)]
    second = [task.task_id for task in CategorySampler(tasks, seed=7).batch(30)]

    assert first == second


def test_a_different_seed_gives_a_different_stream():
    tasks = suite(math=10, code=10, qa=10)
    first = [task.task_id for task in CategorySampler(tasks, seed=7).batch(30)]
    other = [task.task_id for task in CategorySampler(tasks, seed=8).batch(30)]

    assert first != other


def test_balance_holds_across_batches_that_are_too_small_to_hold_it():
    """Five categories and a batch of two: balance has to carry over."""
    sampler = CategorySampler(suite(a=9, b=9, c=9, d=9, e=9), seed=0)

    drawn = category_counts([task for _ in range(10) for task in sampler.batch(2)])

    assert set(drawn) == {"a", "b", "c", "d", "e"}
    assert max(drawn.values()) - min(drawn.values()) <= 1


def test_weights_shift_the_balance_predictably():
    sampler = CategorySampler(
        suite(math=50, code=50), seed=0, weights={"math": 1.0, "code": 3.0}
    )

    drawn = category_counts(sampler.batch(80))

    assert drawn["code"] == 3 * drawn["math"]


def test_weights_naming_an_absent_category_raise():
    with pytest.raises(ValueError, match="not present"):
        CategorySampler(suite(math=5), seed=0, weights={"qa": 2.0})


def test_an_empty_pool_raises_rather_than_looping_forever():
    with pytest.raises(ValueError, match="no tasks"):
        CategorySampler([], seed=0)


def test_batch_size_must_be_positive():
    with pytest.raises(ValueError, match="must be positive"):
        CategorySampler(suite(math=2), seed=0).batch(0)


def test_dict_tasks_work_too():
    """The eval harness passes dicts; the trainer passes Task objects."""
    tasks = [{"task_id": "a", "category": "math"}, {"task_id": "b", "category": "code"}]
    sampler = CategorySampler(tasks, seed=0)

    assert set(category_counts(sampler.batch(10))) == {"math", "code"}


def test_against_the_real_frozen_suite():
    """Guards the actual imbalance this module was written for."""
    from task_suite.registry import load_suite

    train = load_suite()["train"]
    counts = category_counts(train)

    assert counts == {
        "code": 30,
        "math": 150,
        "multi_tool": 35,
        "no_tool": 40,
        "qa": 85,
    }, "the suite is frozen; if this changed, the baseline is invalid"

    sampler = CategorySampler(train, seed=0)
    drawn = category_counts(sampler.batch(250))

    assert max(drawn.values()) - min(drawn.values()) <= 1
    assert drawn["code"] == drawn["math"], "50 tasks of math, 50 of code, from 150 and 30"
