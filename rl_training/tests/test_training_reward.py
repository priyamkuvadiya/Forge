"""run2's training reward: a no-search penalty on qa and multi_tool only.

The penalty changes what the policy is pushed towards, so every boundary of it
is asserted: which categories, which episodes, what counts as a search, and
that with the penalty off the training reward *is* verify()'s reward - the
property that keeps run1 reproducible from the same code.
"""

from types import SimpleNamespace

import pytest

from rl_training.train_grpo import SEARCH_REQUIRED, training_rewards

PENALTY = 0.25


def result(category="qa", reward=0.0, answer="Mercury", calls=()):
    return SimpleNamespace(
        category=category,
        reward=reward,
        answer=answer,
        tool_calls=[{"tool": tool, "ok": ok} for tool, ok in calls],
    )


def test_the_penalty_targets_exactly_the_search_categories():
    assert SEARCH_REQUIRED == {"qa", "multi_tool"}


@pytest.mark.parametrize("category", ["qa", "multi_tool"])
def test_a_wrong_answer_without_search_is_penalised(category):
    assert training_rewards([result(category)], PENALTY) == [-PENALTY]


@pytest.mark.parametrize("category", ["math", "code", "no_tool"])
def test_other_categories_are_untouched(category):
    assert training_rewards([result(category)], PENALTY) == [0.0]


def test_a_successful_search_avoids_the_penalty():
    searched = result(calls=[("search", True)])
    assert training_rewards([searched], PENALTY) == [0.0]


def test_a_failed_search_does_not_count_as_searching():
    # Otherwise an empty or malformed call is a free dodge.
    assert training_rewards([result(calls=[("search", False)])], PENALTY) == [-PENALTY]


def test_a_different_tool_does_not_count_as_searching():
    calc = result("multi_tool", calls=[("calculator", True)])
    assert training_rewards([calc], PENALTY) == [-PENALTY]


def test_declining_to_answer_is_not_penalised():
    assert training_rewards([result(answer=None)], PENALTY) == [0.0]


def test_a_correct_answer_keeps_its_reward_even_without_search():
    assert training_rewards([result(reward=1.0)], PENALTY) == [1.0]


def test_with_no_penalty_it_is_exactly_the_verify_reward():
    results = [
        result("qa"),
        result("multi_tool", calls=[("calculator", True)]),
        result("math", reward=1.0),
        result("code", reward=0.4),
        result("no_tool", answer=None),
    ]
    assert training_rewards(results) == [r.reward for r in results]


def test_the_penalty_makes_an_all_zero_qa_group_usable():
    # The point of run2: a qa group that all scored 0 under run1 had no
    # gradient. Searching vs not searching now separates it.
    group = [result(calls=[("search", True)])] * 2 + [result()] * 6
    rewards = training_rewards(group, PENALTY)
    assert len(set(rewards)) == 2


def test_a_negative_penalty_is_refused():
    with pytest.raises(ValueError):
        training_rewards([result()], -0.25)
