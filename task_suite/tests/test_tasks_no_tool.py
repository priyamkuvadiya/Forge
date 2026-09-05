"""Tests for the no-tool control category.

The category only does its job if the tasks genuinely need no tool. If any of
them secretly did, a tool call on it would be correct behaviour and the
over-calling measurement built on top would be meaningless — so "answerable
without a tool" is asserted here rather than assumed from the prompts looking
easy.
"""

import pytest

from task_suite.qa_world import build_world
from task_suite.registry import load_suite
from task_suite.tasks_no_tool import TEMPLATES, generate_no_tool_tasks, pools_for
from task_suite.verifiers import verify

TRAIN = generate_no_tool_tasks(40, "train", 40_213)
HELDOUT = generate_no_tool_tasks(20, "heldout", 77_419)
ALL_TASKS = TRAIN + HELDOUT


def wrap(answer: str) -> str:
    return f"<answer>{answer}</answer>"


def test_generation_is_deterministic():
    assert [t.prompt for t in generate_no_tool_tasks(40, "train", 40_213)] == [
        t.prompt for t in TRAIN
    ]


def test_prompts_and_ids_are_unique():
    prompts = [t.prompt for t in ALL_TASKS]
    assert len(set(prompts)) == len(prompts)
    ids = [t.task_id for t in ALL_TASKS]
    assert len(set(ids)) == len(ids)


def test_the_splits_draw_on_disjoint_material():
    """Held-out has to mean the same thing here as in every other category.

    An earlier version shared one pool between splits and tripped
    `build_suite`'s duplicate-prompt guard on its first run.
    """
    train, heldout = pools_for("train"), pools_for("heldout")
    assert not set(train.operands) & set(heldout.operands)
    assert not set(train.people) & set(heldout.people)
    assert not set(train.vehicles) & set(heldout.vehicles)
    assert not set(train.items) & set(heldout.items)
    assert not set(train.colours) & set(heldout.colours)


def test_every_template_is_represented():
    used = {t.metadata["template"] for t in ALL_TASKS}
    assert len(used) == len(TEMPLATES)


def test_both_answer_types_are_present():
    """One kind tempts the calculator, the other tempts search.

    Over-calling is not a single behaviour, and a control made only of
    arithmetic would miss a policy that reflexively searches.
    """
    types = {t.metadata["answer_type"] for t in ALL_TASKS}
    assert types == {"number", "text"}


def test_every_task_is_marked_as_needing_no_tools():
    """The eval harness reads this to know where a tool call is over-calling."""
    for task in ALL_TASKS:
        assert task.metadata["tools"] == []


# --------------------------------------------------------------------------
# The property the category rests on: no tool is actually needed
# --------------------------------------------------------------------------

def test_arithmetic_is_small_enough_to_do_mentally():
    """If it needed a calculator, calling one would be right, not over-calling."""
    numeric = [t for t in ALL_TASKS if t.metadata["answer_type"] == "number"]
    assert numeric

    for task in numeric:
        assert task.ground_truth["value"] <= 144, task.prompt
        assert float(task.ground_truth["value"]).is_integer()
        # Exact, not tolerant: there is nothing here to round.
        assert task.ground_truth["tolerance"] == 0.0


def test_the_answer_to_a_comprehension_task_is_stated_in_its_prompt():
    """Nothing has to be retrieved, because it is already on the page."""
    textual = [t for t in ALL_TASKS if t.metadata["answer_type"] == "text"]
    assert textual

    for task in textual:
        accepted = task.ground_truth["answers"]
        assert any(a in task.prompt for a in accepted), task.prompt


def test_searching_for_these_tasks_would_find_nothing():
    """The control has to be a control.

    If the comprehension entities appeared in the QA corpus, a policy could
    reach for search and be rewarded for it, and the over-calling measurement
    would be measuring something else. These names are invented here and kept
    out of the corpus, so a search returns nothing useful.
    """
    corpus = " ".join(d.text + " " + d.title for d in build_world().documents)

    for task in (t for t in ALL_TASKS if t.metadata["answer_type"] == "text"):
        for answer in task.ground_truth["answers"]:
            assert answer not in corpus, f"{answer!r} leaks into the search corpus"


# --------------------------------------------------------------------------
# Scoring, through the real verifier
# --------------------------------------------------------------------------

def test_ground_truth_scores_one():
    for task in ALL_TASKS:
        if task.metadata["answer_type"] == "number":
            value = task.ground_truth["value"]
            assert verify(task, wrap(str(int(value)))) == 1.0, task.prompt
        else:
            for accepted in task.ground_truth["answers"]:
                assert verify(task, wrap(accepted)) == 1.0, task.prompt


def test_a_wrong_answer_scores_zero():
    for task in ALL_TASKS:
        assert verify(task, wrap("Wexford Landing lighthouse keeper")) == 0.0
        assert verify(task, "no answer block at all") == 0.0


def test_the_pinned_suite_carries_the_control_category():
    splits = load_suite()
    counts = {
        name: sum(1 for t in tasks if t.category == "no_tool")
        for name, tasks in splits.items()
    }
    assert counts == {"train": 40, "heldout": 20}


@pytest.mark.parametrize("split", ["train", "heldout"])
def test_split_is_non_trivially_sized(split):
    assert len(generate_no_tool_tasks(20, split, 1)) == 20
