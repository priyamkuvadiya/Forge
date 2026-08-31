"""Checks on the cross-tool category.

Two of these carry the category's whole justification. If the answer is
readable straight out of a retrieved document then no arithmetic was
required and the task is plain QA wearing a number; if the facts were not in
the corpus then no retrieval was required and the task is plain arithmetic.
Both are asserted rather than assumed.
"""

import pytest

from task_suite.qa_world import build_world, split_subjects
from task_suite.tasks_multi_tool import (
    PHRASINGS,
    answer_is_written_down,
    build_problems,
    generate_multi_tool_tasks,
)
from task_suite.verifiers import verify

WORLD = build_world()
TRAIN = generate_multi_tool_tasks("train", WORLD)
HELDOUT = generate_multi_tool_tasks("heldout", WORLD)
ALL_TASKS = TRAIN + HELDOUT
ALL_PROBLEMS = build_problems(WORLD, "train") + build_problems(WORLD, "heldout")


def wrap(answer: str) -> str:
    return f"<answer>{answer}</answer>"


def test_generation_is_deterministic():
    again = generate_multi_tool_tasks("train", WORLD)
    assert [t.to_dict() for t in again] == [t.to_dict() for t in TRAIN]


def test_both_splits_are_populated():
    assert len(TRAIN) >= 20
    assert len(HELDOUT) >= 8


def test_no_answer_is_readable_from_a_supporting_document():
    """Otherwise the arithmetic can be skipped and the task is not cross-tool."""
    for problem in ALL_PROBLEMS:
        assert not answer_is_written_down(WORLD, problem), problem.text


def test_every_fact_needed_lives_in_the_corpus():
    """Otherwise no retrieval is required and the task is plain arithmetic."""
    doc_ids = {d.doc_id for d in WORLD.documents}
    for task in ALL_TASKS:
        supporting = task.metadata["supporting_docs"]
        assert supporting
        assert set(supporting) <= doc_ids


def test_tasks_declare_both_tools():
    for task in ALL_TASKS:
        assert task.metadata["tools"] == ["search", "calculator"]


def test_ground_truth_scores_one_through_the_real_verifier():
    for task in ALL_TASKS:
        assert verify(task, wrap(str(task.ground_truth["value"]))) == 1.0


def test_a_wrong_answer_scores_zero():
    for task in ALL_TASKS:
        assert verify(task, wrap(str(task.ground_truth["value"] + 3))) == 0.0


def test_answers_are_positive():
    """Differences are only emitted one way round, so no task turns on guessing
    which direction a subtraction was meant to go."""
    for task in ALL_TASKS:
        assert task.ground_truth["value"] > 0


def test_every_phrasing_variant_is_used():
    used_by_template: dict[str, set[str]] = {name: set() for name in PHRASINGS}
    for problem in ALL_PROBLEMS:
        used_by_template[problem.template].add(problem.text)
    for template, forms in PHRASINGS.items():
        assert len(used_by_template[template]) >= min(2, len(forms)), template


def test_train_and_heldout_subjects_are_disjoint():
    train_subjects = {t.metadata["subject"] for t in TRAIN}
    heldout_subjects = {t.metadata["subject"] for t in HELDOUT}
    assert train_subjects.isdisjoint(heldout_subjects)


def test_prompts_are_unique():
    prompts = [t.prompt for t in ALL_TASKS]
    assert len(set(prompts)) == len(prompts)


@pytest.mark.parametrize("split", ["train", "heldout"])
def test_only_that_splits_entities_are_asked_about(split):
    expeditions, _, stations = split_subjects(WORLD, split)
    allowed = {e.name for e in expeditions} | {s.name for s in stations}
    for task in generate_multi_tool_tasks(split, WORLD):
        for part in task.metadata["subject"].split(" + "):
            assert part in allowed
