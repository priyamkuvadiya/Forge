"""Checks on the synthetic QA world and the questions built over it.

The important test here is `test_no_single_document_answers_a_question`: it
asserts, for every generated question, that no one document in the corpus
contains both the thing the question names and the answer it wants. That is
the property that makes these questions multi-hop, and it is a property of
the rendered prose, not of the intent — one careless filler sentence would
quietly turn a two-hop question into a lookup while the question text stayed
exactly the same.
"""

import pytest

from task_suite.qa_world import (
    CAPTAIN_NAMES,
    SURNAMES,
    build_questions,
    build_world,
    generate_qa_tasks,
)
from task_suite.verifiers import verify

WORLD = build_world()
TRAIN = generate_qa_tasks("train", WORLD)
HELDOUT = generate_qa_tasks("heldout", WORLD)
ALL_TASKS = TRAIN + HELDOUT
ALL_QUESTIONS = build_questions(WORLD, "train") + build_questions(WORLD, "heldout")


def wrap(answer: str) -> str:
    return f"<answer>{answer}</answer>"


def test_world_is_deterministic():
    again = build_world()
    assert [d.text for d in again.documents] == [d.text for d in WORLD.documents]


def test_document_ids_are_unique_and_cover_every_entity():
    ids = [d.doc_id for d in WORLD.documents]
    assert len(set(ids)) == len(ids)
    expected = (
        len(WORLD.instruments) + len(WORLD.researchers) + len(WORLD.stations)
        + len(WORLD.vessels) + len(WORLD.expeditions)
    )
    assert len(ids) == expected


def test_people_surnames_are_globally_unique():
    """A surname is an accepted alias, so two people sharing one would make an
    answer ambiguous and the reward wrong rather than merely strict."""
    surnames = list(SURNAMES) + [name.split()[-1] for name in CAPTAIN_NAMES]
    assert len(set(surnames)) == len(surnames)


def test_supporting_documents_exist():
    for task in ALL_TASKS:
        for doc_id in task.ground_truth["supporting_docs"]:
            assert WORLD.document(doc_id) is not None


def test_every_answer_is_supported_by_the_corpus():
    """An unanswerable question silently caps the reward this category can earn."""
    for task in ALL_TASKS:
        combined = " ".join(
            WORLD.document(d).text for d in task.ground_truth["supporting_docs"]
        )
        assert any(a in combined for a in task.ground_truth["answers"]), task.prompt


def test_no_single_document_answers_a_question():
    """The multi-hop property, checked against the rendered prose."""
    for question in ALL_QUESTIONS:
        # guards the check below against passing vacuously
        assert any(question.subject in d.text for d in WORLD.documents), question.subject
        for doc in WORLD.documents:
            if question.subject not in doc.text:
                continue
            leaked = [a for a in question.answers if a in doc.text]
            assert not leaked, f"{doc.doc_id} answers {question.text!r} on its own: {leaked}"


def test_hop_count_matches_the_supporting_chain():
    for question in ALL_QUESTIONS:
        assert question.hops == len(question.supporting_docs)
        assert question.hops >= 2


def test_train_and_heldout_subjects_are_disjoint():
    train_subjects = {t.metadata["subject"] for t in TRAIN}
    heldout_subjects = {t.metadata["subject"] for t in HELDOUT}
    assert train_subjects and heldout_subjects
    assert train_subjects.isdisjoint(heldout_subjects)


def test_prompts_and_ids_are_unique():
    prompts = [t.prompt for t in ALL_TASKS]
    assert len(set(prompts)) == len(prompts)
    ids = [t.task_id for t in ALL_TASKS]
    assert len(set(ids)) == len(ids)


def test_every_template_appears_in_both_splits():
    train_templates = {t.metadata["template"] for t in TRAIN}
    heldout_templates = {t.metadata["template"] for t in HELDOUT}
    assert train_templates == heldout_templates


def test_ground_truth_scores_one_through_the_real_verifier():
    for task in ALL_TASKS:
        for accepted in task.ground_truth["answers"]:
            assert verify(task, wrap(accepted)) == 1.0


def test_a_wrong_answer_scores_zero():
    for task in ALL_TASKS:
        assert verify(task, wrap("Wexford Landing lighthouse keeper")) == 0.0
        assert verify(task, "no answer block at all") == 0.0


@pytest.mark.parametrize("split", ["train", "heldout"])
def test_split_is_non_trivially_sized(split):
    assert len(generate_qa_tasks(split, WORLD)) >= 20
