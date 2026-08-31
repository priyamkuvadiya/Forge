"""Checks on the assembled suite and the file it is pinned to."""

import json

from task_suite.registry import (
    CORPUS_PATH,
    SUITE_PATH,
    build_suite,
    load_corpus,
    load_suite,
    save_suite,
)
from task_suite.verifiers import verify

SPLITS = build_suite()


def wrap(answer: str) -> str:
    return f"<answer>{answer}</answer>"


def test_both_splits_are_populated_across_all_three_categories():
    for name, tasks in SPLITS.items():
        categories = {t.category for t in tasks}
        assert categories == {"math", "code", "qa"}, name
        assert len(tasks) >= 100, name


def test_no_prompt_appears_in_both_splits():
    train = {t.prompt for t in SPLITS["train"]}
    heldout = {t.prompt for t in SPLITS["heldout"]}
    assert train.isdisjoint(heldout)


def test_task_ids_are_unique_across_the_whole_suite():
    ids = [t.task_id for split in SPLITS.values() for t in split]
    assert len(set(ids)) == len(ids)


def test_every_task_declares_its_own_split():
    for name, tasks in SPLITS.items():
        assert all(t.split == name for t in tasks)


def test_suite_round_trips_through_json(tmp_path):
    path = save_suite(SPLITS, tmp_path / "suite.json")
    restored = load_suite(path)
    assert {k: [t.to_dict() for t in v] for k, v in restored.items()} == {
        k: [t.to_dict() for t in v] for k, v in SPLITS.items()
    }


def test_the_checked_in_suite_matches_a_fresh_build():
    """A stale pinned file means reported eval numbers describe different tasks."""
    pinned = load_suite(SUITE_PATH)
    assert {k: [t.to_dict() for t in v] for k, v in pinned.items()} == {
        k: [t.to_dict() for t in v] for k, v in SPLITS.items()
    }


def test_the_checked_in_corpus_covers_every_supporting_document():
    corpus_ids = {d["doc_id"] for d in load_corpus(CORPUS_PATH)}
    for split in SPLITS.values():
        for task in split:
            if task.category != "qa":
                continue
            assert set(task.ground_truth["supporting_docs"]) <= corpus_ids


def test_non_code_ground_truth_all_scores_one():
    """One sweep over the whole pinned suite, through the real dispatch."""
    for split in SPLITS.values():
        for task in split:
            if task.category == "math":
                assert verify(task, wrap(str(task.ground_truth["value"]))) == 1.0
            elif task.category == "qa":
                assert verify(task, wrap(task.ground_truth["answers"][0])) == 1.0


def test_pinned_file_records_how_it_was_built():
    payload = json.loads(SUITE_PATH.read_text(encoding="utf-8"))
    config = payload["config"]
    assert config["math_train_seed"] != config["math_heldout_seed"]
    assert config["world_seed"]
