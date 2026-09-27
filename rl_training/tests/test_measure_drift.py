"""The drift probe's scoring, which is the part that can be quietly wrong.

A checker that passes everything would report "no capability lost" for any
adapter, which is the one conclusion this module exists to test rather than
assume. So every checker is exercised on an answer it must accept and one it
must reject. Torch-free: runs in both CI jobs.
"""

import pytest

from rl_training.measure_drift import (
    INSTRUCTION_CHECKS,
    build_probes,
    compare,
    score_probe,
)


def _probe(probe_id):
    return next(p for p in build_probes() if p["id"] == probe_id)


def test_probe_ids_are_unique_and_every_check_exists():
    probes = build_probes()
    assert len({p["id"] for p in probes}) == len(probes) == 55
    for probe in probes:
        if probe["kind"] == "instruction":
            assert probe["check"] in INSTRUCTION_CHECKS


@pytest.mark.parametrize(
    "probe_id, good, bad",
    [
        ("knowledge-00", "The capital of France is Paris.", "It is Lyon."),
        ("knowledge-02", "Its symbol is Au.", "Its symbol is Ag."),
        # Whole-word matching: "au" inside another word is not the answer.
        ("knowledge-02", "Au", "The symbol is Gaul."),
        ("knowledge-06", "There are seven continents.", "There are six."),
        ("knowledge-12", "A spider has 8 legs.", "A spider has 6 legs."),
    ],
)
def test_knowledge(probe_id, good, bad):
    probe = _probe(probe_id)
    assert score_probe(probe, good)
    assert not score_probe(probe, bad)


def test_reasoning_scores_the_last_number_not_any_number():
    probe = _probe("reasoning-00")  # 7 + 5
    assert score_probe(probe, "7 + 5 = 12")
    # Mentions 12 but commits to 13 - listing candidates must not be credited.
    assert not score_probe(probe, "It could be 12 or maybe 13")
    assert not score_probe(probe, "I cannot compute that.")


def test_reasoning_ignores_numbers_restated_from_the_question():
    # The first version scored this as "2" and marked a correct answer wrong.
    assert score_probe(_probe("reasoning-11"), "There are 120 minutes in 2 hours.")
    assert score_probe(_probe("reasoning-04"), "The train travels 180 km in 3 hours.")
    assert not score_probe(_probe("reasoning-11"), "There are 100 minutes in 2 hours.")
    # A restated question with no answer in it is still not an answer.
    assert not score_probe(_probe("reasoning-11"), "You asked about 2 hours.")


def test_reasoning_decimal_and_word_answers():
    assert score_probe(_probe("reasoning-14"), "0.5")
    assert not score_probe(_probe("reasoning-14"), "0.25")
    assert score_probe(_probe("reasoning-12"), "It will be Wednesday.")
    assert not score_probe(_probe("reasoning-12"), "It will be Tuesday.")


@pytest.mark.parametrize(
    "check, good, bad",
    [
        ("one_word", "Blue", "Blue and green"),
        ("two_words", "Goodbye friend.", "Goodbye"),
        ("all_caps", "HELLO", "Hello"),
        ("three_items", "apple, banana, cherry", "apple, banana"),
        ("three_items", "apple, banana, cherry", "apple,\nbanana, cherry"),
        ("starts_yes_no", "Yes, it is.", "The sky is blue."),
        ("under_ten_words", "Vast, deep, and blue.", "The ocean is a vast and deep body of salty water."),
        ("json_name_ada", '```json\n{"name": "Ada"}\n```', '{"name": "Bob"}'),
        ("json_name_ada", '{"name": "Ada"}', "name: Ada"),
        ("only_42", "42", "The number is 42."),
        ("ends_exclaim", "Cats are wonderful!", "Cats are wonderful."),
    ],
)
def test_instruction_checks(check, good, bad):
    assert INSTRUCTION_CHECKS[check](good)
    assert not INSTRUCTION_CHECKS[check](bad)


def test_compare_counts_flips_in_both_directions():
    probes = [_probe("knowledge-00"), _probe("knowledge-01")]
    base = ["Paris", "Osaka"]
    other = ["Lyon", "Tokyo"]
    result = compare(probes, base, other)["knowledge"]
    # Net accuracy is unchanged at 1/2; the flips are what show the movement.
    assert result["base"] == result["adapter"] == 1
    assert result["lost"] == ["knowledge-00"]
    assert result["gained"] == ["knowledge-01"]
    assert result["changed"] == 2


def test_compare_identical_responses_show_nothing():
    probes = build_probes()
    responses = ["x"] * len(probes)
    result = compare(probes, responses, responses)
    assert all(not b["lost"] and not b["gained"] and b["changed"] == 0 for b in result.values())
