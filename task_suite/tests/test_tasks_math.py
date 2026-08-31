"""Checks on the generated math suite.

The invariant tests below are deliberately not "recompute the formula and
compare" — that would only prove the code equals itself. They assert
properties that hold for a correct answer and break for a plausibly wrong
one: compounding grows the balance, two machines together beat either alone,
a weighted average lies between its inputs, a rise then an equal-percentage
fall ends below where it started.
"""

import pytest

from task_suite.schema import Task
from task_suite.tasks_math import TEMPLATES, generate_math_tasks
from task_suite.verifiers import verify

SUITE = generate_math_tasks(count=200, split="train", seed=1234)


def by_template(name: str) -> list[Task]:
    return [t for t in SUITE if t.metadata["template"] == name]


def wrap(answer: str) -> str:
    return f"<answer>{answer}</answer>"


def test_generation_is_deterministic():
    again = generate_math_tasks(count=200, split="train", seed=1234)
    assert [t.to_dict() for t in again] == [t.to_dict() for t in SUITE]


def test_a_different_seed_gives_different_problems():
    other = generate_math_tasks(count=200, split="train", seed=99)
    assert {t.prompt for t in other} != {t.prompt for t in SUITE}


def test_prompts_and_ids_are_unique():
    assert len({t.prompt for t in SUITE}) == len(SUITE)
    assert len({t.task_id for t in SUITE}) == len(SUITE)


def test_every_template_is_represented_evenly():
    counts = {name: 0 for name in (fn.__name__.lstrip("_") for fn in TEMPLATES)}
    for task in SUITE:
        counts[task.metadata["template"]] += 1
    assert min(counts.values()) == max(counts.values()) == len(SUITE) // len(TEMPLATES)


def test_ground_truth_scores_one_through_the_real_verifier():
    for task in SUITE:
        assert verify(task, wrap(str(task.ground_truth["value"]))) == 1.0


def test_a_wrong_answer_scores_zero():
    for task in SUITE[:40]:
        assert verify(task, wrap(str(task.ground_truth["value"] + 1.0))) == 0.0


def test_answers_are_not_trivially_round():
    """If the sums were easy the calculator tool would be pointless to call."""
    integral = [t for t in SUITE if float(t.ground_truth["value"]).is_integer()]
    assert len(integral) / len(SUITE) < 0.1


# --------------------------------------------------------------------------
# per-template invariants
# --------------------------------------------------------------------------

def test_bulk_discount_only_discounts_above_the_threshold():
    for task in by_template("bulk_discount"):
        p = task.metadata["params"]
        undiscounted = p["unit_price"] * p["quantity"]
        if p["quantity"] > p["threshold"]:
            assert task.ground_truth["value"] < undiscounted
        else:
            assert task.ground_truth["value"] == pytest.approx(undiscounted, abs=0.01)


def test_compound_interest_grows_the_principal():
    for task in by_template("compound_interest"):
        assert task.ground_truth["value"] > task.metadata["params"]["principal"]


def test_distance_is_consistent_with_the_stated_duration():
    for task in by_template("distance_from_speed"):
        p = task.metadata["params"]
        implied_minutes = task.ground_truth["value"] / p["speed_kmh"] * 60
        assert implied_minutes == pytest.approx(p["minutes"], abs=0.02)


def test_percentage_change_sign_matches_the_direction():
    for task in by_template("percentage_change"):
        p = task.metadata["params"]
        value = task.ground_truth["value"]
        if p["after"] > p["before"]:
            assert value > 0
        elif p["after"] < p["before"]:
            assert value < 0
        else:
            assert value == 0


def test_two_machines_beat_either_one_alone():
    for task in by_template("combined_work_rate"):
        p = task.metadata["params"]
        fastest = min(p["hours_a"], p["hours_b"])
        assert fastest / 2 <= task.ground_truth["value"] < fastest


def test_weighted_grade_lies_between_its_component_scores():
    for task in by_template("weighted_grade"):
        scores = task.metadata["params"]["scores"]
        assert min(scores) <= task.ground_truth["value"] <= max(scores)
        assert sum(task.metadata["params"]["weights"]) == 100


def test_fuel_cost_implies_the_stated_consumption():
    for task in by_template("fuel_cost"):
        p = task.metadata["params"]
        litres = task.ground_truth["value"] / p["price_per_litre"]
        assert litres == pytest.approx(p["distance_km"] * p["litres_per_100km"] / 100, rel=1e-3)


def test_flooring_cost_implies_the_stated_area():
    for task in by_template("flooring_cost"):
        p = task.metadata["params"]
        area = task.ground_truth["value"] / p["price_per_sqm"]
        assert area == pytest.approx(p["length_m"] * p["width_m"], rel=1e-3)


def test_equal_rise_and_fall_ends_below_the_start():
    """The classic trap the template exists to pose."""
    for task in by_template("successive_percentages"):
        p = task.metadata["params"]
        if p["rise_pct"] == p["fall_pct"]:
            assert task.ground_truth["value"] < p["start"]
        elif p["rise_pct"] > p["fall_pct"]:
            # a bigger rise than fall still need not beat the start, but the
            # result must sit between the two single-step outcomes
            assert task.ground_truth["value"] < p["start"] * (1 + p["rise_pct"] / 100)


def test_currency_conversion_fee_reduces_the_payout():
    for task in by_template("currency_conversion"):
        p = task.metadata["params"]
        assert task.ground_truth["value"] < p["amount_usd"] * p["rate"]
