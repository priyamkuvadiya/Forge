"""Arithmetic word problems whose ground truth is correct by construction.

These are generated from parameterized templates rather than hand-authored,
for two reasons. The ground truth is computed in Python from the same
parameters that render the prompt, so a task's answer cannot disagree with
its question — the failure mode that quietly corrupts a hand-written math
set. And the numbers are drawn to be deliberately awkward (prices to the
cent, non-round rates, multi-step chains), so the calculator tool is
genuinely worth calling rather than something the policy can shortcut with
memorized mental arithmetic. If the sums were easy, a reward improvement
would tell us nothing about tool use.

Every template records the parameters it drew in `metadata["params"]`, so
tests can check invariants against them and later analysis can look at
reward by difficulty rather than only in aggregate.
"""

import random
from dataclasses import dataclass

from .schema import Task

# Money answers are asked for "to the nearest cent", so anything within half a
# cent of the exact value is the right answer stated correctly.
CENT_TOLERANCE = 0.005


@dataclass(frozen=True)
class MathProblem:
    prompt: str
    value: float
    tolerance: float
    template: str
    params: dict


def _bulk_discount(rng: random.Random) -> MathProblem:
    unit_price = rng.randrange(275, 1850) / 100
    threshold = rng.choice([100, 250, 500])
    quantity = rng.randrange(60, 950)
    discount = rng.choice([5, 7, 12, 15])

    subtotal = unit_price * quantity
    total = subtotal * (1 - discount / 100) if quantity > threshold else subtotal

    return MathProblem(
        prompt=(
            f"A supplier charges ${unit_price:.2f} per unit. Orders of more than "
            f"{threshold} units receive a {discount}% discount on the entire order. "
            f"What is the total cost in dollars of an order for {quantity} units? "
            f"Round to the nearest cent."
        ),
        value=round(total, 2),
        tolerance=CENT_TOLERANCE,
        template="bulk_discount",
        params={
            "unit_price": unit_price,
            "threshold": threshold,
            "quantity": quantity,
            "discount": discount,
        },
    )


def _compound_interest(rng: random.Random) -> MathProblem:
    principal = rng.randrange(1200, 48000)
    rate = rng.choice([2.5, 3.25, 4.0, 4.75, 5.5, 6.25])
    years = rng.randint(3, 12)

    return MathProblem(
        prompt=(
            f"An account opens with ${principal:,} and earns {rate}% interest "
            f"compounded annually. What is the balance after {years} years? "
            f"Round to the nearest cent."
        ),
        value=round(principal * (1 + rate / 100) ** years, 2),
        tolerance=CENT_TOLERANCE,
        template="compound_interest",
        params={"principal": principal, "rate": rate, "years": years},
    )


def _distance_from_speed(rng: random.Random) -> MathProblem:
    speed = rng.randrange(45, 125)
    minutes = rng.randrange(35, 400)

    return MathProblem(
        prompt=(
            f"A train runs at a constant {speed} km/h for {minutes} minutes. "
            f"How many kilometres does it cover? Round to two decimal places."
        ),
        value=round(speed * minutes / 60, 2),
        tolerance=CENT_TOLERANCE,
        template="distance_from_speed",
        params={"speed_kmh": speed, "minutes": minutes},
    )


def _percentage_change(rng: random.Random) -> MathProblem:
    before = rng.randrange(1200, 90000)
    delta = rng.randrange(50, 30000) * rng.choice([-1, 1])
    after = max(1, before + delta)

    return MathProblem(
        prompt=(
            f"A town's population changed from {before:,} to {after:,}. What was "
            f"the percentage change, to two decimal places? Give a negative "
            f"number for a decrease."
        ),
        value=round((after - before) / before * 100, 2),
        tolerance=CENT_TOLERANCE,
        template="percentage_change",
        params={"before": before, "after": after},
    )


def _combined_work_rate(rng: random.Random) -> MathProblem:
    hours_a = rng.randrange(3, 40)
    hours_b = rng.randrange(3, 40)

    return MathProblem(
        prompt=(
            f"Machine A finishes a job in {hours_a} hours and Machine B finishes "
            f"the same job in {hours_b} hours. Working together at those rates, "
            f"how many hours do they take? Round to three decimal places."
        ),
        value=round(1 / (1 / hours_a + 1 / hours_b), 3),
        tolerance=0.0005,
        template="combined_work_rate",
        params={"hours_a": hours_a, "hours_b": hours_b},
    )


def _weighted_grade(rng: random.Random) -> MathProblem:
    w1, w2, w3 = rng.choice([(10, 25, 25), (15, 20, 30), (20, 20, 20), (5, 30, 25)])
    w4 = 100 - w1 - w2 - w3
    scores = [rng.randrange(410, 1000) / 10 for _ in range(4)]
    weights = [w1, w2, w3, w4]

    return MathProblem(
        prompt=(
            f"A course grade is {w1}% homework, {w2}% midterm, {w3}% project and "
            f"{w4}% final exam. A student scores {scores[0]}, {scores[1]}, "
            f"{scores[2]} and {scores[3]} on those four components. What is the "
            f"final grade, to two decimal places?"
        ),
        value=round(sum(w * s for w, s in zip(weights, scores)) / 100, 2),
        tolerance=CENT_TOLERANCE,
        template="weighted_grade",
        params={"weights": weights, "scores": scores},
    )


def _fuel_cost(rng: random.Random) -> MathProblem:
    distance = rng.randrange(180, 4200)
    consumption = rng.randrange(52, 165) / 10
    price = rng.randrange(890, 2150) / 1000

    return MathProblem(
        prompt=(
            f"A van drives {distance:,} km. It uses {consumption} litres of fuel "
            f"per 100 km and fuel costs ${price:.3f} per litre. What is the total "
            f"fuel cost in dollars, to the nearest cent?"
        ),
        value=round(distance * consumption / 100 * price, 2),
        tolerance=CENT_TOLERANCE,
        template="fuel_cost",
        params={"distance_km": distance, "litres_per_100km": consumption, "price_per_litre": price},
    )


def _flooring_cost(rng: random.Random) -> MathProblem:
    length = rng.randrange(24, 185) / 10
    width = rng.randrange(21, 140) / 10
    price = rng.randrange(1150, 8900) / 100

    return MathProblem(
        prompt=(
            f"A rectangular floor measures {length} m by {width} m. Flooring costs "
            f"${price:.2f} per square metre. What does it cost to cover the whole "
            f"floor, to the nearest cent?"
        ),
        value=round(length * width * price, 2),
        tolerance=CENT_TOLERANCE,
        template="flooring_cost",
        params={"length_m": length, "width_m": width, "price_per_sqm": price},
    )


def _successive_percentages(rng: random.Random) -> MathProblem:
    start = rng.randrange(850, 24000) / 100
    rise = rng.randrange(3, 45)
    fall = rng.randrange(3, 45)

    return MathProblem(
        prompt=(
            f"A share priced at ${start:.2f} rises by {rise}% and then falls by "
            f"{fall}% from its new price. What is the final price in dollars, to "
            f"the nearest cent?"
        ),
        value=round(start * (1 + rise / 100) * (1 - fall / 100), 2),
        tolerance=CENT_TOLERANCE,
        template="successive_percentages",
        params={"start": start, "rise_pct": rise, "fall_pct": fall},
    )


def _currency_conversion(rng: random.Random) -> MathProblem:
    amount = rng.randrange(240, 18000)
    rate = rng.randrange(7400, 9900) / 10000
    fee = rng.choice([1.5, 2.0, 2.75, 3.25])

    return MathProblem(
        prompt=(
            f"You convert {amount:,} US dollars to euros at a rate of {rate} euros "
            f"per dollar. The bank then charges a {fee}% fee on the converted "
            f"amount. How many euros do you receive, to the nearest cent?"
        ),
        value=round(amount * rate * (1 - fee / 100), 2),
        tolerance=CENT_TOLERANCE,
        template="currency_conversion",
        params={"amount_usd": amount, "rate": rate, "fee_pct": fee},
    )


TEMPLATES = (
    _bulk_discount,
    _compound_interest,
    _distance_from_speed,
    _percentage_change,
    _combined_work_rate,
    _weighted_grade,
    _fuel_cost,
    _flooring_cost,
    _successive_percentages,
    _currency_conversion,
)


def generate_math_tasks(count: int, split: str, seed: int) -> list[Task]:
    """Generate `count` math tasks, cycling templates so the mix stays balanced.

    Duplicate prompts are dropped rather than renumbered: two identical
    questions in a suite inflate whatever the policy happens to do on that one
    problem, and if they land either side of the train/held-out boundary the
    held-out score stops being held out at all.
    """
    rng = random.Random(seed)
    tasks: list[Task] = []
    seen: set[str] = set()

    attempts = 0
    max_attempts = count * 20
    while len(tasks) < count and attempts < max_attempts:
        template = TEMPLATES[len(tasks) % len(TEMPLATES)]
        problem = template(rng)
        attempts += 1

        if problem.prompt in seen:
            continue
        seen.add(problem.prompt)

        tasks.append(
            Task(
                task_id=f"math-{split}-{len(tasks):04d}",
                category="math",
                prompt=problem.prompt,
                ground_truth={"value": problem.value, "tolerance": problem.tolerance},
                split=split,
                metadata={"template": problem.template, "params": problem.params},
            )
        )

    if len(tasks) < count:
        raise RuntimeError(
            f"only generated {len(tasks)} of {count} unique math tasks in {attempts} attempts"
        )
    return tasks
