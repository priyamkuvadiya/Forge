"""Tasks that need no tool at all — the control for over-calling.

Every other category rewards reaching for a tool. That makes a policy which
learned "always search" indistinguishable from one which learned *when* to
search, because on this suite the two score identically. Over-calling is a
real trained-policy failure mode and it is invisible without a control, so
this category supplies one: tasks where the correct behaviour is to answer
directly, and where a tool call is wasted budget rather than progress.

Nothing here needs a new verifier. Arithmetic tasks are numeric and go to the
math verifier; comprehension tasks are textual and go to the QA verifier —
exactly the precedent `multi_tool` set. The category changes what the agent
should *do*, not how the result is graded.

Two things make the tasks genuinely tool-free rather than merely easy:

- The arithmetic is single-operation on small integers, the kind a 0.5B model
  answers correctly without help. If it needed the calculator, a tool call
  would be right and the control would measure nothing.
- The comprehension passages are invented here and deliberately kept out of
  the search corpus, so searching for them returns nothing. A policy that
  reaches for search on these gets no information back, which is the honest
  signal that the call was pointless.

**The reward does not penalise a tool call.** Scoring is correctness alone,
same as everywhere else. What this category enables is a *measurement* — the
tool-call count from `tools.ToolSession`'s trace — so module 9 can report
"the RL policy called a tool on N% of tasks that needed none" next to the
reward. Building the penalty into the reward instead would be training the
policy to satisfy a proxy we invented, which is the thing this whole project
avoids by using verifiable rewards.
"""

import random
from dataclasses import dataclass

from .schema import Task

# Arithmetic answers are exact integers; there is nothing to round.
EXACT = 0.0


@dataclass(frozen=True)
class NoToolProblem:
    prompt: str
    template: str
    # Exactly one of these is set: numeric problems carry a value, textual
    # ones carry the accepted surface forms.
    value: float | None = None
    answers: tuple[str, ...] = ()


# --------------------------------------------------------------------------
# Arithmetic small enough to do in your head
# --------------------------------------------------------------------------

_ARITHMETIC_PHRASINGS = (
    "What is {a} plus {b}?",
    "Add {a} and {b}.",
    "{a} + {b} = ?",
)

_PRODUCT_PHRASINGS = (
    "What is {a} times {b}?",
    "Multiply {a} by {b}.",
    "{a} x {b} = ?",
)


def _small_sum(rng: random.Random, variant: int, pools: "Pools") -> NoToolProblem:
    a, b = rng.choice(pools.operands), rng.choice(pools.operands)
    return NoToolProblem(
        prompt=_ARITHMETIC_PHRASINGS[variant % len(_ARITHMETIC_PHRASINGS)].format(a=a, b=b),
        template="small_sum",
        value=float(a + b),
    )


def _small_product(rng: random.Random, variant: int, pools: "Pools") -> NoToolProblem:
    a, b = rng.choice(pools.operands), rng.choice(pools.operands)
    return NoToolProblem(
        prompt=_PRODUCT_PHRASINGS[variant % len(_PRODUCT_PHRASINGS)].format(a=a, b=b),
        template="small_product",
        value=float(a * b),
    )


def _counting(rng: random.Random, variant: int, pools: "Pools") -> NoToolProblem:
    """Counting items listed in the prompt itself.

    Deliberately not arithmetic on stated numbers - the count is only
    available by reading the list, so there is nothing a calculator could be
    handed.
    """
    chosen = rng.sample(pools.items, rng.randint(3, min(6, len(pools.items))))
    listed = ", ".join(chosen[:-1]) + f" and {chosen[-1]}"
    return NoToolProblem(
        prompt=f"A box contains {listed}. How many items are in the box?",
        template="counting",
        value=float(len(chosen)),
    )


# --------------------------------------------------------------------------
# Comprehension: the answer is stated in the prompt
# --------------------------------------------------------------------------

# Kept disjoint from `qa_world.py`'s name pools so a policy cannot get any
# traction by searching, and so no answer here is ambiguous with a QA answer.
_PEOPLE = (
    "Odalys Brentmoor", "Gustav Ferncastle", "Marguerite Ashdown", "Matthias Selby",
    "Delia Marchmont", "Corwin Tealby", "Coretta Winterbourne", "Rafferty Klose",
    "Sunniva Petrakis", "Emeric Vaudrey", "Isolde Barrowman", "Nikolas Ferreday",
)
_VEHICLES = (
    "Brightwater ferry", "Kingsmere tram", "Aldergate shuttle", "Ravensfoot barge",
    "Cloudbank funicular", "Thornfield omnibus", "Larkhill cable car", "Weirbrook launch",
)
_DAYS = ("Tuesdays", "Wednesdays", "Thursdays", "Fridays", "Saturdays", "Sundays")
_COLOURS = ("green", "crimson", "slate grey", "pale yellow", "navy", "ochre")

_ITEMS = (
    "apples", "pencils", "chairs", "bottles", "folders", "lamps",
    "cushions", "keys", "mugs", "tickets", "candles", "brushes",
)


@dataclass(frozen=True)
class Pools:
    """The material one split may draw on.

    The splits get disjoint pools, so a held-out prompt can never repeat a
    training one. `build_suite` refuses a suite where any prompt appears in
    both splits, and a first version of this module tripped that guard on its
    first run - two splits drawing from one pool of twelve names collide
    quickly. Partitioning mirrors what `qa_world.py` already does with QA
    subjects, and gives held-out the same meaning here as everywhere else:
    material the training split never saw.
    """

    operands: tuple[int, ...]
    items: tuple[str, ...]
    people: tuple[str, ...]
    vehicles: tuple[str, ...]
    colours: tuple[str, ...]


def pools_for(split: str) -> Pools:
    if split == "train":
        return Pools(
            operands=(2, 3, 4, 5, 6),
            items=_ITEMS[:7],
            people=_PEOPLE[:8],
            vehicles=_VEHICLES[:5],
            colours=_COLOURS[:4],
        )
    return Pools(
        # Disjoint operands, so no arithmetic prompt can appear in both
        # splits whatever the phrasing.
        operands=(7, 8, 9, 11, 12),
        items=_ITEMS[7:],
        people=_PEOPLE[8:],
        vehicles=_VEHICLES[5:],
        colours=_COLOURS[4:],
    )


def _stated_operator(rng: random.Random, variant: int, pools: "Pools") -> NoToolProblem:
    person = rng.choice(pools.people)
    vehicle = rng.choice(pools.vehicles)
    day = rng.choice(_DAYS)

    phrasings = (
        "The {vehicle} runs on {day} and is operated by {person}. "
        "Who operates the {vehicle}?",
        "{person} operates the {vehicle}, which runs on {day}. "
        "Who operates it?",
    )
    return NoToolProblem(
        prompt=phrasings[variant % len(phrasings)].format(
            vehicle=vehicle, day=day, person=person
        ),
        template="stated_operator",
        # Surname alone is accepted, matching how `qa_world` treats people.
        answers=(person, person.split()[-1]),
    )


def _stated_colour(rng: random.Random, variant: int, pools: "Pools") -> NoToolProblem:
    vehicle = rng.choice(pools.vehicles)
    colour = rng.choice(pools.colours)
    day = rng.choice(_DAYS)

    phrasings = (
        "The {vehicle} is painted {colour} and does not run on {day}. "
        "What colour is the {vehicle}?",
        "Painted {colour}, the {vehicle} is out of service on {day}. "
        "What colour is it?",
    )
    return NoToolProblem(
        prompt=phrasings[variant % len(phrasings)].format(
            vehicle=vehicle, colour=colour, day=day
        ),
        template="stated_colour",
        answers=(colour,),
    )


TEMPLATES = (
    _small_sum,
    _small_product,
    _counting,
    _stated_operator,
    _stated_colour,
)


def generate_no_tool_tasks(count: int, split: str, seed: int) -> list[Task]:
    """Generate `count` control tasks, cycling templates to keep the mix even.

    Duplicate prompts are dropped rather than renumbered, for the reason the
    math generator gives: two identical questions inflate whatever the policy
    happens to do on that one problem, and across the split boundary they stop
    the held-out score being held out.
    """
    rng = random.Random(seed)
    pools = pools_for(split)
    tasks: list[Task] = []
    seen: set[str] = set()

    attempts = 0
    max_attempts = count * 200
    while len(tasks) < count and attempts < max_attempts:
        index = len(tasks)
        problem = TEMPLATES[index % len(TEMPLATES)](rng, index, pools)
        attempts += 1

        if problem.prompt in seen:
            continue
        seen.add(problem.prompt)

        if problem.value is not None:
            ground_truth = {"value": problem.value, "tolerance": EXACT}
            answer_type = "number"
        else:
            ground_truth = {"answers": list(problem.answers)}
            answer_type = "text"

        tasks.append(
            Task(
                task_id=f"notool-{split}-{index:04d}",
                category="no_tool",
                prompt=problem.prompt,
                ground_truth=ground_truth,
                split=split,
                metadata={
                    "template": problem.template,
                    "answer_type": answer_type,
                    # Read by the eval harness: these are the tasks on which a
                    # tool call is evidence of over-calling.
                    "tools": [],
                },
            )
        )

    if len(tasks) < count:
        raise RuntimeError(
            f"only generated {len(tasks)} of {count} unique no_tool tasks in {attempts} attempts"
        )
    return tasks
