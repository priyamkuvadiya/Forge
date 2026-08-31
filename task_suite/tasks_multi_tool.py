"""Tasks that need the search tool *and* the calculator to answer.

The other three categories map one-to-one onto a tool: math wants the
calculator, QA wants search, code wants the executor. A policy can infer
which tool to reach for from the shape of the question alone, which means RL
on those categories teaches the agent to *use* a tool well but never to
*choose* one. For a project whose claim is a trained tool-use policy, that is
a real gap, and this category closes it: the numbers live in the QA corpus,
so they have to be retrieved, and the answer is arithmetic over them, so it
has to be computed.

The facts come from the same synthetic world as the QA category, which is
what makes them retrievable-only — no base model knows how many berths
Fenwold Station has. Answers are checked by the numeric verifier, since the
result of the arithmetic is a number and gets scored exactly as any other
number does.

Two properties are enforced at generation time rather than hoped for:

- the answer never appears verbatim in any of the task's supporting
  documents, so the arithmetic cannot be skipped by finding the result
  already written down somewhere;
- differences and elapsed-year counts are only emitted when they come out
  positive, so no task turns on whether the reader guessed which way round a
  subtraction was meant.
"""

from dataclasses import dataclass

from .qa_world import World, build_world, split_subjects
from .schema import Task

# Whole-number answers are exact; the one summed-mass template asks for one
# decimal place, matching the precision the corpus states masses to.
EXACT = 0.0
ONE_DECIMAL = 0.05


@dataclass(frozen=True)
class MultiToolProblem:
    template: str
    subject: str
    text: str
    value: float
    tolerance: float
    supporting_docs: tuple[str, ...]
    answer_text: str


PHRASINGS: dict[str, tuple[str, ...]] = {
    "station_age_at_expedition": (
        "The {a} worked out of a station. How many years after that station was "
        "founded did the expedition take place?",
        "How many years separate the founding of the station used by the {a} from "
        "the year the {a} ran?",
    ),
    "expedition_days": (
        "How many days did the {a} last?",
        "The {a} ran for a number of weeks. How many days is that?",
    ),
    "instrument_age_at_expedition": (
        "The {a} was led by a researcher who designed an instrument. How many years "
        "before the expedition did that instrument enter service?",
        "How many years had the instrument designed by the leader of the {a} been in "
        "service by the time the expedition took place?",
    ),
    "combined_instrument_mass": (
        "What is the combined dry mass, in kilograms, of the instruments designed by "
        "the researchers who led the {a} and the {b}? Give your answer to one decimal "
        "place.",
        "The {a} and the {b} were each led by a researcher, and each of those "
        "researchers designed an instrument. What do those two instruments weigh "
        "together in kilograms, to one decimal place?",
    ),
    "vessel_length_difference": (
        "How many metres longer is the vessel that carried the {a} than the vessel "
        "that carried the {b}?",
        "Compare the vessels used by the {a} and the {b}. By how many metres does the "
        "first exceed the second in length?",
    ),
    "total_berths": (
        "How many berths do {a} and {b} have between them?",
        "What is the combined berth capacity of {a} and {b}?",
    ),
}


def phrase(template: str, variant: int, **names: str) -> str:
    forms = PHRASINGS[template]
    return forms[variant % len(forms)].format(**names)


def _single_expedition_problems(world: World, exp, variant: int) -> list[MultiToolProblem]:
    station = world.station(exp.station)
    leader = world.researcher(exp.leader)
    instrument = world.instrument(leader.instrument)

    problems = [
        MultiToolProblem(
            template="expedition_days",
            subject=exp.name,
            text=phrase("expedition_days", variant, a=exp.name),
            value=exp.duration_weeks * 7,
            tolerance=EXACT,
            supporting_docs=(exp.doc_id,),
            answer_text=str(exp.duration_weeks * 7),
        )
    ]

    station_age = exp.year - station.founded
    if station_age > 0:
        problems.append(
            MultiToolProblem(
                template="station_age_at_expedition",
                subject=exp.name,
                text=phrase("station_age_at_expedition", variant, a=exp.name),
                value=station_age,
                tolerance=EXACT,
                supporting_docs=(exp.doc_id, station.doc_id),
                answer_text=str(station_age),
            )
        )

    instrument_age = exp.year - instrument.in_service
    if instrument_age > 0:
        problems.append(
            MultiToolProblem(
                template="instrument_age_at_expedition",
                subject=exp.name,
                text=phrase("instrument_age_at_expedition", variant, a=exp.name),
                value=instrument_age,
                tolerance=EXACT,
                supporting_docs=(exp.doc_id, leader.doc_id, instrument.doc_id),
                answer_text=str(instrument_age),
            )
        )

    return problems


def _expedition_pair_problems(world: World, first, second, variant: int) -> list[MultiToolProblem]:
    problems: list[MultiToolProblem] = []

    leader_a = world.researcher(first.leader)
    leader_b = world.researcher(second.leader)
    if leader_a.name != leader_b.name:
        ins_a = world.instrument(leader_a.instrument)
        ins_b = world.instrument(leader_b.instrument)
        total = round(ins_a.mass_kg + ins_b.mass_kg, 1)
        problems.append(
            MultiToolProblem(
                template="combined_instrument_mass",
                subject=f"{first.name} + {second.name}",
                text=phrase("combined_instrument_mass", variant, a=first.name, b=second.name),
                value=total,
                tolerance=ONE_DECIMAL,
                supporting_docs=(
                    first.doc_id, leader_a.doc_id, ins_a.doc_id,
                    second.doc_id, leader_b.doc_id, ins_b.doc_id,
                ),
                answer_text=f"{total:.1f}",
            )
        )

    ves_a = world.vessel(first.vessel)
    ves_b = world.vessel(second.vessel)
    if ves_a.length_m != ves_b.length_m:
        # order the pair so the difference is positive and the question has one
        # unambiguous reading
        longer, shorter = (first, second) if ves_a.length_m > ves_b.length_m else (second, first)
        gap = abs(ves_a.length_m - ves_b.length_m)
        problems.append(
            MultiToolProblem(
                template="vessel_length_difference",
                subject=f"{longer.name} + {shorter.name}",
                text=phrase("vessel_length_difference", variant, a=longer.name, b=shorter.name),
                value=gap,
                tolerance=EXACT,
                supporting_docs=(
                    longer.doc_id, world.vessel(longer.vessel).doc_id,
                    shorter.doc_id, world.vessel(shorter.vessel).doc_id,
                ),
                answer_text=str(gap),
            )
        )

    return problems


def _station_pair_problems(world: World, first, second, variant: int) -> list[MultiToolProblem]:
    total = first.berths + second.berths
    return [
        MultiToolProblem(
            template="total_berths",
            subject=f"{first.name} + {second.name}",
            text=phrase("total_berths", variant, a=first.name, b=second.name),
            value=total,
            tolerance=EXACT,
            supporting_docs=(first.doc_id, second.doc_id),
            answer_text=str(total),
        )
    ]


def answer_is_written_down(world: World, problem: MultiToolProblem) -> bool:
    """True if the answer can be read straight out of a supporting document."""
    return any(
        problem.answer_text in world.document(doc_id).text
        for doc_id in problem.supporting_docs
    )


def build_problems(world: World, split: str) -> list[MultiToolProblem]:
    expeditions, _, stations = split_subjects(world, split)

    problems: list[MultiToolProblem] = []
    for i, exp in enumerate(expeditions):
        problems.extend(_single_expedition_problems(world, exp, i))
    for i in range(0, len(expeditions) - 1, 2):
        problems.extend(_expedition_pair_problems(world, expeditions[i], expeditions[i + 1], i))
    for i in range(0, len(stations) - 1, 2):
        problems.extend(_station_pair_problems(world, stations[i], stations[i + 1], i))

    # Dropping these rather than rewording them keeps the category's promise
    # exact: every task here requires arithmetic, none can be finished by
    # spotting the answer already printed in a retrieved document.
    return [p for p in problems if not answer_is_written_down(world, p)]


def generate_multi_tool_tasks(split: str, world: World | None = None) -> list[Task]:
    world = world if world is not None else build_world()
    return [
        Task(
            task_id=f"multi-{split}-{i:04d}",
            category="multi_tool",
            prompt=problem.text,
            ground_truth={"value": problem.value, "tolerance": problem.tolerance},
            split=split,
            metadata={
                "template": problem.template,
                "subject": problem.subject,
                "supporting_docs": list(problem.supporting_docs),
                "tools": ["search", "calculator"],
            },
        )
        for i, problem in enumerate(build_problems(world, split))
    ]
