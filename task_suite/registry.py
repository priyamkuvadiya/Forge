"""Assemble the three categories into one suite and pin it to disk.

Everything here is deterministic, so the suite could be rebuilt on demand
instead of stored. It is stored anyway, and the JSON is committed: eval
numbers in the README are only meaningful if the tasks behind them are
exactly the tasks that were run, and "it regenerates the same, trust me" is
weaker than a file anyone can diff. Rebuilding and finding the checked-in
file changed is itself the signal that a reported number has gone stale.

Run `python -m task_suite.registry` to rebuild and print the summary.
"""

import argparse
import json
from pathlib import Path

from .qa_world import WORLD_SEED, build_world, generate_qa_tasks
from .schema import CATEGORIES, Task
from .tasks_code import generate_code_tasks
from .tasks_math import generate_math_tasks
from .tasks_multi_tool import generate_multi_tool_tasks
from .tasks_no_tool import generate_no_tool_tasks

DATA_DIR = Path(__file__).parent / "data"
SUITE_PATH = DATA_DIR / "suite.json"
CORPUS_PATH = DATA_DIR / "corpus.json"

# The math set is generated, so its size is a free parameter while the other
# two are fixed by how many problems and entities were authored. 150/60 keeps
# math from swamping the mix any harder than it already does — it is still the
# largest category, which is a real imbalance and is why every eval number
# this project reports has to be broken down per category, never pooled into
# one headline pass-rate.
DEFAULT_MATH_TRAIN = 150
DEFAULT_MATH_HELDOUT = 60

# The control category is deliberately small. It exists to be *measured* on -
# the tool-call rate on tasks needing no tool - not to be trained on heavily,
# and a large block of trivially easy tasks would lift the macro-average for
# reasons that say nothing about tool use.
DEFAULT_NO_TOOL_TRAIN = 40
DEFAULT_NO_TOOL_HELDOUT = 20

# Distinct seeds so the two splits draw independent problem parameters.
MATH_TRAIN_SEED = 11_071
MATH_HELDOUT_SEED = 90_211
NO_TOOL_TRAIN_SEED = 40_213
NO_TOOL_HELDOUT_SEED = 77_419


def build_suite(
    math_train: int = DEFAULT_MATH_TRAIN,
    math_heldout: int = DEFAULT_MATH_HELDOUT,
    no_tool_train: int = DEFAULT_NO_TOOL_TRAIN,
    no_tool_heldout: int = DEFAULT_NO_TOOL_HELDOUT,
) -> dict[str, list[Task]]:
    world = build_world()

    splits = {
        "train": (
            generate_math_tasks(math_train, "train", MATH_TRAIN_SEED)
            + generate_code_tasks("train")
            + generate_qa_tasks("train", world)
            + generate_multi_tool_tasks("train", world)
            + generate_no_tool_tasks(no_tool_train, "train", NO_TOOL_TRAIN_SEED)
        ),
        "heldout": (
            generate_math_tasks(math_heldout, "heldout", MATH_HELDOUT_SEED)
            + generate_code_tasks("heldout")
            + generate_qa_tasks("heldout", world)
            + generate_multi_tool_tasks("heldout", world)
            + generate_no_tool_tasks(no_tool_heldout, "heldout", NO_TOOL_HELDOUT_SEED)
        ),
    }

    # A prompt on both sides of the split turns the held-out score into a
    # training score for that task, which is the one mistake that would
    # invalidate the headline result of this whole project.
    overlap = {t.prompt for t in splits["train"]} & {t.prompt for t in splits["heldout"]}
    if overlap:
        raise RuntimeError(f"{len(overlap)} prompts appear in both splits, e.g. {next(iter(overlap))!r}")

    return splits


def save_suite(splits: dict[str, list[Task]], path: Path = SUITE_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_by": "task_suite.registry",
        "config": {
            "math_train": sum(1 for t in splits["train"] if t.category == "math"),
            "math_heldout": sum(1 for t in splits["heldout"] if t.category == "math"),
            "math_train_seed": MATH_TRAIN_SEED,
            "math_heldout_seed": MATH_HELDOUT_SEED,
            "no_tool_train": sum(1 for t in splits["train"] if t.category == "no_tool"),
            "no_tool_heldout": sum(1 for t in splits["heldout"] if t.category == "no_tool"),
            "no_tool_train_seed": NO_TOOL_TRAIN_SEED,
            "no_tool_heldout_seed": NO_TOOL_HELDOUT_SEED,
            "world_seed": WORLD_SEED,
        },
        "splits": {name: [t.to_dict() for t in tasks] for name, tasks in splits.items()},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_suite(path: Path = SUITE_PATH) -> dict[str, list[Task]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        name: [Task.from_dict(t) for t in tasks]
        for name, tasks in payload["splits"].items()
    }


def save_corpus(path: Path = CORPUS_PATH) -> Path:
    """Write the QA corpus separately — the search tool needs it, not the tasks."""
    world = build_world()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "world_seed": WORLD_SEED,
        "documents": [
            {"doc_id": d.doc_id, "title": d.title, "text": d.text}
            for d in world.documents
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_corpus(path: Path = CORPUS_PATH) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["documents"]


def summarize(splits: dict[str, list[Task]]) -> str:
    header = f"{'split':<10}" + "".join(f"{c:>12}" for c in CATEGORIES) + f"{'total':>8}"
    lines = [header]
    for name, tasks in splits.items():
        counts = [sum(1 for t in tasks if t.category == c) for c in CATEGORIES]
        lines.append(f"{name:<10}" + "".join(f"{c:>12}" for c in counts) + f"{len(tasks):>8}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and pin the task suite.")
    parser.add_argument("--math-train", type=int, default=DEFAULT_MATH_TRAIN)
    parser.add_argument("--math-heldout", type=int, default=DEFAULT_MATH_HELDOUT)
    parser.add_argument("--no-tool-train", type=int, default=DEFAULT_NO_TOOL_TRAIN)
    parser.add_argument("--no-tool-heldout", type=int, default=DEFAULT_NO_TOOL_HELDOUT)
    args = parser.parse_args()

    splits = build_suite(
        args.math_train, args.math_heldout, args.no_tool_train, args.no_tool_heldout
    )
    suite_path = save_suite(splits)
    corpus_path = save_corpus()

    print(summarize(splits))
    print(f"\nwrote {suite_path}")
    print(f"wrote {corpus_path}")


if __name__ == "__main__":
    main()
