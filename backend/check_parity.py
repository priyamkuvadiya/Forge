"""Does the server serve the policies that were evaluated? Checked, not assumed.

The server attaches every adapter to one shared base and switches between
them; the evaluation loaded one adapter at a time with `HFPolicy(adapter=...)`,
and the baseline with no peft at all. Those should be the same functions, and
this compares them directly: greedy decoding, the same held-out prompts under
the same prefix, served engine against a fresh `HFPolicy` per policy.

    python -m backend.check_parity --adapter run1=artifacts/rl/run1/adapter-final \\
                                   --adapter run2=artifacts/rl/run2/adapter-final

Needs the GPU. Do not run the test suite alongside it.
"""

import argparse
import gc
import sys

from baseline_agent import (
    DEFAULT_MAX_NEW_TOKENS,
    STOP_STRINGS,
    HFPolicy,
    make_episodes,
    select_tasks,
)
from task_suite.registry import load_suite
from tools import build_registry, code_exec

from .__main__ import parse_adapter
from .app import BASELINE
from .engine import ServedPolicies


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m backend.check_parity")
    parser.add_argument(
        "--adapter", type=parse_adapter, action="append", default=[], metavar="NAME=PATH"
    )
    parser.add_argument("--per-category", type=int, default=2)
    arguments = parser.parse_args(argv)
    adapters = dict(arguments.adapter)

    registry = build_registry(include_code=code_exec.sandbox_available())
    tasks = select_tasks(load_suite()["heldout"], per_category=arguments.per_category)
    conversations = [episode.messages for episode in make_episodes(tasks, registry)]

    served = ServedPolicies(adapters, temperature=0.0)
    served_out = {
        name: [served.complete(name, conversation) for conversation in conversations]
        for name in served.names()
    }
    del served
    gc.collect()

    mismatches = 0
    for name in [BASELINE, *adapters]:
        reference = HFPolicy(temperature=0.0, batch_size=1, adapter=adapters.get(name))
        expected = [
            reference.generate([c], stop=STOP_STRINGS, max_new_tokens=DEFAULT_MAX_NEW_TOKENS)[0]
            for c in conversations
        ]
        del reference
        gc.collect()
        same = sum(a == b for a, b in zip(served_out[name], expected))
        mismatches += len(expected) - same
        print(f"{name}: {same}/{len(expected)} identical")
        for task, a, b in zip(tasks, served_out[name], expected):
            if a != b:
                print(f"  {task.task_id}\n    served:    {a[:160]!r}\n    reference: {b[:160]!r}")

    # Also a check that the switch does something: if every arm produced the
    # same text, parity would hold trivially and prove nothing.
    distinct = len({tuple(outputs) for outputs in served_out.values()})
    print(f"distinct output sets across {len(served_out)} policies: {distinct}")
    return 0 if mismatches == 0 and distinct == len(served_out) else 1


if __name__ == "__main__":
    sys.exit(main())
