"""Measure where rollout time goes, before anyone optimises it.

Generation is 90%+ of a training step (19-32 s of rollout against 1.4-2.8 s of
backward), and until this existed nobody had profiled it. Two hypotheses were
on the table and both were guesses: that `stop_strings` does per-step CPU work
that stalls the GPU, and that batch 32 is pathologically slow on the train
split. This measures turn-1 generation - the turn where every episode is live,
which took ~50 of the train-split baseline's 58 minutes - on real train-split
prompts, sorted exactly as `HFPolicy.generate` sorts them.

Per batch it records:

  * wall time and **decode steps**. A batch keeps decoding until its *longest*
    member stops, so decode steps are set by the straggler, not the average.
  * mean completion length, from which **useful-token fraction** follows: the
    share of computed (sequence, step) slots that produced a token for a
    sequence still running. Everything else is padding for a finished row.
  * `memory_reserved()` against the card, the only spill check that has held
    up on this machine, and peak *allocated* beside it. The gap between the
    two is the caching allocator's, not the model's.

Arms: batch 16 and 32, each with and without `stop_strings`. Without them
completions run on to EOS, so lengths differ and arms are compared per decode
step, not per batch.

    python -m rl_training.measure_rollout --out artifacts/rl/rollout_profile.json

Leave the machine alone while it runs; it is ~10 minutes of GPU.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any

from baseline_agent import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MODEL,
    STOP_STRINGS,
    HFPolicy,
    make_episodes,
)
from task_suite.registry import load_suite
from tools import build_registry

ARMS = [(16, True), (32, True), (16, False), (32, False)]


def _prompts(policy: HFPolicy, per_category: int, rollouts: int, seed: int) -> list[str]:
    """Category-stratified train prompts, each repeated `rollouts` times.

    Repeated because that is what a real turn 1 looks like: k copies of the
    same prompt, which sort adjacent and so share a batch.
    """
    rng = random.Random(seed)
    by_category: dict[str, list[Any]] = {}
    for task in load_suite()["train"]:
        by_category.setdefault(task.category, []).append(task)
    picked = []
    for category in sorted(by_category):
        picked += rng.sample(by_category[category], per_category)

    # Prompt construction only; no tool is ever executed here.
    registry = build_registry(include_code=True, code_runner=None)
    episodes = make_episodes(picked, registry, rollouts=rollouts)
    prompts = [
        policy.tokenizer.apply_chat_template(
            episode.messages, tokenize=False, add_generation_prompt=True
        )
        for episode in episodes
    ]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    return [prompts[i] for i in order]


def run_arm(
    policy: HFPolicy,
    prompts: list[str],
    *,
    batch_size: int,
    use_stop: bool,
    max_new_tokens: int,
    seed: int,
) -> dict[str, Any]:
    import torch

    tokenizer, model = policy.tokenizer, policy.model
    total = torch.cuda.get_device_properties(0).total_memory
    torch.manual_seed(seed)
    rows = []
    for start in range(0, len(prompts), batch_size):
        encoded = tokenizer(
            prompts[start : start + batch_size],
            return_tensors="pt",
            padding=True,
            add_special_tokens=False,
        ).to(policy.device)
        stop = {"stop_strings": list(STOP_STRINGS), "tokenizer": tokenizer} if use_stop else {}

        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        began = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=policy.temperature,
                top_p=policy.top_p,
                pad_token_id=tokenizer.pad_token_id,
                **stop,
            )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - began

        fresh = generated[:, encoded["input_ids"].shape[1] :]
        lengths = (fresh != tokenizer.pad_token_id).sum(dim=1).tolist()
        steps = fresh.shape[1]
        reserved = torch.cuda.memory_reserved()
        rows.append(
            {
                "rows": len(lengths),
                "prompt_tokens": encoded["input_ids"].shape[1],
                "seconds": round(seconds, 3),
                "decode_steps": steps,
                "ms_per_step": round(1000 * seconds / max(steps, 1), 2),
                "mean_len": round(sum(lengths) / len(lengths), 1),
                "max_len": max(lengths),
                "hit_cap": sum(1 for n in lengths if n >= max_new_tokens),
                "peak_alloc_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                "reserved_gib": round(reserved / 2**30, 2),
                "spilled": reserved > total,
            }
        )

    seconds = sum(row["seconds"] for row in rows)
    useful = sum(row["mean_len"] * row["rows"] for row in rows)
    computed = sum(row["decode_steps"] * row["rows"] for row in rows)
    return {
        "summary": {
            "batch_size": batch_size,
            "stop_strings": use_stop,
            "batches": len(rows),
            "seconds": round(seconds, 1),
            "sequences_per_second": round(len(prompts) / seconds, 2),
            "useful_tokens_per_second": round(useful / seconds, 1),
            "ms_per_step_median": sorted(row["ms_per_step"] for row in rows)[len(rows) // 2],
            "useful_token_fraction": round(useful / computed, 3),
            "batches_with_cap_hit": sum(1 for row in rows if row["hit_cap"]),
            "max_peak_alloc_gib": max(row["peak_alloc_gib"] for row in rows),
            "max_reserved_gib": max(row["reserved_gib"] for row in rows),
            "any_spill": any(row["spilled"] for row in rows),
        },
        "batches": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rl_training.measure_rollout")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--per-category", type=int, default=8)
    parser.add_argument("--rollouts", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("artifacts/rl/rollout_profile.json"))
    arguments = parser.parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        print("this measures the GPU; no CUDA device is available", file=sys.stderr)
        return 1

    policy = HFPolicy(arguments.model, seed=arguments.seed)
    prompts = _prompts(policy, arguments.per_category, arguments.rollouts, arguments.seed)
    print(f"{len(prompts)} prompts", file=sys.stderr, flush=True)

    common = dict(max_new_tokens=arguments.max_new_tokens, seed=arguments.seed)
    # Warm-up, discarded: the first batches otherwise pay for kernel selection
    # and allocator growth and read as 5x slower per step than they are.
    run_arm(policy, prompts[:32], batch_size=16, use_stop=True, **common)

    arms = {}
    for batch_size, use_stop in ARMS:
        # Each arm starts from an empty cache, so reserved memory is the arm's
        # own and not left over from the one before it.
        torch.cuda.empty_cache()
        name = f"bs{batch_size}_{'stop' if use_stop else 'nostop'}"
        arms[name] = run_arm(policy, prompts, batch_size=batch_size, use_stop=use_stop, **common)
        print(json.dumps({name: arms[name]["summary"]}), file=sys.stderr, flush=True)

    payload = {
        "generated_by": "rl_training.measure_rollout",
        "model": arguments.model,
        "device": torch.cuda.get_device_name(0),
        "card_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "prompts": len(prompts),
        "per_category": arguments.per_category,
        "rollouts": arguments.rollouts,
        "max_new_tokens": arguments.max_new_tokens,
        "arms": arms,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print(f"wrote {arguments.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
