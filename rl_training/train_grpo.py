"""Module 5: the GRPO training loop.

The shape of one step:

    draw a category-balanced set of prompts      (sampling.py)
    roll out `group_size` attempts at each       (baseline_agent.run_episodes)
    score every attempt                          (task_suite.verify)
    centre each reward on its group              (advantages.py)
    mask each transcript to the policy's tokens  (transcript.py)
    accumulate a gradient over microbatches      (objective.py)
    one optimiser step

Everything except the optimiser is code this project already owns, which is the
point: the rollouts come from the *same* `run_episodes` that produced the
baseline, scored by the *same* `verify`, through the *same* tool contract. If
the training loop had its own rollout path, module 9's comparison would be
measuring two agents rather than two sets of weights.

Three decisions in here move the result and none of them is forced, so each is
logged rather than buried:

**Degenerate groups are dropped before the backward pass.** A group whose
rollouts all scored the same has an advantage of zero for every member, so it
contributes nothing to the policy gradient - but it still costs a full forward
and backward. Module 5's signal measurement found 73.6% of held-out prompts in
that state at the baseline's skill, so keeping them would spend roughly three
quarters of the training budget computing zeros. The honest caveat: with
`kl_beta > 0` those sequences would still have contributed a KL penalty, so
dropping them is not a pure optimisation, it slightly changes the objective.
The count is in every step log, and `--keep-degenerate` turns it off.

**Sequences are packed into microbatches by a measured envelope, not a guess.**
16x1024 and 4x2048 fit on this card; 8x2048 spills to system RAM at up to 11x
the latency, with no OOM to notice it by. So the packer uses a token budget
that halves past 1024 tokens, and the training loop asserts against the card's
capacity after every step rather than trusting the arithmetic.

**The reference policy is the base model, reached by turning the adapters off.**
No second copy of the weights, and no drift: the KL is always measured against
the model module 4 benchmarked, not against the policy as it was some number of
steps ago.

    python -m rl_training.train_grpo --steps 200 --out artifacts/rl/run1
    python -m rl_training.train_grpo --steps 2 --prompts-per-step 2 --group-size 4 \
        --categories math --out artifacts/rl/smoke      # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from baseline_agent import (
    DEFAULT_MAX_CALLS,
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_MAX_NUDGES,
    Episode,
    make_episodes,
    run_episodes,
    score_episodes,
)
from task_suite.registry import load_suite
from task_suite.schema import CATEGORIES
from tools import build_registry, code_exec

from rl_training.advantages import group_advantages
from rl_training.sampling import CategorySampler, category_counts
from rl_training.transcript import build_transcript

# The measured envelope. See the module docstring and `artifacts/rl/budget.json`:
# 16x1024 peaks at 6.67 GiB and fits, 4x2048 at 5.59 GiB and fits, 8x2048
# spills. Tokens alone do not predict it - 16x1024 and 8x2048 are both 16,384 -
# because attention and the per-layer activations grow with sequence length as
# well, so the budget halves past the 1024 mark.
SHORT_SEQUENCE = 1024
TOKEN_BUDGET_SHORT = 16384
TOKEN_BUDGET_LONG = 8192


def microbatch_capacity(max_length: int) -> int:
    """How many sequences of `max_length` tokens fit in one backward."""
    if max_length <= 0:
        raise ValueError(f"sequence length must be positive, got {max_length}")
    budget = TOKEN_BUDGET_SHORT if max_length <= SHORT_SEQUENCE else TOKEN_BUDGET_LONG
    return max(1, budget // max_length)


def pack_microbatches(
    lengths: Sequence[int], order: Sequence[int] | None = None
) -> list[list[int]]:
    """Group indices into microbatches that fit, longest sequences together.

    Sorted by length first, so a 2048-token transcript does not drag fifteen
    400-token ones into a padded batch sized for the worst case. That sorting
    is why the packer takes lengths rather than tensors: padding is decided
    after the grouping, not before it.
    """
    indices = list(order) if order is not None else sorted(
        range(len(lengths)), key=lambda i: lengths[i]
    )
    batches: list[list[int]] = []
    current: list[int] = []
    for index in indices:
        candidate = current + [index]
        longest = max(lengths[i] for i in candidate)
        if len(candidate) > microbatch_capacity(longest):
            batches.append(current)
            current = [index]
        else:
            current = candidate
    if current:
        batches.append(current)
    return batches


@dataclass
class StepReport:
    """One optimiser step, as it appears in the run log."""

    step: int
    episodes: int
    kept_episodes: int
    dropped_degenerate: int
    groups: int
    degenerate_groups: int
    mean_reward: float
    per_category_reward: dict[str, float]
    loss: float
    grad_norm: float
    mean_kl: float
    completion_tokens: float
    microbatches: int
    truncated: int
    retokenized: int
    overshoot_chars: int
    seconds_rollout: float
    seconds_backward: float
    peak_gib: float
    reserved_gib: float
    spilled: bool
    tool_calls: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {key: getattr(self, key) for key in self.__dataclass_fields__}


def _per_category_reward(results) -> dict[str, float]:
    sums: dict[str, list[float]] = {}
    for result in results:
        sums.setdefault(result.category, []).append(result.reward)
    return {
        category: round(sum(values) / len(values), 4)
        for category, values in sorted(sums.items())
    }


def _tool_call_counts(results) -> dict[str, int]:
    counts: dict[str, int] = {}
    for result in results:
        for call in result.tool_calls:
            counts[call["tool"]] = counts.get(call["tool"], 0) + 1
    return dict(sorted(counts.items()))


class RunLog:
    """Append-only JSONL, flushed and fsynced per line.

    The same discipline `transformer_scratch/train.py` arrived at the hard way:
    a pipe-buffered log was lost entirely when the machine power-cycled
    mid-training, and the only surviving evidence of the run was a traceback.
    A training run that cannot be reconstructed from its log after a crash is a
    training run that has to be repeated.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self._handle.write(json.dumps(record) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())

    def close(self) -> None:
        self._handle.close()


def save_checkpoint(policy, directory: Path, tag: str) -> Path:
    """Write the adapter atomically: temp directory, then replace.

    A half-written checkpoint is worse than no checkpoint, because it looks
    resumable. Same reason and same shape as the atomic file writes in
    `transformer_scratch/train.py` and `baseline_agent.save_rollouts`.
    """
    directory.mkdir(parents=True, exist_ok=True)
    final = directory / tag
    staging = directory / f".{tag}.tmp"
    if staging.exists():
        shutil.rmtree(staging)
    policy.save_adapter(staging)
    if final.exists():
        shutil.rmtree(final)
    os.replace(staging, final)
    return final


def train_step(
    policy,
    tasks: list,
    registry,
    *,
    step: int,
    group_size: int,
    code_runner=None,
    allow_unconfined: bool = False,
    max_calls: int = DEFAULT_MAX_CALLS,
    max_nudges: int = DEFAULT_MAX_NUDGES,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_sequence_tokens: int = 2048,
    chunk_size: int = 64,
    kl_beta: float = 0.04,
    clip_epsilon: float = 0.2,
    keep_degenerate: bool = False,
    normalize_by_std: bool = True,
    tool_workers: int = 8,
    progress: bool = False,
) -> StepReport:
    """Roll out, score, and accumulate one optimiser step's worth of gradient.

    Does not call `optimizer.step()`. The caller owns the optimiser so that
    clipping, scheduling and the zeroing are in one place and visible.
    """
    import torch

    from rl_training.objective import (
        grpo_backward,
        prepare_for_generation,
        prepare_for_training,
        reference_logprobs,
        shift_for_prediction,
    )
    from rl_training.policy import TurnRecorder

    # --- roll out ----------------------------------------------------------
    prepare_for_generation(policy.model)
    episodes = make_episodes(
        tasks, registry, rollouts=group_size, max_calls=max_calls, few_shot=True
    )
    # Where the policy's own turns begin. Every episode has the same prefix
    # length, but it is read off an episode rather than recomputed, because a
    # prompt-construction change would otherwise silently shift the mask.
    prompt_messages = len(episodes[0].messages)

    recorder = TurnRecorder()
    started = time.perf_counter()
    run_episodes(
        episodes,
        policy,
        max_new_tokens=max_new_tokens,
        max_nudges=max_nudges,
        tool_workers=tool_workers,
        progress=progress,
        on_turn=lambda pending, _texts: recorder.record(pending, policy.last_token_ids),
    )
    seconds_rollout = time.perf_counter() - started

    results = score_episodes(
        episodes,
        code_runner=code_runner,
        workers=tool_workers,
        allow_unconfined=allow_unconfined,
    )
    rewards = [result.reward for result in results]
    advantages, stats = group_advantages(
        rewards,
        [f"{episode.task.task_id}" for episode in episodes],
        normalize_by_std=normalize_by_std,
    )

    # Generation holds a KV cache for the whole batch; training holds
    # activations. They do not have to coexist and on this card they must not.
    if policy.device == "cuda":
        torch.cuda.empty_cache()

    # --- build the trainable batch ----------------------------------------
    kept: list[tuple[Episode, float]] = []
    for episode, advantage in zip(episodes, advantages):
        if advantage == 0.0 and not keep_degenerate:
            continue
        kept.append((episode, advantage))

    transcripts = []
    for episode, advantage in kept:
        transcript = build_transcript(
            episode.messages,
            policy.tokenizer,
            sampled_token_ids=recorder.aligned_with(episode),
            max_tokens=max_sequence_tokens,
            first_completion_index=prompt_messages,
        )
        if transcript.n_completion_tokens == 0:
            # Nothing the policy wrote survived truncation. Training on it
            # would be a backward pass over an all-zero mask.
            continue
        transcripts.append((transcript, advantage))

    report_common = dict(
        step=step,
        episodes=len(episodes),
        kept_episodes=len(transcripts),
        dropped_degenerate=len(episodes) - len(kept),
        groups=stats.n_groups,
        degenerate_groups=stats.n_degenerate,
        mean_reward=round(stats.mean_reward, 4),
        per_category_reward=_per_category_reward(results),
        tool_calls=_tool_call_counts(results),
        seconds_rollout=round(seconds_rollout, 2),
        truncated=sum(1 for t, _ in transcripts if t.is_truncated),
        retokenized=sum(1 for t, _ in transcripts if t.source != "sampled"),
        overshoot_chars=sum(t.overshoot_chars for t, _ in transcripts),
    )

    if not transcripts:
        return StepReport(
            **report_common,
            loss=0.0, grad_norm=0.0, mean_kl=0.0, completion_tokens=0.0,
            microbatches=0, seconds_backward=0.0, peak_gib=0.0,
            reserved_gib=0.0, spilled=False,
        )

    # --- shape the microbatches -------------------------------------------
    device = policy.device
    pad_id = policy.tokenizer.pad_token_id
    lengths = [transcript.n_tokens for transcript, _ in transcripts]
    # Normalizer over the whole step, not per microbatch: gradient accumulation
    # across microbatches has to sum to the same thing a single large batch
    # would have produced.
    total_completion_tokens = float(
        sum(
            sum(transcript.completion_mask[1:])
            for transcript, _ in transcripts
        )
    ) or 1.0

    batches = pack_microbatches(lengths)

    def shape(indices: list[int]):
        width = max(lengths[i] for i in indices)
        input_ids = torch.full((len(indices), width), pad_id, dtype=torch.long)
        attention_mask = torch.zeros((len(indices), width), dtype=torch.long)
        completion_mask = torch.zeros((len(indices), width), dtype=torch.long)
        for row, index in enumerate(indices):
            transcript, _ = transcripts[index]
            span = transcript.n_tokens
            # Right padding: unlike generation, nothing is appended after this
            # sequence, and the loss mask zeroes the pad either way.
            input_ids[row, :span] = torch.tensor(transcript.token_ids)
            attention_mask[row, :span] = 1
            completion_mask[row, :span] = torch.tensor(transcript.completion_mask)
        return (
            input_ids.to(device),
            attention_mask.to(device),
            completion_mask.to(device),
            torch.tensor(
                [transcripts[i][1] for i in indices],
                dtype=torch.float32,
                device=device,
            ),
        )

    shaped = [shape(indices) for indices in batches]

    # --- the KL reference, before the model is reconfigured ----------------
    #
    # Deliberately a separate pass, while the model is still in generation
    # configuration. Gradient checkpointing is on in training configuration,
    # and running a checkpointed forward under `no_grad` makes
    # `torch.utils.checkpoint` warn on every layer of every microbatch that
    # none of its inputs require grad - noise that would bury the step log,
    # for a recomputation that buys nothing when there is no backward to feed.
    # The result is [B, T] floats, so holding all of them costs kilobytes.
    references: list[Any] = [None] * len(shaped)
    if kl_beta != 0.0:
        for index, (input_ids, attention_mask, completion_mask, _) in enumerate(shaped):
            targets, _ = shift_for_prediction(input_ids, completion_mask)
            references[index] = reference_logprobs(
                policy.model, input_ids, attention_mask, targets, chunk_size=chunk_size
            )

    # --- accumulate the gradient ------------------------------------------
    prepare_for_training(policy.model)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    total_loss = 0.0
    total_kl = 0.0
    total_tokens = 0.0

    for (input_ids, attention_mask, completion_mask, advantage_tensor), reference in zip(
        shaped, references
    ):
        result = grpo_backward(
            policy.model,
            input_ids,
            attention_mask,
            completion_mask,
            advantage_tensor,
            ref_logp=reference,
            chunk_size=chunk_size,
            clip_epsilon=clip_epsilon,
            kl_beta=kl_beta,
            normalizer=total_completion_tokens,
        )
        total_loss += result.loss
        total_kl += result.mean_kl * result.tokens
        total_tokens += result.tokens

    seconds_backward = time.perf_counter() - started

    if device == "cuda":
        reserved = torch.cuda.memory_reserved()
        capacity = torch.cuda.get_device_properties(0).total_memory
        peak_gib = torch.cuda.max_memory_allocated() / 2**30
        reserved_gib = reserved / 2**30
        # The only reliable spill check. Windows WDDM backs an over-full
        # allocation with system RAM instead of raising, so exceeding the card
        # costs up to 11x throughput and reports nothing at all.
        spilled = bool(reserved > capacity)
    else:
        peak_gib = reserved_gib = 0.0
        spilled = False

    return StepReport(
        **report_common,
        loss=round(total_loss, 6),
        grad_norm=0.0,
        mean_kl=round(total_kl / total_tokens, 6) if total_tokens else 0.0,
        completion_tokens=total_tokens,
        microbatches=len(batches),
        seconds_backward=round(seconds_backward, 2),
        peak_gib=round(peak_gib, 3),
        reserved_gib=round(reserved_gib, 3),
        spilled=spilled,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GRPO-train the tool-use policy.")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--prompts-per-step", type=int, default=4)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--kl-beta", type=float, default=0.04)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--max-sequence-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--batch-size", type=int, default=16, help="generation batch")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES)
    parser.add_argument("--keep-degenerate", action="store_true")
    parser.add_argument("--no-std-normalization", action="store_true")
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("artifacts/rl/run"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--allow-unconfined", action="store_true")
    parser.add_argument("--no-code-tool", action="store_true")
    parser.add_argument("--progress", action="store_true")
    arguments = parser.parse_args(argv)

    tasks = load_suite()["train"]
    if arguments.categories:
        tasks = [task for task in tasks if task.category in arguments.categories]
    if not tasks:
        print("no tasks selected", file=sys.stderr)
        return 1

    # Same preflight as module 4, for the same reason and one module later: a
    # coding task scored through a sandbox that cannot confine the filesystem
    # is a score the submission could have read off disk, and `verify` raises
    # rather than returning a number. Finding that out after an hour of
    # training would cost the run.
    wants_code = any(task.category == "code" for task in tasks)
    if wants_code and not code_exec.sandbox_available():
        print(
            f"selection includes coding tasks but no sandbox exists on {sys.platform!r}",
            file=sys.stderr,
        )
        return 1

    include_code = not arguments.no_code_tool and code_exec.sandbox_available()
    runner = code_exec.SandboxedCodeRunner() if code_exec.sandbox_available() else None
    registry = build_registry(include_code=include_code, code_runner=runner)

    if runner is not None:
        probe = runner("print('forge')")
        if probe.exit_code != 0:
            print(f"sandbox preflight failed: {probe.stderr[-400:]}", file=sys.stderr)
            return 1
        if wants_code and not arguments.allow_unconfined and not runner.confines_filesystem:
            print(
                f"the {code_exec.sandbox_backend()} sandbox does not confine the "
                "filesystem, so a coding reward here could have been read off "
                "disk rather than earned. Refusing to train on it. Pass "
                "--allow-unconfined to proceed with that caveat recorded, or "
                "select --categories without 'code'.",
                file=sys.stderr,
            )
            return 1

    import torch

    from rl_training.policy import LoRAPolicy

    print(
        f"{len(tasks)} train tasks {category_counts(tasks)}, "
        f"{arguments.prompts_per_step} prompts x {arguments.group_size} rollouts "
        f"per step, {arguments.steps} steps",
        file=sys.stderr,
        flush=True,
    )

    policy = LoRAPolicy(
        arguments.model,
        rank=arguments.rank,
        temperature=arguments.temperature,
        top_p=arguments.top_p,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
        adapter_path=str(arguments.resume) if arguments.resume else None,
    )
    trainable = [p for p in policy.model.parameters() if p.requires_grad]
    if not trainable:
        print("no trainable parameters; the adapter did not attach", file=sys.stderr)
        return 1
    optimizer = torch.optim.AdamW(trainable, lr=arguments.learning_rate)

    sampler = CategorySampler(tasks, seed=arguments.seed)
    arguments.out.mkdir(parents=True, exist_ok=True)
    log = RunLog(arguments.out / "steps.jsonl")
    log.write(
        {
            "record": "config",
            "policy": policy.describe(),
            "tasks": category_counts(tasks),
            "sandbox": {
                "backend": code_exec.sandbox_backend() if runner else None,
                "confinement": code_exec.confinement().summary() if runner else None,
                "allow_unconfined": arguments.allow_unconfined,
            },
            "arguments": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(arguments).items()
            },
        }
    )

    started = time.perf_counter()
    try:
        for step in range(1, arguments.steps + 1):
            report = train_step(
                policy,
                sampler.batch(arguments.prompts_per_step),
                registry,
                step=step,
                group_size=arguments.group_size,
                code_runner=runner,
                allow_unconfined=arguments.allow_unconfined,
                max_new_tokens=arguments.max_new_tokens,
                max_sequence_tokens=arguments.max_sequence_tokens,
                chunk_size=arguments.chunk_size,
                kl_beta=arguments.kl_beta,
                clip_epsilon=arguments.clip_epsilon,
                keep_degenerate=arguments.keep_degenerate,
                normalize_by_std=not arguments.no_std_normalization,
                progress=arguments.progress,
            )

            if report.kept_episodes:
                norm = torch.nn.utils.clip_grad_norm_(trainable, arguments.max_grad_norm)
                report.grad_norm = round(float(norm), 6)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            record = {"record": "step", **report.to_dict()}
            log.write(record)
            print(
                f"step {step}/{arguments.steps}  reward {report.mean_reward:.4f}  "
                f"usable {report.groups - report.degenerate_groups}/{report.groups}  "
                f"loss {report.loss:+.4f}  kl {report.mean_kl:.5f}  "
                f"peak {report.peak_gib:.2f}GiB"
                + ("  SPILLED" if report.spilled else ""),
                file=sys.stderr,
                flush=True,
            )
            if report.spilled:
                print(
                    "  the card is full and CUDA is being backed by system RAM; "
                    "lower --max-sequence-tokens or --group-size",
                    file=sys.stderr,
                    flush=True,
                )

            if arguments.checkpoint_every and step % arguments.checkpoint_every == 0:
                save_checkpoint(policy, arguments.out, f"adapter-step{step}")
    finally:
        save_checkpoint(policy, arguments.out, "adapter-final")
        log.write(
            {
                "record": "summary",
                "wall_clock_seconds": round(time.perf_counter() - started, 1),
            }
        )
        log.close()

    print(f"wrote {arguments.out}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
