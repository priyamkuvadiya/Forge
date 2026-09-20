"""Measure the 8GB budget with the code that will actually spend it.

Every memory claim in `objective.py` is a claim about this card, and a comment
asserting one is worth nothing - the project has already been bitten twice by
exactly that (a "thread-safe" note that was false, and a security test that
asserted a property it never probed). So the claims get re-derived here, by the
real model, through the real functions, and written to a JSON file that a
README number can point at.

Four things get measured, and each one is a claim that could come back false:

  1. **Train mode is worth ~18x.** `from_pretrained` returns a model in eval
     mode and gradient checkpointing is a silent no-op there. Measured as body
     forward peak in each mode.
  2. **Chunked backward beats naive, and by how much.** The naive path is
     expected to OOM or spill at the working batch sizes; that is the result,
     not a failure, and it is recorded rather than crashed on.
  3. **The two agree numerically on the real model.** The CPU toy-model test
     in `tests/test_objective.py` already checks this, but the real model has
     bf16 activations, gradient checkpointing and a tied-or-untied head, none
     of which the toy has.
  4. **Nothing spills.** Windows WDDM backs an over-full CUDA allocation with
     system RAM instead of raising, at up to 11x the latency and with no error
     anywhere, so `memory_reserved()` is compared against the card's actual
     capacity after every configuration.

    python -m rl_training.measure_budget --out artifacts/rl/budget.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# All attention and MLP projections, which is the configuration the 8.8M
# trainable parameter count in the project notes was measured at.
LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]

# The configurations the budget table covers, plus the two that are expected to
# fail. Keeping the failures in is the point: a table with only the successes
# is a table that cannot tell you where the edge is.
CONFIGURATIONS = [(8, 1024), (16, 1024), (4, 2048), (8, 2048), (4, 4096)]


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def _capacity(torch) -> int:
    return torch.cuda.get_device_properties(0).total_memory


def _reset(torch) -> None:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _snapshot(torch) -> dict[str, Any]:
    torch.cuda.synchronize()
    reserved = torch.cuda.memory_reserved()
    capacity = _capacity(torch)
    return {
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
        "reserved_gib": round(reserved / 2**30, 3),
        # The only reliable spill check. Power draw is not the tell - a
        # spilling run pulls slightly *more* watts than a healthy one.
        "spilled": bool(reserved > capacity),
    }


def build_model(model_name: str, rank: int = 16):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to("cuda")
    model = get_peft_model(
        model,
        LoraConfig(
            r=rank,
            lora_alpha=2 * rank,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGETS,
        ),
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return model, {"trainable": trainable, "total": total}


def measure_train_mode_matters(model, batch: int, seq: int) -> dict[str, Any]:
    """Claim 1: checkpointing does nothing in eval mode, and that costs 18x.

    Batch 2, not 4. The eval-mode half of this is the *expensive* half by
    construction - it is what the claim is about - and at batch 4 x 1024 it
    needs ~14 GiB and takes the whole script down with a CUDA OOM before
    anything is written. Measuring the pathological configuration means
    picking one small enough that the pathology still fits.
    """
    import torch

    from rl_training.objective import _body_and_head, prepare_for_training

    body, _ = _body_and_head(model)
    input_ids = torch.randint(0, 1000, (batch, seq), device="cuda")
    attention_mask = torch.ones_like(input_ids)

    results: dict[str, Any] = {"batch": batch, "seq": seq}
    for mode in ("eval", "train"):
        model.gradient_checkpointing_enable()
        if mode == "eval":
            model.eval()
        else:
            prepare_for_training(model)

        _reset(torch)
        try:
            hidden = body(input_ids=input_ids, attention_mask=attention_mask)[0]
            loss = hidden.float().pow(2).mean()
            loss.backward()
            results[mode] = _snapshot(torch)
            del hidden, loss
        except Exception as error:  # noqa: BLE001 - re-raised unless it is an OOM
            if not _is_oom(error):
                raise
            results[mode] = {"outcome": "cuda_oom", "detail": str(error).split("\n")[0][:200]}
        model.zero_grad(set_to_none=True)
        _reset(torch)

    eval_peak = results["eval"].get("peak_allocated_gib")
    train_peak = results["train"].get("peak_allocated_gib")
    results["ratio"] = (
        round(eval_peak / train_peak, 2) if eval_peak and train_peak else None
    )
    return results


# Above this, the naive path is not attempted at all. Not caution for its own
# sake: on Windows an allocation the card cannot hold is backed by system RAM
# rather than refused, so the naive row would not fail fast - it would crawl
# for hours at a fraction of the throughput and report a "fits" that means
# nothing. Refusing it up front and recording the projection is the honest
# version of the same measurement.
NAIVE_LOGIT_CEILING_GIB = 6.0


def projected_logit_gib(batch: int, seq: int, vocab: int) -> float:
    """What the naive path would have to hold: fp32 logits, plus their gradient."""
    return 2 * batch * (seq - 1) * vocab * 4 / 2**30


def _is_oom(error: BaseException) -> bool:
    oom_type = getattr(__import__("torch"), "OutOfMemoryError", None)
    if oom_type is not None and isinstance(error, oom_type):
        return True
    return isinstance(error, RuntimeError) and "out of memory" in str(error).lower()


def measure_step(model, batch: int, seq: int, *, mode: str, chunk_size: int) -> dict[str, Any]:
    """One optimiser step's worth of forward and backward, peak recorded."""
    import torch

    from rl_training.objective import grpo_backward, naive_grpo_backward, prepare_for_training

    projected = projected_logit_gib(batch, seq, model.config.vocab_size)
    if mode == "naive" and projected > NAIVE_LOGIT_CEILING_GIB:
        return {
            "batch": batch, "seq": seq, "mode": mode,
            "outcome": "not attempted",
            "projected_logit_gib": round(projected, 2),
            "detail": (
                f"would need {projected:.1f} GiB of fp32 logits and gradient on an "
                f"{_capacity(torch) / 2**30:.1f} GiB card; on Windows that is backed "
                "by system RAM rather than refused, so it would crawl, not fail"
            ),
        }

    prepare_for_training(model)
    torch.manual_seed(0)
    input_ids = torch.randint(0, 1000, (batch, seq), device="cuda")
    attention_mask = torch.ones_like(input_ids)
    completion_mask = torch.zeros_like(input_ids)
    # Half the sequence is the policy's, which is generous: real transcripts
    # are mostly prompt and tool output, so this over-estimates the head cost.
    completion_mask[:, seq // 2 :] = 1
    advantages = torch.randn(batch, device="cuda")

    _reset(torch)
    started = time.perf_counter()
    try:
        if mode == "chunked":
            report = grpo_backward(
                model, input_ids, attention_mask, completion_mask, advantages,
                chunk_size=chunk_size, kl_beta=0.0,
            )
        else:
            report = naive_grpo_backward(
                model, input_ids, attention_mask, completion_mask, advantages,
                kl_beta=0.0,
            )
    except Exception as error:  # noqa: BLE001 - re-raised unless it is an OOM
        if not _is_oom(error):
            raise
        model.zero_grad(set_to_none=True)
        _reset(torch)
        return {
            "batch": batch, "seq": seq, "mode": mode,
            "outcome": "cuda_oom",
            "projected_logit_gib": round(projected, 2),
            "detail": str(error).split("\n")[0][:200],
        }

    elapsed = time.perf_counter() - started
    snapshot = _snapshot(torch)
    model.zero_grad(set_to_none=True)
    _reset(torch)

    return {
        "batch": batch, "seq": seq, "mode": mode,
        "outcome": "spilled" if snapshot["spilled"] else "fits",
        "seconds": round(elapsed, 3),
        "projected_logit_gib": round(projected, 2),
        **snapshot,
        **report.to_dict(),
    }


def measure_equivalence(model, batch: int, seq: int, chunk_size: int) -> dict[str, Any]:
    """Claim 3: chunked and naive agree on the real model, not just the toy."""
    import torch

    from rl_training.objective import grpo_backward, naive_grpo_backward, prepare_for_training

    prepare_for_training(model)
    torch.manual_seed(0)
    input_ids = torch.randint(0, 1000, (batch, seq), device="cuda")
    attention_mask = torch.ones_like(input_ids)
    completion_mask = torch.zeros_like(input_ids)
    completion_mask[:, seq // 2 :] = 1
    advantages = torch.randn(batch, device="cuda")

    def run(fn, **kwargs):
        model.zero_grad(set_to_none=True)
        _reset(torch)
        report = fn(model, input_ids, attention_mask, completion_mask, advantages, **kwargs)
        grads = {
            name: parameter.grad.detach().float().clone()
            for name, parameter in model.named_parameters()
            if parameter.grad is not None
        }
        return report, grads

    chunked, chunked_grads = run(grpo_backward, chunk_size=chunk_size, kl_beta=0.0)
    naive, naive_grads = run(naive_grpo_backward, kl_beta=0.0)

    if set(chunked_grads) != set(naive_grads):
        return {
            "outcome": "parameter sets differ",
            "only_chunked": sorted(set(chunked_grads) - set(naive_grads))[:5],
            "only_naive": sorted(set(naive_grads) - set(chunked_grads))[:5],
        }

    worst_name, worst = None, 0.0
    for name, grad in chunked_grads.items():
        difference = (grad - naive_grads[name]).abs().max().item()
        if difference > worst:
            worst_name, worst = name, difference

    scale = max(
        (grad.abs().max().item() for grad in naive_grads.values()), default=0.0
    )
    model.zero_grad(set_to_none=True)
    _reset(torch)
    return {
        "outcome": "compared",
        "n_parameters_with_grad": len(chunked_grads),
        "loss_chunked": round(chunked.loss, 8),
        "loss_naive": round(naive.loss, 8),
        "loss_abs_difference": abs(chunked.loss - naive.loss),
        "max_abs_grad_difference": worst,
        "max_abs_grad_difference_at": worst_name,
        "largest_gradient_magnitude": scale,
        "relative": worst / scale if scale else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=64)
    parser.add_argument("--out", type=Path, default=Path("artifacts/rl/budget.json"))
    parser.add_argument(
        "--skip-naive",
        action="store_true",
        help="skip the naive comparison rows, which are the ones that OOM",
    )
    arguments = parser.parse_args(argv)

    import torch

    if not torch.cuda.is_available():
        print("no CUDA device; this script only measures the training card", file=sys.stderr)
        return 2

    _log(f"loading {arguments.model} with LoRA r={arguments.rank}")
    model, parameters = build_model(arguments.model, arguments.rank)
    _log(f"  {parameters['trainable']:,} trainable of {parameters['total']:,}")

    record: dict[str, Any] = {
        "generated_by": "rl_training.measure_budget",
        "model": arguments.model,
        "device": torch.cuda.get_device_name(0),
        "capacity_gib": round(_capacity(torch) / 2**30, 3),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "lora": {"rank": arguments.rank, "targets": LORA_TARGETS, **parameters},
        "chunk_size": arguments.chunk_size,
    }

    arguments.out.parent.mkdir(parents=True, exist_ok=True)

    def checkpoint() -> None:
        """Write after every phase, not once at the end.

        The first run of this script died on a CUDA OOM in its *first*
        measurement and produced no file at all, which is the same lesson
        `baseline_agent.save_rollouts` already learned: expensive output that
        is only written after the last thing that can raise is output you lose.
        """
        temporary = arguments.out.with_suffix(arguments.out.suffix + ".tmp")
        temporary.write_text(json.dumps(record, indent=2), encoding="utf-8")
        os.replace(temporary, arguments.out)

    record["steps"] = []
    checkpoint()

    _log("measuring whether train mode changes the body's memory")
    record["train_mode"] = measure_train_mode_matters(model, 2, 1024)
    checkpoint()
    _log(f"  eval {record['train_mode']['eval'].get('peak_allocated_gib', 'OOM')} GiB vs "
         f"train {record['train_mode']['train'].get('peak_allocated_gib', 'OOM')} GiB "
         f"(ratio {record['train_mode']['ratio']})")

    _log("checking chunked and naive agree on the real model")
    record["equivalence"] = measure_equivalence(model, 2, 256, arguments.chunk_size)
    checkpoint()
    _log(f"  {record['equivalence']}")

    for batch, seq in CONFIGURATIONS:
        for mode in ("chunked",) if arguments.skip_naive else ("chunked", "naive"):
            _log(f"measuring {mode} at batch {batch} x {seq}")
            row = measure_step(model, batch, seq, mode=mode, chunk_size=arguments.chunk_size)
            _log(f"  {row['outcome']}  peak {row.get('peak_allocated_gib', '-')} GiB")
            record["steps"].append(row)
            checkpoint()

    _log(f"wrote {arguments.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
