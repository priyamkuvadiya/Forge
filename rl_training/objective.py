"""The GRPO objective, and the memory choreography that makes it fit in 8GB.

Two separate things live here, and they are separate on purpose.

The **objective** is a handful of pure tensor functions - shift, log-softmax,
gather, a clipped ratio, a KL estimator. They take tensors and return tensors,
they run on CPU, and they are checked against numbers worked out by hand. That
is the part an interview walks through line by line, and it is about forty
lines of actual maths.

The **choreography** is the part that is only necessary because of this card,
and it is where the measured facts from the 8GB budget work show up as code:

  1. `from_pretrained` hands back a model in **eval mode**, and
     `transformers` 5.x guards gradient checkpointing on
     `self.gradient_checkpointing and self.training`. A missing `.train()`
     means checkpointing silently does nothing - no warning, every flag still
     reading True - and the body's forward goes from 0.40 GB to 7.27 GB.
     `prepare_for_training` asserts the checkpointing actually engaged rather
     than trusting the flag, because the flag is the thing that lies.

  2. **The vocabulary projection is the big tensor, not the transformer.**
     Qwen2.5-0.5B has hidden size 896 and a vocabulary of 151,936, so the
     logits for one 1024-token sequence are ~170x the size of the hidden
     states that produced them - and the loss upcasts them to fp32 and needs a
     gradient of the same shape. Chunking the head is not enough on its own:
     autograd keeps every chunk's graph alive until the single backward, so
     all the chunks end up resident anyway and nothing is saved.

     So `grpo_backward` runs **backward per chunk** against a detached copy of
     the hidden states, accumulating into that copy's `.grad`, and only then
     pushes the accumulated gradient through the body once. Peak logit memory
     becomes one chunk's worth instead of the whole sequence's, and the body is
     still traversed exactly once.

The KL term needs the reference policy's logprobs, and with LoRA that costs no
second copy of the model: `disable_adapter()` turns the same resident weights
back into the base model. It does mean the reference pass is a second forward,
which is why it is done under `no_grad` and chunked to a per-token vector -
[B, T] floats, not [B, T, 151936].
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F

# One chunk of 64 positions was the configuration the budget table was measured
# at. Smaller trades throughput for headroom; larger is what OOMs.
DEFAULT_CHUNK_SIZE = 64

# PPO/GRPO clipping. Inert at one update per batch - see `grpo_token_loss`.
DEFAULT_CLIP_EPSILON = 0.2

# Weight on the KL to the frozen base policy.
DEFAULT_KL_BETA = 0.04


def shift_for_prediction(
    input_ids: torch.Tensor, completion_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Line up positions with the tokens they predict.

    A causal LM at position `t` predicts token `t + 1`, so the targets are
    `input_ids[:, 1:]` and the logits come from positions `[:, :-1]`. The mask
    shifts with the *targets*, not the inputs: a token is in the objective when
    the token being predicted is one the policy wrote, which is not the same as
    when the token being conditioned on is.

    This shift happens here and nowhere else. `transcript.py` deliberately
    returns an unshifted mask over inputs, because doing it in both places is
    exactly the off-by-one that would train the model on the first token of
    every tool result - the one token whose predecessor is the policy's.
    """
    return input_ids[:, 1:], completion_mask[:, 1:]


def token_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """log p(target) per position, upcast to fp32.

    bf16 has ~3 decimal digits of mantissa. A log-softmax over 151,936 classes
    in bf16 loses enough precision that the ratio `exp(logp - old_logp)`, which
    should be exactly 1 on the first update, comes out visibly off it - so the
    clipping fires on numerical noise. Upcasting costs one chunk's worth of
    fp32 logits, which is what the chunking is for.
    """
    logits = logits.float()
    return torch.gather(
        F.log_softmax(logits, dim=-1), dim=-1, index=targets.unsqueeze(-1)
    ).squeeze(-1)


def k3_kl(logp: torch.Tensor, ref_logp: torch.Tensor) -> torch.Tensor:
    """Schulman's k3 estimator of KL(policy || reference), per token.

    `exp(d) - d - 1` where `d = ref - policy`. Non-negative by construction,
    unlike the naive `ref - policy` difference, which is an unbiased estimate
    of the KL but takes both signs sample by sample - so a batch of it can
    report a negative KL and a penalty that pays the policy to move away from
    the reference.
    """
    delta = ref_logp - logp
    return torch.exp(delta) - delta - 1.0


def grpo_token_loss(
    logp: torch.Tensor,
    old_logp: torch.Tensor,
    ref_logp: torch.Tensor | None,
    advantages: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    clip_epsilon: float = DEFAULT_CLIP_EPSILON,
    kl_beta: float = DEFAULT_KL_BETA,
    normalizer: torch.Tensor | float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """The GRPO surrogate for one block of positions.

    `advantages` is per-token but constant along a sequence - GRPO assigns one
    advantage to a whole rollout, because the reward is only known at the end
    of it. Broadcasting it here rather than indexing keeps this function
    shape-agnostic, so it can be called on a chunk or on a whole batch and
    return the same thing.

    **The clipping is inert in the default configuration and that is not a
    bug.** With one optimiser step per batch of rollouts, `old_logp` is
    `logp.detach()` from the same forward pass, the ratio is exactly 1.0, and
    the objective reduces to the plain policy gradient `-A * logp`. The clipped
    form is written out anyway because it is what makes more than one inner
    epoch safe, and because a reader should be able to see which term is doing
    the work rather than take a comment's word for it.

    `normalizer` is passed in rather than computed from the mask, because with
    chunked backward a chunk does not know the batch's total token count and a
    per-chunk normalizer would weight short chunks like long ones.
    """
    ratio = torch.exp(logp - old_logp)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
    per_token = -torch.min(unclipped, clipped)

    kl = None
    if ref_logp is not None and kl_beta != 0.0:
        kl = k3_kl(logp, ref_logp)
        per_token = per_token + kl_beta * kl

    masked = per_token * loss_mask
    loss = masked.sum() / normalizer

    n_tokens = loss_mask.sum()
    diagnostics = {
        "tokens": float(n_tokens.item()),
        "mean_ratio": float(
            ((ratio * loss_mask).sum() / n_tokens.clamp(min=1)).item()
        ),
        "clip_fraction": float(
            (((unclipped > clipped).float() * loss_mask).sum() / n_tokens.clamp(min=1))
            .item()
        ),
        "mean_kl": float(
            ((kl * loss_mask).sum() / n_tokens.clamp(min=1)).item()
        )
        if kl is not None
        else 0.0,
    }
    return loss, diagnostics


def _body_and_head(model: Any) -> tuple[Any, Any]:
    """Split a causal LM into its transformer and its vocabulary projection.

    Goes through `get_decoder` / `get_output_embeddings` rather than reaching
    for `model.model` and `model.lm_head`, because under a PEFT wrapper those
    attribute paths gain a level and the wrapper forwards the accessors.
    """
    body = model.get_decoder()
    head = model.get_output_embeddings()
    if body is None or head is None:
        raise TypeError(
            f"{type(model).__name__} does not expose a decoder and an output "
            "embedding; chunked backward needs both halves separately"
        )
    return body, head


def prepare_for_training(model: Any) -> dict[str, Any]:
    """Put the model in the state the memory budget was measured in, and check.

    Returns what it verified, so a training run can record it. The check is the
    point: `model.gradient_checkpointing` on the parent is a stale attribute in
    transformers 5.x, and reading it back is how a run ends up quietly spending
    18x the memory it budgeted for. The layers are what decide, so the layers
    are what get asserted.
    """
    model.train()
    model.gradient_checkpointing_enable()

    body, _ = _body_and_head(model)
    layers = list(getattr(body, "layers", []))
    engaged = [bool(getattr(layer, "gradient_checkpointing", False)) for layer in layers]

    if layers and not all(engaged):
        raise RuntimeError(
            f"gradient checkpointing engaged on {sum(engaged)}/{len(layers)} "
            "layers. On this card the difference is 0.40 GB of activations "
            "against 7.27 GB, so this is not a warning."
        )
    if not model.training:
        raise RuntimeError(
            "model is not in train mode; checkpointing is a no-op in eval mode"
        )

    return {
        "training": bool(model.training),
        "layers": len(layers),
        "layers_checkpointing": sum(engaged),
    }


@torch.no_grad()
def reference_logprobs(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    targets: torch.Tensor,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    use_base_policy: bool = True,
) -> torch.Tensor:
    """Per-token logprobs under the frozen base policy.

    With LoRA there is no second model to hold: `disable_adapter()` makes the
    resident weights the base model for the duration of the block, so the KL
    reference costs a forward pass and no VRAM. That is also why this cannot be
    computed once and cached across steps - it can, in fact, since the base
    policy never changes, and a training loop should do exactly that; it is not
    done here because this function does not know the batch's identity.
    """
    body, head = _body_and_head(model)
    context = (
        model.disable_adapter()
        if use_base_policy and hasattr(model, "disable_adapter")
        else nullcontext()
    )

    with context:
        hidden = body(input_ids=input_ids, attention_mask=attention_mask)[0]
        hidden = hidden[:, :-1]

        out = torch.empty(targets.shape, dtype=torch.float32, device=targets.device)
        for start in range(0, hidden.shape[1], chunk_size):
            stop = start + chunk_size
            out[:, start:stop] = token_logprobs(
                head(hidden[:, start:stop]), targets[:, start:stop]
            )
    return out


@dataclass
class BackwardReport:
    """What one chunked backward actually did, for the step log."""

    loss: float
    tokens: float
    chunks: int
    mean_ratio: float
    clip_fraction: float
    mean_kl: float
    peak_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": round(self.loss, 6),
            "tokens": self.tokens,
            "chunks": self.chunks,
            "mean_ratio": round(self.mean_ratio, 6),
            "clip_fraction": round(self.clip_fraction, 6),
            "mean_kl": round(self.mean_kl, 6),
            "peak_gib": round(self.peak_bytes / 2**30, 3),
        }


def grpo_backward(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    *,
    ref_logp: torch.Tensor | None = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    clip_epsilon: float = DEFAULT_CLIP_EPSILON,
    kl_beta: float = DEFAULT_KL_BETA,
    normalizer: float | None = None,
) -> BackwardReport:
    """Accumulate GRPO gradients without ever holding the whole logit tensor.

    Runs the body once, detaches, walks the sequence in chunks running a
    backward on each, then pushes the accumulated hidden-state gradient back
    through the body in a single pass. Gradients land in `.grad` as usual; the
    caller owns the optimiser step and the zeroing.

    `advantages` is one scalar per sequence, shape [B].
    """
    body, head = _body_and_head(model)
    targets, loss_mask = shift_for_prediction(input_ids, completion_mask)
    loss_mask = loss_mask.float()

    if normalizer is None:
        # Total completion tokens in the batch: every token gets equal weight,
        # so a long rollout contributes more than a short one *in proportion to
        # how much of it the policy wrote*. Dividing per sequence by its own
        # length instead makes a 40-token answer count as much as a 700-token
        # one, which on this suite is a systematic thumb on the scale towards
        # the `no_tool` category.
        normalizer = float(loss_mask.sum().clamp(min=1.0).item())

    hidden = body(input_ids=input_ids, attention_mask=attention_mask)[0]
    detached = hidden[:, :-1].detach().requires_grad_(True)

    per_token_advantages = advantages.unsqueeze(1).float()

    total_loss = 0.0
    totals = {"tokens": 0.0, "ratio": 0.0, "clip": 0.0, "kl": 0.0}
    chunks = 0

    for start in range(0, detached.shape[1], chunk_size):
        stop = start + chunk_size
        chunk_mask = loss_mask[:, start:stop]
        if float(chunk_mask.sum()) == 0.0:
            # Nothing in this block belongs to the policy - all prompt, tool
            # output or padding. Running the head over it would cost a full
            # chunk of fp32 logits to multiply by zero.
            continue

        logp = token_logprobs(
            head(detached[:, start:stop]), targets[:, start:stop]
        )
        loss, diagnostics = grpo_token_loss(
            logp,
            logp.detach(),
            ref_logp[:, start:stop] if ref_logp is not None else None,
            per_token_advantages,
            chunk_mask,
            clip_epsilon=clip_epsilon,
            kl_beta=kl_beta,
            normalizer=normalizer,
        )
        loss.backward()

        total_loss += float(loss.item())
        weight = diagnostics["tokens"]
        totals["tokens"] += weight
        totals["ratio"] += diagnostics["mean_ratio"] * weight
        totals["clip"] += diagnostics["clip_fraction"] * weight
        totals["kl"] += diagnostics["mean_kl"] * weight
        chunks += 1

    if chunks and detached.grad is not None:
        hidden[:, :-1].backward(gradient=detached.grad)

    tokens = totals["tokens"] or 1.0
    return BackwardReport(
        loss=total_loss,
        tokens=totals["tokens"],
        chunks=chunks,
        mean_ratio=totals["ratio"] / tokens,
        clip_fraction=totals["clip"] / tokens,
        mean_kl=totals["kl"] / tokens,
        peak_bytes=(
            torch.cuda.max_memory_allocated() if input_ids.is_cuda else 0
        ),
    )


def naive_grpo_backward(
    model: Any,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    completion_mask: torch.Tensor,
    advantages: torch.Tensor,
    *,
    ref_logp: torch.Tensor | None = None,
    clip_epsilon: float = DEFAULT_CLIP_EPSILON,
    kl_beta: float = DEFAULT_KL_BETA,
    normalizer: float | None = None,
) -> BackwardReport:
    """The same objective, computed the obvious way, for the equivalence test.

    Holds the whole [B, T, 151936] fp32 logit tensor and its gradient, which is
    precisely what the chunked version exists to avoid - so this is a test
    fixture and a reference implementation, not a fallback. It is here because
    "the chunked version computes the same thing" is a claim that should be
    checked against something, and checking it against a second copy of the
    chunked code would check nothing.
    """
    targets, loss_mask = shift_for_prediction(input_ids, completion_mask)
    loss_mask = loss_mask.float()
    if normalizer is None:
        normalizer = float(loss_mask.sum().clamp(min=1.0).item())

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1]
    logp = token_logprobs(logits, targets)

    loss, diagnostics = grpo_token_loss(
        logp,
        logp.detach(),
        ref_logp,
        advantages.unsqueeze(1).float(),
        loss_mask,
        clip_epsilon=clip_epsilon,
        kl_beta=kl_beta,
        normalizer=normalizer,
    )
    loss.backward()

    return BackwardReport(
        loss=float(loss.item()),
        tokens=diagnostics["tokens"],
        chunks=1,
        mean_ratio=diagnostics["mean_ratio"],
        clip_fraction=diagnostics["clip_fraction"],
        mean_kl=diagnostics["mean_kl"],
    )
