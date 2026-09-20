"""The trainable policy: the same agent loop's `Policy`, with LoRA and a memory.

Two things separate this from `baseline_agent.HFPolicy`, and everything else is
deliberately identical - same chat template, same stop strings, same left
padding, same bf16, same sorting of a batch by length. The comparison in module
9 is only honest if the arms differ in their weights and nothing else.

**It carries LoRA adapters.** r=16 over every attention and MLP projection:
8.8M trainable parameters against the base model's 502.8M. That is not a
concession to the card so much as the thing that makes training possible on it
at all - full fine-tuning would need optimiser state for all 502.8M - and it
buys the KL reference for free, since `disable_adapter()` turns the resident
weights back into the base policy without a second copy.

**It remembers the token ids it sampled.** The agent loop deals in decoded
strings, and re-encoding a decoded string is not reliably the trajectory the
policy actually took (see `transcript.py`). So `generate` keeps the raw ids for
the batch it just produced, in the order it was handed the conversations, and
`record_turn` pairs them with the episodes they belong to via the loop's
`on_turn` hook. The pairing is the whole contract: ids attached to the wrong
episode would mask the wrong tokens and nothing downstream would notice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from baseline_agent import DEFAULT_MAX_NEW_TOKENS, STOP_STRINGS, Episode

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

LORA_TARGETS = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def attach_adapter(base: Any, *, rank: int = 16, lora_alpha: int | None = None,
                   lora_dropout: float = 0.0, adapter_path: str | None = None) -> Any:
    """Put LoRA adapters on a base model, new or resumed.

    Split out of `LoRAPolicy.__init__` so it can be tested against a small
    model without downloading a 0.5B one, because the resume branch has a
    silent failure in it: `PeftModel.from_pretrained` defaults to
    `is_trainable=False`, and a run resumed without it loads the adapters,
    proceeds normally, and produces a zero gradient on every step.
    """
    from peft import LoraConfig, PeftModel, get_peft_model

    if adapter_path is not None:
        return PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
    return get_peft_model(
        base,
        LoraConfig(
            r=rank,
            lora_alpha=lora_alpha if lora_alpha is not None else 2 * rank,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGETS,
        ),
    )


@dataclass
class TurnRecorder:
    """Collects, per episode, the ids each of its assistant turns came from.

    Keyed by `id(episode)`, which is safe because every episode in a step is
    alive for the whole step - the trainer holds the list. Not keyed by task id
    and rollout, because a step can draw the same task twice.
    """

    by_episode: dict[int, list[list[int]]] = field(default_factory=dict)

    def record(self, episodes: list[Episode], token_ids: list[list[int]]) -> None:
        if len(episodes) != len(token_ids):
            raise ValueError(
                f"{len(token_ids)} sampled turns for {len(episodes)} episodes"
            )
        for episode, ids in zip(episodes, token_ids):
            self.by_episode.setdefault(id(episode), []).append(list(ids))

    def for_episode(self, episode: Episode) -> list[list[int]]:
        return self.by_episode.get(id(episode), [])

    def aligned_with(self, episode: Episode) -> list[list[int]] | None:
        """The ids for this episode, or None if they do not line up.

        A turn is recorded before the loop decides what to do with it, and a
        completion that was empty produces an assistant turn of nothing - so
        the counts can legitimately disagree. Returning None there makes the
        transcript builder fall back to re-tokenizing that episode and record
        `source="retokenized"`, which is visible in the run log, rather than
        silently misaligning the mask.
        """
        recorded = self.for_episode(episode)
        return recorded if len(recorded) == len(episode.assistant_turns) else None


class LoRAPolicy:
    """A LoRA-wrapped causal LM behind the agent loop's `Policy` protocol."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        rank: int = 16,
        lora_alpha: int | None = None,
        lora_dropout: float = 0.0,
        device: str | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        batch_size: int = 16,
        seed: int = 0,
        adapter_path: str | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size
        self.seed = seed
        self.rank = rank
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16, attn_implementation="sdpa"
        ).to(self.device)

        self.model = attach_adapter(
            base,
            rank=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            adapter_path=adapter_path,
        )

        torch.manual_seed(seed)
        self.last_token_ids: list[list[int]] = []

    # --- reporting ---------------------------------------------------------

    def describe(self) -> dict[str, Any]:
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return {
            "model": self.model_name,
            "dtype": "bfloat16",
            "device": self.device,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "lora": {
                "rank": self.rank,
                "targets": LORA_TARGETS,
                "trainable": trainable,
                "total": total,
            },
        }

    # --- the Policy protocol -----------------------------------------------

    def generate(
        self,
        conversations: list[list[dict[str, str]]],
        *,
        stop: tuple[str, ...] = STOP_STRINGS,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> list[str]:
        prompts = [
            self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            for conversation in conversations
        ]

        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        outputs: list[str] = [""] * len(prompts)
        sampled: list[list[int]] = [[] for _ in prompts]

        for start in range(0, len(order), self.batch_size):
            indices = order[start : start + self.batch_size]
            encoded = self.tokenizer(
                [prompts[i] for i in indices],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(self.device)

            with self._torch.no_grad():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature if self.temperature > 0 else None,
                    top_p=self.top_p if self.temperature > 0 else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    stop_strings=list(stop),
                    tokenizer=self.tokenizer,
                )

            fresh = generated[:, encoded["input_ids"].shape[1] :]
            for index, row in zip(indices, fresh):
                ids = row.tolist()
                # Trailing pad is not something the policy chose. Left in, it
                # would be masked-in padding at the end of every short turn.
                while ids and ids[-1] == self.tokenizer.pad_token_id:
                    ids.pop()
                sampled[index] = ids
                outputs[index] = self.tokenizer.decode(row, skip_special_tokens=True)

        # Kept in the order the caller passed, which is the order `on_turn`
        # reports its episodes in.
        self.last_token_ids = sampled
        return outputs

    # --- checkpointing -----------------------------------------------------

    def save_adapter(self, path) -> None:
        """Write the adapter, and only the adapter.

        8.8M parameters rather than 502.8M, so a checkpoint per evaluation is
        affordable - which is the pattern `transformer_scratch/train.py`
        settled on after a crash cost a whole run.
        """
        self.model.save_pretrained(str(path))
