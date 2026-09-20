"""Turning a finished episode into the exact thing GRPO can take a gradient on.

An episode is a list of chat messages, and only some of it is the policy's
doing. The system prompt, the few-shot examples, the task, the nudge and -
above all - the tool output are text the policy *read*, not text it *wrote*.
Training on them would teach the model to predict search results and
calculator output, which is both useless and actively harmful: it is the one
part of the transcript the model has no business memorising, and on the QA
category it is a verbatim corpus document.

So every transcript has to carry a mask, and the mask has to be exact. An
off-by-one at a message boundary either drops a real token from the objective
or trains on `<|im_start|>`, and neither failure announces itself - the reward
curve looks the same either way. That is the reason this is its own module
with its own tests rather than a helper inside the training loop.

Two facts about Qwen's chat template make an exact mask possible, and both were
measured rather than assumed (see `tests/test_transcript.py::test_template_is_
prefix_stable`, which re-derives them against the live tokenizer):

  1. **Prefix stability.** Rendering `messages[:i]` with `add_generation_prompt`
     yields an exact string prefix of rendering the whole conversation. So the
     character offset at which each assistant turn begins is computable by
     rendering the prefix and taking its length.
  2. **Additivity.** Tokenizing a rendered prefix yields a prefix of tokenizing
     the full render, because every message ends with `<|im_end|>\\n` and that
     is a hard token boundary.

Which lets the sequence be assembled segment by segment - static text tokenized
normally, assistant turns spliced in - instead of tokenized whole and then
carved up by guesswork.

**The assistant turns are spliced in as the ids the policy actually sampled,
not as a re-tokenization of its decoded text.** These are not the same thing.
The loop stores decoded strings, and re-encoding a decoded string is only
*usually* the identity - a different but equally valid segmentation of the same
characters is a different trajectory, and a policy gradient computed on it is a
gradient on actions the policy never took. `retokenized_divergence()` measures
how often that happens on real rollouts so the cost of the shortcut is a number
rather than a hunch; `build_transcript` simply declines to pay it when the
sampled ids are available, and records `source="retokenized"` when they are not
(CI, a scripted policy, a re-scored checkpoint) so a caller can tell.

One place the two representations genuinely disagree, and it is not avoidable:
`baseline_agent._step` truncates an assistant turn at the end of its first
`</tool>`, discarding whatever the model imagined afterwards. That cut is a
*character* position, and character positions do not have to land on token
boundaries. `align_completion` maps the cut back into token space and keeps the
straddling token rather than dropping it - a trajectory ending in a truncated
`</too` is not a tool call, while one ending in a few extra characters is still
the call the policy made. The overshoot is counted, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, Sequence

# What this module needs from a tokenizer, and nothing more. Spelling it out
# keeps the tests honest: they can drive the real Qwen tokenizer *and* a stub,
# and the stub cannot accidentally be handed capabilities the real one lacks.
class Tokenizer(Protocol):
    def apply_chat_template(
        self, conversation: Sequence[dict[str, str]], *, tokenize: bool, **kwargs: Any
    ) -> str: ...

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]: ...

    def decode(self, token_ids: Sequence[int], *, skip_special_tokens: bool) -> str: ...


@dataclass(frozen=True)
class Segment:
    """One contiguous run of tokens with a single provenance.

    `kind` is `"static"` for anything the template or the environment put in
    the transcript, and `"completion"` for tokens the policy emitted. Only the
    latter carry gradient.
    """

    kind: str
    token_ids: list[int]
    text: str

    @property
    def is_completion(self) -> bool:
        return self.kind == "completion"


@dataclass(frozen=True)
class MaskedTranscript:
    """A tokenized episode plus the mask saying which tokens are the policy's.

    `completion_mask[i] == 1` means token `i` was sampled by the policy and
    belongs in the objective. Note that this is a mask over *inputs*: the loss
    predicts token `i` from tokens `< i`, so the training code shifts it by one
    before use. That shift lives in `grpo.py`, deliberately in exactly one
    place - doing it here as well is the off-by-one this module exists to
    prevent.
    """

    token_ids: list[int]
    completion_mask: list[int]
    segments: list[Segment]
    source: str
    overshoot_chars: int = 0
    truncated_from: int | None = None

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.completion_mask):
            raise ValueError(
                f"mask length {len(self.completion_mask)} does not match "
                f"{len(self.token_ids)} tokens"
            )

    @property
    def n_tokens(self) -> int:
        return len(self.token_ids)

    @property
    def n_completion_tokens(self) -> int:
        return sum(self.completion_mask)

    @property
    def is_truncated(self) -> bool:
        return self.truncated_from is not None

    def completion_token_ids(self) -> list[int]:
        return [t for t, m in zip(self.token_ids, self.completion_mask) if m]

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_tokens": self.n_tokens,
            "n_completion_tokens": self.n_completion_tokens,
            "source": self.source,
            "overshoot_chars": self.overshoot_chars,
            "truncated_from": self.truncated_from,
        }


def align_completion(
    sampled_ids: Sequence[int], kept_text: str, tokenizer: Tokenizer
) -> tuple[list[int], int]:
    """Cut `sampled_ids` down to the prefix that covers `kept_text`.

    Returns the ids and how many characters of overshoot the last token
    contributed past the end of `kept_text`.

    The loop keeps a *string* - `text.strip()` truncated at the first
    `</tool>` - and the cut can land inside a token. Binary search is over the
    length of the decoded prefix, which is non-decreasing in the number of
    tokens (adding a token can only add characters or, if it is pure
    whitespace that `strip` then removes, add none), so the search is
    well-founded. It is a search and not a scan because decoding is not free
    and a scan would decode a 384-token turn 384 times.
    """
    if not kept_text:
        return [], 0

    def decoded(k: int) -> str:
        return tokenizer.decode(list(sampled_ids[:k]), skip_special_tokens=True).strip()

    target = len(kept_text)
    low, high = 0, len(sampled_ids)
    if len(decoded(high)) < target:
        # The kept text is longer than anything these ids decode to, so they
        # are not the ids for this turn. Caller error, and a silent wrong
        # alignment here would be a silent wrong gradient later.
        raise ValueError(
            f"sampled ids decode to {len(decoded(high))} characters, "
            f"short of the {target} kept by the episode"
        )

    while low < high:
        middle = (low + high) // 2
        if len(decoded(middle)) < target:
            low = middle + 1
        else:
            high = middle

    prefix = list(sampled_ids[:low])
    overshoot = len(decoded(low)) - target
    return prefix, overshoot


def _assistant_indices(
    messages: Sequence[dict[str, str]], first_completion_index: int = 0
) -> list[int]:
    """Assistant messages the *policy* wrote, which is not all of them.

    The agent's prompt prefix contains worked examples, and a worked example is
    an assistant message. Masking those in would train the policy on the
    few-shot demonstrations - plain supervised fine-tuning on three hand-written
    turns, smuggled in under a reward signal, on every single rollout. Module 4
    measured those examples as worth 0.1908 against 0.0125, so they are not a
    detail: a policy quietly trained to reproduce them would look like RL
    working.
    """
    return [
        i
        for i, message in enumerate(messages)
        if message["role"] == "assistant" and i >= first_completion_index
    ]


def build_transcript(
    messages: Sequence[dict[str, str]],
    tokenizer: Tokenizer,
    *,
    sampled_token_ids: Sequence[Sequence[int]] | None = None,
    max_tokens: int | None = None,
    first_completion_index: int = 0,
) -> MaskedTranscript:
    """Tokenize a conversation and mark exactly the tokens the policy wrote.

    `sampled_token_ids` is one list of ids per assistant message, in order, as
    recorded at generation time. Pass `None` to fall back to re-tokenizing the
    stored text - correct enough to exercise the loop, not correct enough to
    publish a gradient from. See the module docstring.

    `first_completion_index` is where the policy's own turns start - the number
    of messages in the prompt prefix. Assistant messages before it are few-shot
    demonstrations and stay static. It defaults to 0 because a conversation
    with no prefix is the simpler case to reason about, but every real episode
    from `make_episodes` has one and passing the wrong value here is a silent
    bug, not a crash.

    `max_tokens` truncates from the *left* when a transcript will not fit the
    measured 8GB budget. Left, because the tail is where the answer and the
    reward live; dropping the head costs some of the few-shot examples, which
    is a smaller loss than dropping the thing being rewarded. A truncated
    transcript records where it was cut so the training loop can report how
    many of them there were rather than silently training on fragments.
    """
    assistants = _assistant_indices(messages, first_completion_index)
    if sampled_token_ids is not None and len(sampled_token_ids) != len(assistants):
        raise ValueError(
            f"{len(sampled_token_ids)} sampled turns for "
            f"{len(assistants)} assistant messages"
        )

    full_render = tokenizer.apply_chat_template(list(messages), tokenize=False)

    segments: list[Segment] = []
    overshoot_total = 0
    used_sampled = False
    used_retokenized = False
    cursor = 0

    for turn, index in enumerate(assistants):
        opening = tokenizer.apply_chat_template(
            list(messages[:index]), tokenize=False, add_generation_prompt=True
        )
        # Prefix stability is the load-bearing assumption of this whole
        # function, so it is checked on every call rather than trusted. A
        # template change upstream should break loudly here, not quietly
        # misalign a mask.
        if not full_render.startswith(opening):
            raise ValueError(
                "chat template is not prefix-stable: rendering the prefix of "
                f"assistant turn {turn} is not a prefix of the full render"
            )

        static_text = full_render[cursor : len(opening)]
        if static_text:
            segments.append(
                Segment(
                    kind="static",
                    token_ids=tokenizer.encode(static_text, add_special_tokens=False),
                    text=static_text,
                )
            )

        content = messages[index]["content"]
        if sampled_token_ids is not None:
            ids, overshoot = align_completion(
                sampled_token_ids[turn], content, tokenizer
            )
            overshoot_total += overshoot
            used_sampled = True
        else:
            ids = tokenizer.encode(content, add_special_tokens=False)
            used_retokenized = True
        segments.append(Segment(kind="completion", token_ids=ids, text=content))

        cursor = len(opening) + len(content)

    trailing = full_render[cursor:]
    if trailing:
        segments.append(
            Segment(
                kind="static",
                token_ids=tokenizer.encode(trailing, add_special_tokens=False),
                text=trailing,
            )
        )

    token_ids: list[int] = []
    completion_mask: list[int] = []
    for segment in segments:
        token_ids.extend(segment.token_ids)
        completion_mask.extend(
            [1 if segment.is_completion else 0] * len(segment.token_ids)
        )

    source = (
        "mixed"
        if used_sampled and used_retokenized
        else "sampled"
        if used_sampled
        else "retokenized"
    )

    truncated_from = None
    if max_tokens is not None and len(token_ids) > max_tokens:
        truncated_from = len(token_ids)
        token_ids = token_ids[-max_tokens:]
        completion_mask = completion_mask[-max_tokens:]

    return MaskedTranscript(
        token_ids=token_ids,
        completion_mask=completion_mask,
        segments=segments,
        source=source,
        overshoot_chars=overshoot_total,
        truncated_from=truncated_from,
    )


@dataclass
class DivergenceReport:
    """How much re-tokenizing a decoded completion changes it.

    The question this answers: if the training loop threw away the sampled ids
    and re-encoded the text instead, how many of the tokens it took a gradient
    on would be tokens the policy never emitted?
    """

    turns: int = 0
    sampled_tokens: int = 0
    retokenized_tokens: int = 0
    differing_turns: int = 0
    examples: list[dict[str, Any]] = field(default_factory=list)

    @property
    def turn_divergence_rate(self) -> float:
        return self.differing_turns / self.turns if self.turns else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "turns": self.turns,
            "differing_turns": self.differing_turns,
            "turn_divergence_rate": round(self.turn_divergence_rate, 4),
            "sampled_tokens": self.sampled_tokens,
            "retokenized_tokens": self.retokenized_tokens,
            "token_count_delta": self.retokenized_tokens - self.sampled_tokens,
            "examples": self.examples[:5],
        }


def retokenized_divergence(
    kept_texts: Sequence[str],
    sampled_token_ids: Sequence[Sequence[int]],
    tokenizer: Tokenizer,
    *,
    keep_examples: int = 5,
) -> DivergenceReport:
    """Measure the shortcut this module declines to take, on real rollouts."""
    report = DivergenceReport()
    for text, sampled in zip(kept_texts, sampled_token_ids):
        if not text:
            continue
        aligned, _ = align_completion(sampled, text, tokenizer)
        retokenized = tokenizer.encode(text, add_special_tokens=False)
        report.turns += 1
        report.sampled_tokens += len(aligned)
        report.retokenized_tokens += len(retokenized)
        if aligned != retokenized:
            report.differing_turns += 1
            if len(report.examples) < keep_examples:
                report.examples.append(
                    {
                        "text": text[:160],
                        "sampled": aligned[:32],
                        "retokenized": retokenized[:32],
                    }
                )
    return report
