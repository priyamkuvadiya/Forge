"""Does the mask cover exactly what the policy wrote, and nothing else?

Most of this runs against a stub tokenizer, because CI installs pytest and
nothing else and the masking logic is ordinary string and list work. The stub
is deliberately *not* character-level: it produces multi-character tokens, so
the interesting failure - a truncation point that lands inside a token - can
actually occur.

Two tests at the bottom drive the real Qwen tokenizer and skip when
`transformers` is missing, which on CI is always. They are the only thing that
checks the two template properties this module's correctness rests on, so they
are run by hand on the training machine and the result is recorded in the
README rather than asserted into a green tick nobody produced.
"""

import re

import pytest

from rl_training.transcript import (
    MaskedTranscript,
    align_completion,
    build_transcript,
    retokenized_divergence,
)

TOOL_OUTPUT = "RESULT 481516 from the corpus"
SENTINEL = "481516"


class StubTokenizer:
    """A chat template and a word-ish tokenizer, with the properties that matter.

    Prefix-stable and additive, like Qwen's, and with multi-character tokens so
    that alignment has something to align. Ids are assigned on first sight,
    which is enough for round-tripping and keeps the vocabulary out of the
    test's way.
    """

    def __init__(self) -> None:
        self._to_id: dict[str, int] = {}
        self._to_text: dict[int, str] = {}

    def _piece(self, text: str) -> int:
        if text not in self._to_id:
            index = len(self._to_id)
            self._to_id[text] = index
            self._to_text[index] = text
        return self._to_id[text]

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt=False):
        assert tokenize is False, "this module only ever renders to text"
        rendered = "".join(
            f"<s>{message['role']}\n{message['content']}<e>\n"
            for message in conversation
        )
        if add_generation_prompt:
            rendered += "<s>assistant\n"
        return rendered

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [self._piece(piece) for piece in re.findall(r"\w+|\W", text)]

    def decode(self, token_ids, *, skip_special_tokens):
        return "".join(self._to_text[i] for i in token_ids)


class UnstableTokenizer(StubTokenizer):
    """A template that rewrites earlier messages once later ones exist.

    Not a hypothetical: a template that injects a default system prompt only
    when none is present, or that renders the final message differently, breaks
    prefix stability exactly this way. The mask would still be produced, and it
    would be silently wrong.
    """

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt=False):
        rendered = super().apply_chat_template(
            conversation, tokenize=tokenize, add_generation_prompt=add_generation_prompt
        )
        if not add_generation_prompt:
            return "<PREAMBLE>" + rendered
        return rendered


def conversation():
    """The shape every real episode has: prompt, call, tool output, answer."""
    return [
        {"role": "system", "content": "You may call tools."},
        {"role": "user", "content": "Who wrote the report?"},
        {"role": "assistant", "content": '<tool name="search">report</tool>'},
        {"role": "user", "content": TOOL_OUTPUT},
        {"role": "assistant", "content": "Thinking about it."},
        {"role": "user", "content": "Give your final answer."},
        {"role": "assistant", "content": "<answer>Yuki</answer>"},
    ]


@pytest.fixture
def tokenizer():
    return StubTokenizer()


def test_masked_tokens_decode_to_exactly_the_assistant_turns(tokenizer):
    messages = conversation()
    transcript = build_transcript(messages, tokenizer)

    expected = "".join(m["content"] for m in messages if m["role"] == "assistant")
    assert tokenizer.decode(
        transcript.completion_token_ids(), skip_special_tokens=True
    ) == expected


def test_whole_transcript_round_trips(tokenizer):
    """The mask must not change what the model reads, only what it learns from."""
    messages = conversation()
    transcript = build_transcript(messages, tokenizer)

    assert tokenizer.decode(
        transcript.token_ids, skip_special_tokens=True
    ) == tokenizer.apply_chat_template(messages, tokenize=False)


def test_tool_output_is_never_in_the_completion_mask(tokenizer):
    """The point of the module. Probed by sentinel, not by counting tokens."""
    messages = conversation()
    transcript = build_transcript(messages, tokenizer)

    whole = tokenizer.decode(transcript.token_ids, skip_special_tokens=True)
    masked = tokenizer.decode(
        transcript.completion_token_ids(), skip_special_tokens=True
    )

    assert SENTINEL in whole, "the tool output should be in what the model reads"
    assert SENTINEL not in masked, "the tool output must not be in what it learns"


def test_nudge_and_system_prompt_are_static(tokenizer):
    messages = conversation()
    transcript = build_transcript(messages, tokenizer)
    masked = tokenizer.decode(
        transcript.completion_token_ids(), skip_special_tokens=True
    )

    assert "final answer" not in masked
    assert "may call tools" not in masked
    assert "<s>assistant" not in masked, "template scaffolding is not the policy's"


def test_prefix_instability_raises_rather_than_misaligning():
    """Mutation test for the guard: break the assumption, demand a failure.

    Without the `startswith` check in `build_transcript` this call returns a
    transcript with a plausible-looking but wrong mask.
    """
    messages = conversation()
    with pytest.raises(ValueError, match="prefix-stable"):
        build_transcript(messages, UnstableTokenizer())


def test_no_assistant_turns_produces_an_all_static_transcript(tokenizer):
    messages = [
        {"role": "system", "content": "You may call tools."},
        {"role": "user", "content": "Who wrote the report?"},
    ]
    transcript = build_transcript(messages, tokenizer)

    assert transcript.n_tokens > 0
    assert transcript.n_completion_tokens == 0


def test_mask_must_match_token_count():
    with pytest.raises(ValueError, match="does not match"):
        MaskedTranscript(token_ids=[1, 2, 3], completion_mask=[1, 1], segments=[], source="x")


def test_sampled_ids_are_spliced_in_verbatim(tokenizer):
    messages = conversation()
    turns = [m["content"] for m in messages if m["role"] == "assistant"]
    # What the policy "sampled": the kept text plus the tail the loop discards.
    sampled = [
        tokenizer.encode(turn + " and then it rambled on", add_special_tokens=False)
        for turn in turns
    ]

    transcript = build_transcript(messages, tokenizer, sampled_token_ids=sampled)

    assert transcript.source == "sampled"
    assert tokenizer.decode(
        transcript.completion_token_ids(), skip_special_tokens=True
    ) == "".join(turns)
    assert "rambled" not in tokenizer.decode(
        transcript.completion_token_ids(), skip_special_tokens=True
    )


def test_sampled_turn_count_must_match(tokenizer):
    messages = conversation()
    with pytest.raises(ValueError, match="sampled turns"):
        build_transcript(messages, tokenizer, sampled_token_ids=[[1, 2]])


def test_align_completion_keeps_the_token_that_straddles_the_cut(tokenizer):
    """A cut inside a token keeps the token: a truncated `</too` is not a call."""
    # `dogcat` is one token under the stub's `\w+` rule; cutting after `dog`
    # lands inside it.
    sampled = tokenizer.encode("dogcat", add_special_tokens=False)
    assert len(sampled) == 1

    ids, overshoot = align_completion(sampled, "dog", tokenizer)

    assert ids == sampled, "the straddling token is kept, not dropped"
    assert overshoot == 3


def test_align_completion_reports_no_overshoot_on_a_clean_boundary(tokenizer):
    sampled = tokenizer.encode("dog cat", add_special_tokens=False)
    ids, overshoot = align_completion(sampled, "dog", tokenizer)

    assert overshoot == 0
    assert tokenizer.decode(ids, skip_special_tokens=True) == "dog"


def test_align_completion_refuses_ids_that_are_too_short(tokenizer):
    sampled = tokenizer.encode("dog", add_special_tokens=False)
    with pytest.raises(ValueError, match="short of"):
        align_completion(sampled, "dog cat elephant", tokenizer)


def test_align_completion_on_empty_text_is_empty(tokenizer):
    assert align_completion([1, 2, 3], "", tokenizer) == ([], 0)


def test_truncation_drops_the_head_and_keeps_the_answer(tokenizer):
    messages = conversation()
    full = build_transcript(messages, tokenizer)
    cap = full.n_tokens - 10

    cut = build_transcript(messages, tokenizer, max_tokens=cap)

    assert cut.is_truncated
    assert cut.truncated_from == full.n_tokens
    assert cut.n_tokens == cap
    assert len(cut.completion_mask) == cap
    assert cut.token_ids == full.token_ids[-cap:], "truncation is from the left"
    assert "<answer>Yuki</answer>" in tokenizer.decode(
        cut.completion_token_ids(), skip_special_tokens=True
    )


def test_untruncated_transcript_reports_no_truncation(tokenizer):
    transcript = build_transcript(conversation(), tokenizer, max_tokens=10_000)
    assert not transcript.is_truncated
    assert transcript.truncated_from is None


def test_divergence_report_counts_a_real_difference(tokenizer):
    """Force a divergence rather than hoping one occurs.

    `dogcat` re-encodes as one token; a policy that sampled it as `dog` + `cat`
    took two actions, and a gradient on the re-encoding is a gradient on an
    action it never took.
    """
    sampled = [
        tokenizer.encode("dog", add_special_tokens=False)
        + tokenizer.encode("cat", add_special_tokens=False)
    ]

    report = retokenized_divergence(["dogcat"], sampled, tokenizer)

    assert report.turns == 1
    assert report.differing_turns == 1
    assert report.turn_divergence_rate == 1.0
    assert report.sampled_tokens == 2
    assert report.retokenized_tokens == 1


def test_divergence_report_is_clean_when_encodings_agree(tokenizer):
    sampled = [tokenizer.encode("dog cat", add_special_tokens=False)]
    report = retokenized_divergence(["dog cat"], sampled, tokenizer)

    assert report.turns == 1
    assert report.differing_turns == 0
    assert report.to_dict()["token_count_delta"] == 0


# --- The two properties the real template has to have -----------------------
#
# Skipped on CI, which installs no transformers and has no network budget for
# a tokenizer download. Run on the training machine; see the README.

SKIP_REASON = "transformers not installed (expected on CI)"


def _real_tokenizer():
    transformers = pytest.importorskip("transformers", reason=SKIP_REASON)
    return transformers.AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")


def test_real_template_is_prefix_stable():
    tokenizer = _real_tokenizer()
    messages = conversation()
    full = tokenizer.apply_chat_template(messages, tokenize=False)

    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        opening = tokenizer.apply_chat_template(
            messages[:index], tokenize=False, add_generation_prompt=True
        )
        assert full.startswith(opening)


def test_real_tokenizer_masks_exactly_the_assistant_turns():
    tokenizer = _real_tokenizer()
    messages = conversation()
    transcript = build_transcript(messages, tokenizer)

    masked = tokenizer.decode(
        transcript.completion_token_ids(), skip_special_tokens=True
    )
    assert masked == "".join(
        m["content"] for m in messages if m["role"] == "assistant"
    )
    assert SENTINEL not in masked
    assert tokenizer.decode(transcript.token_ids, skip_special_tokens=False) == (
        tokenizer.apply_chat_template(messages, tokenize=False)
    )
