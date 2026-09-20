"""Microbatch packing against the measured 8GB envelope.

The packer is the thing standing between a training run and a silent spill to
system RAM, and it is pure arithmetic, so it is checked here rather than
discovered at hour two of a run. The envelope itself is not arithmetic - it
came off the card - so the numbers are asserted as the measurements they are.
"""

import pytest

from rl_training.train_grpo import (
    TOKEN_BUDGET_LONG,
    TOKEN_BUDGET_SHORT,
    microbatch_capacity,
    pack_microbatches,
)


def test_the_measured_envelope_is_what_the_packer_allows():
    """16x1024 and 4x2048 fit; 8x2048 spills. Measured, not derived."""
    assert microbatch_capacity(1024) == 16
    assert microbatch_capacity(2048) == 4


def test_the_budget_halves_past_the_short_sequence_mark():
    """Token count alone does not predict fit - 16x1024 and 8x2048 are both
    16,384 tokens, and only the first one stays on the card."""
    assert microbatch_capacity(SHORT := 1024) * SHORT == TOKEN_BUDGET_SHORT
    assert microbatch_capacity(2048) * 2048 == TOKEN_BUDGET_LONG


def test_a_sequence_longer_than_the_whole_budget_still_gets_a_batch():
    """One-at-a-time is the floor; returning zero would drop the sequence."""
    assert microbatch_capacity(100_000) == 1


def test_a_non_positive_length_raises():
    with pytest.raises(ValueError, match="must be positive"):
        microbatch_capacity(0)


def test_every_index_lands_in_exactly_one_microbatch():
    lengths = [300, 1200, 400, 2048, 150, 900, 2000, 64]

    batches = pack_microbatches(lengths)

    flattened = [index for batch in batches for index in batch]
    assert sorted(flattened) == list(range(len(lengths)))
    assert len(flattened) == len(set(flattened))


def test_no_microbatch_exceeds_its_capacity():
    """The property that matters: every batch must fit on the card."""
    lengths = [64, 128, 256, 512, 1024, 1024, 2048, 2048, 1500, 300, 900, 700]

    for batch in pack_microbatches(lengths):
        longest = max(lengths[i] for i in batch)
        assert len(batch) <= microbatch_capacity(longest)


def test_long_sequences_are_not_padded_up_with_short_ones():
    """Sorting by length is the difference between a 2048-wide batch of two
    and a 2048-wide batch of sixteen mostly-padding rows."""
    lengths = [2048, 64, 2048, 64, 2048, 64, 2048, 64]

    batches = pack_microbatches(lengths)

    for batch in batches:
        widths = {lengths[i] for i in batch}
        assert len(widths) == 1, "a batch mixed a long sequence with short ones"


def test_sixteen_short_sequences_pack_into_one_batch():
    assert pack_microbatches([1024] * 16) == [list(range(16))]


def test_seventeen_short_sequences_need_two():
    batches = pack_microbatches([1024] * 17)
    assert len(batches) == 2
    assert sorted(len(b) for b in batches) == [1, 16]


def test_eight_long_sequences_are_split_rather_than_spilled():
    """8x2048 is the configuration measured to spill, so it must never be one
    batch."""
    batches = pack_microbatches([2048] * 8)

    assert len(batches) == 2
    assert all(len(batch) == 4 for batch in batches)


def test_an_empty_batch_is_empty_not_a_crash():
    assert pack_microbatches([]) == []


def test_a_caller_supplied_order_is_respected():
    lengths = [1024] * 4
    assert pack_microbatches(lengths, order=[3, 2, 1, 0]) == [[3, 2, 1, 0]]


def test_the_trainer_imports_without_torch():
    """Module scope must stay torch-free: the packer and the CLI's preflight
    run on CI, and a stray top-level `import torch` would take both out."""
    import rl_training.train_grpo as module

    assert "torch" not in dir(module)
