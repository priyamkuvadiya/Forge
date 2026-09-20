"""Module 5: training the tool-use policy with GRPO.

Nothing here imports torch at module scope, for the same reason
`baseline_agent` does not: the parts with the bugs in them - masking a
transcript, turning rewards into advantages, weighting a category-imbalanced
sample - are ordinary data manipulation, and requiring a GPU to exercise them
would mean they are only ever tested on one laptop.
"""

from rl_training.transcript import (
    DivergenceReport,
    MaskedTranscript,
    Segment,
    align_completion,
    build_transcript,
    retokenized_divergence,
)

__all__ = [
    "DivergenceReport",
    "MaskedTranscript",
    "Segment",
    "align_completion",
    "build_transcript",
    "retokenized_divergence",
]
