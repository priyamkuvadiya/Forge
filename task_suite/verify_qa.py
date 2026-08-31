"""Reward for multi-hop QA answered from the local search corpus.

Ground truth shape:
    {"answers": [str, ...], "supporting_docs": [doc_id, ...]}

`answers` is the set of accepted surface forms for one correct answer (the
full name and the surname, say), authored alongside the question.
`supporting_docs` is not scored — it records which corpus documents the
question actually requires, so the eval harness can report whether the agent
retrieved them, separately from whether it got the answer right.

Scoring is exact match on a normalized string, not token F1. Token overlap
would hand partial reward to an answer that merely contains a right-looking
word, and a small policy trained against that will happily learn to emit
long answers stuffed with plausible entities. Binary is harsher and honest.
"""

import re
import string

from .protocol import extract_answer

_ARTICLES = {"a", "an", "the"}
_PUNCT_TABLE = str.maketrans("", "", string.punctuation)


def normalize_answer(text: str) -> str:
    """Lowercase, drop punctuation and articles, collapse whitespace."""
    lowered = text.lower().translate(_PUNCT_TABLE)
    tokens = [t for t in lowered.split() if t not in _ARTICLES]
    return re.sub(r"\s+", " ", " ".join(tokens)).strip()


def verify_qa(response: str, ground_truth: dict) -> float:
    answer = extract_answer(response)
    if answer is None:
        return 0.0

    normalized = normalize_answer(answer)
    if not normalized:
        return 0.0

    accepted = {normalize_answer(a) for a in ground_truth["answers"]}
    return 1.0 if normalized in accepted else 0.0
