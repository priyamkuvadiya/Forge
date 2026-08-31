"""Reward for arithmetic / math word problems.

Ground truth shape:
    {"value": float, "tolerance": float}

`tolerance` is absolute and is set by whoever authored the task: 0.0 for
answers that are exactly an integer, 0.005 for money answers rounded to
cents, and so on. It is stored per task rather than fixed globally because
"how close counts as right" is a property of the question, not of the
verifier.
"""

import math
import re

from .protocol import extract_answer

# A signed number, optionally with thousands separators and a decimal part,
# optionally in scientific notation.
_NUMBER_PATTERN = re.compile(r"[-+]?\d[\d,]*\.?\d*(?:[eE][-+]?\d+)?")

# Currency symbols and percent signs are stripped before parsing: a model
# answering "$1,240.50" has answered 1240.50 and it would be dishonest to
# score that as wrong to make the numbers look tidier.
_STRIP_CHARS = "$€£%, \t\n"


def parse_number(text: str) -> float | None:
    """Parse a model's answer text into a number, or None if it isn't one.

    Accepts a bare number, a number with currency/percent decoration, or a
    short phrase containing exactly one number ("42 apples"). A phrase with
    two or more numbers is rejected rather than guessed at — picking one of
    them would be inventing a commitment the model never made.
    """
    cleaned = text.strip().strip(_STRIP_CHARS)
    direct = _to_float(cleaned)
    if direct is not None:
        return direct

    matches = _NUMBER_PATTERN.findall(text)
    if len(matches) != 1:
        return None
    return _to_float(matches[0])


def _to_float(token: str) -> float | None:
    token = token.strip().strip(_STRIP_CHARS).replace(",", "")
    if not token:
        return None
    try:
        value = float(token)
    except ValueError:
        return None
    # inf/nan parse fine as floats but are never a legitimate answer here
    if not math.isfinite(value):
        return None
    return value


def verify_math(response: str, ground_truth: dict) -> float:
    answer = extract_answer(response)
    if answer is None:
        return 0.0

    parsed = parse_number(answer)
    if parsed is None:
        return 0.0

    expected = float(ground_truth["value"])
    tolerance = float(ground_truth.get("tolerance", 0.0))
    # A small relative term keeps large-magnitude answers from failing on
    # float representation alone; tolerance stays the dominant term.
    return 1.0 if math.isclose(parsed, expected, abs_tol=tolerance, rel_tol=1e-9) else 0.0
