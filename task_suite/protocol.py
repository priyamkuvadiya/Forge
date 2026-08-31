"""The answer contract between the agent policy and the verifiers.

The policy is free to reason and call tools however it likes, but it has to
commit to a final answer inside `<answer>...</answer>` for that answer to be
scored. This is deliberately strict: format compliance is part of what RL
has to learn, and a response with no parseable commitment earns 0.0 rather
than being generously re-parsed. Anything looser (grab the last number in
the response, fuzzy-match a prefix) would hand out reward for text that
never actually answered the question, which is exactly the kind of soft
credit that makes a reward curve meaningless.
"""

import re

# DOTALL so multi-line answers (code blocks) survive; IGNORECASE because a
# small model's casing is not what we're trying to measure here.
_ANSWER_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

ANSWER_INSTRUCTIONS = (
    "When you have solved the task, give your final answer inside "
    "<answer></answer> tags. Only the content of the last <answer> block is "
    "scored."
)


def extract_answer(response: str) -> str | None:
    """Return the content of the *last* <answer> block, or None if absent.

    Last rather than first: models routinely restate an early guess and then
    revise it, and the final block is the one the policy actually committed
    to. An unterminated `<answer>` with no closing tag deliberately does not
    match — a truncated generation is a failed generation.
    """
    matches = _ANSWER_PATTERN.findall(response)
    if not matches:
        return None
    return matches[-1].strip()
