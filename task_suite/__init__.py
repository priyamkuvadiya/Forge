"""Verifiable-reward task suite.

Every task in this suite can be checked programmatically: a math answer is
right or wrong, generated code either passes its tests or doesn't, a QA
answer either matches the supporting corpus or doesn't. That is the whole
point — the reward signal driving RL training is a real verifier, not a
learned preference model, so it can't be gamed the way an eval score can.
"""

from .schema import Task
from .protocol import ANSWER_INSTRUCTIONS, extract_answer
from .verifiers import verify

__all__ = ["Task", "ANSWER_INSTRUCTIONS", "extract_answer", "verify"]
