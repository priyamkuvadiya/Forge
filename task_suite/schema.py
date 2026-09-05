"""The Task record every category in the suite shares.

One flat record instead of a class hierarchy, because tasks have to survive a
round trip through JSON: the generated suite is written to disk and reloaded
by the baseline agent, the RL training loop and the eval harness, and all
three must see byte-identical tasks or the numbers they report aren't
comparable. `ground_truth` is a category-specific dict whose shape is owned
by that category's verifier.
"""

from dataclasses import asdict, dataclass, field
from typing import Any

CATEGORIES = ("math", "code", "qa", "multi_tool", "no_tool")
SPLITS = ("train", "heldout")


@dataclass(frozen=True)
class Task:
    task_id: str
    category: str
    prompt: str
    ground_truth: dict[str, Any]
    split: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES:
            raise ValueError(f"unknown category {self.category!r}, expected one of {CATEGORIES}")
        if self.split not in SPLITS:
            raise ValueError(f"unknown split {self.split!r}, expected one of {SPLITS}")
        if not self.task_id:
            raise ValueError("task_id must be non-empty")
        if not self.prompt.strip():
            raise ValueError(f"task {self.task_id} has an empty prompt")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Task":
        return cls(
            task_id=payload["task_id"],
            category=payload["category"],
            prompt=payload["prompt"],
            ground_truth=payload["ground_truth"],
            split=payload["split"],
            metadata=payload.get("metadata", {}),
        )
