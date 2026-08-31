"""Single entry point for reward: `verify(task, response) -> float in [0, 1]`.

Every caller — the baseline agent, the GRPO training loop, the eval harness —
goes through this function, so all three score identically by construction.
The dispatch is intentionally dumb; the interesting logic lives in the
per-category modules next to it.
"""

from .schema import Task
from .verify_code import DEFAULT_TIMEOUT_SECONDS, CodeRunner, verify_code
from .verify_math import verify_math
from .verify_qa import verify_qa


def verify(
    task: Task,
    response: str,
    *,
    code_runner: CodeRunner | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> float:
    """Score one response against one task. Deterministic, given the same inputs."""
    # multi_tool tasks are arithmetic over retrieved facts, so the answer is a
    # number and is scored exactly as any other number is. The category exists
    # to change what the agent has to *do*, not how the result is graded.
    if task.category in ("math", "multi_tool"):
        return verify_math(response, task.ground_truth)

    if task.category == "qa":
        return verify_qa(response, task.ground_truth)

    if task.category == "code":
        if code_runner is None:
            # Returning 0.0 here would look like "the model failed" and would
            # quietly flatten a whole category's reward to zero for an entire
            # training run. Fail loudly instead.
            raise ValueError(
                f"task {task.task_id} is a code task and needs a sandboxed code_runner"
            )
        return verify_code(response, task.ground_truth, code_runner, timeout)

    raise ValueError(f"no verifier for category {task.category!r}")
