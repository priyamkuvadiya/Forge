"""Module 4: the control group.

The open-weight base model, *prompted* to use the tools from module 3 on the
tasks from module 2. No training happens here and none should. The number this
produces is the thing every later claim about RL is measured against, so the
one property that matters is that it is a fair fight: the RL policy in module 5
gets the same system prompt, the same tool contract, the same budget, the same
verifiers and the same sampling parameters, and the only difference between the
two runs is the weights.

Three decisions in here are worth stating plainly, because each of them moves
the baseline number and none of them is forced:

**The system prompt says "not every task needs a tool".** Without that line the
`no_tool` control measures how suggestible the model is rather than whether it
can tell tasks apart, since the prompt would otherwise advertise three tools and
never mention declining them. With it, the over-calling number means something.
It is in the prompt for both arms.

**The prompt says what a coding answer should contain.** A model that reasons
its way to correct code and then writes prose inside `<answer>` scores zero for
a formatting reason, which would understate the baseline rather than measure it.

**One nudge.** If a turn ends with neither a tool call nor an `<answer>`, the
agent asks once for a final answer and then gives up. This does not loosen the
answer contract - `task_suite.protocol.extract_answer` is still the only thing
that reads an answer, and it is still strict - it only declines to score a
model zero for trailing off mid-thought on its first attempt. The nudge rate is
reported next to the reward so the cost is visible, and module 5's policy gets
exactly one nudge too.

Everything the eval harness in module 9 needs comes out of `run()` as plain
data: per-episode reward, the tool-call trace, and the sandbox confinement that
produced any coding score.

    python -m baseline_agent --split heldout --rollouts 8 --out artifacts/baseline/heldout.json
    python -m baseline_agent --split train --per-category 2 --show-transcripts   # dry run
"""

import argparse
import json
import os
import random
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from task_suite import Task, extract_answer, verify
from task_suite.protocol import ANSWER_INSTRUCTIONS
from task_suite.registry import load_suite
from task_suite.schema import CATEGORIES
from tools import DEFAULT_MAX_CALLS, ToolCall, ToolRegistry, ToolSession, build_registry

# Module scope, not inside `main`: `write_results` records the sandbox
# confinement beside every coding score, so this is needed on the
# `--score-rollouts` path too, where `main` never reaches the model. The
# module itself imports cleanly everywhere - its platform-specific half is
# loaded lazily, behind `sandbox_available()`.
from tools import code_exec
from tools.toolbox import TraceEntry, find_tool_calls

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# Generation stops at either closing tag. Cheaper than generating to the token
# cap and throwing the tail away, and it is the same reason the tail gets
# thrown away: whatever a small model writes after `</tool>` is invented.
STOP_STRINGS = ("</tool>", "</answer>")

DEFAULT_MAX_NEW_TOKENS = 384

# One retry, not a conversation. See the module docstring.
DEFAULT_MAX_NUDGES = 1
#
# The wording is the third draft, and both rewrites were forced by watching a
# nudge destroy an answer the model had already got right:
#
# - "give it now ... using what you already know" was read as an invitation to
#   start over. A correct fenced `sum_of_evens` came back as `<answer>0</answer>`.
# - Adding "if the task was to write a function, put the function in the tags"
#   fixed that and broke arithmetic instead: three math episodes that had
#   reasoned their way to a number in prose came back as Python function
#   definitions, because the clause was the most concrete instruction present.
#
# - "Repeat it unchanged, in the same form, inside those tags and nothing
#   else" stopped the damage but obeyed the wrong half of itself: the model
#   replied with the bare answer ("Yuki Verhoeven") and no tags at all.
#
# - Showing the shape as a fill-in template - `<answer>YOUR ANSWER</answer>` -
#   got the tags back and lost the answer: five episodes answered the literal
#   string "YOUR ANSWER". A placeholder is indistinguishable from an answer to
#   a model this size.
#
# So: no placeholder to copy and no hint about what kind of answer to give,
# only where the two literal strings go. The coding format is taught by the
# worked example, not by the recovery message.
NUDGE_MESSAGE = (
    "That reply cannot be scored, because it did not contain the answer tags. "
    "Send the same answer again with the literal text <answer> immediately "
    "before it and the literal text </answer> immediately after it. "
    "Do not start over and do not call any more tools."
)

# A cap on what one tool result can add to the transcript. Nothing in the
# current corpus comes close - the longest search result is well under this -
# but a policy is free to ask for anything, and an unbounded tool output on an
# 8-call budget is an unbounded prompt.
MAX_TOOL_OUTPUT_CHARS = 4000

# Ordering only: which of `<answer>` and `<tool ...>` the model wrote first.
# Reading an answer is still `extract_answer`'s job and only its job - this
# regex deliberately does not know what a well-formed answer block looks like.
_ANSWER_OPEN = re.compile(r"<answer>", re.IGNORECASE)

SYSTEM_PROMPT_TEMPLATE = """\
You are a careful assistant that solves tasks accurately.

{tools}

Not every task needs a tool. If you already know the answer, give it directly \
rather than calling one; if the task asks about something you have no \
knowledge of, search for it rather than guessing or refusing.

{answer_instructions} A reply with no <answer> block scores nothing, however \
good the reasoning above it was. For a programming task, the answer is the \
complete Python function definition, written inside the answer tags."""

Message = dict[str, str]

# Three worked examples, replayed as real conversation turns rather than
# described in the system prompt.
#
# This is the single change the dry run forced, and it is worth recording why.
# Prompted with the description alone, Qwen2.5-0.5B-Instruct emitted zero
# `<answer>` blocks and zero tool calls across a ten-task probe - it solved
# "What is 5 plus 4?" correctly in prose and then scored 0.0, and it answered
# a question about a fictional expedition by apologising for not knowing
# rather than searching for it. A baseline of 0.000 in every category is
# arithmetically honest and analytically worthless: it measures whether a 0.5B
# model can follow an unfamiliar tag convention from a description, which is
# not what this project claims RL improves, and it would hand module 5 a
# guaranteed win over a control that never attempted the task.
#
# So the control gets the prompt a competent engineer would actually ship. The
# constraints on it are that it must be honest and it must be identical for
# both arms:
#
# - The examples are invented, and deliberately unlike anything in the suite -
#   no expedition, no researcher, no bulk-discount problem. They demonstrate
#   the *format*, and give away no task.
# - The tool results are produced by the real `calculate`, `render_hits` and
#   `ToolResult.render`, not typed out by hand, so an exemplar cannot drift
#   from the format the policy is really shown mid-episode.
# - Module 5's policy is prompted with exactly these turns, and `--no-few-shot`
#   exists so the cost of removing them can be quoted rather than guessed at.
#
# The order ends on a search example on purpose: the failure the probe showed
# was under-calling, never over-calling, and the `no_tool` control measures
# whether that trade got made.
#
# Four examples is where this stops. A small model copies from context: on the
# probe, a nudged QA episode answered "Ivory Halst" - the search exemplar's
# answer, misspelled - to an unrelated question. More examples would buy more
# of that, not more format compliance.
# Not "What is 3 plus 6?", which was the first draft: that is verbatim
# `notool-train-0030`, so the worked example was quietly handing the policy a
# solved training task from the very category that exists to measure it.
# `test_few_shot_examples_leak_no_task_from_the_suite` caught it and now
# guards it.
_EXAMPLE_DIRECT = (
    "Which is larger, 8 or 11?",
    "11 is larger than 8.\n<answer>11</answer>",
)
# The numbers are chosen so the product is exact in binary. With the first
# draft's 8.45 the calculator honestly returned 27783.599999999999, and an
# example that shows that and then answers 27783.60 is teaching the policy to
# quietly launder float noise rather than teaching it the format.
_EXAMPLE_CALCULATOR = (
    "A pallet holds 18 boxes and each box costs $6.25. What do 245 pallets cost?",
    "18 * 245 * 6.25",
    "<answer>27562.50</answer>",
)
_EXAMPLE_SEARCH = (
    "Who compiled the Tollan Codex?",
    "Tollan Codex",
    "<answer>Ivor Halst</answer>",
)
# Shown as a direct answer, not through the `python` tool: the answer to a
# coding task is the function, not what running it printed. Without this the
# probe's coding answer rate was 0/2 - the model wrote correct code in a fenced
# block every time and never once put it in the tags.
_EXAMPLE_CODE = (
    "Write a Python function `double_all(values)` that returns a new list "
    "with every number doubled.",
    "<answer>\n```python\ndef double_all(values):\n"
    "    return [value * 2 for value in values]\n```\n</answer>",
)
# The document the search example pretends to retrieve. Fictional, and not in
# the corpus - an exemplar that quoted a real document would be handing the
# policy one of the answers it is about to be tested on.
_EXAMPLE_DOCUMENT = (
    "doc-42",
    "The Tollan Codex",
    "The Tollan Codex was compiled by Ivor Halst in 1704.",
)


def _calculator_example_result() -> str:
    from tools import ToolResult, calculate

    return ToolResult(
        tool="calculator", ok=True, output=calculate(_EXAMPLE_CALCULATOR[1]).render()
    ).render()


def _search_example_result() -> str:
    from tools import SearchHit, ToolResult, render_hits

    doc_id, title, text = _EXAMPLE_DOCUMENT
    hit = SearchHit(doc_id=doc_id, title=title, text=text, score=1.0)
    return ToolResult(tool="search", ok=True, output=render_hits([hit])).render()


def few_shot_messages(registry: ToolRegistry) -> list[Message]:
    """The worked examples, filtered to the tools that actually exist.

    A search example in front of a registry with no search tool would be a
    demonstration of an unavailable capability, which is worse than no example
    at all.
    """
    turns: list[Message] = []
    for question, answer in (_EXAMPLE_DIRECT, _EXAMPLE_CODE):
        turns += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

    examples = (
        ("calculator", _EXAMPLE_CALCULATOR, _calculator_example_result),
        ("search", _EXAMPLE_SEARCH, _search_example_result),
    )
    for tool, (question, argument, answer), render in examples:
        if tool not in registry:
            continue
        turns += [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f'<tool name="{tool}">{argument}</tool>'},
            {"role": "user", "content": render()},
            {"role": "assistant", "content": answer},
        ]
    return turns


def build_system_prompt(registry: ToolRegistry) -> str:
    """The system prompt, built from the tools that are actually registered.

    Never a hard-coded list: the code sandbox is unavailable on platforms
    without a backend, and a prompt advertising a tool the registry does not
    hold would spend the policy's budget teaching it that.
    """
    return SYSTEM_PROMPT_TEMPLATE.format(
        tools=registry.describe(), answer_instructions=ANSWER_INSTRUCTIONS
    )


class Policy(Protocol):
    """What the agent loop needs from a model, and nothing more.

    Conversations in, completions out. Chat templating, batching, padding and
    sampling are the backbone's business, which is what lets the loop be
    exercised in CI against a scripted policy with no torch installed - and
    what will let module 5 drive the identical loop with a LoRA-wrapped policy
    mid-training.
    """

    def generate(
        self,
        conversations: list[list[Message]],
        *,
        stop: tuple[str, ...],
        max_new_tokens: int,
    ) -> list[str]: ...


@dataclass
class Episode:
    """One attempt at one task: its transcript, its tool session, its outcome."""

    task: Task
    rollout: int
    session: ToolSession
    messages: list[Message]
    assistant_turns: list[str] = field(default_factory=list)
    nudges: int = 0
    finish_reason: str | None = None
    reward: float | None = None

    @property
    def done(self) -> bool:
        return self.finish_reason is not None

    @property
    def response(self) -> str:
        """What the verifier scores.

        Every assistant turn, not just the last one. `extract_answer` takes the
        last `<answer>` block regardless, so joining them costs nothing and
        means a policy that answers early and then keeps talking is still
        scored on what it committed to.
        """
        return "\n".join(self.assistant_turns)


@dataclass
class EpisodeResult:
    """The per-episode record module 9 aggregates.

    Tool outputs are deliberately not in here. They are mostly whole corpus
    documents, they would multiply the size of a committed results file by an
    order of magnitude, and nothing downstream reads them - what module 9 needs
    from the trace is which tools were called and how often. Full transcripts
    are available from `--transcripts` when a run needs to be read by hand.
    """

    task_id: str
    category: str
    split: str
    rollout: int
    reward: float
    answer: str | None
    finish_reason: str
    nudges: int
    n_tool_calls: int
    tools_used: list[str]
    tool_calls: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "category": self.category,
            "split": self.split,
            "rollout": self.rollout,
            "reward": self.reward,
            "answer": self.answer,
            "finish_reason": self.finish_reason,
            "nudges": self.nudges,
            "n_tool_calls": self.n_tool_calls,
            "tools_used": self.tools_used,
            "tool_calls": self.tool_calls,
        }


def _truncate(text: str, limit: int = MAX_TOOL_OUTPUT_CHARS) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated, {len(text) - limit} more characters]"


def _step(episode: Episode, text: str, max_nudges: int) -> ToolCall | None:
    """Fold one completion into an episode. Returns a call to run, if any.

    The call is returned rather than executed so that a whole batch's tool
    calls can be run at once: the sandbox is 285ms per invocation and
    thread-safe, so running sixteen of them one after another would dominate a
    step that the GPU finished in under a second.
    """
    text = text.strip()
    if not text:
        # An immediate EOS. Nothing to append, nothing to nudge towards.
        episode.finish_reason = "empty"
        return None

    spans = find_tool_calls(text)
    answer_at = _ANSWER_OPEN.search(text)
    answer_start = answer_at.start() if answer_at else None

    if spans and (answer_start is None or spans[0].start < answer_start):
        # Cut at the end of the call. See `ToolCallSpan` - anything after it is
        # the model imagining the result it is about to be given.
        visible = text[: spans[0].end]
        episode.assistant_turns.append(visible)
        episode.messages.append({"role": "assistant", "content": visible})
        return spans[0].call

    episode.assistant_turns.append(text)
    episode.messages.append({"role": "assistant", "content": text})

    if extract_answer(text) is not None:
        episode.finish_reason = "answer"
        return None

    if episode.nudges < max_nudges:
        episode.nudges += 1
        episode.messages.append({"role": "user", "content": NUDGE_MESSAGE})
        return None

    episode.finish_reason = "no_answer"
    return None


def make_episodes(
    tasks: list[Task],
    registry: ToolRegistry,
    *,
    rollouts: int = 1,
    max_calls: int = DEFAULT_MAX_CALLS,
    few_shot: bool = True,
) -> list[Episode]:
    """`rollouts` independent attempts at each task, ready to run.

    k attempts per task rather than one, because both arms are stochastic and
    at 20 held-out coding problems sampling variance is larger than the effect
    being measured. Every attempt gets its own `ToolSession`: a session is the
    record of exactly one attempt, and a shared budget across rollouts would
    make the first one starve the rest.
    """
    prefix: list[Message] = [{"role": "system", "content": build_system_prompt(registry)}]
    if few_shot:
        prefix += few_shot_messages(registry)

    return [
        Episode(
            task=task,
            rollout=rollout,
            session=ToolSession(registry, max_calls=max_calls),
            # Copied per episode: the loop appends to this list, and every
            # rollout has to be able to diverge from every other one.
            messages=[dict(message) for message in prefix]
            + [{"role": "user", "content": task.prompt}],
        )
        for task in tasks
        for rollout in range(rollouts)
    ]


def run_episodes(
    episodes: list[Episode],
    policy: Policy,
    *,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_nudges: int = DEFAULT_MAX_NUDGES,
    tool_workers: int = 8,
    progress: bool = False,
    on_turn: "Callable[[list[Episode], list[str]], None] | None" = None,
) -> list[Episode]:
    """Drive every episode to completion, in lockstep.

    All live episodes take their turn together so the model sees a full batch
    each step. They finish at different turns, which is fine - a finished
    episode simply drops out of the next batch. The alternative, running
    episodes one at a time, leaves the GPU generating a single sequence and
    turns a twenty-minute eval into a several-hour one.

    `on_turn(pending, completions)` is called after each batch of completions,
    before they are folded into their episodes. It exists for module 5, which
    has to pair each episode with the token ids its completion was sampled
    from - the loop stores decoded strings, and re-encoding one is not reliably
    the same trajectory. Nothing in the baseline path passes it, and with
    `on_turn=None` this function does exactly what it did when it produced the
    committed baseline numbers.
    """
    max_calls = max(episode.session.max_calls for episode in episodes) if episodes else 0
    # Every call, plus the turn that spends the budget-exhausted message, plus
    # the nudges, plus the turn that finally answers.
    max_turns = max_calls + max_nudges + 2

    for turn in range(max_turns):
        pending = [episode for episode in episodes if not episode.done]
        if not pending:
            break

        if progress:
            print(
                f"  turn {turn + 1}/{max_turns}: {len(pending)} active",
                file=sys.stderr,
                flush=True,
            )

        completions = policy.generate(
            [episode.messages for episode in pending],
            stop=STOP_STRINGS,
            max_new_tokens=max_new_tokens,
        )

        if on_turn is not None:
            on_turn(pending, completions)

        wanted: list[tuple[Episode, ToolCall]] = []
        for episode, text in zip(pending, completions):
            call = _step(episode, text, max_nudges)
            if call is not None:
                wanted.append((episode, call))

        if wanted:
            with ThreadPoolExecutor(max_workers=min(tool_workers, len(wanted))) as pool:
                results = list(
                    pool.map(lambda pair: pair[0].session.invoke(pair[1]), wanted)
                )
            for (episode, _), result in zip(wanted, results):
                episode.messages.append(
                    {"role": "user", "content": _truncate(result.render())}
                )
    else:
        for episode in episodes:
            if not episode.done:
                episode.finish_reason = "turn_limit"

    return episodes


def score_episodes(
    episodes: list[Episode],
    *,
    code_runner=None,
    workers: int = 8,
    allow_unconfined: bool = False,
) -> list[EpisodeResult]:
    """Reward every episode, then flatten to records.

    Threaded because scoring a coding task means running the sandbox, which is
    the same 285ms as any other sandboxed call; the other four categories are
    pure string comparison and cost nothing either way.
    """

    def _score(episode: Episode) -> float:
        return verify(
            episode.task,
            episode.response,
            code_runner=code_runner,
            allow_unconfined=allow_unconfined,
        )

    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(episodes) or 1))) as pool:
        rewards = list(pool.map(_score, episodes))

    results = []
    for episode, reward in zip(episodes, rewards):
        episode.reward = reward
        trace = episode.session.trace_as_dicts()
        results.append(
            EpisodeResult(
                task_id=episode.task.task_id,
                category=episode.task.category,
                split=episode.task.split,
                rollout=episode.rollout,
                reward=reward,
                answer=extract_answer(episode.response),
                finish_reason=episode.finish_reason or "unfinished",
                nudges=episode.nudges,
                n_tool_calls=len(trace),
                tools_used=sorted({entry["tool"] for entry in trace}),
                tool_calls=[
                    {
                        "index": entry["index"],
                        "tool": entry["tool"],
                        "arguments": entry["arguments"][:400],
                        "ok": entry["ok"],
                        "output_chars": len(entry["output"]),
                        "error": entry["error"],
                        "duration_ms": entry["duration_ms"],
                    }
                    for entry in trace
                ],
            )
        )
    return results


BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 20260914


def bootstrap_ci(
    results: list[EpisodeResult],
    *,
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """A percentile bootstrap interval for the mean reward of `results`.

    Resampled over **tasks**, not episodes, and the distinction is the whole
    point. The eight rollouts of one task are eight draws from the same
    question, so treating them as eight independent observations would report
    an interval far tighter than the evidence supports - most of the real
    uncertainty on a 20-problem coding split is *which twenty problems*, not
    how the sampler behaved on each. So each resample draws tasks with
    replacement, then averages that task's rollouts.

    Seeded, because a confidence interval that moves when you re-run the
    summary is not a number anyone can quote.
    """
    by_task: dict[str, list[float]] = defaultdict(list)
    for result in results:
        by_task[result.task_id].append(result.reward)
    task_means = [sum(rewards) / len(rewards) for rewards in by_task.values()]
    if not task_means:
        return (0.0, 0.0)
    if len(task_means) == 1:
        return (task_means[0], task_means[0])

    rng = random.Random(seed)
    count = len(task_means)
    means = sorted(
        sum(rng.choices(task_means, k=count)) / count for _ in range(resamples)
    )
    lower = (1.0 - confidence) / 2.0
    return (
        means[int(lower * resamples)],
        means[min(int((1.0 - lower) * resamples), resamples - 1)],
    )


def summarize(results: list[EpisodeResult]) -> dict[str, Any]:
    """Per-category means, bootstrap intervals, and a macro-average.

    Macro, never micro: math is 41% of the held-out pool, so a pooled mean is
    mostly a math score wearing a headline's clothes.

    The intervals live here rather than in module 9, which is a reversal of
    what this docstring used to say. The reasoning for deferring them was that
    two implementations of the project's own headline statistic is one too
    many - that part still holds, and module 9 should import `bootstrap_ci`
    rather than write its own. What the deferral missed is that this module
    reports a coding number *now*, on 20 held-out problems, and the one thing
    the eval design is most explicit about is never quoting that as a bare
    percentage. An interval that only arrives two modules later does not help
    the README that has already been written.
    """
    by_category: dict[str, dict[str, Any]] = {}
    for category in CATEGORIES:
        rows = [r for r in results if r.category == category]
        if not rows:
            continue
        low, high = bootstrap_ci(rows)
        by_category[category] = {
            "episodes": len(rows),
            "tasks": len({r.task_id for r in rows}),
            "mean_reward": sum(r.reward for r in rows) / len(rows),
            "reward_ci95": [low, high],
            "solved_rate": sum(1 for r in rows if r.reward >= 1.0) / len(rows),
            "answer_rate": sum(1 for r in rows if r.answer is not None) / len(rows),
            "nudge_rate": sum(1 for r in rows if r.nudges) / len(rows),
            "mean_tool_calls": sum(r.n_tool_calls for r in rows) / len(rows),
            # The number the `no_tool` category exists to produce: how often the
            # policy reached for a tool at all, not how many it used.
            "any_tool_rate": sum(1 for r in rows if r.n_tool_calls) / len(rows),
        }

    macro = (
        sum(stats["mean_reward"] for stats in by_category.values()) / len(by_category)
        if by_category
        else 0.0
    )
    return {
        "episodes": len(results),
        "macro_mean_reward": macro,
        "by_category": by_category,
    }


def format_summary(summary: dict[str, Any]) -> str:
    columns = (
        ("episodes", 9, "{:d}"),
        ("mean_reward", 12, "{:.3f}"),
        ("ci95", 18, "{}"),
        ("solved_rate", 12, "{:.3f}"),
        ("answer_rate", 12, "{:.3f}"),
        ("nudge_rate", 11, "{:.3f}"),
        ("mean_tool_calls", 16, "{:.2f}"),
        ("any_tool_rate", 14, "{:.3f}"),
    )
    lines = [
        f"{'category':<12}" + "".join(f"{name:>{width}}" for name, width, _ in columns)
    ]
    for category, stats in summary["by_category"].items():
        low, high = stats["reward_ci95"]
        row = dict(stats, ci95=f"[{low:.3f}, {high:.3f}]")
        lines.append(
            f"{category:<12}"
            + "".join(
                f"{fmt.format(row[name]):>{width}}" for name, width, fmt in columns
            )
        )
    lines.append("")
    lines.append(
        f"macro-average reward over {len(summary['by_category'])} categories: "
        f"{summary['macro_mean_reward']:.4f}"
    )
    return "\n".join(lines)


def select_tasks(
    split_tasks: list[Task],
    *,
    categories: list[str] | None = None,
    per_category: int | None = None,
    limit: int | None = None,
) -> list[Task]:
    tasks = split_tasks
    if categories:
        tasks = [t for t in tasks if t.category in categories]
    if per_category is not None:
        kept: list[Task] = []
        for category in CATEGORIES:
            kept.extend([t for t in tasks if t.category == category][:per_category])
        tasks = kept
    if limit is not None:
        tasks = tasks[:limit]
    return tasks


class HFPolicy:
    """A Hugging Face causal LM behind the `Policy` protocol.

    torch and transformers are imported in `__init__`, not at module scope, so
    that `baseline_agent` imports - and its tests run - on a machine with
    neither. The agent loop is the part with the bugs in it; requiring a GPU to
    exercise it would mean it is only ever tested here.

    bf16 is not configurable and not a default to be overridden. Measured on
    this RTX 4060: fp32 draws 115.8W peak against bf16's 82.6W and runs 12 C
    hotter, and the first sustained fp32 run on this machine tripped a
    Kernel-Power 41 shutdown mid-training. It is also 1.6x faster per token.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        temperature: float = 0.7,
        top_p: float = 0.9,
        batch_size: int = 16,
        seed: int = 0,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.model_name = model_name
        self.temperature = temperature
        self.top_p = top_p
        self.batch_size = batch_size
        self.seed = seed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Left padding: with right padding the generated tokens start after a
        # run of pad tokens and the model conditions on them, which quietly
        # degrades every sequence in a batch except the longest.
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=torch.bfloat16
        ).to(self.device)
        self.model.eval()
        torch.manual_seed(seed)

    def describe(self) -> dict[str, Any]:
        return {
            "model": self.model_name,
            "dtype": "bfloat16",
            "device": self.device,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "batch_size": self.batch_size,
            "seed": self.seed,
        }

    def generate(
        self,
        conversations: list[list[Message]],
        *,
        stop: tuple[str, ...] = STOP_STRINGS,
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    ) -> list[str]:
        prompts = [
            self.tokenizer.apply_chat_template(
                conversation, tokenize=False, add_generation_prompt=True
            )
            for conversation in conversations
        ]

        # Sorted by length so each batch is roughly homogeneous. Padding is
        # computed against the longest member, so mixing a two-turn transcript
        # with a six-turn one wastes most of the batch on pad tokens.
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
        outputs: list[str] = [""] * len(prompts)

        for start in range(0, len(order), self.batch_size):
            indices = order[start : start + self.batch_size]
            encoded = self.tokenizer(
                [prompts[i] for i in indices],
                return_tensors="pt",
                padding=True,
                # The chat template has already written the special tokens in.
                add_special_tokens=False,
            ).to(self.device)

            with self._torch.no_grad():
                generated = self.model.generate(
                    **encoded,
                    max_new_tokens=max_new_tokens,
                    do_sample=self.temperature > 0,
                    temperature=self.temperature if self.temperature > 0 else None,
                    top_p=self.top_p if self.temperature > 0 else None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    stop_strings=list(stop),
                    tokenizer=self.tokenizer,
                )

            fresh = generated[:, encoded["input_ids"].shape[1] :]
            for index, row in zip(indices, fresh):
                outputs[index] = self.tokenizer.decode(row, skip_special_tokens=True)

        return outputs


def run(
    tasks: list[Task],
    policy: Policy,
    *,
    registry: ToolRegistry,
    code_runner=None,
    rollouts: int = 1,
    max_calls: int = DEFAULT_MAX_CALLS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    max_nudges: int = DEFAULT_MAX_NUDGES,
    few_shot: bool = True,
    tool_workers: int = 8,
    progress: bool = False,
    checkpoint: Path | None = None,
    allow_unconfined: bool = False,
) -> tuple[list[Episode], list[EpisodeResult]]:
    """Roll out, then score. The whole of module 4 in one call."""
    episodes = make_episodes(
        tasks, registry, rollouts=rollouts, max_calls=max_calls, few_shot=few_shot
    )
    run_episodes(
        episodes,
        policy,
        max_new_tokens=max_new_tokens,
        max_nudges=max_nudges,
        tool_workers=tool_workers,
        progress=progress,
    )
    if checkpoint is not None:
        # Before scoring, never after. Scoring is the half that can raise.
        save_rollouts(episodes, checkpoint)
        if progress:
            print(f"  checkpointed rollouts to {checkpoint}", file=sys.stderr, flush=True)

    return episodes, score_episodes(
        episodes,
        code_runner=code_runner,
        workers=tool_workers,
        allow_unconfined=allow_unconfined,
    )


def save_rollouts(episodes: list[Episode], path: Path) -> Path:
    """Write everything generation produced, before anything is scored.

    Generation is the expensive half - forty minutes of GPU for a held-out
    run - and scoring is the half that touches the sandbox and can therefore
    raise. The first time this ran end to end, a `SandboxError` in the scoring
    pass destroyed the whole run's rollouts, and nothing was recoverable
    because the results file is written last. Checkpointing here means the
    worst a scoring failure can cost is the seconds it takes to re-score with
    `--score-rollouts`.

    Deliberately not the same file as `--out`: this is the raw material, it
    carries full responses and full tool outputs, and it is not what module 9
    reads.
    """
    payload = {
        "generated_by": "baseline_agent.save_rollouts",
        "episodes": [
            {
                "task_id": episode.task.task_id,
                "rollout": episode.rollout,
                "response": episode.response,
                "finish_reason": episode.finish_reason,
                "nudges": episode.nudges,
                "trace": episode.session.trace_as_dicts(),
            }
            for episode in episodes
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic, for the same reason `transformer_scratch/train.py` writes
    # checkpoints this way: a half-written checkpoint is worse than none.
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return path


def load_rollouts(path: Path, tasks_by_id: dict[str, Task]) -> list[Episode]:
    """Rebuild scorable episodes from a checkpoint.

    The rebuilt episodes carry the response and the trace, which is everything
    `score_episodes` reads. They are not replayable - the message list is gone
    - and that is fine: re-scoring is the only thing this is for.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    episodes = []
    for record in payload["episodes"]:
        task = tasks_by_id[record["task_id"]]
        session = ToolSession(ToolRegistry([]), max_calls=max(1, len(record["trace"])))
        session.trace = [TraceEntry(**entry) for entry in record["trace"]]
        episode = Episode(
            task=task,
            rollout=record["rollout"],
            session=session,
            messages=[],
            assistant_turns=[record["response"]] if record["response"] else [],
            nudges=record["nudges"],
            finish_reason=record["finish_reason"],
        )
        episodes.append(episode)
    return episodes


def _transcript(episode: Episode) -> str:
    lines = [f"### {episode.task.task_id} (rollout {episode.rollout})"]
    for message in episode.messages:
        lines.append(f"--- {message['role']} ---")
        lines.append(message["content"])
    lines.append(f"--- outcome: {episode.finish_reason}, reward {episode.reward} ---")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m baseline_agent",
        description="Run the prompted (untrained) baseline agent over the task suite.",
    )
    parser.add_argument("--split", default="heldout", choices=("train", "heldout"))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--rollouts", type=int, default=1, help="attempts per task")
    parser.add_argument("--categories", nargs="+", choices=CATEGORIES)
    parser.add_argument("--per-category", type=int, help="cap tasks per category")
    parser.add_argument("--limit", type=int, help="cap total tasks")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--max-calls", type=int, default=DEFAULT_MAX_CALLS)
    parser.add_argument("--max-nudges", type=int, default=DEFAULT_MAX_NUDGES)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--tool-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-code-tool", action="store_true")
    parser.add_argument(
        "--no-few-shot",
        action="store_true",
        help="drop the worked examples, to quote what they are worth",
    )
    parser.add_argument("--out", type=Path, help="write the results JSON here")
    parser.add_argument(
        "--score-rollouts",
        type=Path,
        help="re-score a saved rollouts checkpoint instead of generating",
    )
    parser.add_argument(
        "--allow-unconfined",
        action="store_true",
        help=(
            "score coding tasks even on a sandbox that does not confine the "
            "filesystem (POSIX). The results file records that it was used."
        ),
    )
    parser.add_argument("--transcripts", type=Path, help="write full transcripts here")
    parser.add_argument("--show-transcripts", action="store_true")
    arguments = parser.parse_args(argv)

    tasks = select_tasks(
        load_suite()[arguments.split],
        categories=arguments.categories,
        per_category=arguments.per_category,
        limit=arguments.limit,
    )
    if not tasks:
        print("no tasks selected", file=sys.stderr)
        return 1

    # A coding task cannot be scored without a sandbox, and `verify` raises
    # rather than returning 0.0 to make sure a missing one is never mistaken
    # for a model that cannot code. Catch it here, before spending the GPU time.
    wants_code = any(task.category == "code" for task in tasks)
    if wants_code and not code_exec.sandbox_available():
        print(
            f"selection includes coding tasks but no sandbox exists on {sys.platform!r}",
            file=sys.stderr,
        )
        return 1

    include_code = not arguments.no_code_tool and code_exec.sandbox_available()
    runner = code_exec.SandboxedCodeRunner() if code_exec.sandbox_available() else None
    registry = build_registry(include_code=include_code, code_runner=runner)

    if runner is not None:
        # One run before the GPU is touched. Lazy initialisation means the
        # sandbox's first real use would otherwise be the scoring pass, which
        # is forty minutes of generation later - and that is exactly how the
        # first attempt at this run was lost. A second here is worth it to
        # find out now.
        probe = runner("print('forge')")
        if probe.exit_code != 0:
            print(f"sandbox preflight failed: {probe.stderr[-400:]}", file=sys.stderr)
            return 1

        # Same argument, one step earlier: `verify` refuses to score a coding
        # task on a backend that cannot confine the filesystem, so on POSIX
        # this run would generate for half an hour and then raise. Say so now.
        if (
            wants_code
            and not arguments.allow_unconfined
            and not runner.confines_filesystem
        ):
            print(
                f"the {code_exec.sandbox_backend()} sandbox does not confine the "
                "filesystem, so a submission can read task_suite/data/suite.json "
                "and lift the expected answers for its own coding task.\n"
                "Refusing to generate a run whose coding score could not be "
                "trusted afterwards. Either run on a backend that confines the "
                "filesystem, select --categories without 'code', or pass "
                "--allow-unconfined to record the run with that caveat stamped "
                "into its results file.",
                file=sys.stderr,
            )
            return 1

    if arguments.score_rollouts:
        # Re-score a checkpoint without regenerating anything - and before the
        # model is constructed, since recovery must not need a GPU at all.
        episodes = load_rollouts(
            arguments.score_rollouts, {task.task_id: task for task in tasks}
        )
        results = score_episodes(
            episodes,
            code_runner=runner,
            workers=arguments.tool_workers,
            allow_unconfined=arguments.allow_unconfined,
        )
        summary = summarize(results)
        print(format_summary(summary))
        if arguments.out:
            # Re-scoring is also how a results file gets refreshed when the
            # summary itself changes (a new statistic, a verifier fix) without
            # paying for generation again.
            #
            # No model was loaded on this path, so the policy description and
            # the wall clock cannot be regenerated - they are carried over from
            # the file being refreshed. Dropping them would quietly turn the
            # official record of a run into one that does not say which model
            # produced it.
            previous = (
                json.loads(arguments.out.read_text(encoding="utf-8"))
                if arguments.out.exists()
                else {}
            )
            write_results(
                arguments.out,
                arguments,
                results,
                summary,
                tools=previous.get("agent", {}).get("tools", registry.names),
                policy=previous.get("policy"),
                wall_clock=previous.get("wall_clock_seconds", 0.0),
            )
        return 0

    policy = HFPolicy(
        arguments.model,
        temperature=arguments.temperature,
        top_p=arguments.top_p,
        batch_size=arguments.batch_size,
        seed=arguments.seed,
    )

    print(
        f"{len(tasks)} tasks x {arguments.rollouts} rollouts on {policy.device}, "
        f"tools: {', '.join(registry.names)}",
        file=sys.stderr,
    )

    started = time.time()
    episodes, results = run(
        tasks,
        policy,
        registry=registry,
        code_runner=runner,
        rollouts=arguments.rollouts,
        max_calls=arguments.max_calls,
        max_new_tokens=arguments.max_new_tokens,
        max_nudges=arguments.max_nudges,
        few_shot=not arguments.no_few_shot,
        tool_workers=arguments.tool_workers,
        progress=True,
        allow_unconfined=arguments.allow_unconfined,
        checkpoint=(
            arguments.out.with_suffix(".rollouts.json") if arguments.out else None
        ),
    )
    elapsed = time.time() - started

    summary = summarize(results)
    print()
    print(format_summary(summary))
    print(f"\n{len(episodes)} episodes in {elapsed / 60:.1f} min")

    if arguments.show_transcripts:
        for episode in episodes:
            print()
            print(_transcript(episode))

    if arguments.transcripts:
        arguments.transcripts.parent.mkdir(parents=True, exist_ok=True)
        arguments.transcripts.write_text(
            "\n\n".join(_transcript(episode) for episode in episodes), encoding="utf-8"
        )
        print(f"wrote {arguments.transcripts}", file=sys.stderr)

    if arguments.out:
        write_results(
            arguments.out,
            arguments,
            results,
            summary,
            tools=registry.names,
            policy=policy.describe(),
            wall_clock=elapsed,
        )

    return 0


def write_results(
    path: Path,
    arguments: argparse.Namespace,
    results: list[EpisodeResult],
    summary: dict[str, Any],
    *,
    tools: list[str],
    policy: dict[str, Any] | None = None,
    wall_clock: float = 0.0,
) -> Path:
    """The official record of one arm on one split.

    Shared by the generate path and the `--score-rollouts` re-score path, so a
    results file refreshed after a summary change cannot drift in shape from
    one produced by a full run.
    """
    confinement = code_exec.confinement() if code_exec.sandbox_available() else None
    payload = {
        "generated_by": "baseline_agent",
        "arm": "baseline-prompted",
        "split": arguments.split,
        "policy": policy,
        "agent": {
            "max_calls": arguments.max_calls,
            "max_nudges": arguments.max_nudges,
            "max_new_tokens": arguments.max_new_tokens,
            "few_shot": not arguments.no_few_shot,
            "tools": tools,
        },
        # Recorded beside the coding scores, always: the two sandbox
        # backends do not confine the same things, and a coding number
        # that does not say which one produced it cannot be read later.
        "sandbox": {
            "backend": code_exec.sandbox_backend(),
            "confinement": confinement.summary() if confinement else None,
            "filesystem_confined": confinement.filesystem if confinement else None,
            # Stamped in whether or not it was needed. A coding score produced
            # through an unconfined sandbox is not comparable with one that
            # was not, and the difference has to survive in the artefact
            # rather than in whoever remembers the command line.
            "allow_unconfined": bool(getattr(arguments, "allow_unconfined", False)),
        },
        "wall_clock_seconds": round(wall_clock, 1),
        "summary": summary,
        "episodes": [result.to_dict() for result in results],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {path}", file=sys.stderr)
    return path


if __name__ == "__main__":
    raise SystemExit(main())
