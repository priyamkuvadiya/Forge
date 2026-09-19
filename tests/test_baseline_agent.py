"""The agent loop, exercised against scripted policies instead of a model.

`baseline_agent` deliberately keeps torch and transformers inside `HFPolicy.
__init__`, and the reason is this file: the loop is where the bugs are - the
truncation rule, the ordering rule between an answer and a tool call, the
budget, the nudge - and every one of them is a text-handling decision that a
GPU adds nothing to. Testing them against a scripted policy means they run in
CI on both platforms, on every push, in under a second.

A scripted policy is also the only way to test the cases that matter and are
rare: a model that hallucinates its own tool result, one that calls a tool
that does not exist, one that never answers at all. Waiting for a real 0.5B
model to happen to do each of those is not a test.
"""

import pytest

import baseline_agent
from baseline_agent import (
    Episode,
    build_system_prompt,
    few_shot_messages,
    make_episodes,
    run_episodes,
    score_episodes,
    select_tasks,
    summarize,
)
from task_suite.schema import Task
from tools import ToolRegistry, ToolResult, build_registry, find_tool_calls, parse_tool_calls


class ScriptedPolicy:
    """Replays a fixed list of completions, one per turn, for every episode.

    Records the conversations it was handed so a test can assert what the loop
    actually put in front of the model - which is the only way to check that a
    hallucinated continuation was really dropped rather than merely not scored.
    """

    def __init__(self, turns: list[str]) -> None:
        self.turns = turns
        self.seen: list[list[list[dict[str, str]]]] = []
        self._turn = 0

    def generate(self, conversations, *, stop=(), max_new_tokens=0):
        # Copy: the loop mutates the message lists in place afterwards.
        self.seen.append([list(c) for c in conversations])
        text = self.turns[min(self._turn, len(self.turns) - 1)]
        self._turn += 1
        return [text] * len(conversations)


class EchoTool:
    name = "echo"
    description = "Echo the argument back."

    def __init__(self) -> None:
        self.calls: list[str] = []

    def invoke(self, arguments: str) -> ToolResult:
        self.calls.append(arguments)
        return ToolResult(tool=self.name, ok=True, output=f"echo: {arguments}")


def make_task(**overrides) -> Task:
    payload = {
        "task_id": "t-0",
        "category": "math",
        "prompt": "What is 2 plus 2?",
        "ground_truth": {"value": 4.0, "tolerance": 0.0},
        "split": "train",
    }
    payload.update(overrides)
    return Task(**payload)


@pytest.fixture
def registry_and_tool():
    tool = EchoTool()
    return ToolRegistry([tool]), tool


def drive(policy, registry, task=None, *, max_calls=8, max_nudges=1) -> Episode:
    # few_shot off unless a test is specifically about the examples: they add
    # six turns to every transcript, and a test asserting on message positions
    # should be asserting about the loop, not about the prompt prefix.
    episodes = make_episodes(
        [task or make_task()], registry, max_calls=max_calls, few_shot=False
    )
    run_episodes(episodes, policy, max_nudges=max_nudges, tool_workers=1)
    return episodes[0]


# --- the span addition to the tool contract -------------------------------


def test_find_tool_calls_reports_offsets_that_bound_the_call():
    text = 'thinking <tool name="echo">hi</tool> trailing'
    span = find_tool_calls(text)[0]
    assert text[span.start : span.end] == '<tool name="echo">hi</tool>'
    assert span.call.name == "echo"


def test_parse_tool_calls_still_returns_bare_calls():
    text = '<tool name="echo">a</tool><tool name="echo">b</tool>'
    assert [c.arguments for c in parse_tool_calls(text)] == ["a", "b"]


# --- the loop -------------------------------------------------------------


def test_direct_answer_finishes_without_touching_a_tool(registry_and_tool):
    registry, tool = registry_and_tool
    episode = drive(ScriptedPolicy(["The sum is <answer>4</answer>"]), registry)

    assert episode.finish_reason == "answer"
    assert episode.session.calls_made == 0
    assert tool.calls == []


def test_a_tool_call_is_executed_and_its_result_fed_back(registry_and_tool):
    registry, tool = registry_and_tool
    policy = ScriptedPolicy(
        ['<tool name="echo">ping</tool>', "<answer>4</answer>"]
    )
    episode = drive(policy, registry)

    assert tool.calls == ["ping"]
    assert episode.session.calls_made == 1
    assert episode.finish_reason == "answer"
    # The result reached the model on the turn after the call.
    assert "echo: ping" in policy.seen[1][0][-1]["content"]


def test_text_after_a_tool_call_is_discarded(registry_and_tool):
    """The case this rule exists for: a model inventing its own tool result."""
    registry, _ = registry_and_tool
    policy = ScriptedPolicy(
        [
            '<tool name="echo">ping</tool>\n'
            '<tool_result name="echo">\necho: 99\n</tool_result>\n'
            "So the answer is <answer>99</answer>",
            "<answer>4</answer>",
        ]
    )
    episode = drive(policy, registry)

    assert episode.assistant_turns[0] == '<tool name="echo">ping</tool>'
    # The fabricated result and the answer derived from it never reached the
    # verifier, and never went back into the model's own context either.
    assert "99" not in episode.response
    assert "99" not in policy.seen[1][0][-2]["content"]
    assert episode.finish_reason == "answer"
    assert "<answer>4</answer>" in episode.assistant_turns[-1]


def test_an_answer_written_before_a_tool_call_wins(registry_and_tool):
    registry, tool = registry_and_tool
    episode = drive(
        ScriptedPolicy(['<answer>4</answer> <tool name="echo">ping</tool>']), registry
    )

    assert episode.finish_reason == "answer"
    assert tool.calls == []


def test_a_call_to_an_unknown_tool_is_reported_and_the_episode_continues(
    registry_and_tool,
):
    registry, _ = registry_and_tool
    policy = ScriptedPolicy(
        ['<tool name="nope">x</tool>', "<answer>4</answer>"]
    )
    episode = drive(policy, registry)

    assert episode.session.trace[0].ok is False
    assert "no tool named 'nope'" in policy.seen[1][0][-1]["content"]
    assert episode.finish_reason == "answer"


def test_the_budget_caps_calls_and_says_so(registry_and_tool):
    registry, tool = registry_and_tool
    # A policy that only ever calls the tool, never answers.
    policy = ScriptedPolicy(['<tool name="echo">ping</tool>'])
    episode = drive(policy, registry, max_calls=2, max_nudges=0)

    assert len(tool.calls) == 2
    assert episode.session.calls_made == 2
    assert "budget exhausted" in policy.seen[-1][0][-1]["content"]
    assert episode.finish_reason == "turn_limit"


def test_one_nudge_is_sent_then_the_episode_gives_up(registry_and_tool):
    registry, _ = registry_and_tool
    policy = ScriptedPolicy(["I am thinking about it."])
    episode = drive(policy, registry, max_nudges=1)

    assert episode.nudges == 1
    assert baseline_agent.NUDGE_MESSAGE in [m["content"] for m in episode.messages]
    assert episode.finish_reason == "no_answer"


def test_zero_nudges_gives_up_immediately(registry_and_tool):
    registry, _ = registry_and_tool
    episode = drive(ScriptedPolicy(["I am thinking about it."]), registry, max_nudges=0)

    assert episode.nudges == 0
    assert episode.finish_reason == "no_answer"


def test_a_nudge_can_rescue_an_answer(registry_and_tool):
    registry, _ = registry_and_tool
    policy = ScriptedPolicy(["I am thinking about it.", "<answer>4</answer>"])
    episode = drive(policy, registry, max_nudges=1)

    assert episode.nudges == 1
    assert episode.finish_reason == "answer"


def test_an_empty_completion_ends_the_episode(registry_and_tool):
    registry, _ = registry_and_tool
    episode = drive(ScriptedPolicy(["   "]), registry)

    assert episode.finish_reason == "empty"
    assert episode.assistant_turns == []


def test_a_long_tool_output_is_truncated_before_it_reaches_the_transcript():
    class FloodTool:
        name = "flood"
        description = "Returns far too much."

        def invoke(self, arguments: str) -> ToolResult:
            return ToolResult(tool=self.name, ok=True, output="x" * 50_000)

    registry = ToolRegistry([FloodTool()])
    policy = ScriptedPolicy(['<tool name="flood">go</tool>', "<answer>4</answer>"])
    drive(policy, registry)

    fed_back = policy.seen[1][0][-1]["content"]
    assert "truncated" in fed_back
    assert len(fed_back) < baseline_agent.MAX_TOOL_OUTPUT_CHARS + 200


def test_each_rollout_gets_its_own_session_and_budget(registry_and_tool):
    registry, tool = registry_and_tool
    episodes = make_episodes([make_task()], registry, rollouts=3, max_calls=1, few_shot=False)
    run_episodes(
        episodes,
        ScriptedPolicy(['<tool name="echo">ping</tool>', "<answer>4</answer>"]),
        max_nudges=0,
        tool_workers=1,
    )

    assert [e.rollout for e in episodes] == [0, 1, 2]
    assert all(e.session.calls_made == 1 for e in episodes)
    assert len(tool.calls) == 3


def test_episodes_finishing_at_different_turns_drop_out_of_the_batch():
    """The lockstep loop must not keep generating for an episode that is done."""

    class Diverging:
        def __init__(self) -> None:
            self.batch_sizes: list[int] = []
            self._turn = 0

        def generate(self, conversations, *, stop=(), max_new_tokens=0):
            self.batch_sizes.append(len(conversations))
            self._turn += 1
            if self._turn == 1:
                # First episode answers, second calls a tool.
                return ["<answer>4</answer>", '<tool name="echo">ping</tool>']
            return ["<answer>4</answer>"] * len(conversations)

    registry = ToolRegistry([EchoTool()])
    episodes = make_episodes(
        [make_task(task_id="t-0"), make_task(task_id="t-1")], registry, few_shot=False
    )
    policy = Diverging()
    run_episodes(episodes, policy, max_nudges=0, tool_workers=1)

    assert policy.batch_sizes == [2, 1]
    assert all(e.finish_reason == "answer" for e in episodes)


# --- scoring and reporting ------------------------------------------------


def test_scoring_goes_through_the_suite_verifier(registry_and_tool):
    registry, _ = registry_and_tool
    episodes = make_episodes(
        [make_task(task_id="right"), make_task(task_id="wrong")], registry, few_shot=False
    )

    class PerTask:
        def generate(self, conversations, *, stop=(), max_new_tokens=0):
            return [
                "<answer>4</answer>" if "right" in str(c) or i == 0 else "<answer>7</answer>"
                for i, c in enumerate(conversations)
            ]

    run_episodes(episodes, PerTask(), max_nudges=0, tool_workers=1)
    results = score_episodes(episodes, workers=1)

    assert [r.reward for r in results] == [1.0, 0.0]
    assert [r.answer for r in results] == ["4", "7"]


def test_a_coding_task_without_a_runner_raises_rather_than_scoring_zero(
    registry_and_tool,
):
    registry, _ = registry_and_tool
    task = make_task(
        task_id="c-0",
        category="code",
        prompt="Write f.",
        ground_truth={"entry_point": "f", "tests": [{"args": [1], "expected": 1}]},
    )
    episodes = make_episodes([task], registry, few_shot=False)
    run_episodes(
        episodes, ScriptedPolicy(["<answer>def f(x): return x</answer>"]), tool_workers=1
    )

    with pytest.raises(ValueError, match="sandboxed code_runner"):
        score_episodes(episodes, workers=1)


def test_the_summary_macro_averages_over_categories_not_episodes(registry_and_tool):
    registry, _ = registry_and_tool
    tasks = [make_task(task_id=f"m-{i}") for i in range(4)] + [
        make_task(task_id="n-0", category="no_tool", ground_truth={"value": 4.0, "tolerance": 0.0})
    ]
    episodes = make_episodes(tasks, registry, few_shot=False)

    class MathWrong:
        def generate(self, conversations, *, stop=(), max_new_tokens=0):
            return [
                "<answer>4</answer>" if "no_tool" in c[1]["content"] else "<answer>9</answer>"
                for c in conversations
            ]

    # The no_tool task is identified by its prompt, so give it a distinct one.
    episodes[-1].messages[1]["content"] = "no_tool question"
    run_episodes(episodes, MathWrong(), max_nudges=0, tool_workers=1)
    summary = summarize(score_episodes(episodes, workers=1))

    assert summary["by_category"]["math"]["mean_reward"] == 0.0
    assert summary["by_category"]["no_tool"]["mean_reward"] == 1.0
    # Micro would be 1/5 = 0.2; macro over the two present categories is 0.5.
    assert summary["macro_mean_reward"] == 0.5


def test_the_summary_reports_tool_use_on_the_control_category(registry_and_tool):
    registry, _ = registry_and_tool
    task = make_task(task_id="n-0", category="no_tool")
    episodes = make_episodes([task], registry, rollouts=2, few_shot=False)
    run_episodes(
        episodes,
        ScriptedPolicy(['<tool name="echo">why</tool>', "<answer>4</answer>"]),
        tool_workers=1,
    )
    summary = summarize(score_episodes(episodes, workers=1))

    assert summary["by_category"]["no_tool"]["any_tool_rate"] == 1.0
    assert summary["by_category"]["no_tool"]["mean_tool_calls"] == 1.0


def test_results_records_carry_the_trace_without_the_tool_output(registry_and_tool):
    registry, _ = registry_and_tool
    policy = ScriptedPolicy(['<tool name="echo">ping</tool>', "<answer>4</answer>"])
    episodes = make_episodes([make_task()], registry, few_shot=False)
    run_episodes(episodes, policy, tool_workers=1)
    record = score_episodes(episodes, workers=1)[0].to_dict()

    assert record["tools_used"] == ["echo"]
    assert record["n_tool_calls"] == 1
    assert record["tool_calls"][0]["arguments"] == "ping"
    assert record["tool_calls"][0]["output_chars"] == len("echo: ping")
    assert "output" not in record["tool_calls"][0]


# --- bootstrap intervals ---------------------------------------------------


def _results(rewards_by_task: dict[str, list[float]]) -> list:
    """EpisodeResult rows with the given per-task rollout rewards."""
    rows = []
    for task_id, rewards in rewards_by_task.items():
        for rollout, reward in enumerate(rewards):
            rows.append(
                baseline_agent.EpisodeResult(
                    task_id=task_id,
                    category="math",
                    split="heldout",
                    rollout=rollout,
                    reward=reward,
                    answer="x",
                    finish_reason="answered",
                    nudges=0,
                    n_tool_calls=0,
                    tools_used=[],
                    tool_calls=[],
                )
            )
    return rows


def test_the_bootstrap_resamples_tasks_not_episodes():
    """Rollouts of one task are not independent observations.

    Eight rollouts of the same question are eight draws from one question. A
    bootstrap over episodes would treat them as eight tasks and report an
    interval far tighter than the evidence supports - which on a 20-problem
    coding split is precisely the error the eval design exists to prevent.

    Same total episodes, same mean, different number of underlying tasks: the
    two-task interval must be dramatically wider than the sixteen-task one.
    """
    two_tasks = _results({"a": [1.0] * 8, "b": [0.0] * 8})
    many_tasks = _results({f"t{i}": [1.0 if i % 2 else 0.0] for i in range(16)})

    assert len(two_tasks) == len(many_tasks) == 16
    assert sum(r.reward for r in two_tasks) == sum(r.reward for r in many_tasks)

    narrow_low, narrow_high = baseline_agent.bootstrap_ci(many_tasks)
    wide_low, wide_high = baseline_agent.bootstrap_ci(two_tasks)
    assert (wide_high - wide_low) > (narrow_high - narrow_low) * 1.5


def test_the_bootstrap_is_deterministic():
    """An interval that moves between runs is not a number anyone can quote."""
    rows = _results({f"t{i}": [i / 10.0] for i in range(10)})
    assert baseline_agent.bootstrap_ci(rows) == baseline_agent.bootstrap_ci(rows)


def test_an_unvarying_category_gets_a_zero_width_interval():
    """Held-out QA scored a flat 0.000, and its interval must say so.

    A bootstrap over identical values can only resample identical values, so
    [0.000, 0.000] is the honest report - not a defect, and not something to
    paper over with a floor.
    """
    assert baseline_agent.bootstrap_ci(_results({f"t{i}": [0.0] * 8 for i in range(20)})) == (
        0.0,
        0.0,
    )


def test_the_interval_brackets_the_mean_it_is_reported_beside():
    rows = _results({f"t{i}": [1.0, 0.0, 1.0, 1.0] for i in range(12)})
    mean = sum(r.reward for r in rows) / len(rows)
    low, high = baseline_agent.bootstrap_ci(rows)
    assert low <= mean <= high

    summary = summarize(rows)["by_category"]["math"]
    assert summary["reward_ci95"] == [low, high]


# --- the rollouts checkpoint ------------------------------------------------


def test_rollouts_survive_a_save_and_load_round_trip(tmp_path, registry_and_tool):
    """Generation is the expensive half and scoring is the half that raises.

    A `SandboxError` during scoring once destroyed a completed 40-minute
    held-out run, because the results file is written last and nothing else
    was written at all. The checkpoint exists so that costs seconds, not the
    GPU time - so what it round-trips has to be exactly what scoring reads.
    """
    registry, _ = registry_and_tool
    task = make_task(task_id="m-0")
    episodes = make_episodes([task], registry, rollouts=2, few_shot=False)
    run_episodes(
        episodes,
        ScriptedPolicy(['<tool name="echo">hi</tool>', "<answer>4</answer>"]),
        max_nudges=0,
        tool_workers=1,
    )
    expected = [r.reward for r in score_episodes(episodes, workers=1)]

    path = tmp_path / "rollouts.json"
    baseline_agent.save_rollouts(episodes, path)
    restored = baseline_agent.load_rollouts(path, {task.task_id: task})

    assert [r.reward for r in score_episodes(restored, workers=1)] == expected
    assert [e.response for e in restored] == [e.response for e in episodes]
    assert [e.session.trace_as_dicts() for e in restored] == [
        e.session.trace_as_dicts() for e in episodes
    ]


def test_a_rescored_checkpoint_reports_the_same_tool_counts(tmp_path, registry_and_tool):
    """Module 9 reads tool counts off these records, not just rewards."""
    registry, _ = registry_and_tool
    task = make_task(task_id="m-0")
    episodes = make_episodes([task], registry, few_shot=False)
    run_episodes(
        episodes,
        ScriptedPolicy(['<tool name="echo">a</tool>', "<answer>4</answer>"]),
        max_nudges=0,
        tool_workers=1,
    )

    path = tmp_path / "rollouts.json"
    baseline_agent.save_rollouts(episodes, path)
    restored = baseline_agent.load_rollouts(path, {task.task_id: task})

    original = score_episodes(episodes, workers=1)[0]
    recovered = score_episodes(restored, workers=1)[0]
    assert recovered.n_tool_calls == original.n_tool_calls == 1
    assert recovered.tools_used == original.tools_used == ["echo"]


def test_the_checkpoint_is_written_before_scoring_can_fail(tmp_path, registry_and_tool):
    """The ordering is the entire point; assert it, not just that a file appears.

    A coding task with no runner makes `verify` raise, which is the same shape
    of failure the sandbox produced. The checkpoint must already be on disk
    when it does.
    """
    registry, _ = registry_and_tool
    task = make_task(
        task_id="c-0",
        category="code",
        prompt="Write f.",
        ground_truth={"entry_point": "f", "tests": [{"args": [1], "expected": 1}]},
    )
    path = tmp_path / "rollouts.json"

    with pytest.raises(ValueError, match="sandboxed code_runner"):
        baseline_agent.run(
            [task],
            ScriptedPolicy(["<answer>def f(x): return x</answer>"]),
            registry=registry,
            code_runner=None,
            few_shot=False,
            tool_workers=1,
            checkpoint=path,
        )

    assert path.exists(), "scoring raised and took the generation with it"
    restored = baseline_agent.load_rollouts(path, {task.task_id: task})
    assert restored[0].response == "<answer>def f(x): return x</answer>"


# --- prompt and selection -------------------------------------------------


def test_the_system_prompt_advertises_exactly_the_registered_tools(registry_and_tool):
    registry, _ = registry_and_tool
    prompt = build_system_prompt(registry)

    assert "echo" in prompt
    assert "calculator" not in prompt
    assert "<answer></answer>" in prompt


def test_the_system_prompt_says_tools_are_optional(registry_and_tool):
    """The `no_tool` control measures nothing if the prompt never says this."""
    registry, _ = registry_and_tool
    assert "Not every task needs a tool" in build_system_prompt(registry)


def test_few_shot_examples_only_demonstrate_registered_tools(registry_and_tool):
    registry, _ = registry_and_tool  # holds `echo` and nothing else
    turns = few_shot_messages(registry)

    assert [t["role"] for t in turns] == ["user", "assistant"] * 2
    assert not any("<tool " in t["content"] for t in turns)


def test_few_shot_examples_demonstrate_the_tools_that_are_present():
    registry = build_registry(include_code=False)
    turns = few_shot_messages(registry)
    text = "\n".join(t["content"] for t in turns)

    assert '<tool name="calculator">' in text
    assert '<tool name="search">' in text
    # Every example, tool-using or not, commits to an answer.
    assert text.count("<answer>") == 4


def test_few_shot_tool_results_are_rendered_by_the_real_code_paths():
    """An exemplar that drifts from the live format teaches the wrong shape."""
    from tools import CalculatorTool, ToolRegistry as Registry, ToolResult

    registry = Registry([CalculatorTool()])
    turns = few_shot_messages(registry)
    # Two tool-free examples, then question / call / result / answer.
    question, call_turn, result_turn, answer_turn = turns[4:8]

    call = find_tool_calls(call_turn["content"])[0].call
    live = CalculatorTool().invoke(call.arguments)
    assert isinstance(live, ToolResult) and live.ok
    assert result_turn["content"] == live.render()
    # And the answer the example commits to is what the tool actually said,
    # not a tidied-up version of it.
    assert "27562.5" in live.output
    assert answer_turn["content"] == "<answer>27562.50</answer>"
    assert question["role"] == "user"


def test_few_shot_examples_leak_no_task_from_the_suite():
    """A worked example quoting a real corpus document would be an answer key."""
    from task_suite.registry import load_corpus, load_suite

    registry = build_registry(include_code=False)
    text = "\n".join(t["content"] for t in few_shot_messages(registry))

    tasks = [task for split in load_suite().values() for task in split]
    assert not any(task.prompt in text for task in tasks)
    assert not any(document["title"] in text for document in load_corpus())
    # The coding exemplar must not solve a task either: `extract_code` would
    # happily lift its function straight out of the example.
    entry_points = {
        task.ground_truth["entry_point"] for task in tasks if task.category == "code"
    }
    assert not any(entry_point in text for entry_point in entry_points)


def test_the_few_shot_prefix_is_copied_per_episode(registry_and_tool):
    registry, _ = registry_and_tool
    first, second = make_episodes(
        [make_task(task_id="a"), make_task(task_id="b")], registry, few_shot=True
    )
    first.messages[0]["content"] = "tampered"

    assert second.messages[0]["content"] != "tampered"


def test_select_tasks_caps_per_category():
    tasks = [make_task(task_id=f"m-{i}") for i in range(5)] + [
        make_task(task_id=f"q-{i}", category="qa", ground_truth={"answers": ["x"]})
        for i in range(5)
    ]
    picked = select_tasks(tasks, per_category=2)

    assert [t.task_id for t in picked] == ["m-0", "m-1", "q-0", "q-1"]


def test_select_tasks_filters_by_category():
    tasks = [make_task(task_id="m-0")] + [
        make_task(task_id="q-0", category="qa", ground_truth={"answers": ["x"]})
    ]
    assert [t.task_id for t in select_tasks(tasks, categories=["qa"])] == ["q-0"]
