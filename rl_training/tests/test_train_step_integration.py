"""One whole training step, on CPU, against a real (tiny) transformer.

The unit tests check each piece in isolation; this checks that they compose.
It drives `train_step` end to end - make_episodes, run_episodes and its
`on_turn` hook, the real verifiers, group advantages, transcript masking,
microbatch packing, the LoRA reference pass through `disable_adapter`, and the
chunked backward - and asserts that a gradient actually reaches the adapter.

The model is a randomly initialised Qwen2 of two 64-wide layers, which is
nonsense as a language model and exactly right as a test fixture: every shape,
every attribute path and every transformers/peft API this loop depends on is
the real one. What it cannot check is memory, which is a property of the 8GB
card and lives in `measure_budget.py`.

Skips without transformers and peft, which on both CI jobs is always. It is
the integration check that would otherwise only ever run as the first minute
of a multi-hour GPU run.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed")
transformers = pytest.importorskip("transformers", reason="transformers not installed")
peft = pytest.importorskip("peft", reason="peft not installed")

from collections import Counter  # noqa: E402

from task_suite.registry import load_suite  # noqa: E402
from tools import build_registry  # noqa: E402

from rl_training.train_grpo import train_step  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"


@pytest.fixture(scope="module")
def tokenizer():
    return transformers.AutoTokenizer.from_pretrained(MODEL, padding_side="left")


@pytest.fixture
def tiny_policy(tokenizer):
    """A real Qwen2 with real LoRA adapters, small enough to train on CPU."""
    config = transformers.Qwen2Config(
        vocab_size=len(tokenizer),
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=128,
        max_position_embeddings=4096,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = transformers.Qwen2ForCausalLM(config)
    model = peft.get_peft_model(
        model,
        peft.LoraConfig(
            r=4,
            lora_alpha=8,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
        ),
    )

    class ScriptedPolicy:
        """Answers each task correctly once and wrongly thereafter.

        That is the point: a group whose rollouts all score the same is
        degenerate and gets dropped, so a policy that always answered the same
        way would leave nothing for the backward pass and the test would prove
        only that `train_step` survives having no work to do.
        """

        def __init__(self, answers: dict[str, str]) -> None:
            self.model = model
            self.tokenizer = tokenizer
            self.device = "cpu"
            self.last_token_ids: list[list[int]] = []
            self._answers = answers
            self._seen: Counter = Counter()

        def generate(self, conversations, *, stop=(), max_new_tokens=0):
            texts = []
            for conversation in conversations:
                prompt = conversation[-1]["content"]
                self._seen[prompt] += 1
                correct = self._answers.get(prompt)
                if correct is not None and self._seen[prompt] == 1:
                    texts.append(f"Reasoning.<answer>{correct}</answer>")
                else:
                    texts.append("Reasoning.<answer>-12345</answer>")
            self.last_token_ids = [
                tokenizer.encode(text, add_special_tokens=False) for text in texts
            ]
            return texts

    return ScriptedPolicy


@pytest.fixture
def tasks():
    train = load_suite()["train"]
    return [task for task in train if task.category == "no_tool"][:2]


def test_a_whole_step_produces_a_gradient_on_the_adapter(tiny_policy, tasks):
    answers = {task.prompt: str(task.ground_truth["value"]) for task in tasks}
    policy = tiny_policy(answers)
    registry = build_registry(include_code=False)

    report = train_step(
        policy,
        tasks,
        registry,
        step=1,
        group_size=4,
        max_new_tokens=32,
        max_sequence_tokens=1024,
        chunk_size=16,
        kl_beta=0.04,
        tool_workers=1,
    )

    assert report.episodes == 8, "2 tasks x 4 rollouts"
    assert report.groups == 2
    assert report.degenerate_groups == 0, "one right and three wrong is not degenerate"
    assert report.kept_episodes == 8
    assert report.completion_tokens > 0
    assert report.microbatches >= 1

    trainable = [p for p in policy.model.parameters() if p.requires_grad]
    assert trainable, "LoRA did not attach"
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in trainable
    ), "no gradient reached the adapter"


def test_the_base_weights_are_never_touched(tiny_policy, tasks):
    """LoRA's whole claim: only the adapter moves."""
    answers = {task.prompt: str(task.ground_truth["value"]) for task in tasks}
    policy = tiny_policy(answers)

    frozen = [p for p in policy.model.parameters() if not p.requires_grad]
    train_step(
        policy, tasks, build_registry(include_code=False), step=1, group_size=4,
        max_new_tokens=32, chunk_size=16, kl_beta=0.0, tool_workers=1,
    )

    assert all(p.grad is None for p in frozen)


def test_the_kl_reference_is_the_base_policy_not_the_current_one(tiny_policy, tasks):
    """With LoRA freshly initialised, B is zero, so policy == reference and the
    KL must be zero. A non-zero KL here would mean `disable_adapter` is not
    doing what the objective assumes."""
    answers = {task.prompt: str(task.ground_truth["value"]) for task in tasks}
    policy = tiny_policy(answers)

    report = train_step(
        policy, tasks, build_registry(include_code=False), step=1, group_size=4,
        max_new_tokens=32, chunk_size=16, kl_beta=0.04, tool_workers=1,
    )

    assert report.mean_kl == pytest.approx(0.0, abs=1e-6)


def test_a_fully_degenerate_step_does_no_backward_and_does_not_crash(
    tiny_policy, tasks
):
    """The common case at the baseline's skill: every rollout scores zero."""
    policy = tiny_policy({})  # no correct answers at all

    report = train_step(
        policy, tasks, build_registry(include_code=False), step=1, group_size=4,
        max_new_tokens=32, chunk_size=16, kl_beta=0.04, tool_workers=1,
    )

    assert report.degenerate_groups == report.groups
    assert report.kept_episodes == 0
    assert report.dropped_degenerate == report.episodes
    assert report.microbatches == 0
    assert all(p.grad is None for p in policy.model.parameters())


def test_keeping_degenerate_groups_still_produces_no_policy_gradient(
    tiny_policy, tasks
):
    """`--keep-degenerate` keeps the KL term, not a policy gradient: the
    advantages are still zero, so the difference is the regulariser only."""
    policy = tiny_policy({})

    report = train_step(
        policy, tasks, build_registry(include_code=False), step=1, group_size=4,
        max_new_tokens=32, chunk_size=16, kl_beta=0.0, tool_workers=1,
        keep_degenerate=True,
    )

    assert report.kept_episodes == report.episodes
    assert report.microbatches >= 1
    trainable = [p for p in policy.model.parameters() if p.requires_grad]
    assert all(
        p.grad is None or p.grad.abs().sum() == 0 for p in trainable
    ), "zero advantages with no KL must leave the policy alone"


def test_dropping_degenerate_groups_does_not_change_the_gradient(tiny_policy):
    """Dropping them must be a compute saving and nothing else.

    A degenerate group contributes zero to the numerator, so if it were also
    left out of the *denominator* the remaining gradient would be scaled up by
    the reciprocal of the usable fraction - a silent ~3.8x on the effective
    learning rate at the measured 26.4%, drifting step to step with however
    many groups happened to be degenerate. This is the test that keeps the
    normalizer counting every episode the step sampled.
    """
    train = load_suite()["train"]
    no_tool = [t for t in train if t.category == "no_tool"][:2]
    answers = {t.prompt: str(t.ground_truth["value"]) for t in no_tool[:1]}

    grads = {}
    for keep in (True, False):
        policy = tiny_policy(answers)
        # The fixture builds the model once, so both passes share it and
        # `train_step` deliberately does not zero gradients - the caller owns
        # the optimiser. Without this the second pass accumulates onto the
        # first and the comparison reads a clean 2x that is entirely the
        # test's own doing.
        policy.model.zero_grad(set_to_none=True)
        torch.manual_seed(3)
        report = train_step(
            policy, no_tool, build_registry(include_code=False), step=1,
            group_size=4, max_new_tokens=32, chunk_size=16, kl_beta=0.0,
            tool_workers=1, keep_degenerate=keep,
        )
        grads[keep] = {
            n: p.grad.clone()
            for n, p in policy.model.named_parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        }
        if keep:
            assert report.dropped_degenerate == 0
        else:
            assert report.dropped_degenerate > 0, "the fixture must drop something"

    assert grads[True], "the run must produce some gradient to compare"
    assert set(grads[True]) == set(grads[False])
    for name, grad in grads[True].items():
        assert torch.allclose(grad, grads[False][name], atol=1e-6), name


def test_the_sampled_ids_path_is_the_one_taken(tiny_policy, tasks):
    """If the recorder ever falls out of step the run silently switches to
    re-tokenizing, so the count is asserted rather than logged and forgotten."""
    answers = {task.prompt: str(task.ground_truth["value"]) for task in tasks}
    policy = tiny_policy(answers)

    report = train_step(
        policy, tasks, build_registry(include_code=False), step=1, group_size=4,
        max_new_tokens=32, chunk_size=16, kl_beta=0.0, tool_workers=1,
    )

    assert report.retokenized == 0, "every kept transcript used the sampled ids"
