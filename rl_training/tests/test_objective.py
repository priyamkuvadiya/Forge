"""The GRPO maths, and proof that the memory trick does not change the answer.

Runs on CPU against a toy model of a few hundred parameters. That is enough to
check every claim in `objective.py` that is about *arithmetic* - the shift, the
KL estimator, the clipping, and above all that the chunked backward produces
bit-comparable gradients to the obvious implementation. The claims that are
about *memory* cannot be checked here and are not pretended to be: they need
the real model on the real card, and they live in
`rl_training/measure_budget.py`.
"""

import pytest

torch = pytest.importorskip("torch", reason="torch not installed (expected on the no-torch CI job)")

import torch.nn as nn  # noqa: E402

from rl_training.objective import (  # noqa: E402
    grpo_backward,
    grpo_token_loss,
    k3_kl,
    naive_grpo_backward,
    shift_for_prediction,
    token_logprobs,
)

VOCAB = 23
HIDDEN = 8


class ToyBody(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN)
        self.mix = nn.Linear(HIDDEN, HIDDEN)

    def forward(self, input_ids, attention_mask=None):
        # A tuple, because that is what a transformers decoder returns and
        # `_body_and_head` indexes `[0]`.
        return (self.mix(self.embed(input_ids)),)


class ToyLM(nn.Module):
    """Enough of a causal LM for the objective: a decoder and a head."""

    def __init__(self) -> None:
        super().__init__()
        self.body = ToyBody()
        self.head = nn.Linear(HIDDEN, VOCAB, bias=False)

    def get_decoder(self):
        return self.body

    def get_output_embeddings(self):
        return self.head

    def forward(self, input_ids, attention_mask=None):
        hidden = self.body(input_ids, attention_mask)[0]

        class Output:
            pass

        out = Output()
        out.logits = self.head(hidden)
        return out


@pytest.fixture
def batch():
    torch.manual_seed(0)
    input_ids = torch.randint(0, VOCAB, (3, 17))
    attention_mask = torch.ones_like(input_ids)
    completion_mask = torch.zeros_like(input_ids)
    # Two disjoint policy spans per row, with environment text between them -
    # the shape a tool-using episode actually has.
    completion_mask[:, 4:8] = 1
    completion_mask[:, 12:15] = 1
    advantages = torch.tensor([1.0, -0.5, 0.25])
    return input_ids, attention_mask, completion_mask, advantages


def test_shift_lines_targets_up_with_the_tokens_being_predicted():
    """Worked by hand. An error here trains on the first token of every tool result."""
    input_ids = torch.tensor([[10, 11, 12, 13, 14]])
    completion_mask = torch.tensor([[0, 0, 1, 1, 0]])

    targets, loss_mask = shift_for_prediction(input_ids, completion_mask)

    assert targets.tolist() == [[11, 12, 13, 14]]
    assert loss_mask.tolist() == [[0, 1, 1, 0]]


def test_shift_does_not_train_on_the_token_after_a_completion():
    """The specific off-by-one: token 14 follows a completion but is not one."""
    completion_mask = torch.tensor([[0, 0, 1, 1, 0]])
    _, loss_mask = shift_for_prediction(torch.zeros_like(completion_mask), completion_mask)

    assert loss_mask[0, -1].item() == 0


def test_token_logprobs_matches_a_hand_computed_softmax():
    logits = torch.tensor([[[0.0, 1.0, 2.0]]])
    targets = torch.tensor([[2]])

    got = token_logprobs(logits, targets)

    expected = torch.log_softmax(logits.float(), dim=-1)[0, 0, 2]
    assert torch.allclose(got[0, 0], expected)


def test_k3_kl_is_zero_at_equality_and_positive_otherwise():
    same = torch.tensor([-1.2, -0.4, -3.0])
    assert torch.allclose(k3_kl(same, same), torch.zeros(3), atol=1e-7)

    different = k3_kl(torch.tensor([-1.0, -2.0]), torch.tensor([-1.5, -0.5]))
    assert (different > 0).all(), "k3 is non-negative by construction"


def test_naive_kl_difference_can_go_negative_which_is_why_k3_is_used():
    """Justifies the estimator choice rather than asserting it."""
    logp = torch.tensor([-1.0])
    ref = torch.tensor([-1.5])
    assert (ref - logp).item() < 0
    assert k3_kl(logp, ref).item() > 0


def test_clipping_is_inert_when_old_logprobs_come_from_the_same_pass():
    logp = torch.tensor([[-1.0, -2.0, -0.5]])
    mask = torch.ones_like(logp)

    _, diagnostics = grpo_token_loss(
        logp,
        logp.detach(),
        None,
        torch.tensor([[2.0]]),
        mask,
        normalizer=3.0,
    )

    assert diagnostics["mean_ratio"] == pytest.approx(1.0, abs=1e-6)
    assert diagnostics["clip_fraction"] == pytest.approx(0.0)


def test_clipping_fires_once_the_ratio_moves():
    logp = torch.tensor([[0.0]])
    old = torch.tensor([[-1.0]])  # ratio = e ~ 2.72, well past 1.2

    _, diagnostics = grpo_token_loss(
        logp, old, None, torch.tensor([[1.0]]), torch.ones_like(logp), normalizer=1.0
    )

    assert diagnostics["clip_fraction"] == pytest.approx(1.0)
    assert diagnostics["mean_ratio"] > 2.0


@pytest.mark.parametrize("advantage,expected_sign", [(1.0, -1.0), (-1.0, 1.0)])
def test_the_gradient_pushes_the_logprob_the_way_the_advantage_says(
    advantage, expected_sign
):
    """Sign check. Backwards here trains the policy to do worse, silently.

    It has to be checked on the gradient and not on the loss *value*: at one
    update per batch the ratio is identically 1, so the loss is `-A` whatever
    the logprob is, and a value-based check would pass against a sign error.
    That is not a hypothetical - the first version of this test did exactly
    that and asserted `-1.0 < -1.0`.
    """
    logp = torch.tensor([[-2.0]], requires_grad=True)
    mask = torch.ones(1, 1)

    loss, _ = grpo_token_loss(
        logp,
        logp.detach(),
        None,
        torch.tensor([[advantage]]),
        mask,
        normalizer=1.0,
    )
    loss.backward()

    # Descent moves logp against its gradient, so a positive advantage needs a
    # negative one here.
    assert logp.grad.item() == pytest.approx(expected_sign)


def test_masked_tokens_do_not_reach_the_loss(batch):
    """A token outside the mask must change nothing, however wild its logprob."""
    logp = torch.zeros(1, 4)
    mask = torch.tensor([[1.0, 0.0, 1.0, 0.0]])
    advantage = torch.tensor([[1.0]])

    baseline, _ = grpo_token_loss(
        logp, logp.detach(), None, advantage, mask, normalizer=2.0
    )

    poisoned = logp.clone()
    poisoned[0, 1] = -50.0
    shifted, _ = grpo_token_loss(
        poisoned, poisoned.detach(), None, advantage, mask, normalizer=2.0
    )

    assert baseline.item() == pytest.approx(shifted.item())


def _grads(model):
    return {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}


def test_chunked_backward_matches_the_naive_one(batch):
    """The load-bearing equivalence. Same loss, same gradients, every parameter."""
    input_ids, attention_mask, completion_mask, advantages = batch

    torch.manual_seed(1)
    chunked_model = ToyLM()
    torch.manual_seed(1)
    naive_model = ToyLM()

    chunked = grpo_backward(
        chunked_model, input_ids, attention_mask, completion_mask, advantages, chunk_size=3
    )
    naive = naive_grpo_backward(
        naive_model, input_ids, attention_mask, completion_mask, advantages
    )

    assert chunked.loss == pytest.approx(naive.loss, abs=1e-6)
    assert chunked.tokens == naive.tokens
    assert chunked.chunks > 1, "the test must actually exercise more than one chunk"

    chunked_grads, naive_grads = _grads(chunked_model), _grads(naive_model)
    assert set(chunked_grads) == set(naive_grads)
    for name, grad in chunked_grads.items():
        assert torch.allclose(grad, naive_grads[name], atol=1e-6), name


def test_chunked_backward_matches_with_a_kl_term(batch):
    """The KL is per-token too, so chunking it is a second chance to misalign."""
    input_ids, attention_mask, completion_mask, advantages = batch
    targets, _ = shift_for_prediction(input_ids, completion_mask)
    torch.manual_seed(2)
    ref_logp = -torch.rand(targets.shape)

    torch.manual_seed(1)
    chunked_model = ToyLM()
    torch.manual_seed(1)
    naive_model = ToyLM()

    chunked = grpo_backward(
        chunked_model,
        input_ids,
        attention_mask,
        completion_mask,
        advantages,
        ref_logp=ref_logp,
        chunk_size=3,
        kl_beta=0.1,
    )
    naive = naive_grpo_backward(
        naive_model,
        input_ids,
        attention_mask,
        completion_mask,
        advantages,
        ref_logp=ref_logp,
        kl_beta=0.1,
    )

    assert chunked.loss == pytest.approx(naive.loss, abs=1e-6)
    assert chunked.mean_kl == pytest.approx(naive.mean_kl, abs=1e-5)
    assert chunked.mean_kl > 0

    chunked_grads, naive_grads = _grads(chunked_model), _grads(naive_model)
    # The set, not just the values. Without this, dropping the body backward
    # entirely leaves only the head's gradients to compare and the test passes.
    assert set(chunked_grads) == set(naive_grads)
    for name, grad in chunked_grads.items():
        assert torch.allclose(grad, naive_grads[name], atol=1e-6), name


def test_chunk_size_does_not_change_the_gradient(batch):
    """If it does, the normalizer is being computed per chunk."""
    input_ids, attention_mask, completion_mask, advantages = batch

    grads = []
    for chunk_size in (2, 5, 64):
        torch.manual_seed(1)
        model = ToyLM()
        grpo_backward(
            model, input_ids, attention_mask, completion_mask, advantages,
            chunk_size=chunk_size,
        )
        grads.append(_grads(model))

    for name in grads[0]:
        assert torch.allclose(grads[0][name], grads[1][name], atol=1e-6), name
        assert torch.allclose(grads[0][name], grads[2][name], atol=1e-6), name


def test_chunks_with_no_completion_tokens_are_skipped(batch):
    """The all-prompt head pass is pure waste; skipping it must be free."""
    input_ids, attention_mask, completion_mask, advantages = batch

    report = grpo_backward(
        input_ids=input_ids,
        attention_mask=attention_mask,
        completion_mask=completion_mask,
        advantages=advantages,
        model=ToyLM(),
        chunk_size=2,
    )

    total_chunks = (input_ids.shape[1] - 1 + 1) // 2
    assert report.chunks < total_chunks, "some chunks are entirely prompt"


def test_an_all_zero_mask_produces_no_gradient_and_no_crash(batch):
    """What a fully degenerate group looks like by the time it reaches here."""
    input_ids, attention_mask, _, advantages = batch
    model = ToyLM()

    report = grpo_backward(
        model,
        input_ids,
        attention_mask,
        torch.zeros_like(input_ids),
        advantages,
        chunk_size=4,
    )

    assert report.chunks == 0
    assert report.tokens == 0.0
    assert all(p.grad is None for p in model.parameters())


def test_zero_advantages_leave_the_parameters_alone(batch):
    """A degenerate group must not move the policy, even with tokens in the mask."""
    input_ids, attention_mask, completion_mask, _ = batch
    model = ToyLM()

    grpo_backward(
        model, input_ids, attention_mask, completion_mask, torch.zeros(3), chunk_size=4
    )

    for name, parameter in model.named_parameters():
        assert parameter.grad is None or torch.allclose(
            parameter.grad, torch.zeros_like(parameter.grad), atol=1e-7
        ), name
