"""Shape checks plus an actual causal-masking correctness check."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import GPT, GPTConfig, CausalSelfAttention, MLP, Block


def make_config(**overrides) -> GPTConfig:
    defaults = dict(vocab_size=65, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)
    defaults.update(overrides)
    return GPTConfig(**defaults)


def test_attention_output_shape():
    config = make_config()
    attn = CausalSelfAttention(config)
    x = torch.randn(2, config.block_size, config.n_embd)
    out = attn(x)
    assert out.shape == (2, config.block_size, config.n_embd)


def test_mlp_output_shape():
    config = make_config()
    mlp = MLP(config)
    x = torch.randn(2, config.block_size, config.n_embd)
    out = mlp(x)
    assert out.shape == x.shape


def test_block_output_shape():
    config = make_config()
    block = Block(config)
    x = torch.randn(2, config.block_size, config.n_embd)
    out = block(x)
    assert out.shape == x.shape


def test_gpt_forward_shape_and_loss():
    config = make_config()
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (2, config.block_size))
    targets = torch.randint(0, config.vocab_size, (2, config.block_size))

    logits, loss = model(idx, targets)
    assert logits.shape == (2, config.block_size, config.vocab_size)
    assert loss.item() > 0


def test_causal_mask_blocks_future_tokens():
    """A position's output must not change when a *future* token is perturbed."""
    config = make_config()
    attn = CausalSelfAttention(config)
    attn.eval()

    torch.manual_seed(0)
    x = torch.randn(1, config.block_size, config.n_embd)

    with torch.no_grad():
        out_before = attn(x)

        x_perturbed = x.clone()
        future_pos = config.block_size - 1
        x_perturbed[:, future_pos, :] += 100.0
        out_after = attn(x_perturbed)

    # every position except the perturbed one (and anything after it) must be unchanged
    assert torch.allclose(out_before[:, :future_pos, :], out_after[:, :future_pos, :], atol=1e-5)
    # sanity: the perturbed position's own output *should* have changed
    assert not torch.allclose(out_before[:, future_pos, :], out_after[:, future_pos, :], atol=1e-5)
