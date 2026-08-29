"""Overfit-one-batch sanity check: if the model can't drive loss to ~0 on a
single small batch, something in the forward/backward path is broken.
"""

import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from model import GPT, GPTConfig


def test_overfits_single_batch():
    torch.manual_seed(0)

    config = GPTConfig(vocab_size=20, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0)
    model = GPT(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    x = torch.randint(0, config.vocab_size, (4, config.block_size))
    y = torch.randint(0, config.vocab_size, (4, config.block_size))

    initial_loss = None
    final_loss = None
    for step in range(300):
        _, loss = model(x, y)
        if step == 0:
            initial_loss = loss.item()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = loss.item()

    random_baseline = math.log(config.vocab_size)
    assert initial_loss > 1.0
    assert final_loss < 0.1
    assert final_loss < random_baseline
