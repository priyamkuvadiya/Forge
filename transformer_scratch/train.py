"""Trains the from-scratch GPT on tinyshakespeare and logs a real loss curve."""

import argparse
import json
from pathlib import Path

import torch

from data import ShakespeareDataset
from model import GPT, GPTConfig

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def estimate_loss(model: GPT, dataset: ShakespeareDataset, batch_size: int, block_size: int, device: str, eval_iters: int = 50) -> dict[str, float]:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = dataset.get_batch(split, batch_size, block_size, device)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(
    max_iters: int = 3000,
    eval_interval: int = 250,
    batch_size: int = 64,
    block_size: int = 256,
    lr: float = 3e-4,
) -> None:
    device = get_device()
    print(f"device: {device}")

    dataset = ShakespeareDataset()
    config = GPTConfig(vocab_size=dataset.tokenizer.vocab_size, block_size=block_size)
    model = GPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params / 1e6:.2f}M, vocab_size={config.vocab_size}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = {"iter": [], "train_loss": [], "val_loss": []}

    for it in range(max_iters):
        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model, dataset, batch_size, block_size, device)
            print(f"iter {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            history["iter"].append(it)
            history["train_loss"].append(losses["train"])
            history["val_loss"].append(losses["val"])

        x, y = dataset.get_batch("train", batch_size, block_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(), "config": config, "stoi": dataset.tokenizer.stoi, "itos": dataset.tokenizer.itos},
        CHECKPOINT_DIR / "gpt_shakespeare.pt",
    )

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "loss_history.json", "w") as f:
        json.dump(history, f, indent=2)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(history["iter"], history["train_loss"], label="train loss")
    plt.plot(history["iter"], history["val_loss"], label="val loss")
    plt.xlabel("iteration")
    plt.ylabel("cross-entropy loss")
    plt.title("From-scratch GPT on tinyshakespeare (char-level)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "loss_curve.png", dpi=150)
    print(f"saved checkpoint and loss curve to {CHECKPOINT_DIR} / {ARTIFACTS_DIR}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iters", type=int, default=3000)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    train(args.max_iters, args.eval_interval, args.batch_size, args.block_size, args.lr)
