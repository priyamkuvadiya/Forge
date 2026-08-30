"""Trains the from-scratch GPT on tinyshakespeare and logs a real loss curve."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from data import ShakespeareDataset
from model import GPT, GPTConfig

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"
CHECKPOINT_DIR = Path(__file__).parent / "checkpoints"


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class RunLog:
    """Line-buffered log that flushes to disk on every write.

    A hard power loss killed an earlier run and took the whole (pipe-buffered)
    log with it, so every line is fsync'd as it is written rather than being
    held in a buffer until the process exits.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = path.open("w", encoding="utf-8")

    def write(self, line: str) -> None:
        print(line, flush=True)
        self._fh.write(line + "\n")
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def close(self) -> None:
        self._fh.close()


def save_checkpoint(model: GPT, config: GPTConfig, dataset: ShakespeareDataset) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    target = CHECKPOINT_DIR / "gpt_shakespeare.pt"
    # write to a temp file and swap, so a crash mid-write cannot leave a
    # half-written checkpoint where a valid one used to be
    tmp = target.with_suffix(".pt.tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "stoi": dataset.tokenizer.stoi,
            "itos": dataset.tokenizer.itos,
        },
        tmp,
    )
    os.replace(tmp, target)


def save_history(history: dict) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(ARTIFACTS_DIR / "loss_history.json", "w") as f:
        json.dump(history, f, indent=2)


def lr_at(it: int, max_iters: int, lr: float, warmup_iters: int, min_lr: float) -> float:
    """Linear warmup, then cosine decay from `lr` down to `min_lr`.

    A flat LR plateaus well above the achievable loss floor; annealing is the
    single cheapest improvement available here since it costs no extra compute.
    """
    if it < warmup_iters:
        return lr * (it + 1) / warmup_iters
    progress = (it - warmup_iters) / max(1, max_iters - warmup_iters)
    progress = min(1.0, progress)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def estimate_loss(model: GPT, dataset: ShakespeareDataset, batch_size: int, block_size: int, device: str, use_amp: bool = False, eval_iters: int = 50) -> dict[str, float]:
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = dataset.get_batch(split, batch_size, block_size, device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def train(
    max_iters: int = 5000,
    eval_interval: int = 250,
    batch_size: int = 64,
    block_size: int = 256,
    lr: float = 1e-3,
    min_lr: float = 1e-4,
    warmup_iters: int = 100,
    grad_clip: float = 1.0,
    amp: bool = True,
) -> None:
    device = get_device()
    # bf16 measured at ~81W/66C vs ~116W/78C for fp32 on this card, and ~1.7x
    # faster; fp32 stays available for CPU runs and for comparison.
    use_amp = amp and device == "cuda"

    log = RunLog(ARTIFACTS_DIR / "train_log.txt")
    log.write(f"device: {device}, precision: {'bf16 autocast' if use_amp else 'fp32'}")

    dataset = ShakespeareDataset()
    config = GPTConfig(vocab_size=dataset.tokenizer.vocab_size, block_size=block_size)
    model = GPT(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.write(f"model params: {n_params / 1e6:.2f}M, vocab_size={config.vocab_size}")
    log.write(
        f"max_iters={max_iters} batch_size={batch_size} block_size={block_size} "
        f"lr={lr}->{min_lr} warmup={warmup_iters} grad_clip={grad_clip}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    history = {"iter": [], "train_loss": [], "val_loss": [], "lr": []}
    started = time.time()

    # this corpus is small enough that the model overfits well before the last
    # iteration, so the reported model is the best-val one, not the final one
    best_val = float("inf")
    best_iter = -1

    for it in range(max_iters):
        cur_lr = lr_at(it, max_iters, lr, warmup_iters, min_lr)
        for group in optimizer.param_groups:
            group["lr"] = cur_lr

        if it % eval_interval == 0 or it == max_iters - 1:
            losses = estimate_loss(model, dataset, batch_size, block_size, device, use_amp)
            history["iter"].append(it)
            history["train_loss"].append(losses["train"])
            history["val_loss"].append(losses["val"])
            history["lr"].append(cur_lr)

            improved = losses["val"] < best_val
            if improved:
                best_val = losses["val"]
                best_iter = it
                save_checkpoint(model, config, dataset)

            elapsed = time.time() - started
            log.write(
                f"iter {it}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}, lr {cur_lr:.2e} "
                f"[{elapsed / 60:.1f} min]{' *best' if improved else ''}"
            )
            save_history(history)

        x, y = dataset.get_batch("train", batch_size, block_size, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

    history["best_val_loss"] = best_val
    history["best_iter"] = best_iter
    save_history(history)
    log.write(f"best val loss {best_val:.4f} at iter {best_iter} (checkpoint kept)")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(history["iter"], history["train_loss"], label="train loss")
    plt.plot(history["iter"], history["val_loss"], label="val loss")
    if best_iter >= 0:
        plt.axvline(best_iter, color="gray", linestyle="--", linewidth=1)
        plt.scatter([best_iter], [best_val], color="red", zorder=5, s=25,
                    label=f"best val {best_val:.3f} @ {best_iter}")
    plt.xlabel("iteration")
    plt.ylabel("cross-entropy loss")
    plt.title("From-scratch GPT on tinyshakespeare (char-level)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ARTIFACTS_DIR / "loss_curve.png", dpi=150)
    log.write(f"done in {(time.time() - started) / 60:.1f} min")
    log.write(f"saved checkpoint and loss curve to {CHECKPOINT_DIR} / {ARTIFACTS_DIR}")
    log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iters", type=int, default=5000)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--no_amp", action="store_true", help="disable bf16 autocast")
    args = parser.parse_args()
    train(
        args.max_iters,
        args.eval_interval,
        args.batch_size,
        args.block_size,
        args.lr,
        args.min_lr,
        args.warmup_iters,
        args.grad_clip,
        amp=not args.no_amp,
    )
