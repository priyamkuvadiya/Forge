"""Loads a trained from-scratch GPT checkpoint and samples text from it."""

import argparse
from pathlib import Path

import torch

from data import CharTokenizer
from model import GPT

CHECKPOINT_PATH = Path(__file__).parent / "checkpoints" / "gpt_shakespeare.pt"
ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def load_model(device: str) -> tuple[GPT, CharTokenizer]:
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    config = checkpoint["config"]

    model = GPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokenizer = CharTokenizer.__new__(CharTokenizer)
    tokenizer.stoi = checkpoint["stoi"]
    tokenizer.itos = checkpoint["itos"]
    tokenizer.vocab_size = len(tokenizer.stoi)

    return model, tokenizer


def generate(prompt: str, max_new_tokens: int, temperature: float, save: bool) -> str:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = load_model(device)

    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=temperature)
    text = tokenizer.decode(out[0].tolist())

    print(text)
    if save:
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        (ARTIFACTS_DIR / "sample_output.txt").write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="\n")
    parser.add_argument("--max_new_tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    generate(args.prompt, args.max_new_tokens, args.temperature, args.save)
