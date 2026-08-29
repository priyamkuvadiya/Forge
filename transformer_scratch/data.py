"""Character-level tokenizer and data loading for the from-scratch GPT.

Deliberately no external tokenizer library — a char-level vocab built
directly from the corpus is simple, fully self-authored, and enough to
prove the model learns.
"""

import urllib.request
from pathlib import Path

import torch

DATA_DIR = Path(__file__).parent / "data"
DATA_FILE = DATA_DIR / "tinyshakespeare.txt"
DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def download_dataset() -> Path:
    if not DATA_FILE.exists():
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(DATA_URL, DATA_FILE)
    return DATA_FILE


class CharTokenizer:
    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        return [self.stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[i] for i in ids)


class ShakespeareDataset:
    """Loads tinyshakespeare, builds a char vocab, and serves random batches."""

    def __init__(self, val_fraction: float = 0.1):
        path = download_dataset()
        text = path.read_text(encoding="utf-8")

        self.tokenizer = CharTokenizer(text)
        data = torch.tensor(self.tokenizer.encode(text), dtype=torch.long)

        split = int(len(data) * (1 - val_fraction))
        self.train_data = data[:split]
        self.val_data = data[split:]

    def get_batch(self, split: str, batch_size: int, block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        data = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(data) - block_size - 1, (batch_size,))
        x = torch.stack([data[i:i + block_size] for i in ix])
        y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
        return x.to(device), y.to(device)
