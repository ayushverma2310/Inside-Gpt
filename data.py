from __future__ import annotations

from pathlib import Path

import torch

from .tokenizer import CharTokenizer


def load_text(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8")


def make_splits(
    text: str, tokenizer: CharTokenizer, val_fraction: float = 0.1
) -> dict[str, torch.Tensor]:
    """First (1 - val_fraction) of the corpus is train, the rest is validation."""
    ids = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    n = int((1.0 - val_fraction) * len(ids))
    return {"train": ids[:n], "val": ids[n:]}


def get_batch(
    data: torch.Tensor,
    block_size: int,
    batch_size: int,
    device: str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random (x, y) windows; y is x shifted left by one token."""
    if len(data) <= block_size + 1:
        raise ValueError(
            f"split has {len(data)} tokens, need more than block_size + 1 = {block_size + 1}"
        )
    ix = torch.randint(len(data) - block_size, (batch_size,), generator=generator)
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
