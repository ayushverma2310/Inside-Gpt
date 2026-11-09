from __future__ import annotations

from dataclasses import asdict, dataclass, fields

import torch


@dataclass
class GPTConfig:
    """Architecture hyperparameters. Defaults match the original Karpathy script."""

    vocab_size: int = 65
    block_size: int = 256  # maximum context length
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(
                f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
            )

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "GPTConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


@dataclass
class TrainConfig:
    """Training-run hyperparameters. Defaults match the original Karpathy script."""

    data_path: str = "data/input.txt"
    out_dir: str = "runs/baseline"
    batch_size: int = 64
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    learning_rate: float = 3e-4
    grad_clip: float = 0.0  # 0 disables clipping (the original script has none)
    val_fraction: float = 0.1
    seed: int = 1337
    device: str = "auto"
    sample_tokens: int = 500

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "TrainConfig":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})


def resolve_device(name: str = "auto") -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
