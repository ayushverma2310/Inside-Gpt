from __future__ import annotations

import os
from pathlib import Path

import torch

from .config import GPTConfig, TrainConfig, resolve_device
from .model import GPT
from .tokenizer import CharTokenizer

TOKENIZER_FILE = "tokenizer.json"


def save_checkpoint(
    path: str | Path,
    model: GPT,
    optimizer: torch.optim.Optimizer | None,
    step: int,
    train_cfg: TrainConfig | None = None,
    metrics: dict | None = None,
) -> None:
    """Atomically write a checkpoint (weights, configs, step, metrics, optional optimizer)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "model_config": model.cfg.to_dict(),
        "train_config": train_cfg.to_dict() if train_cfg is not None else None,
        "step": step,
        "metrics": metrics or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> dict:
    return torch.load(path, map_location=device, weights_only=True)


def load_model(
    path: str | Path, device: str = "auto", which: str = "best"
) -> tuple[GPT, CharTokenizer, dict]:
    """Load (model, tokenizer, metadata) from a run directory or a checkpoint file.

    For a directory, ``which`` picks best.pt or last.pt (falling back to whichever
    exists). The tokenizer is read from tokenizer.json next to the checkpoint.
    The model is returned in eval mode.
    """
    device = resolve_device(device)
    path = Path(path)
    if path.is_dir():
        run_dir = path
        ckpt_path = run_dir / f"{which}.pt"
        if not ckpt_path.exists():
            other = run_dir / ("last.pt" if which == "best" else "best.pt")
            if not other.exists():
                raise FileNotFoundError(f"no checkpoint found in {run_dir}")
            ckpt_path = other
    else:
        run_dir, ckpt_path = path.parent, path

    tok_path = run_dir / TOKENIZER_FILE
    if not tok_path.exists():
        raise FileNotFoundError(f"missing {TOKENIZER_FILE} next to checkpoint in {run_dir}")

    ckpt = load_checkpoint(ckpt_path, device)
    model = GPT(GPTConfig.from_dict(ckpt["model_config"]))
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    tokenizer = CharTokenizer.load(tok_path)
    meta = {
        "step": ckpt["step"],
        "metrics": ckpt["metrics"],
        "train_config": ckpt["train_config"],
        "checkpoint": str(ckpt_path),
    }
    return model, tokenizer, meta
