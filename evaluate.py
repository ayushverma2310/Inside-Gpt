"""Baseline evaluation: exact loss / perplexity / bits-per-char / accuracy plus samples.

Run on a finished run directory:
    python -m insidegpt.evaluate --run runs/baseline --data data/input.txt
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .checkpoint import load_model
from .config import resolve_device
from .data import load_text, make_splits
from .model import GPT
from .tokenizer import CharTokenizer

DEFAULT_PROBES = ["\n", "ROMEO:", "To be, or not to be", "First Citizen:\n"]


@torch.no_grad()
def evaluate_full(
    model: GPT, data: torch.Tensor, batch_size: int = 64, device: str = "cpu"
) -> dict:
    """Deterministic evaluation over a whole split using non-overlapping windows.

    Every token (except a final partial window) is scored exactly once, so the number
    is reproducible. Early positions in each window see little context, so this loss
    is slightly higher than the random-window estimate used during training.
    """
    was_training = model.training
    model.eval()
    T = model.cfg.block_size
    n_windows = (len(data) - 1) // T
    if n_windows < 1:
        raise ValueError("split is shorter than one block")
    total_loss, correct, n_tokens = 0.0, 0, 0
    for start in range(0, n_windows, batch_size):
        ids = range(start, min(start + batch_size, n_windows))
        x = torch.stack([data[i * T : i * T + T] for i in ids]).to(device)
        y = torch.stack([data[i * T + 1 : i * T + T + 1] for i in ids]).to(device)
        logits, _ = model(x)
        loss_sum = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum"
        )
        total_loss += loss_sum.item()
        correct += (logits.argmax(-1) == y).sum().item()
        n_tokens += y.numel()
    model.train(was_training)
    loss = total_loss / n_tokens
    return {
        "loss": loss,
        "perplexity": math.exp(loss),
        "bits_per_char": loss / math.log(2),
        "top1_accuracy": correct / n_tokens,
        "n_tokens": n_tokens,
    }


@torch.no_grad()
def sample_text(
    model: GPT,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int | None = 40,
    seed: int = 0,
    device: str = "cpu",
) -> str:
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    g = torch.Generator(device="cpu").manual_seed(seed)
    was_training = model.training
    model.eval()  # sample without dropout
    out = model.generate(ids, max_new_tokens, temperature=temperature, top_k=top_k, generator=g)
    model.train(was_training)
    return tokenizer.decode(out[0].tolist())


def baseline_eval(
    model: GPT,
    tokenizer: CharTokenizer,
    splits: dict[str, torch.Tensor],
    out_path: str | Path | None = None,
    device: str = "cpu",
    probes: list[str] | None = None,
    sample_tokens: int = 200,
) -> dict:
    result: dict = {"n_params": model.n_params(), "splits": {}, "samples": {}}
    for name, data in splits.items():
        result["splits"][name] = evaluate_full(model, data, device=device)
    for p in probes or DEFAULT_PROBES:
        try:
            result["samples"][p] = sample_text(
                model, tokenizer, p, max_new_tokens=sample_tokens, device=device
            )
        except ValueError:  # probe uses a character outside the vocabulary
            continue
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", required=True, help="run directory (or checkpoint file)")
    ap.add_argument("--data", default="data/input.txt")
    ap.add_argument("--which", default="best", choices=["best", "last"])
    ap.add_argument("--device", default="auto")
    ap.add_argument("--val_fraction", type=float, default=0.1)
    args = ap.parse_args()

    device = resolve_device(args.device)
    model, tok, meta = load_model(args.run, device=device, which=args.which)
    splits = make_splits(load_text(args.data), tok, args.val_fraction)
    run_dir = Path(args.run) if Path(args.run).is_dir() else Path(args.run).parent
    res = baseline_eval(model, tok, splits, run_dir / "baseline.json", device=device)
    for name, m in res["splits"].items():
        print(
            f"{name}: loss {m['loss']:.4f}  ppl {m['perplexity']:.2f}  "
            f"bpc {m['bits_per_char']:.3f}  acc {m['top1_accuracy']:.3f}"
        )
    print(f"wrote {run_dir / 'baseline.json'}")


if __name__ == "__main__":
    main()
