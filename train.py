"""Train a small GPT from scratch on a text file (character level).

    python -m insidegpt.train --out_dir runs/baseline
    python -m insidegpt.train --out_dir runs/smoke --n_layer 2 --n_embd 64 --n_head 4 \
        --block_size 64 --batch_size 32 --max_iters 300 --eval_interval 100 --eval_iters 20

Writes to out_dir: tokenizer.json, config.json, best.pt, last.pt, metrics.jsonl,
baseline.json, sample.txt.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import torch

from .checkpoint import save_checkpoint
from .config import GPTConfig, TrainConfig, resolve_device
from .data import get_batch, load_text, make_splits
from .evaluate import baseline_eval
from .model import GPT
from .tokenizer import CharTokenizer


@torch.no_grad()
def estimate_loss(
    model: GPT,
    splits: dict[str, torch.Tensor],
    cfg: TrainConfig,
    device: str,
) -> dict[str, float]:
    """Mean loss over ``eval_iters`` random batches per split (as in the original)."""
    model.eval()
    out = {}
    for split, data in splits.items():
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            x, y = get_batch(data, model.cfg.block_size, cfg.batch_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def run_training(train_cfg: TrainConfig, model_kwargs: dict | None = None) -> Path:
    device = resolve_device(train_cfg.device)
    out_dir = Path(train_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(train_cfg.seed)

    text = load_text(train_cfg.data_path)
    tokenizer = CharTokenizer.from_text(text)
    tokenizer.save(out_dir / "tokenizer.json")
    splits = make_splits(text, tokenizer, train_cfg.val_fraction)

    model_cfg = GPTConfig(vocab_size=tokenizer.vocab_size, **(model_kwargs or {}))
    (out_dir / "config.json").write_text(
        json.dumps({"model": model_cfg.to_dict(), "train": train_cfg.to_dict()}, indent=2)
    )

    model = GPT(model_cfg).to(device)
    print(f"device={device}  vocab={tokenizer.vocab_size}  params={model.n_params() / 1e6:.2f}M")
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)

    best_val = float("inf")
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.write_text("")
    t0 = time.time()

    for it in range(train_cfg.max_iters):
        if it % train_cfg.eval_interval == 0 or it == train_cfg.max_iters - 1:
            losses = estimate_loss(model, splits, train_cfg, device)
            rec = {"step": it, "train_loss": losses["train"], "val_loss": losses["val"],
                   "elapsed_s": round(time.time() - t0, 1)}
            with metrics_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"step {it}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(out_dir / "best.pt", model, None, it, train_cfg, rec)

        x, y = get_batch(splits["train"], model_cfg.block_size, train_cfg.batch_size, device)
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

    final = {"step": train_cfg.max_iters, "best_val_loss": best_val}
    save_checkpoint(out_dir / "last.pt", model, optimizer, train_cfg.max_iters, train_cfg, final)

    # Baseline evaluation of the final weights, plus a longer sample.
    result = baseline_eval(model, tokenizer, splits, out_dir / "baseline.json", device=device)
    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample =tokenizer.decode(model.generate(context, train_cfg.sample_tokens)[0].tolist())
    (out_dir / "sample.txt").write_text(sample, encoding="utf-8")
    for name, m in result["splits"].items():
        print(f"final {name}: loss {m['loss']:.4f}  ppl {m['perplexity']:.2f}  "
              f"bpc {m['bits_per_char']:.3f}  acc {m['top1_accuracy']:.3f}")
    print(f"saved run to {out_dir}")
    return out_dir


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    for cls in (TrainConfig, GPTConfig):
        for f in dataclasses.fields(cls):
            if f.name == "vocab_size":  # derived from the data
                continue
            ap.add_argument(f"--{f.name}", type=type(f.default), default=f.default)
    args = vars(ap.parse_args())
    train_names = {f.name for f in dataclasses.fields(TrainConfig)}
    train_cfg = TrainConfig(**{k: v for k, v in args.items() if k in train_names})
    model_kwargs = {k: v for k, v in args.items() if k not in train_names}
    run_training(train_cfg, model_kwargs)


if __name__ == "__main__":
    main()
