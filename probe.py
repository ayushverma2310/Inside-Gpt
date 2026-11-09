"""High-level entry point for Phase 2: run a prompt and get every internal back."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .hooks import ActivationCache
from .model import GPT
from .tokenizer import CharTokenizer


@dataclass
class InspectionResult:
    prompt: str
    tokens: list[str]  # printable token strings, one per position
    token_ids: list[int]
    offset: int  # characters dropped from the front if the prompt exceeded block_size
    logits: torch.Tensor  # (T, V) for the single prompt
    cache: ActivationCache  # tensors keep their batch dim of 1
    top_predictions: list[tuple[str, float]]  # next-token candidates after the last position

    def summary(self) -> str:
        preds = ", ".join(f"{t!r}: {p:.3f}" for t, p in self.top_predictions)
        return (
            f"{len(self.tokens)} tokens, {len(self.cache)} cached tensors\n"
            f"next-token top-{len(self.top_predictions)}: {preds}"
        )


def inspect_prompt(
    model: GPT,
    tokenizer: CharTokenizer,
    prompt: str,
    names_filter=None,
    top_k: int = 5,
    device: str | None = None,
) -> InspectionResult:
    """Run ``prompt`` through the model and capture its internals.

    Prompts longer than block_size are cropped to the last block_size tokens
    (``offset`` records how many were dropped). ``names_filter`` accepts the same
    selectors as ``GPT.run_with_cache``.
    """
    if not prompt:
        raise ValueError("prompt is empty")
    device = device or next(model.parameters()).device
    ids = tokenizer.encode(prompt)  # raises a clear error on unknown characters
    offset = max(0, len(ids) - model.cfg.block_size)
    ids = ids[offset:]

    was_training = model.training
    model.eval()
    x = torch.tensor([ids], dtype=torch.long, device=device)
    logits, cache = model.run_with_cache(x, names_filter=names_filter)
    model.train(was_training)

    last = torch.softmax(logits[0, -1], dim=-1)
    p, i = torch.topk(last, min(top_k, last.numel()))
    return InspectionResult(
        prompt=prompt,
        tokens=[tokenizer.token_str(t) for t in ids],
        token_ids=ids,
        offset=offset,
        logits=logits[0],
        cache=cache,
        top_predictions=[(tokenizer.token_str(a), b) for a, b in zip(i.tolist(), p.tolist())],
    )
