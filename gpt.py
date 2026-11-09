from __future__ import annotations

import fnmatch
from contextlib import contextmanager, nullcontext
from typing import Callable, Iterable, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import GPTConfig
from ..hooks import ActivationCache, HookFn, HookPoint
from .block import Block

NamesFilter = Union[None, str, Sequence[str], Callable[[str], bool]]


class GPT(nn.Module):
    """Decoder-only transformer language model.

    Differences from the original script, all deliberate:
      * ``forward`` returns logits of shape (B, T, V) even when targets are given
        (the original flattened them to (B*T, V)); loss is computed separately.
      * ``generate`` gains temperature / top_k / a seedable generator.
      * Every intermediate tensor passes through a named ``HookPoint``.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size)

        self.hook_tok_emb = HookPoint()  # (B, T, C)
        self.hook_pos_emb = HookPoint()  # (T, C)
        self.hook_resid_embed = HookPoint()  # (B, T, C) tok_emb + pos_emb
        self.hook_ln_final = HookPoint()  # (B, T, C)
        self.hook_logits = HookPoint()  # (B, T, V)

        self.apply(self._init_weights)
        self.setup_hooks()

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------ forward

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")

        tok = self.hook_tok_emb(self.tok_emb(idx))  # (B, T, C)
        pos = self.hook_pos_emb(self.pos_emb(torch.arange(T, device=idx.device)))  # (T, C)
        x = self.hook_resid_embed(tok + pos)
        for block in self.blocks:
            x = block(x)
        x = self.hook_ln_final(self.ln_f(x))
        logits = self.hook_logits(self.lm_head(x))  # (B, T, V)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.cfg.block_size :]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if temperature == 0:
                idx_next = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                    logits = logits.masked_fill(logits < kth, float("-inf"))
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1, generator=generator)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    # ------------------------------------------------------------ instrumentation

    def setup_hooks(self) -> None:
        """Name every HookPoint by its module path, e.g. 'blocks.2.attn.hook_pattern'."""
        self._hook_dict: dict[str, HookPoint] = {}
        for name, module in self.named_modules():
            if isinstance(module, HookPoint):
                module.name = name
                self._hook_dict[name] = module

    def hook_points(self) -> dict[str, HookPoint]:
        return dict(self._hook_dict)

    def hook_names(self) -> list[str]:
        return list(self._hook_dict)

    def _resolve_names(self, sel: NamesFilter) -> list[str]:
        """None -> all; str / list of str -> glob patterns (fnmatch); callable -> predicate."""
        all_names = self.hook_names()
        if sel is None:
            return all_names
        if callable(sel):
            return [n for n in all_names if sel(n)]
        patterns = [sel] if isinstance(sel, str) else list(sel)
        out: list[str] = []
        for pat in patterns:
            matched = fnmatch.filter(all_names, pat)
            if not matched:
                raise ValueError(f"no hook point matches {pat!r}")
            out.extend(n for n in matched if n not in out)
        return out

    @contextmanager
    def hooks(self, fwd_hooks: Iterable[tuple[NamesFilter, HookFn]]):
        """Temporarily attach hook functions; always detached on exit, even on error.

        ``fwd_hooks`` is a list of (selector, fn). The selector is a hook name,
        a glob like "blocks.*.attn.hook_z", a list of those, or a predicate.
        """
        added: list[tuple[HookPoint, HookFn]] = []
        try:
            for sel, fn in fwd_hooks:
                for name in self._resolve_names(sel):
                    hp = self._hook_dict[name]
                    hp.add_hook(fn)
                    added.append((hp, fn))
            yield self
        finally:
            for hp, fn in added:
                hp.remove_hook(fn)

    def run_with_hooks(
        self,
        idx: torch.Tensor,
        fwd_hooks: Iterable[tuple[NamesFilter, HookFn]],
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        with self.hooks(fwd_hooks):
            return self(idx, targets)

    def run_with_cache(
        self,
        idx: torch.Tensor,
        names_filter: NamesFilter = None,
        fwd_hooks: Iterable[tuple[NamesFilter, HookFn]] | None = None,
        device: str | torch.device | None = None,
        grad: bool = False,
    ) -> tuple[torch.Tensor, ActivationCache]:
        """Forward pass that also returns an ActivationCache.

        ``fwd_hooks`` (e.g. interventions) are attached before the caching hook, so the
        cache records the *modified* activations. Pass ``device="cpu"`` to move cached
        tensors off the GPU. Gradients are off unless ``grad=True``.
        """
        store: dict[str, torch.Tensor] = {}

        def save(x: torch.Tensor, hp: HookPoint) -> None:
            t = x if grad else x.detach()
            store[hp.name] = t.to(device) if device is not None else t

        hook_list = list(fwd_hooks or []) + [(names_filter, save)]
        with (nullcontext() if grad else torch.no_grad()), self.hooks(hook_list):
            logits, _ = self(idx)
        return logits, ActivationCache(store)
