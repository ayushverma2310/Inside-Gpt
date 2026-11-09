"""Instrumentation primitives: HookPoint and ActivationCache.

Every interesting tensor in the model passes through a ``HookPoint``. A HookPoint
is an identity module until a function is attached to it. A hook function has the
signature ``fn(tensor, hook_point) -> Tensor | None``; returning a tensor replaces
the activation (this is how Phase 4 interventions will work), returning None
leaves it alone (this is how caching works).
"""

from __future__ import annotations

from typing import Callable, Iterable

import torch
import torch.nn as nn

HookFn = Callable[[torch.Tensor, "HookPoint"], "torch.Tensor | None"]


class HookPoint(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.name: str | None = None  # filled in by GPT.setup_hooks()
        self._fwd_hooks: list[HookFn] = []

    def add_hook(self, fn: HookFn) -> HookFn:
        self._fwd_hooks.append(fn)
        return fn

    def remove_hook(self, fn: HookFn) -> None:
        try:
            self._fwd_hooks.remove(fn)
        except ValueError:
            pass

    def clear_hooks(self) -> None:
        self._fwd_hooks.clear()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fn in list(self._fwd_hooks):
            out = fn(x, self)
            if out is not None:
                x = out
        return x


# Short "kind" names -> hook-name templates. ``{l}`` is the layer index.
# Shapes use B=batch, T=time, H=heads, S=head size, C=n_embd, V=vocab.
KINDS: dict[str, str] = {
    # per layer, inside attention
    "q": "blocks.{l}.attn.hook_q",  # (B,H,T,S)
    "k": "blocks.{l}.attn.hook_k",  # (B,H,T,S)
    "v": "blocks.{l}.attn.hook_v",  # (B,H,T,S)
    "scores": "blocks.{l}.attn.hook_scores",  # (B,H,T,T) causal-masked, pre-softmax
    "pattern": "blocks.{l}.attn.hook_pattern",  # (B,H,T,T) post-softmax
    "z": "blocks.{l}.attn.hook_z",  # (B,T,H,S) per-head output, before W_O
    # per layer, inside MLP
    "mlp_pre": "blocks.{l}.mlp.hook_pre",  # (B,T,4C) before ReLU
    "mlp_post": "blocks.{l}.mlp.hook_post",  # (B,T,4C) after ReLU (the "neurons")
    # per layer, residual stream and sublayer outputs
    "ln1": "blocks.{l}.hook_ln1",  # (B,T,C)
    "ln2": "blocks.{l}.hook_ln2",  # (B,T,C)
    "attn_out": "blocks.{l}.hook_attn_out",  # (B,T,C)
    "mlp_out": "blocks.{l}.hook_mlp_out",  # (B,T,C)
    "resid_pre": "blocks.{l}.hook_resid_pre",  # (B,T,C)
    "resid_mid": "blocks.{l}.hook_resid_mid",  # (B,T,C)
    "resid_post": "blocks.{l}.hook_resid_post",  # (B,T,C)
}

# Kinds that exist once per model (no layer index).
GLOBAL_KINDS: dict[str, str] = {
    "tok_emb": "hook_tok_emb",  # (B,T,C)
    "pos_emb": "hook_pos_emb",  # (T,C)
    "resid_embed": "hook_resid_embed",  # (B,T,C) tok_emb + pos_emb
    "ln_final": "hook_ln_final",  # (B,T,C)
    "logits": "hook_logits",  # (B,T,V)
}


def resolve_hook_name(kind: str, layer: int | None = None) -> str:
    """Turn ("pattern", 2) into "blocks.2.attn.hook_pattern"; full names pass through."""
    if kind in GLOBAL_KINDS:
        return GLOBAL_KINDS[kind]
    if kind in KINDS:
        if layer is None:
            raise KeyError(f"{kind!r} is per-layer; pass a layer index, e.g. cache[{kind!r}, 0]")
        return KINDS[kind].format(l=layer)
    return kind  # assume a full hook name


class ActivationCache(dict):
    """A dict of hook name -> detached tensor, with convenience lookups.

    cache["blocks.2.attn.hook_pattern"]   full name
    cache["pattern", 2]                   (kind, layer)
    cache["logits"]                       global kinds need no layer
    """

    def __getitem__(self, key):
        if isinstance(key, tuple):
            kind, layer = key
            name = resolve_hook_name(kind, layer)
        else:
            name = resolve_hook_name(key)
        try:
            return super().__getitem__(name)
        except KeyError:
            raise KeyError(
                f"{name!r} not in cache (cached: {len(self)} tensors; "
                "did names_filter exclude it?)"
            ) from None

    @property
    def n_layer(self) -> int:
        layers = {int(k.split(".")[1]) for k in self if k.startswith("blocks.")}
        return max(layers) + 1 if layers else 0

    def stack(self, kind: str, layers: Iterable[int] | None = None) -> torch.Tensor:
        """Stack a per-layer kind along a new leading layer dimension."""
        layers = list(range(self.n_layer)) if layers is None else list(layers)
        return torch.stack([self[kind, l] for l in layers])

    def resid_stack(self) -> torch.Tensor:
        """(n_layer + 1, B, T, C): embedding stream followed by every layer's resid_post."""
        parts = [self["resid_embed"]] + [self["resid_post", l] for l in range(self.n_layer)]
        return torch.stack(parts)

    def logits_at(self, positions) -> torch.Tensor:
        """Logits restricted to the given time positions: (B, len(positions), V)."""
        return self["logits"][:, positions]

    def to(self, device) -> "ActivationCache":
        return ActivationCache({k: v.to(device) for k, v in self.items()})

    def summary(self) -> str:
        lines = [f"{name:<36} {tuple(t.shape)}" for name, t in self.items()]
        return "\n".join(lines)
