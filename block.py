from __future__ import annotations

import torch
import torch.nn as nn

from ..config import GPTConfig
from ..hooks import HookPoint
from .attention import CausalSelfAttention
from .mlp import MLP


class Block(nn.Module):
    """Pre-LN transformer block: communication (attention) then computation (MLP)."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

        self.hook_resid_pre = HookPoint()  # residual stream entering the block
        self.hook_ln1 = HookPoint()  # normalized input to attention
        self.hook_attn_out = HookPoint()  # what attention writes to the stream
        self.hook_resid_mid = HookPoint()  # after attention, before MLP
        self.hook_ln2 = HookPoint()  # normalized input to MLP
        self.hook_mlp_out = HookPoint()  # what the MLP writes to the stream
        self.hook_resid_post = HookPoint()  # residual stream leaving the block

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hook_resid_pre(x)
        a = self.attn(self.hook_ln1(self.ln1(x)))
        a = self.hook_attn_out(a)
        x = self.hook_resid_mid(x + a)
        m = self.mlp(self.hook_ln2(self.ln2(x)))
        m = self.hook_mlp_out(m)
        return self.hook_resid_post(x + m)
