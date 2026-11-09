from __future__ import annotations

import torch
import torch.nn as nn

from ..config import GPTConfig
from ..hooks import HookPoint


class MLP(nn.Module):
    """Position-wise feed-forward network: Linear -> ReLU -> Linear -> Dropout."""

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        self.fc_in = nn.Linear(cfg.n_embd, 4 * cfg.n_embd)
        self.act = nn.ReLU()
        self.fc_out = nn.Linear(4 * cfg.n_embd, cfg.n_embd)
        self.dropout = nn.Dropout(cfg.dropout)

        self.hook_pre = HookPoint()  # (B, T, 4C) pre-activation
        self.hook_post = HookPoint()  # (B, T, 4C) post-activation: the neuron activations

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hook_pre(self.fc_in(x))
        h = self.hook_post(self.act(h))
        return self.dropout(self.fc_out(h))
