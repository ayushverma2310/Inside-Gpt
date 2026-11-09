from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import GPTConfig
from ..hooks import HookPoint


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with every intermediate exposed as a hook.

    Mathematically identical to the original ``Head`` + ``MultiHeadAttention``
    pair, but all heads are computed in one batched matmul. Head ``h`` owns rows
    ``h*S:(h+1)*S`` of W_q / W_k / W_v and columns ``h*S:(h+1)*S`` of W_o.
    """

    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        C = cfg.n_embd
        self.n_head = cfg.n_head
        self.head_size = cfg.head_size
        self.W_q = nn.Linear(C, C, bias=False)
        self.W_k = nn.Linear(C, C, bias=False)
        self.W_v = nn.Linear(C, C, bias=False)
        self.W_o = nn.Linear(C, C)  # bias=True, like the original proj layer
        self.attn_dropout = nn.Dropout(cfg.dropout)
        self.resid_dropout = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

        self.hook_q = HookPoint()  # (B, H, T, S)
        self.hook_k = HookPoint()  # (B, H, T, S)
        self.hook_v = HookPoint()  # (B, H, T, S)
        self.hook_scores = HookPoint()  # (B, H, T, T) masked, pre-softmax
        self.hook_pattern = HookPoint()  # (B, H, T, T) post-softmax, pre-dropout
        self.hook_z = HookPoint()  # (B, T, H, S) per-head output before W_o

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, S = self.n_head, self.head_size

        q = self.hook_q(self.W_q(x).view(B, T, H, S).transpose(1, 2))
        k = self.hook_k(self.W_k(x).view(B, T, H, S).transpose(1, 2))
        v = self.hook_v(self.W_v(x).view(B, T, H, S).transpose(1, 2))

        scores = q @ k.transpose(-2, -1) * S**-0.5  # (B, H, T, T)
        scores = scores.masked_fill(~self.mask[:T, :T], float("-inf"))
        scores = self.hook_scores(scores)
        pattern = self.hook_pattern(F.softmax(scores, dim=-1))

        z = self.attn_dropout(pattern) @ v  # (B, H, T, S)
        z = self.hook_z(z.transpose(1, 2))  # (B, T, H, S)
        return self.resid_dropout(self.W_o(z.reshape(B, T, C)))
