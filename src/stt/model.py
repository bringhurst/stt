"""Tiny Transformer model instrumented for attention and hidden-state metrics."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ModelOutput:
    """Forward-pass outputs needed by task loss, STT losses, and metrics.

    Attributes:
        logits: Next-token logits shaped `(batch, seq, vocab_size)`.
        hidden: Final normalized hidden states shaped `(batch, seq, dim)`.
        attention: Attention maps shaped `(batch, layers, heads, seq, seq)`.
    """

    logits: Tensor
    hidden: Tensor
    attention: Tensor


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block with per-head attention output."""

    def __init__(self, dim: int, heads: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Apply self-attention and feed-forward updates.

        Returns:
            Updated hidden states and raw per-head attention weights.
        """
        attended, weights = self.attn(
            self.norm_attn(x),
            self.norm_attn(x),
            self.norm_attn(x),
            need_weights=True,
            average_attn_weights=False,
        )
        x = x + attended
        x = x + self.ff(self.norm_ff(x))
        return x, weights


class TinyTransformer(nn.Module):
    """Small sequence model used to test STT regularizers quickly.

    The model is intentionally modest so tests and experiments can run on CPU,
    while still using standard Transformer components: token embeddings,
    positional embeddings, multi-head attention, residuals, MLP blocks, and a
    next-token prediction head.
    """

    def __init__(
        self,
        vocab_size: int,
        seq_len: int,
        dim: int = 48,
        heads: int = 4,
        layers: int = 2,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList([TransformerBlock(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size)

    def forward(self, tokens: Tensor) -> ModelOutput:
        """Run a forward pass and collect attention from every layer."""
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        attentions = []
        for block in self.blocks:
            x, attention = block(x)
            attentions.append(attention)
        hidden = self.norm(x)
        logits = self.output(hidden)
        return ModelOutput(logits=logits, hidden=hidden, attention=torch.stack(attentions, dim=1))
