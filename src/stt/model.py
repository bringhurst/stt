"""Tiny Transformer model instrumented for attention and hidden-state metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

CompartmentMode = Literal["router", "dendritic"]


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
    branch_gates: Tensor | None = None
    branch_outputs: Tensor | None = None
    branch_scores: Tensor | None = None
    branch_inhibition: Tensor | None = None


@dataclass(frozen=True)
class CompartmentOutput:
    """Output and routing internals from one compartmentalized FFN."""

    hidden: Tensor
    gates: Tensor
    branch_outputs: Tensor
    branch_scores: Tensor
    branch_inhibition: Tensor


def sparse_top_k_gates(scores: Tensor, top_k: int) -> Tensor:
    """Return sparse top-k gates with straight-through dense score gradients."""
    branches = scores.shape[-1]
    soft_gates = torch.softmax(scores, dim=-1)
    if top_k == branches:
        return soft_gates
    top_values, top_indices = torch.topk(scores, top_k, dim=-1)
    top_weights = torch.softmax(top_values, dim=-1)
    sparse_gates = torch.zeros_like(scores)
    sparse_gates.scatter_(-1, top_indices, top_weights)
    return sparse_gates.detach() - soft_gates.detach() + soft_gates


class CompartmentBranch(nn.Module):
    """Tiny dendritic branch used inside a compartmentalized FFN."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.up = nn.Linear(dim, hidden)
        self.activation = nn.SiLU()
        self.down = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> Tensor:
        """Apply local nonlinear branch processing."""
        return self.down(self.activation(self.up(x)))


class DendriticBranch(nn.Module):
    """Branch that computes both local output and local participation score."""

    def __init__(self, dim: int, hidden: int) -> None:
        super().__init__()
        self.up = nn.Linear(dim, hidden)
        self.activation = nn.SiLU()
        self.spike = nn.Linear(hidden, 1)
        self.down = nn.Linear(hidden, dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Apply local branch processing and emit a branch-local spike score."""
        local = self.activation(self.up(x))
        return self.down(local), self.spike(local).squeeze(-1)


class CompartmentalizedFeedForward(nn.Module):
    """A lightweight routed ensemble inside one Transformer FFN slot."""

    def __init__(
        self,
        dim: int,
        branches: int = 4,
        top_k: int = 1,
        hidden: int | None = None,
    ) -> None:
        super().__init__()
        if branches < 1:
            raise ValueError("branches must be at least 1")
        if top_k < 1 or top_k > branches:
            raise ValueError("top_k must be between 1 and branches")

        self.branches = branches
        self.top_k = top_k
        branch_hidden = hidden if hidden is not None else max(1, (dim * 4) // branches)
        self.router = nn.Linear(dim, branches)
        self.branch_modules = nn.ModuleList(
            [CompartmentBranch(dim, branch_hidden) for _ in range(branches)]
        )

    def forward(self, x: Tensor) -> CompartmentOutput:
        """Route each token through a sparse subset of local branches."""
        scores = self.router(x)
        gates = sparse_top_k_gates(scores, self.top_k)
        branch_outputs = torch.stack(
            [branch(x) for branch in self.branch_modules],
            dim=-2,
        )
        hidden = (gates.unsqueeze(-1) * branch_outputs).sum(dim=-2)
        return CompartmentOutput(
            hidden=hidden,
            gates=gates,
            branch_outputs=branch_outputs,
            branch_scores=scores,
            branch_inhibition=torch.zeros_like(scores),
        )


class DendriticFeedForward(nn.Module):
    """Compartment FFN where local branches self-score and inhibit neighbors."""

    def __init__(
        self,
        dim: int,
        branches: int = 4,
        top_k: int = 1,
        hidden: int | None = None,
        inhibition_strength: float = 0.5,
    ) -> None:
        super().__init__()
        if branches < 1:
            raise ValueError("branches must be at least 1")
        if top_k < 1 or top_k > branches:
            raise ValueError("top_k must be between 1 and branches")
        if inhibition_strength < 0.0:
            raise ValueError("inhibition_strength cannot be negative")

        self.branches = branches
        self.top_k = top_k
        self.inhibition_strength = inhibition_strength
        branch_hidden = hidden if hidden is not None else max(1, (dim * 4) // branches)
        self.branch_modules = nn.ModuleList(
            [DendriticBranch(dim, branch_hidden) for _ in range(branches)]
        )

    def forward(self, x: Tensor) -> CompartmentOutput:
        """Let branches self-score, inhibit similar neighbors, and merge at soma."""
        outputs = []
        scores = []
        for branch in self.branch_modules:
            branch_output, branch_score = branch(x)
            outputs.append(branch_output)
            scores.append(branch_score)

        branch_outputs = torch.stack(outputs, dim=-2)
        raw_scores = torch.stack(scores, dim=-1)
        if self.branches > 1 and self.inhibition_strength > 0.0:
            normalized = torch.nn.functional.normalize(branch_outputs, dim=-1, eps=1e-8)
            similarity = torch.matmul(normalized, normalized.transpose(-1, -2))
            mask = ~torch.eye(self.branches, dtype=torch.bool, device=x.device)
            inhibition = torch.relu(similarity).masked_fill(~mask, 0.0).sum(dim=-1)
            inhibition = inhibition / (self.branches - 1)
            scores = raw_scores - (self.inhibition_strength * inhibition)
        else:
            inhibition = torch.zeros_like(raw_scores)
            scores = raw_scores

        gates = sparse_top_k_gates(scores, self.top_k)
        hidden = (gates.unsqueeze(-1) * branch_outputs).sum(dim=-2)
        return CompartmentOutput(
            hidden=hidden,
            gates=gates,
            branch_outputs=branch_outputs,
            branch_scores=scores,
            branch_inhibition=inhibition,
        )


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block with per-head attention output."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dropout: float = 0.0,
        compartments: int = 0,
        compartment_top_k: int = 1,
        compartment_mode: CompartmentMode = "router",
        branch_inhibition_strength: float = 0.5,
    ) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm_ff = nn.LayerNorm(dim)
        self.compartmentalized = compartments > 0
        if compartment_mode not in {"router", "dendritic"}:
            raise ValueError("compartment_mode must be 'router' or 'dendritic'")
        self.ff = (
            (
                DendriticFeedForward(
                    dim,
                    branches=compartments,
                    top_k=compartment_top_k,
                    inhibition_strength=branch_inhibition_strength,
                )
                if compartment_mode == "dendritic"
                else CompartmentalizedFeedForward(
                    dim,
                    branches=compartments,
                    top_k=compartment_top_k,
                )
            )
            if self.compartmentalized
            else nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
            )
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, CompartmentOutput | None]:
        """Apply self-attention and feed-forward updates.

        Returns:
            Updated hidden states and raw per-head attention weights.
        """
        attn_input = self.norm_attn(x)
        attended, weights = self.attn(
            attn_input,
            attn_input,
            attn_input,
            need_weights=True,
            average_attn_weights=False,
        )
        x = x + attended
        ff_input = self.norm_ff(x)
        if self.compartmentalized:
            compartment_output = self.ff(ff_input)
            if not isinstance(compartment_output, CompartmentOutput):
                raise TypeError("compartmentalized FFN must return CompartmentOutput")
            x = x + compartment_output.hidden
            return x, weights, compartment_output

        ff_output = self.ff(ff_input)
        if not isinstance(ff_output, Tensor):
            raise TypeError("standard FFN must return a tensor")
        x = x + ff_output
        return x, weights, None


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
        compartments: int = 0,
        compartment_top_k: int = 1,
        compartment_mode: CompartmentMode = "router",
        branch_inhibition_strength: float = 0.5,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, dim)
        self.position_embedding = nn.Embedding(seq_len, dim)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim,
                    heads,
                    compartments=compartments,
                    compartment_top_k=compartment_top_k,
                    compartment_mode=compartment_mode,
                    branch_inhibition_strength=branch_inhibition_strength,
                )
                for _ in range(layers)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.output = nn.Linear(dim, vocab_size)

    def forward(self, tokens: Tensor) -> ModelOutput:
        """Run a forward pass and collect attention from every layer."""
        positions = torch.arange(tokens.shape[1], device=tokens.device).unsqueeze(0)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        attentions = []
        branch_gates = []
        branch_outputs = []
        branch_scores = []
        branch_inhibition = []
        for block in self.blocks:
            x, attention, compartment_output = block(x)
            attentions.append(attention)
            if compartment_output is not None:
                branch_gates.append(compartment_output.gates)
                branch_outputs.append(compartment_output.branch_outputs)
                branch_scores.append(compartment_output.branch_scores)
                branch_inhibition.append(compartment_output.branch_inhibition)
        hidden = self.norm(x)
        logits = self.output(hidden)
        return ModelOutput(
            logits=logits,
            hidden=hidden,
            attention=torch.stack(attentions, dim=1),
            branch_gates=torch.stack(branch_gates, dim=1) if branch_gates else None,
            branch_outputs=torch.stack(branch_outputs, dim=1) if branch_outputs else None,
            branch_scores=torch.stack(branch_scores, dim=1) if branch_scores else None,
            branch_inhibition=torch.stack(branch_inhibition, dim=1) if branch_inhibition else None,
        )
