"""Regularization losses for Surface Tension Transformer experiments.

These losses are intentionally architecture-agnostic. They operate on attention
maps or hidden states returned by a model and can therefore be moved from this
toy experiment to larger Transformer fine-tuning code later.
"""

from __future__ import annotations

import torch
from torch import Tensor


def attention_diversity_loss(attention: Tensor) -> Tensor:
    """Mean pairwise cosine similarity between attention heads.

    Minimizing this term discourages heads in the same layer from attending to
    identical patterns. It is a direct implementation of the first STT experiment:
    weak pressure against redundant attention maps.

    Args:
        attention: Tensor shaped `(batch, layers, heads, seq, seq)`.

    Returns:
        Scalar tensor containing mean off-diagonal head similarity.
    """
    if attention.ndim != 5:
        raise ValueError("attention must have shape (batch, layers, heads, seq, seq)")

    batch, layers, heads, _, _ = attention.shape
    if heads < 2:
        return attention.new_zeros(())

    flat = attention.reshape(batch, layers, heads, -1)
    flat = torch.nn.functional.normalize(flat, dim=-1)
    similarity = torch.matmul(flat, flat.transpose(-1, -2))
    mask = ~torch.eye(heads, dtype=torch.bool, device=attention.device)
    return similarity[..., mask].mean()


def representation_repulsion_loss(hidden: Tensor, sample_size: int = 128) -> Tensor:
    """Repel hidden vectors by penalizing close pairwise distances.

    The exponential kernel is largest when two hidden vectors are close and
    quickly decays as they separate. Minimizing it creates weak manifold-spread
    pressure without needing labels for semantic relatedness. Vectors are L2
    normalized first so the loss is meaningful for both small toy models and
    pretrained LMs with much larger hidden-state norms.

    Args:
        hidden: Tensor shaped `(batch, seq, dim)`.
        sample_size: Maximum number of flattened token vectors to compare. This
            bounds the quadratic pairwise distance cost.

    Returns:
        Scalar tensor containing mean close-neighbor penalty.
    """
    if hidden.ndim != 3:
        raise ValueError("hidden must have shape (batch, seq, dim)")

    vectors = hidden.reshape(-1, hidden.shape[-1])
    if vectors.shape[0] < 2:
        return hidden.new_zeros(())
    if vectors.shape[0] > sample_size:
        vectors = vectors[:sample_size]

    vectors = torch.nn.functional.normalize(vectors, dim=-1)
    distances = torch.cdist(vectors, vectors).pow(2)
    mask = ~torch.eye(vectors.shape[0], dtype=torch.bool, device=hidden.device)
    return torch.exp(-distances[mask]).mean()


def sparse_activation_loss(hidden: Tensor) -> Tensor:
    """L1 pressure on hidden activations.

    This is the simplest sparse-competition proxy. It does not implement true
    winner-take-all routing, but it makes dense hidden activation patterns more
    expensive and is easy to test.
    """
    if hidden.ndim != 3:
        raise ValueError("hidden must have shape (batch, seq, dim)")
    return hidden.abs().mean()


def sample_token_vectors(
    hidden: Tensor,
    attention_mask: Tensor | None = None,
    max_vectors: int = 256,
) -> Tensor:
    """Return sampled non-padding token vectors from hidden states.

    Args:
        hidden: Tensor shaped `(batch, seq, dim)`.
        attention_mask: Optional mask shaped `(batch, seq)` where nonzero values
            mark real tokens.
        max_vectors: Maximum vectors to return. Sampling keeps gossip cheap on
            memory-constrained MPS runs.
    """
    if hidden.ndim != 3:
        raise ValueError("hidden must have shape (batch, seq, dim)")
    if attention_mask is not None and attention_mask.shape != hidden.shape[:2]:
        raise ValueError("attention_mask must have shape (batch, seq)")

    vectors = hidden.reshape(-1, hidden.shape[-1])
    if attention_mask is not None:
        mask = attention_mask.reshape(-1).to(dtype=torch.bool, device=hidden.device)
        vectors = vectors[mask]
    if vectors.shape[0] <= max_vectors:
        return vectors
    indices = torch.randperm(vectors.shape[0], device=hidden.device)[:max_vectors]
    return vectors.index_select(0, indices)


def gossip_repulsion_loss(
    hidden: Tensor,
    attention_mask: Tensor | None = None,
    tau: float = 0.85,
    k: int = 8,
    max_vectors: int = 256,
) -> Tensor:
    """Thresholded sampled anti-consensus loss over hidden token vectors.

    The loss samples `k` peers per token vector, penalizing only cosine
    similarities above `tau`. This repairs local collapse without forcing every
    representation pair apart.
    """
    vectors = sample_token_vectors(hidden, attention_mask=attention_mask, max_vectors=max_vectors)
    if vectors.shape[0] <= 1 or k <= 0:
        return hidden.new_zeros(())

    vectors = torch.nn.functional.normalize(vectors, dim=-1, eps=1e-8)
    n = vectors.shape[0]
    anchors = torch.arange(n, device=hidden.device)
    peers = torch.randint(0, n, size=(n, k), device=hidden.device)
    peers = torch.where(peers == anchors[:, None], (peers + 1) % n, peers)

    anchor_vectors = vectors[anchors]
    peer_vectors = vectors[peers]
    cosine = (anchor_vectors[:, None, :] * peer_vectors).sum(dim=-1)
    overlap = torch.relu(cosine - tau)
    return overlap.pow(2).mean()
