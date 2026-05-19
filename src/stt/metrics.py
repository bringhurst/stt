"""Representation geometry metrics used by STT experiments."""

from __future__ import annotations

import torch
from torch import Tensor

from stt.losses import attention_diversity_loss


@torch.no_grad()
def head_similarity(attention: Tensor) -> float:
    """Return mean pairwise attention-head cosine similarity.

    Lower values indicate less redundant heads under this specific metric.
    """
    return float(attention_diversity_loss(attention).detach().cpu())


@torch.no_grad()
def effective_rank(hidden: Tensor, eps: float = 1e-8) -> float:
    """Estimate usable representation dimensionality from singular values.

    Effective rank is `exp(entropy(normalized_singular_values))`. It is high
    when variance is spread across many dimensions and low when hidden states
    collapse into a narrow subspace.
    """
    vectors = hidden.detach().float().cpu().reshape(-1, hidden.shape[-1])
    vectors = vectors - vectors.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(vectors)
    probabilities = singular_values / singular_values.sum().clamp_min(eps)
    entropy = -(probabilities * probabilities.clamp_min(eps).log()).sum()
    return float(torch.exp(entropy).detach().cpu())


@torch.no_grad()
def isotropy(hidden: Tensor, eps: float = 1e-8) -> float:
    """Return a simple anisotropy score from singular values.

    Values closer to 1 mean the singular spectrum is flatter. Larger values mean
    one or a few dominant directions carry disproportionate variance.
    """
    vectors = hidden.detach().float().cpu().reshape(-1, hidden.shape[-1])
    vectors = vectors - vectors.mean(dim=0, keepdim=True)
    singular_values = torch.linalg.svdvals(vectors)
    return float((singular_values.max() / singular_values.mean().clamp_min(eps)).detach().cpu())


@torch.no_grad()
def active_fraction(hidden: Tensor, threshold: float = 0.05) -> float:
    """Return the fraction of hidden activations above a magnitude threshold."""
    return float((hidden.abs() > threshold).float().mean().detach().cpu())
