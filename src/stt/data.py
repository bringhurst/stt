"""Synthetic data generation for the minimal STT experiments.

The task is deliberately small and deterministic. It is not a language-modeling
benchmark; it is a controlled next-token prediction problem that gives the toy
Transformer enough structure to form nontrivial hidden states and attention maps.
"""

from __future__ import annotations

import torch


class SyntheticSequenceTask:
    """Generate modular next-token sequences with repeatable randomness.

    The sequence mixes arithmetic progressions, short motifs, and small noise.
    That combination is useful for quick experiments because the model can learn
    something measurable in seconds while still producing varied representations.

    Args:
        vocab_size: Number of discrete token ids.
        seq_len: Input sequence length. Targets are the same sequence shifted by
            one step.
        seed: Seed for the task-local generator. Keeping this local avoids
            accidental coupling to model initialization.
    """

    def __init__(self, vocab_size: int = 32, seq_len: int = 24, seed: int = 0) -> None:
        if vocab_size < 16:
            raise ValueError("vocab_size must be at least 16")
        if seq_len < 4:
            raise ValueError("seq_len must be at least 4")
        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.generator = torch.Generator().manual_seed(seed)

    def batch(
        self,
        batch_size: int,
        device: torch.device | str = "cpu",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return a batch of input tokens and shifted next-token targets."""
        starts = torch.randint(0, self.vocab_size, (batch_size, 1), generator=self.generator)
        offsets = torch.arange(self.seq_len + 1).unsqueeze(0)
        modes = torch.randint(0, 4, (batch_size, 1), generator=self.generator)

        arithmetic = starts + offsets * (modes + 1)
        motif = (offsets % 4) * (modes + 2)
        noise = torch.randint(0, 3, (batch_size, self.seq_len + 1), generator=self.generator)
        sequence = (arithmetic + motif + noise) % self.vocab_size

        tokens = sequence[:, :-1].to(device=device, dtype=torch.long)
        targets = sequence[:, 1:].to(device=device, dtype=torch.long)
        return tokens, targets
