import torch

from stt.losses import (
    attention_diversity_loss,
    representation_repulsion_loss,
    sparse_activation_loss,
)


def test_attention_diversity_loss_is_higher_for_identical_heads() -> None:
    identical = torch.ones(2, 1, 3, 4, 4)
    diverse = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(2, 1, 3, 1, 1)
    diverse[:, :, 1] = torch.flip(diverse[:, :, 1], dims=(-1,))
    diverse[:, :, 2] = torch.flip(diverse[:, :, 2], dims=(-2,))

    assert attention_diversity_loss(identical) > attention_diversity_loss(diverse)


def test_repulsion_loss_decreases_for_distant_vectors() -> None:
    close = torch.zeros(1, 4, 3)
    far = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3) * 10.0

    assert representation_repulsion_loss(close) > representation_repulsion_loss(far)


def test_sparse_activation_loss_is_l1_mean() -> None:
    hidden = torch.tensor([[[1.0, -3.0, 2.0]]])

    assert torch.isclose(sparse_activation_loss(hidden), torch.tensor(2.0))
