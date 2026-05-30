import torch

from stt.losses import (
    attention_diversity_loss,
    branch_inhibition_loss,
    branch_load_balance_loss,
    branch_output_repulsion_loss,
    gossip_repulsion_loss,
    representation_repulsion_loss,
    sample_token_vectors,
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


def test_branch_output_repulsion_is_higher_for_identical_branches() -> None:
    identical = torch.ones(1, 2, 3, 4, 4)
    diverse = torch.eye(4).reshape(1, 1, 1, 4, 4).repeat(1, 2, 3, 1, 1)

    assert branch_output_repulsion_loss(identical) > branch_output_repulsion_loss(diverse)


def test_branch_load_balance_loss_penalizes_collapsed_routing() -> None:
    collapsed = torch.zeros(1, 1, 8, 4)
    collapsed[..., 0] = 1.0
    balanced = torch.nn.functional.one_hot(torch.arange(8) % 4, num_classes=4).float()
    balanced = balanced.reshape(1, 1, 8, 4)

    assert branch_load_balance_loss(collapsed) > branch_load_balance_loss(balanced)


def test_branch_inhibition_loss_penalizes_correlated_coactive_branches() -> None:
    identical = torch.ones(1, 1, 2, 4)
    orthogonal = torch.eye(4)[:2].reshape(1, 1, 2, 4)
    gates = torch.full((1, 1, 2), 0.5)

    assert branch_inhibition_loss(identical, gates) > branch_inhibition_loss(orthogonal, gates)


def test_gossip_repulsion_loss_penalizes_high_similarity() -> None:
    torch.manual_seed(0)
    collapsed = torch.ones(1, 8, 4)
    spread = torch.eye(4).repeat(1, 2, 1)

    assert gossip_repulsion_loss(collapsed, tau=0.5, k=2) > gossip_repulsion_loss(
        spread, tau=0.5, k=2
    )


def test_sample_token_vectors_respects_attention_mask() -> None:
    hidden = torch.arange(12, dtype=torch.float32).reshape(1, 3, 4)
    mask = torch.tensor([[1, 0, 1]])

    vectors = sample_token_vectors(hidden, attention_mask=mask, max_vectors=10)

    assert vectors.shape == (2, 4)
    assert torch.equal(vectors[0], hidden[0, 0])
    assert torch.equal(vectors[1], hidden[0, 2])
