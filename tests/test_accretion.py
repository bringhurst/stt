from typing import cast

import torch

from stt.accretion import (
    AccretionResult,
    lora_effective_deltas,
    mean_lora_cosine,
    summarize_accretion,
    tensor_cosine,
)
from stt.accretion_data import (
    generate_lines,
    group_for,
    task_a_line,
    task_b_rehearsal_line,
    task_c_line,
)


def test_accretion_data_is_deterministic_and_conflicting() -> None:
    first = generate_lines(num_entities=16, seed=0)
    second = generate_lines(num_entities=16, seed=0)

    assert first == second
    assert set(first) == {
        "accretion_task_a.txt",
        "accretion_task_b_related.txt",
        "accretion_task_b_rehearsal.txt",
        "accretion_task_c_conflict.txt",
        "accretion_task_n_neutral.txt",
    }
    assert "Entity-" in first["accretion_task_a.txt"][0]
    assert "Entity-" in first["accretion_task_b_related.txt"][0]
    assert "Entity-" in first["accretion_task_c_conflict.txt"][0]
    assert "Artifact-" in first["accretion_task_n_neutral.txt"][0]

    index = 0
    assert group_for(index) != group_for(index, offset=3)
    assert task_a_line(index) != task_c_line(index)
    assert task_a_line(index) in task_b_rehearsal_line(index)


def test_summarize_accretion_metric_signs() -> None:
    result = cast(AccretionResult, {
        "variant": "baseline",
        "eval_a_after_a": 1.0,
        "eval_a_after_b": 0.9,
        "eval_a_after_c": 1.1,
        "eval_b_after_b": 0.8,
        "eval_b_after_c": 1.0,
        "eval_c_after_c": 0.7,
        "learning_a": 2.0,
        "learning_b": 1.5,
        "learning_c": 1.2,
        "accretion_a_after_b": 0.1,
        "backward_transfer_a_after_b": -0.1,
        "interference_a_after_c": 0.2,
        "interference_b_after_c": 0.2,
        "retention_a_after_b": 1.0 / 0.9,
        "retention_a_after_c": 1.0 / 1.1,
        "retention_b_after_c": 0.8 / 1.0,
    })

    summary = summarize_accretion([result])

    assert summary["baseline"]["accretion_a_after_b_mean"] == 0.1
    assert summary["baseline"]["backward_transfer_a_after_b_mean"] == -0.1
    assert summary["baseline"]["interference_a_after_c_mean"] == 0.2


def test_tensor_cosine_handles_zero_vectors() -> None:
    assert tensor_cosine(torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])) == 0.0
    assert tensor_cosine(torch.zeros(2), torch.ones(2)) is None


def test_lora_effective_deltas_and_mean_cosine() -> None:
    class FakeLora(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lora_A = torch.nn.ModuleDict({"default": torch.nn.Linear(2, 1, bias=False)})
            self.lora_B = torch.nn.ModuleDict({"default": torch.nn.Linear(1, 2, bias=False)})
            self.scaling = {"default": 2.0}

    model = torch.nn.Sequential(FakeLora())
    fake = cast(FakeLora, model[0])
    lora_a = cast(torch.nn.Linear, fake.lora_A["default"])
    lora_b = cast(torch.nn.Linear, fake.lora_B["default"])
    with torch.no_grad():
        lora_a.weight.fill_(1.0)
        lora_b.weight.copy_(torch.tensor([[1.0], [2.0]]))

    deltas = lora_effective_deltas(model)

    assert set(deltas) == {"0"}
    assert torch.equal(deltas["0"], torch.tensor([[2.0, 2.0], [4.0, 4.0]]))
    assert mean_lora_cosine(deltas, deltas) == 1.0
