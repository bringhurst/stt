import torch

from stt.oracle_route import (
    compose_group_route_state,
    group_key,
    group_keys,
    route_metrics,
)


def test_group_key_supports_layer_module_and_tensor_modes() -> None:
    name = "base.model.model.layers.3.self_attn.q_proj.lora_A.default.weight"

    assert group_key(name, "layer") == "layer_3"
    assert group_key(name, "module") == "base.model.model.layers.3.self_attn.q_proj"
    assert group_key(name, "tensor") == name


def test_compose_group_route_state_scales_c_by_group() -> None:
    base = {
        "model.layers.0.q.lora_A.default.weight": torch.tensor([1.0]),
        "model.layers.1.q.lora_A.default.weight": torch.tensor([2.0]),
    }
    delta_b = {
        "model.layers.0.q.lora_A.default.weight": torch.tensor([10.0]),
        "model.layers.1.q.lora_A.default.weight": torch.tensor([20.0]),
    }
    delta_c = {
        "model.layers.0.q.lora_A.default.weight": torch.tensor([100.0]),
        "model.layers.1.q.lora_A.default.weight": torch.tensor([200.0]),
    }

    composed = compose_group_route_state(
        base,
        delta_b_state=delta_b,
        delta_c_state=delta_c,
        b_scale=0.5,
        group_scales={"layer_0": 0.25, "layer_1": 0.75},
        group_by="layer",
    )

    assert group_keys(base, "layer") == ["layer_0", "layer_1"]
    assert composed["model.layers.0.q.lora_A.default.weight"].item() == 31.0
    assert composed["model.layers.1.q.lora_A.default.weight"].item() == 162.0


def test_route_metrics_uses_phase_local_c_learning() -> None:
    metrics = route_metrics(
        (0.9, 0.8, 0.7),
        eval_a_after_a=1.0,
        eval_b_after_a=2.0,
        eval_a_after_b=0.95,
        eval_b_after_b=0.75,
        eval_c_after_b=1.5,
        sequential_accretion_a=0.0,
        sequential_interference_a=0.1,
        sequential_interference_b=0.2,
        sequential_learning_b=1.0,
        sequential_learning_c=0.7,
    )

    assert metrics["oracle_learning_c"] == 0.8
    assert metrics["oracle_learning_b"] == 1.2
