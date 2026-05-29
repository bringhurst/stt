from typing import cast

import torch

from stt.routed_accretion import (
    RoutedAccretionResult,
    c_scale_for_parameter,
    compose_grouped_route_state,
    frontier_score,
    grouped_route_spec,
    layer_route_spec,
    parse_group_route_pairs,
    parse_layer_route_pairs,
    parse_route_pairs,
    route_passes_selection_constraints,
    route_variant_name,
    routed_metric_values,
    scalar_route_spec,
    select_calibrated_candidate,
    selection_candidate_record,
    summarize_routed_accretion,
)


def test_summarize_routed_accretion_reports_wins() -> None:
    result = cast(RoutedAccretionResult, {
        "variant": "gossip",
        "route_b_scale": 0.9,
        "route_c_scale": 0.25,
        "route_c_lora_a_scale": 0.25,
        "route_c_lora_b_scale": 0.25,
        "route_c_early_scale": 0.25,
        "route_c_middle_scale": 0.25,
        "route_c_late_scale": 0.25,
        "sequential_eval_a": 1.4,
        "sequential_eval_b": 1.5,
        "sequential_eval_c": 0.2,
        "routed_eval_a": 1.0,
        "routed_eval_b": 1.0,
        "routed_eval_c": 0.6,
        "sequential_accretion_a": 0.1,
        "sequential_interference_a": 0.4,
        "sequential_interference_a_after_c": 0.4,
        "sequential_interference_b": 0.5,
        "sequential_interference_b_after_c": 0.5,
        "sequential_learning_b": 2.0,
        "sequential_learning_c": 1.0,
        "sequential_retention_a": 0.7,
        "sequential_retention_b": 0.8,
        "routed_accretion_a": 0.2,
        "routed_interference_a": 0.01,
        "routed_interference_a_after_c": 0.01,
        "routed_interference_b": 0.02,
        "routed_interference_b_after_c": 0.02,
        "routed_learning_b": 1.9,
        "routed_learning_c": 1.5,
        "routed_retention_a": 0.9,
        "routed_retention_b": 0.95,
        "delta_accretion_a": 0.1,
        "delta_interference_a": 0.39,
        "delta_interference_b": 0.48,
        "delta_learning_b": -0.1,
        "delta_learning_c": 0.5,
        "frontier_score": 1.0,
    })

    summary = summarize_routed_accretion([result])

    assert summary["gossip"]["routed_accretion_a_mean"] == 0.2
    assert summary["gossip"]["delta_interference_a_mean"] == 0.39
    assert summary["gossip"]["routed_accretion_win_count"] == 1.0
    assert summary["gossip"]["accretion_a_win_count"] == 1.0
    assert summary["gossip"]["routed_interference_a_win_count"] == 1.0
    assert summary["gossip"]["interference_a_win_count"] == 1.0
    assert summary["gossip"]["routed_learning_c_preserved_count"] == 1.0
    assert summary["gossip"]["learning_c_preserved_count"] == 1.0
    assert summary["gossip"]["frontier_score_win_count"] == 1.0
    assert summary["gossip"]["frontier_score_mean"] == 1.0


def test_route_sweep_helpers() -> None:
    assert parse_route_pairs(["0.9:0.25", "1:0.15"]) == [
        scalar_route_spec(0.9, 0.25),
        scalar_route_spec(1.0, 0.15),
    ]
    assert parse_group_route_pairs(["0.9:0.25:0.8", "1:0.4:0.7"]) == [
        grouped_route_spec(0.9, 0.25, 0.8),
        grouped_route_spec(1.0, 0.4, 0.7),
    ]
    assert parse_layer_route_pairs(["0.9:0.25:0.4:0.9"]) == [
        layer_route_spec(0.9, 0.25, 0.4, 0.9),
    ]
    assert (
        route_variant_name("gossip", scalar_route_spec(0.9, 0.25), multi_route=False)
        == "gossip"
    )
    assert (
        route_variant_name("gossip", scalar_route_spec(0.9, 0.25), multi_route=True)
        == "gossip_b0.9_c0.25"
    )
    assert (
        route_variant_name("gossip", grouped_route_spec(0.9, 0.25, 0.8), multi_route=True)
        == "gossip_b0.9_ca0.25_cb0.8"
    )
    assert (
        route_variant_name("gossip", layer_route_spec(0.9, 0.25, 0.4, 0.9), multi_route=True)
        == "gossip_b0.9_ce0.25_cm0.4_cl0.9"
    )


def test_grouped_c_route_scales_lora_tensor_families() -> None:
    assert (
        c_scale_for_parameter(
            "base.layers.0.lora_A.default.weight",
            route=grouped_route_spec(0.9, 0.2, 0.8),
        )
        == 0.2
    )
    assert (
        c_scale_for_parameter(
            "base.layers.0.lora_B.default.weight",
            route=grouped_route_spec(0.9, 0.2, 0.8),
        )
        == 0.8
    )

    base = {
        "layer.lora_A.default.weight": torch.tensor([1.0]),
        "layer.lora_B.default.weight": torch.tensor([2.0]),
    }
    delta_b = {
        "layer.lora_A.default.weight": torch.tensor([10.0]),
        "layer.lora_B.default.weight": torch.tensor([20.0]),
    }
    delta_c = {
        "layer.lora_A.default.weight": torch.tensor([100.0]),
        "layer.lora_B.default.weight": torch.tensor([200.0]),
    }

    composed = compose_grouped_route_state(
        base,
        delta_b_state=delta_b,
        delta_c_state=delta_c,
        route=grouped_route_spec(0.5, 0.25, 0.75),
    )

    assert composed["layer.lora_A.default.weight"].item() == 31.0
    assert composed["layer.lora_B.default.weight"].item() == 162.0


def test_layer_c_route_scales_layer_bands() -> None:
    route = layer_route_spec(0.5, 0.25, 0.5, 1.0)

    assert (
        c_scale_for_parameter(
            "base.model.model.layers.0.self_attn.q_proj.lora_A.default.weight",
            route=route,
            max_layer_index=8,
        )
        == 0.25
    )
    assert (
        c_scale_for_parameter(
            "base.model.model.layers.4.self_attn.q_proj.lora_A.default.weight",
            route=route,
            max_layer_index=8,
        )
        == 0.5
    )
    assert (
        c_scale_for_parameter(
            "base.model.model.layers.7.self_attn.q_proj.lora_A.default.weight",
            route=route,
            max_layer_index=8,
        )
        == 1.0
    )


def test_frontier_score_balances_metrics() -> None:
    score = frontier_score(
        sequential_accretion_a=0.2,
        sequential_interference_a=0.4,
        sequential_interference_b=0.5,
        sequential_learning_b=2.0,
        sequential_learning_c=1.0,
        routed_accretion_a=0.1,
        routed_interference_a=0.0,
        routed_interference_b=0.1,
        routed_learning_b=1.8,
        routed_learning_c=1.6,
    )

    assert round(score, 4) == 1.25


def test_routed_metric_values_uses_phase_local_c_learning() -> None:
    metrics = routed_metric_values(
        eval_a_after_a=1.0,
        eval_b_after_a=2.0,
        eval_a_after_b=0.9,
        eval_b_after_b=0.7,
        eval_c_after_b=1.5,
        sequential_eval_a=1.2,
        sequential_eval_b=1.0,
        sequential_eval_c=0.3,
        routed_eval_a=0.8,
        routed_eval_b=0.75,
        routed_eval_c=0.4,
    )

    assert round(metrics["sequential_learning_c"], 4) == 1.2
    assert round(metrics["routed_learning_c"], 4) == 1.1
    assert round(metrics["delta_learning_c"], 4) == -0.1


def test_calibrated_selection_prefers_constrained_old_task_score() -> None:
    route_a = scalar_route_spec(0.9, 0.4)
    route_b = scalar_route_spec(0.95, 0.5)
    route_c = scalar_route_spec(0.85, 0.2)
    metrics_a = {
        "delta_accretion_a": 0.2,
        "delta_interference_a": 0.2,
        "delta_interference_b": 0.2,
        "delta_learning_b": -0.03,
        "delta_learning_c": -0.08,
        "frontier_score": 0.52,
        "routed_learning_b": 1.97,
        "routed_learning_c": 0.92,
        "sequential_learning_b": 2.0,
        "sequential_learning_c": 1.0,
    }
    metrics_b = {
        **metrics_a,
        "delta_accretion_a": 0.1,
        "delta_interference_a": 0.1,
        "delta_interference_b": 0.1,
        "delta_learning_c": -0.02,
        "frontier_score": 0.6,
        "routed_learning_c": 0.98,
    }
    metrics_c = {
        **metrics_a,
        "delta_accretion_a": 0.4,
        "delta_interference_a": 0.4,
        "delta_interference_b": 0.4,
        "frontier_score": 0.9,
        "routed_learning_c": 0.5,
    }
    candidates = [
        selection_candidate_record(
            route_spec="0.9:0.4",
            route=route_a,
            metrics=metrics_a,
            constraint_passed=route_passes_selection_constraints(
                metrics_a,
                c_retention_min=0.9,
                learning_b_tolerance=0.05,
            ),
        ),
        selection_candidate_record(
            route_spec="0.95:0.5",
            route=route_b,
            metrics=metrics_b,
            constraint_passed=route_passes_selection_constraints(
                metrics_b,
                c_retention_min=0.9,
                learning_b_tolerance=0.05,
            ),
        ),
        selection_candidate_record(
            route_spec="0.85:0.2",
            route=route_c,
            metrics=metrics_c,
            constraint_passed=route_passes_selection_constraints(
                metrics_c,
                c_retention_min=0.9,
                learning_b_tolerance=0.05,
            ),
        ),
    ]

    selected = select_calibrated_candidate(candidates)

    assert selected["route_spec"] == "0.9:0.4"
    assert selected["selection_constraint_passed"] is True
