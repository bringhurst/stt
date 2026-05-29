import argparse
from typing import cast

import torch

from stt.lora_experiment import LoraSettings
from stt.memory_bank import (
    MemoryBank,
    MemoryBankResult,
    MemoryDelta,
    aggregate_domain_metrics,
    compose_memory_route_state,
    expected_route_for_domain,
    merge_route_exprs,
    parse_route_expr,
    probes_from_args,
    route_expr_string,
    route_optimality_audit,
    route_variant_suffix,
    summarize_memory_bank,
)


def test_parse_route_expr_compiles_named_delta_scales() -> None:
    route = parse_route_expr("A+B+0.9C+0.4*D")

    assert route.expression == "A+B+0.9C+0.4D"
    assert route.scales == {"B": 1.0, "C": 0.9, "D": 0.4}
    assert route_expr_string(route) == "A+B+0.9C+0.4D"
    assert route_variant_suffix(route) == "A_B_0p9C_0p4D"


def test_merge_route_exprs_keeps_candidates_before_audit_routes() -> None:
    routes = [parse_route_expr("A+B"), parse_route_expr("A+C")]
    audit_routes = [parse_route_expr("A+0.5B+0.5C"), parse_route_expr("A+B")]

    merged = merge_route_exprs(routes, audit_routes)

    assert [route.expression for route in merged] == [
        "A+B",
        "A+C",
        "A+0.5B+0.5C",
    ]


def test_probes_from_args_normalizes_expected_routes() -> None:
    args = argparse.Namespace(
        probe_files=["data/memory_probe_gamma_boundary.txt"],
        probe_names=None,
        probe_routes=["A+1.0*C"],
    )

    probes = probes_from_args(args, stable_phase="A")

    assert probes[0].name == "memory_probe_gamma_boundary"
    assert probes[0].expected_route == "A+C"


def test_route_optimality_audit_reports_selected_and_expected_gaps(monkeypatch) -> None:
    calls = []

    def fake_score_route_states(*args, **kwargs):
        index = len(calls)
        calls.append(index)
        if index == 0:
            return [("A+B", 1.0), ("A+C", 1.4), ("A", 1.8)]
        return [("A+C", 2.0), ("A+B", 2.5), ("A", 3.0)]

    monkeypatch.setattr("stt.memory_bank.score_route_states", fake_score_route_states)
    encoded = {
        "input_ids": torch.tensor([[1], [2]]),
        "attention_mask": torch.tensor([[1], [1]]),
        "labels": torch.tensor([[1], [2]]),
    }

    audit = route_optimality_audit(
        model=torch.nn.Linear(1, 1),
        route_states={"A": {}, "A+B": {}, "A+C": {}},
        encoded=encoded,
        settings=LoraSettings(model_name="tiny", batch_size=1, eval_batches=1),
        expected_route="A+B",
        selected_eval_routes=["A+C"],
    )

    assert audit["best_route_counts"] == {"A+B": 1, "A+C": 1}
    assert audit["most_best_route"] == "A+C"
    assert audit["best_eval_loss"] == 1.5
    assert round(float(audit["selected_loss_gap"]), 4) == 0.2
    assert round(float(audit["expected_loss_gap"]), 4) == 0.25
    assert audit["optimal_route_count"] == 1
    assert audit["optimal_route_rate"] == 0.5


def test_compose_memory_route_state_adds_scaled_deltas() -> None:
    bank = MemoryBank(
        stable_phase="A",
        stable_adapter={"w": torch.tensor([1.0]), "b": torch.tensor([2.0])},
        deltas={
            "B": MemoryDelta(
                name="B",
                from_phase="A",
                to_phase="B",
                tensors={"w": torch.tensor([10.0]), "b": torch.tensor([20.0])},
                metadata={},
            ),
            "C": MemoryDelta(
                name="C",
                from_phase="B",
                to_phase="C",
                tensors={"w": torch.tensor([100.0]), "b": torch.tensor([200.0])},
                metadata={},
            ),
        },
    )

    composed = compose_memory_route_state(bank, parse_route_expr("A+0.5B+0.25C"))

    assert composed["w"].item() == 31.0
    assert composed["b"].item() == 62.0


def test_expected_route_prefers_related_b_for_a_and_direct_conflict_routes() -> None:
    routes = ["A", "A+B", "A+C", "A+D"]

    assert expected_route_for_domain("A", ["A", "B", "C", "D"], routes) == "A+B"
    assert expected_route_for_domain("B", ["A", "B", "C", "D"], routes) == "A+B"
    assert expected_route_for_domain("C", ["A", "B", "C", "D"], routes) == "A+C"
    assert expected_route_for_domain("D", ["A", "B", "C", "D"], routes) == "A+D"


def test_aggregate_domain_metrics_weights_by_selection_count() -> None:
    per_domain = {
        "A": {
            "selected_route_counts": {"A+B": 2},
            "expected_route": "A+B",
            "most_selected_route": "A+B",
            "selection_count": 2,
            "route_accuracy": 1.0,
            "ambiguous_count": 0,
            "ambiguous_rate": 0.0,
            "eval_loss": 1.0,
            "sequential_eval_loss": 1.5,
            "learned_eval_loss": 0.8,
            "initial_eval_loss": 2.0,
            "loss_delta_vs_sequential": 0.5,
            "learning_retained": 0.8,
            "interference": 0.2,
        },
        "C": {
            "selected_route_counts": {"A+C": 1},
            "expected_route": "A+C",
            "most_selected_route": "A+C",
            "selection_count": 1,
            "route_accuracy": 1.0,
            "ambiguous_count": 0,
            "ambiguous_rate": 0.0,
            "eval_loss": 0.7,
            "sequential_eval_loss": 0.8,
            "learned_eval_loss": 0.6,
            "initial_eval_loss": 1.2,
            "loss_delta_vs_sequential": 0.1,
            "learning_retained": 0.83,
            "interference": 0.1,
        },
    }

    metrics = aggregate_domain_metrics(per_domain)

    assert round(metrics["contextual_eval_loss"], 4) == 0.9
    assert round(metrics["sequential_eval_loss"], 4) == 1.2667
    assert round(metrics["frontier_score"], 4) == 0.3667


def test_summarize_memory_bank_reports_domains_and_wins() -> None:
    result = cast(MemoryBankResult, {
        "variant": "gossip_contextual_memory_bank_oracle",
        "heldout_report": True,
        "contextual_eval_loss": 1.0,
        "sequential_eval_loss": 1.5,
        "loss_delta_vs_sequential": 0.5,
        "mean_learning_retained": 0.9,
        "mean_interference": 0.1,
        "route_accuracy": 1.0,
        "ambiguous_rate": 0.0,
        "frontier_score": 0.5,
        "per_domain": {
            "A": {
                "selected_route_counts": {"A+B": 2},
                "expected_route": "A+B",
                "most_selected_route": "A+B",
                "selection_count": 2,
                "route_accuracy": 1.0,
                "ambiguous_count": 0,
                "ambiguous_rate": 0.0,
                "eval_loss": 1.0,
                "sequential_eval_loss": 1.5,
                "learned_eval_loss": 0.8,
                "initial_eval_loss": 2.0,
                "loss_delta_vs_sequential": 0.5,
                "learning_retained": 0.83,
                "interference": 0.2,
            }
        },
    })

    summary = summarize_memory_bank([result])

    values = summary["gossip_contextual_memory_bank_oracle"]
    assert values["contextual_win_count"] == 1.0
    assert values["route_accuracy_mean"] == 1.0
    assert values["domain_A_eval_loss_mean"] == 1.0
    assert values["domain_A_learning_retained_mean"] == 0.83


def test_summarize_memory_bank_reports_probe_metrics() -> None:
    result = cast(MemoryBankResult, {
        "variant": "gossip_contextual_memory_bank_calibration",
        "heldout_report": True,
        "contextual_eval_loss": 1.0,
        "sequential_eval_loss": 1.5,
        "loss_delta_vs_sequential": 0.5,
        "mean_learning_retained": 0.9,
        "mean_interference": 0.1,
        "route_accuracy": 1.0,
        "ambiguous_rate": 0.0,
        "optimal_route_rate": 0.8,
        "selected_loss_gap": 0.2,
        "expected_loss_gap": 0.3,
        "probe_eval_loss": 0.7,
        "probe_route_accuracy": 1.0,
        "probe_ambiguous_rate": 0.0,
        "probe_optimal_route_rate": 1.0,
        "probe_selected_loss_gap": 0.0,
        "probe_expected_loss_gap": 0.0,
        "frontier_score": 0.5,
        "per_domain": {},
        "per_probe": {
            "gamma_boundary": {
                "selected_route_counts": {"A+C": 2},
                "expected_route": "A+C",
                "most_selected_route": "A+C",
                "selection_count": 2,
                "route_accuracy": 1.0,
                "ambiguous_count": 0,
                "ambiguous_rate": 0.0,
                "eval_loss": 0.7,
                "sequential_eval_loss": 1.0,
                "learned_eval_loss": 0.7,
                "initial_eval_loss": 1.2,
                "loss_delta_vs_sequential": 0.3,
                "learning_retained": 1.0,
                "interference": 0.0,
                "optimal_route_rate": 1.0,
                "selected_loss_gap": 0.0,
                "expected_loss_gap": 0.0,
            }
        },
    })

    summary = summarize_memory_bank([result])

    values = summary["gossip_contextual_memory_bank_calibration"]
    assert values["probe_eval_loss_mean"] == 0.7
    assert values["probe_route_accuracy_mean"] == 1.0
    assert values["probe_gamma_boundary_eval_loss_mean"] == 0.7
