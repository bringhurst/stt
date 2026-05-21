from typing import cast

import torch

from stt.oracle_compose import (
    CompositionResult,
    classify_route,
    compose_state,
    parse_fixed_compositions,
    select_b_candidate,
    select_c_candidate,
    snapshot_trainable_state,
    split_eval_encoded,
    subtract_state,
)


def test_trainable_state_arithmetic_round_trip() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    for parameter in model[1].parameters():
        parameter.requires_grad = False
    first_layer = cast(torch.nn.Linear, model[0])
    with torch.no_grad():
        first_layer.weight.fill_(1.0)
        assert first_layer.bias is not None
        first_layer.bias.fill_(0.5)

    first = snapshot_trainable_state(model)
    second = {name: value + 2.0 for name, value in first.items()}
    delta = subtract_state(second, first)
    composed = compose_state(first, [(0.25, delta), (0.25, delta)])

    assert set(first) == {"0.weight", "0.bias"}
    assert torch.equal(delta["0.weight"], torch.full_like(delta["0.weight"], 2.0))
    assert torch.equal(composed["0.bias"], torch.full_like(composed["0.bias"], 1.5))


def test_classify_route_uses_old_and_new_deltas() -> None:
    assert classify_route(0.2, 0.2, eps_old=0.1, eps_new=0.1) == "shared"
    assert classify_route(0.0, 0.2, eps_old=0.1, eps_new=0.1) == "private"
    assert classify_route(-0.2, 0.2, eps_old=0.1, eps_new=0.1) == "conflict_private"
    assert classify_route(0.2, 0.0, eps_old=0.1, eps_new=0.1) == "reject_or_downweight"


def test_split_eval_encoded_creates_heldout_halves() -> None:
    encoded = {
        "input_ids": torch.arange(6).reshape(6, 1),
        "attention_mask": torch.ones(6, 1),
    }

    select, report, heldout = split_eval_encoded(encoded, batch_size=2)

    assert heldout
    assert select["input_ids"].shape[0] == 3
    assert report["input_ids"].shape[0] == 3


def test_parse_fixed_compositions_reads_scale_pairs() -> None:
    assert parse_fixed_compositions(["0.9:0.25", "1:0"]) == [(0.9, 0.25), (1.0, 0.0)]


def test_oracle_selectors_prefer_safe_learning() -> None:
    b_candidates = cast(list[CompositionResult], [
        {
            "variant": "gossip",
            "seed": 0,
            "composition": "bad",
            "route": "conflict_private",
            "eval_a": 1.0,
            "eval_b": 1.0,
            "eval_c": 1.0,
            "learning_b": 10.0,
            "learning_c": 0.0,
            "accretion_a": -1.0,
            "b_scale": 1.0,
            "c_scale": 0.0,
            "interference_a": None,
            "interference_b": None,
        },
        {
            "variant": "gossip",
            "seed": 0,
            "composition": "safe",
            "route": "private",
            "eval_a": 1.0,
            "eval_b": 1.0,
            "eval_c": 1.0,
            "learning_b": 5.0,
            "learning_c": 0.0,
            "accretion_a": 0.0,
            "b_scale": 0.5,
            "c_scale": 0.0,
            "interference_a": None,
            "interference_b": None,
        },
    ])
    c_candidates = cast(list[CompositionResult], [
        {
            "variant": "gossip",
            "seed": 0,
            "composition": "conflict",
            "route": "conflict_private",
            "eval_a": 1.0,
            "eval_b": 1.0,
            "eval_c": 1.0,
            "learning_b": 0.0,
            "learning_c": 10.0,
            "accretion_a": 0.0,
            "b_scale": 0.5,
            "c_scale": 1.0,
            "interference_a": 1.0,
            "interference_b": 1.0,
        },
        {
            "variant": "gossip",
            "seed": 0,
            "composition": "zero",
            "route": "reject_or_downweight",
            "eval_a": 1.0,
            "eval_b": 1.0,
            "eval_c": 1.0,
            "learning_b": 0.0,
            "learning_c": 0.0,
            "accretion_a": 0.0,
            "b_scale": 0.5,
            "c_scale": 0.0,
            "interference_a": 0.0,
            "interference_b": 0.0,
        },
    ])

    assert select_b_candidate(b_candidates)["b_scale"] == 0.5
    assert select_c_candidate(c_candidates)["c_scale"] == 0.0
