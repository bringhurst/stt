import pytest

from stt.continual import ContinualResult, run_continual_variant, summarize_continual
from stt.experiment import Variant
from stt.lora_experiment import LoraSettings


def test_continual_smoke_run() -> None:
    pytest.importorskip("peft")
    pytest.importorskip("transformers")

    settings = LoraSettings(
        model_name="sshleifer/tiny-gpt2",
        max_length=32,
        batch_size=1,
        eval_batches=1,
        grad_accum=1,
        target_modules=("c_attn",),
    )
    result = run_continual_variant(
        Variant("baseline"),
        settings=settings,
        task_a_texts=["task a example one", "task a example two"],
        task_b_texts=["task b example one", "task b example two"],
        phase_steps=1,
        seed=0,
        device="cpu",
    )

    assert result["trainable_parameters"] > 0
    assert result["eval_a_after_a"] > 0.0
    assert result["eval_b_after_b"] > 0.0
    assert "forgetting_a" in result
    assert result["backward_transfer_a"] == result["forgetting_a"]


def test_summarize_continual_groups_metrics() -> None:
    result: ContinualResult = {
        "variant": "baseline",
        "model": "model",
        "device": "cpu",
        "seed": 0,
        "diversity_weight": 0.0,
        "repulsion_weight": 0.0,
        "sparse_weight": 0.0,
        "gossip_weight": 0.0,
        "gossip_tau": 0.85,
        "gossip_k": 8,
        "max_gossip_vectors": 256,
        "trainable_parameters": 1,
        "total_parameters": 2,
        "trainable_fraction": 0.5,
        "eval_a_before": 3.0,
        "eval_b_before": 4.0,
        "eval_a_after_a": 2.0,
        "eval_b_after_a": 4.0,
        "eval_a_after_b": 2.5,
        "eval_b_after_b": 2.0,
        "forgetting_a": 0.5,
        "backward_transfer_a": 0.5,
        "learning_b": 2.0,
        "retention_ratio": 0.8,
    }

    summary = summarize_continual([result])

    assert summary["baseline"]["count"] == 1.0
    assert summary["baseline"]["forgetting_a_mean"] == 0.5
    assert summary["baseline"]["backward_transfer_a_mean"] == 0.5
