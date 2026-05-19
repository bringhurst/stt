import pytest
import torch

from stt.experiment import Variant
from stt.lora_experiment import (
    LoraExperimentResult,
    LoraSettings,
    build_lora_model,
    build_variants,
    encode_texts,
    infer_target_modules,
    load_tokenizer,
    parameter_counts,
    parse_sweep,
    split_corpus,
    stt_loss,
    summarize_results,
    train_lora_variant,
)


def test_infer_target_modules_handles_gpt2_and_qwen() -> None:
    assert infer_target_modules("sshleifer/tiny-gpt2") == ("c_attn", "c_proj")
    assert infer_target_modules("Qwen/Qwen2.5-0.5B") == (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    )


@pytest.mark.parametrize("variant", [Variant("baseline"), Variant("combined", diversity=0.01)])
def test_lora_smoke_run_updates_only_adapter_params(variant: Variant) -> None:
    pytest.importorskip("peft")
    pytest.importorskip("transformers")

    settings = LoraSettings(
        model_name="sshleifer/tiny-gpt2",
        max_length=32,
        batch_size=1,
        grad_accum=1,
        target_modules=("c_attn",),
    )
    result = train_lora_variant(
        variant,
        settings=settings,
        steps=1,
        seed=0,
        device="cpu",
        texts=["tiny lora smoke test", "surface tension transformer test"],
    )

    assert result["trainable_parameters"] > 0
    assert result["trainable_parameters"] < result["total_parameters"]
    assert result["eval_lm_loss"] > 0.0
    assert result["effective_rank"] > 0.0


def test_parameter_counts_separates_frozen_params() -> None:
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    for parameter in model[0].parameters():
        parameter.requires_grad = False

    trainable, total = parameter_counts(model)

    assert trainable == 3
    assert total == 9


def test_split_corpus_is_seeded_and_preserves_items() -> None:
    corpus = [f"line {index}" for index in range(10)]

    train_a, eval_a = split_corpus(corpus, seed=0)
    train_b, eval_b = split_corpus(corpus, seed=0)
    train_c, eval_c = split_corpus(corpus, seed=1)

    assert train_a == train_b
    assert eval_a == eval_b
    assert train_a != train_c or eval_a != eval_c
    assert sorted(train_a + eval_a) == sorted(corpus)


def test_parse_sweep_and_build_variants() -> None:
    assert parse_sweep("repulsion=0,0.5,1") == ("repulsion", [0.0, 0.5, 1.0])

    variants = build_variants(
        ["repulsion"],
        diversity=None,
        repulsion=None,
        sparse=None,
        sweep="repulsion=0,1",
    )

    assert [variant.name for variant in variants] == [
        "repulsion_repulsion_0",
        "repulsion_repulsion_1",
    ]
    assert [variant.repulsion for variant in variants] == [0.0, 1.0]


def test_build_variants_does_not_sweep_baseline() -> None:
    variants = build_variants(
        ["baseline", "repulsion"],
        diversity=None,
        repulsion=None,
        sparse=None,
        sweep="repulsion=0,1",
    )

    assert [variant.name for variant in variants] == [
        "baseline",
        "repulsion_repulsion_0",
        "repulsion_repulsion_1",
    ]


def test_summarize_results_groups_by_variant() -> None:
    base: LoraExperimentResult = {
        "variant": "baseline",
        "model": "model",
        "device": "cpu",
        "seed": 0,
        "diversity_weight": 0.0,
        "repulsion_weight": 0.0,
        "sparse_weight": 0.0,
        "train_lm_loss": 1.0,
        "eval_lm_loss": 2.0,
        "head_similarity": 0.5,
        "effective_rank": 3.0,
        "isotropy": 4.0,
        "active_fraction": 0.9,
        "eval_diversity_loss": 0.5,
        "eval_repulsion_loss": 0.2,
        "eval_sparse_loss": 1.0,
        "trainable_parameters": 1,
        "total_parameters": 2,
        "trainable_fraction": 0.5,
    }

    second = base.copy()
    second["eval_lm_loss"] = 4.0

    summary = summarize_results([base, second])

    assert summary["baseline"]["count"] == 2.0
    assert summary["baseline"]["eval_lm_loss_mean"] == 3.0


def test_lora_training_step_changes_adapter_not_base() -> None:
    pytest.importorskip("peft")
    pytest.importorskip("transformers")

    settings = LoraSettings(
        model_name="sshleifer/tiny-gpt2",
        max_length=32,
        batch_size=1,
        grad_accum=1,
        target_modules=("c_attn",),
    )
    tokenizer = load_tokenizer(settings.model_name)
    model = build_lora_model(settings, "cpu")
    encoded = encode_texts(tokenizer, ["surface tension transformer"], settings.max_length, "cpu")
    frozen_name, frozen_before = next(
        (name, parameter.detach().clone())
        for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    )
    lora_name, lora_before = next(
        (name, parameter.detach().clone())
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "lora" in name
    )

    output = model(**encoded, output_attentions=True, output_hidden_states=True)
    loss = output.loss + stt_loss(output, Variant("combined", diversity=0.01))
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=1e-2,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    parameters = dict(model.named_parameters())
    assert torch.equal(parameters[frozen_name].detach(), frozen_before)
    assert not torch.equal(parameters[lora_name].detach(), lora_before)
