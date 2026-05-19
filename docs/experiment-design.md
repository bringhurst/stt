# Experiment Design

This repo tests a narrow version of the Surface Tension Transformer idea: add small geometric regularizers to a Transformer and measure whether representation geometry changes.

It does not test large-language-model quality, continual learning, or online cognition yet. The current goal is to make the smallest useful scaffold that is understandable, testable, and easy to extend.

## Model

The model in `src/stt/model.py` is a small Transformer encoder trained on next-token prediction.

It returns three values:

- `logits` for the normal task loss.
- `hidden` for representation metrics and repulsion/sparsity losses.
- `attention` for head-diversity metrics and losses.

## Data

`SyntheticSequenceTask` creates short modular token sequences. The data combines arithmetic progressions, local motifs, and small noise.

This gives the model learnable structure while keeping runs fast on CPU or Apple Silicon MPS.

## Variants

The CLI currently supports these variants:

- `baseline`: next-token loss only.
- `diversity`: penalizes similar attention maps between heads.
- `repulsion`: penalizes hidden vectors that are too close together.
- `sparse`: adds L1 activation pressure.
- `combined`: uses smaller weights for all three regularizers.

## LoRA Fine-Tuning

`stt-lora` applies the same STT losses to pretrained causal language models through LoRA adapters. This is the bridge from the toy Transformer scaffold to real small-model experiments.

The recommended progression is:

- `sshleifer/tiny-gpt2` for smoke tests and CI-style checks.
- `Qwen/Qwen2.5-0.5B` for the first meaningful base-model experiment.
- `Qwen/Qwen2.5-0.5B-Instruct` later as an instruction-tuned comparison.

The LoRA path reports trainable and total parameter counts so runs can verify that only adapter parameters are updated.

Use `--diversity-weight`, `--repulsion-weight`, and `--sparse-weight` for dose-response checks. A good mechanical test is whether increasing one weight moves its target metric while leaving `eval_lm_loss` within a tolerable range.

The repulsion loss normalizes hidden vectors before pairwise distance calculation. Without normalization, pretrained language-model hidden norms can make the exponential kernel underflow to zero, which hides whether the regularizer is wired correctly.

## Gossip Self-Stabilization

`gossip` is a sampled, thresholded anti-consensus loss over hidden token vectors. It samples a small peer neighborhood for each selected token vector and penalizes only cosine similarity above a threshold `tau`.

The implemented loss is:

```text
overlap = relu(cos(anchor, peer) - tau)
gossip_loss = mean(overlap ** 2)
```

The first implementation uses the final hidden state only, samples non-padding token vectors, and caps sampled vectors with `--max-gossip-vectors` for MPS memory safety.

Defaults:

```text
gossip_weight: 1.0
gossip_tau: 0.85
gossip_k: 8
max_gossip_vectors: 256
```

The goal is not maximum separation. The goal is local homeostatic repair when representations become too similar.

On `Qwen/Qwen2.5-0.5B`, `tau=0.85` produced raw gossip losses around `2e-4`, which was too small to move geometry at weights `0.5..2.0`. A zero-step calibration found more useful raw scales around `tau=0.5` (`~0.04`) and `tau=0.3` (`~0.14`). The current best short-run setting is `gossip_tau=0.5`, `gossip_weight=5`.

Use `--seeds` for repeated trials and `--output-dir runs` to persist a run record. The record includes config, git status, raw results, and mean/std summaries per variant. Use `stt-analyze` to print baseline-relative deltas and pass/fail checks against simple geometry-vs-loss criteria.

For line-based corpora passed through `--text-file`, the runner deterministically shuffles lines per seed, uses 75% for training, and keeps 25% as holdout text. Evaluation averages `--eval-batches` bounded batches from that holdout split so larger corpora do not create a single oversized MPS batch.

## Continual Learning Prototype

`stt-continual` trains one LoRA adapter sequentially on task A and then task B. It reports `backward_transfer_a` as `eval_a_after_b - eval_a_after_a`, so lower is better and negative values mean task B improved task A. The legacy alias `forgetting_a` is still emitted for compatibility. It also reports `learning_b` as `eval_b_before - eval_b_after_b`, so higher is better.

The included WikiText split files are:

- `data/wikitext2_task_a.txt`
- `data/wikitext2_task_b.txt`

These are deterministic halves of `data/wikitext2_corpus.txt`, intended as a first smoke-test split rather than a final continual-learning benchmark.

The WikiText half split produced positive backward transfer: task B often improved task A. That is useful, but not a strong interference test. The synthetic conflicting-facts task pair in `docs/continual-tasks.md` is intended to create more direct A/B interference.

A second synthetic conflict family, `data/conflict2_task_a.txt` and `data/conflict2_task_b.txt`, uses conflicting quotas, routes, permissions, windows, and cause/action rules. It is intended as the next transfer check for the current gossip setting.

## Current Results

The strongest replicated Phase 1 signal so far is representation repulsion during LoRA fine-tuning of `Qwen/Qwen2.5-0.5B` on `data/wikitext2_corpus.txt`.

Confirmed run:

```text
steps: 300
eval_batches: 32
seeds: 0 1 2 3 4
repulsion sweep: 1.5, 2.0, 2.5
```

The best tradeoff was `repulsion=2.0`:

```text
eval_lm_loss:    +3.13% vs baseline
effective_rank:  +22.81% vs baseline
isotropy:        -36.74% vs baseline
```

`repulsion=2.5` improved geometry more strongly but increased eval loss more. This makes `2.0` the current default candidate for follow-up continual-learning tests.

Gossip self-stabilization now has an initial positive short-run signal. In a 100-step, 3-seed WikiText run, `gossip tau=0.5 weight=5` got most of the fixed-repulsion geometry gain with lower eval-loss cost:

```text
gossip tau=0.5 weight=5:
  eval_lm_loss:    +1.73% vs baseline
  effective_rank:  +4.67% vs baseline
  isotropy:        -11.21% vs baseline

repulsion=2.0:
  eval_lm_loss:    +3.45% vs baseline
  effective_rank:  +5.96% vs baseline
  isotropy:        -12.84% vs baseline
```

On the synthetic conflict continual task with `phase_steps=100`, the same gossip setting matched fixed repulsion on `backward_transfer_a` improvement while preserving better B-task learning:

```text
gossip tau=0.5 weight=5:
  backward_transfer_a: -6.67% vs baseline
  learning_b:          -0.43% vs baseline
  eval_b_after_b:      +2.23% vs baseline

repulsion=2.0:
  backward_transfer_a: -6.62% vs baseline
  learning_b:          -1.53% vs baseline
  eval_b_after_b:      +8.00% vs baseline
```

The longer `phase_steps=150`, `max_length=128`, `eval_batches=16`, `max_gossip_vectors=256` confirmation strengthened the gossip result:

```text
gossip tau=0.5 weight=5:
  backward_transfer_a: -15.61% vs baseline
  learning_b:          +0.03% vs baseline
  eval_b_after_b:      -0.23% vs baseline
  retention_ratio:     +3.22% vs baseline

repulsion=2.0:
  backward_transfer_a: +1.40% vs baseline
  learning_b:          -0.52% vs baseline
  eval_b_after_b:      +4.51% vs baseline
  retention_ratio:     +0.30% vs baseline
```

Paired seed deltas for gossip improved `backward_transfer_a` on all three seeds: `[-0.0397, -0.0833, -0.0045]`. Fixed repulsion was mixed: `[-0.0378, +0.0563, -0.0070]`.

Repeating the same setting on seeds `3 4 5` replicated the pattern. Combining seeds `0 1 2 3 4 5` gives:

```text
gossip tau=0.5 weight=5:
  backward_transfer_a: -12.25% vs baseline
  learning_b:          -0.06% vs baseline
  eval_b_after_b:      +0.50% vs baseline
  retention_ratio:     +2.34% vs baseline

repulsion=2.0:
  backward_transfer_a: +2.95% vs baseline
  learning_b:          -0.67% vs baseline
  eval_b_after_b:      +5.86% vs baseline
  retention_ratio:     -0.11% vs baseline
```

A first threshold-neighborhood check with `gossip_tau=0.4`, `gossip_weight=5` on seeds `0 1 2` was still better than fixed repulsion but worse than `tau=0.5`: `backward_transfer_a -9.34%`, `learning_b -0.18%`, and `eval_b_after_b +1.56%` vs baseline. This keeps `tau=0.5`, `weight=5` as the current best setting.

A first weight-neighborhood check with `gossip_tau=0.5`, `gossip_weight=7` on seeds `0 1 2` was also worse than `weight=5` on the primary retention metric: `backward_transfer_a -10.67%`, `learning_b +0.04%`, and `eval_b_after_b -0.33%` vs baseline. It still beat fixed repulsion, but `weight=5` remains the best stability-plasticity point seen so far.

## Metrics

The experiment prints JSON with:

- `eval_task_loss`: held-out synthetic next-token loss.
- `head_similarity`: mean pairwise cosine similarity between attention heads; lower is less redundant.
- `effective_rank`: entropy-based rank of hidden states; higher means more usable dimensions.
- `isotropy`: singular-spectrum imbalance; lower means less directional collapse.
- `active_fraction`: fraction of hidden activations above a fixed magnitude threshold.

## Interpreting Results

A useful early signal would be a variant that improves one or more geometry metrics without badly increasing task loss.

These results are not proof that STT works. They are a debugging and measurement scaffold for deciding which regularizers are worth trying on larger language-model fine-tuning runs.

## Apple Silicon

The experiment defaults to `--device auto`. On a Mac with PyTorch MPS support, that selects `mps` and can use unified memory through the Metal backend.

Training runs on MPS when available. SVD-based metrics move detached hidden states to CPU because PyTorch MPS does not currently implement the needed SVD operator.

Use CPU explicitly when comparing against CI or debugging determinism:

```bash
poetry run stt-experiment --device cpu --steps 80
```

## Quality Gates

The project is configured for `pytest`, `ruff`, and `ty`:

```bash
poetry run pytest
poetry run ruff check .
poetry run ty check
```

Keep these clean before changing experiment logic so metric changes are easier to trust.
