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

## Compartmentalized FFN Prototype

`TinyTransformer` can optionally replace each standard feed-forward block with a routed compartmentalized
FFN. This is a toy test for the hypothesis that intra-layer branch routing can act like small dendritic
compartments inside a Transformer MLP.

CLI flags:

```bash
poetry run stt-experiment \
  --steps 80 \
  --variants baseline combined \
  --device cpu \
  --compartments 4 \
  --compartment-mode router \
  --compartment-top-k 1 \
  --branch-repulsion-weight 0.01 \
  --branch-load-balance-weight 0.01
```

The explicit-router mode uses one central branch router. The dendritic mode removes that central router:
each branch emits a local spike score, similar branch outputs inhibit one another, and the soma merges the
top active branches.

Dendritic command:

```bash
poetry run stt-experiment \
  --steps 200 \
  --variants baseline combined \
  --device cpu \
  --compartments 4 \
  --compartment-mode dendritic \
  --compartment-top-k 2 \
  --branch-repulsion-weight 0.01 \
  --branch-load-balance-weight 0.05 \
  --branch-inhibition-strength 0.5 \
  --branch-inhibition-weight 0.01
```

The compartment path tracks:

- `branch_entropy`: normalized entropy of average branch usage; high means no collapse.
- `branch_active_fraction`: active gate fraction; for top-1 over four branches this should be `0.25`.
- `branch_usage_min`, `branch_usage_max`, `branch_usage_std`: load balance diagnostics.
- `branch_score_entropy`: normalized entropy of pre-top-k score probabilities.
- `branch_inhibition_mean`: mean dendritic inhibition magnitude before top-k selection.
- `branch_repulsion_loss`: off-diagonal branch-output cosine similarity.
- `branch_load_balance_loss`: squared deviation from uniform average usage.
- `branch_inhibition_loss`: positively correlated co-active branch outputs.

First paired toy comparison:

```text
steps=80
seeds=0 1 2 3 4
variants=baseline,combined
dense FFN vs 4 top-1 compartments with branch repulsion/load-balance 0.01
```

Mean results:

| Condition | Variant | Eval loss | Effective rank | Isotropy | Branch entropy | Usage min/max |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| dense | baseline | `0.6351` | `32.30` | `3.97` | n/a | n/a |
| compartments | baseline | `0.6822` | `36.38` | `3.38` | `0.9858` | `0.188/0.307` |
| dense | combined | `0.7013` | `32.02` | `4.10` | n/a | n/a |
| compartments | combined | `0.7945` | `36.58` | `3.43` | `0.9836` | `0.184/0.309` |

Longer convergence check:

```text
steps=200
seeds=0 1 2
```

Mean results:

| Condition | Variant | Eval loss | Effective rank | Isotropy | Branch entropy | Usage min/max |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| dense | baseline | `0.0750` | `31.95` | `3.41` | n/a | n/a |
| compartments | baseline | `0.0798` | `35.05` | `3.05` | `0.9855` | `0.191/0.310` |
| dense | combined | `0.0761` | `31.93` | `3.39` | n/a | n/a |
| compartments | combined | `0.0804` | `35.39` | `2.97` | `0.9841` | `0.189/0.312` |

Interpretation:

- The prototype is mechanically viable: top-1 routing emits exactly `0.25` active branch fraction and branch entropy stays near `0.98`, so no immediate branch collapse.
- Compartments consistently improve geometry on the toy task: effective rank is about `+10%` to `+14%`, and isotropy improves by about `10%` to `16%`.
- The cost is a small eval-loss penalty. At 200 steps the penalty is much smaller than at 80 steps, suggesting slower convergence rather than total failure.
- This is not yet evidence for continual-learning benefit. The next useful test is an A/B/C toy continual or accretion scaffold that compares dense vs compartment FFNs on retention and interference.

First dendritic comparison:

```text
steps=80
seeds=0 1 2 3 4
dense vs explicit-router top-1 vs dendritic top-1
```

Mean results:

| Condition | Variant | Eval loss | Effective rank | Isotropy | Branch entropy | Usage min/max |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| dense | baseline | `0.6351` | `32.30` | `3.97` | n/a | n/a |
| router top-1 | baseline | `0.6822` | `36.38` | `3.38` | `0.9858` | `0.188/0.307` |
| dendritic top-1 | baseline | `2.4899` | `35.33` | `4.24` | `0.4928` | `0.033/0.623` |
| dense | combined | `0.7013` | `32.02` | `4.10` | n/a | n/a |
| router top-1 | combined | `0.7945` | `36.58` | `3.43` | `0.9836` | `0.184/0.309` |
| dendritic top-1 | combined | `2.4492` | `35.77` | `4.02` | `0.4585` | `0.001/0.651` |

Interpretation:

- Naive dendritic top-1 is a negative result. Without a central router, local spike scores collapsed onto a few branches and training loss stayed high.
- Stronger load balance fixes entropy for top-1 but still trails the explicit-router loss curve.
- Top-2 dendritic competition is the better minimal dendritic setting because inhibition has co-active branches to act on.

Stabilized dendritic top-2 check:

```text
branches=4
top_k=2
branch_repulsion_weight=0.01
branch_load_balance_weight=0.05
branch_inhibition_strength=0.5
branch_inhibition_weight=0.01
```

At 80 steps with seeds `0 1 2`:

| Condition | Variant | Eval loss | Effective rank | Isotropy | Branch entropy | Usage min/max |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| dendritic top-2 | baseline | `0.7157` | `35.53` | `3.57` | `0.9776` | `0.160/0.321` |
| dendritic top-2 | combined | `0.8149` | `35.68` | `3.61` | `0.9791` | `0.164/0.320` |

At 200 steps with seeds `0 1 2`:

| Condition | Variant | Eval loss | Effective rank | Isotropy | Branch entropy | Usage min/max |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| dense | baseline | `0.0750` | `31.95` | `3.41` | n/a | n/a |
| dendritic top-2 | baseline | `0.0744` | `35.07` | `2.98` | `0.9950` | `0.213/0.284` |
| dense | combined | `0.0761` | `31.93` | `3.39` | n/a | n/a |
| dendritic top-2 | combined | `0.0779` | `35.40` | `2.93` | `0.9944` | `0.209/0.283` |

Interpretation:

- Dendritic top-2 is the first positive dendritic-surface signal. It matches dense eval loss by 200 steps while improving effective rank and isotropy.
- Branch usage is more balanced than explicit-router top-1 at convergence, with entropy near `0.995`.
- This supports continuing with dendritic top-2 for the toy continual/accretion test rather than scaling explicit-router compartments first.

## Toy Accretion

`stt-toy-accretion` trains one `TinyTransformer` sequentially on three marked synthetic tasks:

- A uses marker token `0` and the base next-token target.
- B uses marker token `1` and the same target rule on a different sample stream.
- C uses marker token `2` and a large target offset, making it a stronger conflict task.

The fixed eval seeds keep A/B/C losses comparable across phases. The runner reports A-after-B accretion, A/B-after-C interference, retention ratios, geometry metrics, and branch diagnostics after the final C phase.

Dense smoke:

```bash
poetry run stt-toy-accretion \
  --phase-steps 80 \
  --variants baseline \
  --device cpu \
  --summary
```

Dendritic top-2 smoke:

```bash
poetry run stt-toy-accretion \
  --phase-steps 80 \
  --variants baseline \
  --device cpu \
  --compartments 4 \
  --compartment-mode dendritic \
  --compartment-top-k 2 \
  --branch-repulsion-weight 0.01 \
  --branch-load-balance-weight 0.05 \
  --branch-inhibition-strength 0.5 \
  --branch-inhibition-weight 0.01 \
  --summary
```

Use this before any Qwen compartment work. It is cheap enough to run paired dense/router/dendritic seeds and directly tests whether the branch geometry helps retain earlier task behavior.

First toy accretion comparison:

```text
phase_steps=80
seeds=0 1 2
variant=baseline
```

Mean results:

| Condition | A after B | A after C | B after C | C after C | A accretion | A retention after C | B retention after C | Rank | Isotropy | Branch entropy |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense | `0.0944` | `7.1073` | `7.1159` | `0.1093` | `0.8278` | `0.1300` | `0.0118` | `26.37` | `4.77` | n/a |
| router top-1 | `0.0921` | `6.4084` | `6.3684` | `0.1195` | `0.8254` | `0.1434` | `0.0145` | `31.93` | `3.79` | `0.9885` |
| dendritic top-2 | `0.0904` | `6.3370` | `6.3229` | `0.1204` | `0.7699` | `0.1355` | `0.0141` | `31.44` | `3.78` | `0.9762` |
| dendritic top-1 load `0.1` | `0.0969` | `6.2182` | `6.2342` | `0.1271` | `0.8804` | `0.1571` | `0.0155` | `32.68` | `3.74` | `0.9758` |

Interpretation:

- The revised toy scaffold now shows real A/B accretion before C; `accretion_a_after_b` is positive for all conditions.
- Compartments improve A/B retention after conflicting C relative to dense, while preserving better rank/isotropy.
- Dendritic top-1 with stronger load balance is currently the best retention setting in this toy scaffold, but it pays a small C-learning penalty.
- Dendritic top-2 remains a good single-task geometry setting, but it is not yet the best accretion/retention setting.

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

For continual runs split across multiple result files, pass all files to `stt-analyze`. It combines raw results, recomputes aggregate baseline-relative metrics, and emits paired seed deltas across the combined seed set.

For line-based corpora passed through `--text-file`, the runner deterministically shuffles lines per seed, uses 75% for training, and keeps 25% as holdout text. Evaluation averages `--eval-batches` bounded batches from that holdout split so larger corpora do not create a single oversized MPS batch.

## Continual Learning Prototype

`stt-continual` trains one LoRA adapter sequentially on task A and then task B. It reports `backward_transfer_a` as `eval_a_after_b - eval_a_after_a`, so lower is better and negative values mean task B improved task A. The legacy alias `forgetting_a` is still emitted for compatibility. It also reports `learning_b` as `eval_b_before - eval_b_after_b`, so higher is better.

The included WikiText split files are:

- `data/wikitext2_task_a.txt`
- `data/wikitext2_task_b.txt`

These are deterministic halves of `data/wikitext2_corpus.txt`, intended as a first smoke-test split rather than a final continual-learning benchmark.

The WikiText half split produced positive backward transfer: task B often improved task A. That is useful, but not a strong interference test. The synthetic conflicting-facts task pair in `docs/continual-tasks.md` is intended to create more direct A/B interference.

A second synthetic conflict family, `data/conflict2_task_a.txt` and `data/conflict2_task_b.txt`, uses conflicting quotas, routes, permissions, windows, and cause/action rules. It is intended as the next transfer check for the current gossip setting.

`stt-accretion` extends the continual scaffold to A→B_related→C_conflict. It records whether related B improves A (`accretion_a_after_b`) and whether conflicting C damages A/B (`interference_a_after_c`, `interference_b_after_c`). See `docs/accretion.md`.

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

The second conflict family, `data/conflict2_task_a.txt` and `data/conflict2_task_b.txt`, gave a more mixed transfer result. Across seeds `0 1 2 3 4 5`, gossip improved mean `backward_transfer_a` slightly more than fixed repulsion, while repulsion had slightly better B-task loss and retention ratio:

```text
gossip tau=0.5 weight=5:
  backward_transfer_a: -6.74% vs baseline
  learning_b:          +0.04% vs baseline
  eval_b_after_b:      -0.74% vs baseline
  retention_ratio:     +6.53% vs baseline

repulsion=2.0:
  backward_transfer_a: -5.40% vs baseline
  learning_b:          +0.05% vs baseline
  eval_b_after_b:      -0.84% vs baseline
  retention_ratio:     +7.11% vs baseline
```

This supports transfer beyond the original conflict template, but narrows the claim: gossip is not uniformly better than fixed repulsion across task families yet.

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
