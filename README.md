# Surface Tension Transformers

Minimal, CPU-friendly experiments for testing whether small geometric constraints improve Transformer representation geometry.

This repo intentionally starts tiny: a synthetic next-token task, a small Transformer encoder, and measurable regularizers for attention head diversity, representation repulsion, and sparse activations.

Current finding: LoRA fine-tuning `Qwen/Qwen2.5-0.5B` with representation repulsion improves held-out representation geometry on WikiText-2. In the confirmed run, `repulsion=2.0` improved effective rank by about `+22.8%` and isotropy by about `-36.7%` with about `+3.1%` eval loss. Sampled gossip self-stabilization at `tau=0.5`, `gossip=5` now matches most of the short-run geometry and conflict-task stability gain with lower language-model and task-B penalties.

## Setup

```bash
poetry install
```

## Run Tests

```bash
poetry run pytest
```

## Code Quality

```bash
poetry run ruff check .
poetry run ty check
```

## Run Experiments

```bash
poetry run stt-experiment --steps 80 --variants baseline diversity repulsion sparse combined
```

## Run LoRA Experiments

Use the tiny GPT-2 checkpoint for a quick wiring smoke test:

```bash
poetry run stt-lora --model sshleifer/tiny-gpt2 --steps 5 --variants baseline combined
```

Use Qwen 0.5B for the first meaningful small-model run on Apple Silicon:

```bash
poetry run stt-lora \
  --model Qwen/Qwen2.5-0.5B \
  --device auto \
  --steps 100 \
  --max-length 128 \
  --batch-size 1 \
  --grad-accum 8 \
  --variants baseline diversity combined
```

For dose-response checks, override regularizer weights explicitly:

```bash
poetry run stt-lora \
  --model Qwen/Qwen2.5-0.5B \
  --steps 20 \
  --variants repulsion \
  --repulsion-weight 1.0
```

Run gossip self-stabilization with sampled thresholded anti-consensus pressure. The current useful setting is lower-threshold and stronger-weighted than the default because `tau=0.85` produced a very small raw loss on Qwen hidden states.

```bash
poetry run stt-lora \
  --model Qwen/Qwen2.5-0.5B \
  --device auto \
  --steps 100 \
  --max-length 96 \
  --batch-size 1 \
  --eval-batches 12 \
  --grad-accum 4 \
  --learning-rate 2e-4 \
  --variants baseline repulsion gossip \
  --sweep gossip=5.0 \
  --repulsion-weight 2.0 \
  --gossip-tau 0.5 \
  --gossip-k 8 \
  --max-gossip-vectors 192 \
  --seeds 0 1 2 \
  --text-file data/wikitext2_corpus.txt \
  --output-dir runs
```

The current sweep parser supports one swept parameter at a time. Use fixed overrides like `--gossip-tau`, `--gossip-k`, and `--max-gossip-vectors` for the other gossip settings.

In the 100-step, 3-seed WikiText check, `gossip tau=0.5 weight=5` improved effective rank by `+4.67%` and isotropy by `-11.21%` with `+1.73%` eval loss. Fixed `repulsion=2.0` improved effective rank by `+5.96%` and isotropy by `-12.84%` with `+3.45%` eval loss.

Run multi-seed sweeps and persist results:

```bash
poetry run stt-lora \
  --model Qwen/Qwen2.5-0.5B \
  --steps 100 \
  --eval-batches 16 \
  --seeds 0 1 2 \
  --variants baseline repulsion \
  --sweep repulsion=0,0.1,0.3,1.0 \
  --output-dir runs
```

Analyze a saved run:

```bash
poetry run stt-analyze runs/<timestamp>/results.json
```

Analyze multiple continual runs as one paired seed table:

```bash
poetry run stt-analyze \
  runs/<timestamp-a>/results.json \
  runs/<timestamp-b>/results.json
```

Confirmed WikiText geometry command:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 poetry run stt-lora \
  --model Qwen/Qwen2.5-0.5B \
  --device auto \
  --steps 300 \
  --max-length 128 \
  --batch-size 1 \
  --eval-batches 32 \
  --grad-accum 4 \
  --learning-rate 2e-4 \
  --variants baseline repulsion \
  --sweep repulsion=1.5,2.0,2.5 \
  --seeds 0 1 2 3 4 \
  --text-file data/wikitext2_corpus.txt \
  --output-dir runs
```

## Run Continual-Learning Experiments

Train one LoRA adapter on task A, then continue training on task B and measure task-A backward transfer:

```bash
poetry run stt-continual \
  --model Qwen/Qwen2.5-0.5B \
  --device auto \
  --phase-steps 150 \
  --max-length 128 \
  --batch-size 1 \
  --eval-batches 16 \
  --grad-accum 4 \
  --variants baseline repulsion \
  --sweep repulsion=1.5,2.0 \
  --seeds 0 1 2 \
  --task-a-file data/wikitext2_task_a.txt \
  --task-b-file data/wikitext2_task_b.txt \
  --output-dir runs
```

For a stronger interference test, use the synthetic conflicting-facts pair:

```bash
poetry run stt-continual \
  --model Qwen/Qwen2.5-0.5B \
  --device auto \
  --phase-steps 150 \
  --max-length 128 \
  --batch-size 1 \
  --eval-batches 16 \
  --grad-accum 4 \
  --variants baseline repulsion \
  --sweep repulsion=1.0,1.5,2.0 \
  --seeds 0 1 2 \
  --task-a-file data/conflict_task_a.txt \
  --task-b-file data/conflict_task_b.txt \
  --output-dir runs
```

Current conflict-task result: `gossip tau=0.5 weight=5` improved `backward_transfer_a` more reliably than fixed `repulsion=2.0` while preserving B-task learning. Over seeds `0 1 2 3 4 5` with `phase_steps=150`, gossip improved `backward_transfer_a` by about `-12.25%`, changed `learning_b` by about `-0.06%`, and changed `eval_b_after_b` by about `+0.50%`; repulsion changed `backward_transfer_a` by about `+2.95%`, `learning_b` by about `-0.67%`, and `eval_b_after_b` by about `+5.86%`.

Small neighborhood checks around the current gossip setting found weaker results for `tau=0.4` and `gossip_weight=7`, so `tau=0.5`, `gossip_weight=5` remains the current best setting.

See `docs/continual-tasks.md` for task-pair details.

The repo also includes a second conflict family, `data/conflict2_task_a.txt` and `data/conflict2_task_b.txt`, that conflicts on numeric quotas, routes, permissions, windows, and cause/action rules instead of profile attributes. Use it to test whether the current gossip result transfers beyond the first synthetic template.

Combined 6-seed conflict2 results show partial transfer: gossip improved `backward_transfer_a` by `-6.74%` with `learning_b +0.04%`, while fixed repulsion improved `backward_transfer_a` by `-5.40%` with `learning_b +0.05%`. This is positive for transfer, but not a clean gossip-over-repulsion win because fixed repulsion had slightly better `eval_b_after_b` and `retention_ratio`.

## Run Accretion Experiments

Generate A/B/C accretion task files:

```bash
poetry run python -m stt.accretion_data --output-dir data --num-entities 256 --seed 0
```

Run the A→B_related→C_conflict scaffold. The generator also writes
`data/accretion_task_b_rehearsal.txt` as a positive-control B condition with exact
A fact rehearsal plus related context.

```bash
poetry run stt-accretion \
  --model sshleifer/tiny-gpt2 \
  --device cpu \
  --phase-steps 2 \
  --max-length 64 \
  --batch-size 1 \
  --eval-batches 2 \
  --grad-accum 1 \
  --variants baseline gossip \
  --sweep gossip=1.0 \
  --gossip-tau 0.5 \
  --task-a-file data/accretion_task_a.txt \
  --task-b-file data/accretion_task_b_related.txt \
  --task-c-file data/accretion_task_c_conflict.txt \
  --output-dir runs
```

See `docs/accretion.md` for metric interpretation and the Qwen command.

Current Qwen accretion status: `B_related` is near-neutral and shows gossip preserving A better than baseline, while `B_rehearsal` is the positive-control condition and produces positive baseline accretion. In the 3-seed rehearsal run, baseline `accretion_a_after_b=+0.1536`, gossip `+0.1718`, and repulsion `+0.1476`.

`forgetting_a` is still emitted as a compatibility alias for `backward_transfer_a`.

The command prints final task loss plus representation metrics:

- `head_similarity`: lower means attention heads are less redundant.
- `effective_rank`: higher means hidden states use more dimensions.
- `isotropy`: lower means less directional collapse.
- `active_fraction`: lower means sparser activations.
- `eval_diversity_loss`, `eval_repulsion_loss`, and `eval_sparse_loss`: raw unweighted STT components for checking whether a regularizer has a useful scale.
- `eval_gossip_loss`: raw sampled thresholded gossip loss.

## Design

The code is split into small modules:

- `stt.model`: tiny Transformer with returned attention maps and hidden states.
- `stt.losses`: STT regularizers.
- `stt.metrics`: geometry metrics.
- `stt.data`: deterministic synthetic sequence task.
- `stt.experiment`: training loop and CLI.
- `stt.lora_experiment`: LoRA fine-tuning CLI for pretrained causal LMs.
- `stt.analyze`: baseline-relative summaries for persisted LoRA experiment records.
- `stt.continual`: sequential A-then-B LoRA continual-learning experiments.
- `stt.accretion`: sequential A-then-B-then-C compatibility experiments.

See `docs/experiment-design.md` for the current research framing, metric interpretation, and Apple Silicon notes.

This is not intended to prove the full STT thesis. It is a first measurable scaffold: dream big, measure tiny.
