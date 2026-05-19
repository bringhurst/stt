# Surface Tension Transformers

Minimal, CPU-friendly experiments for testing whether small geometric constraints improve Transformer representation geometry.

This repo intentionally starts tiny: a synthetic next-token task, a small Transformer encoder, and measurable regularizers for attention head diversity, representation repulsion, and sparse activations.

Current finding: LoRA fine-tuning `Qwen/Qwen2.5-0.5B` with representation repulsion improves held-out representation geometry on WikiText-2. In the confirmed run, `repulsion=2.0` improved effective rank by about `+22.8%` and isotropy by about `-36.7%` with about `+3.1%` eval loss. Continual-learning results are preliminary and currently show small stability gains on the synthetic conflict task.

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

Run gossip self-stabilization with sampled thresholded anti-consensus pressure:

```bash
poetry run stt-lora \
  --model Qwen/Qwen2.5-0.5B \
  --device auto \
  --steps 300 \
  --max-length 128 \
  --batch-size 1 \
  --eval-batches 32 \
  --grad-accum 4 \
  --learning-rate 2e-4 \
  --variants baseline repulsion gossip \
  --sweep gossip=0.5,1.0,2.0 \
  --gossip-tau 0.85 \
  --gossip-k 8 \
  --max-gossip-vectors 256 \
  --seeds 0 1 2 \
  --text-file data/wikitext2_corpus.txt \
  --output-dir runs
```

The current sweep parser supports one swept parameter at a time. Use fixed overrides like `--gossip-tau`, `--gossip-k`, and `--max-gossip-vectors` for the other gossip settings.

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

See `docs/continual-tasks.md` for task-pair details.

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

See `docs/experiment-design.md` for the current research framing, metric interpretation, and Apple Silicon notes.

This is not intended to prove the full STT thesis. It is a first measurable scaffold: dream big, measure tiny.
