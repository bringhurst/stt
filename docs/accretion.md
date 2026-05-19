# Accretion Experiments

`stt-accretion` is the first A-to-B-to-C compatibility scaffold. It tests whether related later learning can improve an earlier task, and whether conflicting later learning damages earlier tasks.

The sequence is:

```text
Task A: base entity facts
Task B: related guild rules that reinforce A through latent groups
Task C: conflicting facts for the same entity IDs
```

A useful accretion signal is:

```text
eval_a_after_b < eval_a_after_a
accretion_a_after_b > 0
```

A useful interference-resistance signal is:

```text
interference_a_after_c = eval_a_after_c - eval_a_after_b
interference_b_after_c = eval_b_after_c - eval_b_after_b
```

Lower interference is better, but only if `learning_c` remains close to baseline. Preserving A by refusing to learn C is not a win.

Generate deterministic task files:

```bash
poetry run python -m stt.accretion_data \
  --output-dir data \
  --num-entities 256 \
  --seed 0
```

Smoke test:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 poetry run stt-accretion \
  --model sshleifer/tiny-gpt2 \
  --device cpu \
  --phase-steps 2 \
  --max-length 64 \
  --batch-size 1 \
  --eval-batches 2 \
  --grad-accum 1 \
  --learning-rate 2e-4 \
  --variants baseline gossip \
  --sweep gossip=1.0 \
  --gossip-tau 0.5 \
  --gossip-k 4 \
  --max-gossip-vectors 64 \
  --seeds 0 \
  --task-a-file data/accretion_task_a.txt \
  --task-b-file data/accretion_task_b_related.txt \
  --task-c-file data/accretion_task_c_conflict.txt \
  --output-dir runs
```

Meaningful first run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 poetry run stt-accretion \
  --model Qwen/Qwen2.5-0.5B \
  --device auto \
  --phase-steps 150 \
  --max-length 128 \
  --batch-size 1 \
  --eval-batches 16 \
  --grad-accum 4 \
  --learning-rate 2e-4 \
  --variants baseline repulsion gossip \
  --sweep gossip=5.0 \
  --repulsion-weight 2.0 \
  --gossip-tau 0.5 \
  --gossip-k 8 \
  --max-gossip-vectors 256 \
  --seeds 0 1 2 \
  --task-a-file data/accretion_task_a.txt \
  --task-b-file data/accretion_task_b_related.txt \
  --task-c-file data/accretion_task_c_conflict.txt \
  --output-dir runs
```

Analyze:

```bash
poetry run stt-analyze runs/<timestamp>/results.json
```

This is not adapter routing or compaction. It is only the measurement scaffold needed before those mechanisms are worth implementing.
