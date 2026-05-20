# Accretion Experiments

`stt-accretion` is the first A-to-B-to-C compatibility scaffold. It tests whether related later learning can improve an earlier task, and whether conflicting later learning damages earlier tasks.

The sequence is:

```text
Task A: base entity facts
Task B: related guild rules that reinforce A through latent groups
Task B strong: answer-compatible related reminders without exact A-line rehearsal
Task B rehearsal: exact A fact rehearsal plus related guild context
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

Compatibility metrics are reported alongside transfer metrics:

- `lora_cosine_a_b_mean`: mean per-layer cosine between effective LoRA increments from phase A and phase B.
- `lora_cosine_a_c_mean`: mean per-layer cosine between phase A and phase C increments.
- `lora_cosine_b_c_mean`: mean per-layer cosine between phase B and phase C increments.
- `grad_cosine_a_b_after_a`: optional task-gradient cosine for A versus B after phase A.
- `grad_cosine_a_c_after_b`: optional task-gradient cosine for A versus C after phase B.

LoRA cosine metrics are always computed. Gradient cosine metrics are disabled by default; pass `--compat-batches N` to compute them from `N` train batches per task.

Generate deterministic task files:

```bash
poetry run python -m stt.accretion_data \
  --output-dir data \
  --num-entities 256 \
  --seed 0
```

The generator writes three B conditions:

- `data/accretion_task_b_related.txt`: related/schema-compatible facts without exact A-line rehearsal.
- `data/accretion_task_b_related_strong.txt`: stronger answer-compatible related reminders without exact A-line rehearsal.
- `data/accretion_task_b_rehearsal.txt`: positive-control B condition that includes the exact A fact plus related context.

Smoke test:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 poetry run stt-accretion \
  --model sshleifer/tiny-gpt2 \
  --device cpu \
  --phase-steps 2 \
  --max-length 64 \
  --batch-size 1 \
  --eval-batches 2 \
  --compat-batches 1 \
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

Middle-condition strong related run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 poetry run stt-accretion \
  --model Qwen/Qwen2.5-0.5B \
  --device auto \
  --phase-steps 150 \
  --max-length 128 \
  --batch-size 1 \
  --eval-batches 8 \
  --compat-batches 1 \
  --grad-accum 4 \
  --learning-rate 2e-4 \
  --variants baseline repulsion gossip \
  --repulsion-weight 2.0 \
  --gossip-weight 5.0 \
  --gossip-tau 0.5 \
  --gossip-k 8 \
  --max-gossip-vectors 256 \
  --seeds 0 1 2 \
  --task-a-file data/accretion_task_a.txt \
  --task-b-file data/accretion_task_b_related_strong.txt \
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

Positive-control rehearsal run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 poetry run stt-accretion \
  --model Qwen/Qwen2.5-0.5B \
  --device auto \
  --phase-steps 150 \
  --max-length 128 \
  --batch-size 1 \
  --eval-batches 8 \
  --compat-batches 2 \
  --grad-accum 4 \
  --learning-rate 2e-4 \
  --variants baseline repulsion gossip \
  --repulsion-weight 2.0 \
  --gossip-weight 5.0 \
  --gossip-tau 0.5 \
  --gossip-k 8 \
  --max-gossip-vectors 256 \
  --seeds 0 1 2 \
  --task-a-file data/accretion_task_a.txt \
  --task-b-file data/accretion_task_b_rehearsal.txt \
  --task-c-file data/accretion_task_c_conflict.txt \
  --output-dir runs
```

Analyze:

```bash
poetry run stt-analyze runs/<timestamp>/results.json
```

Current Qwen findings:

- `B_related` is a clean semantic/schema-compatible condition. With the revised text shape, it is near-neutral rather than positive: baseline `accretion_a_after_b=-0.0637`, gossip `-0.0623`, repulsion `-0.1217` over seeds `0 1 2 3 4 5` in `runs/20260520T053747223341Z/results.json`.
- `B_related_strong` is the middle condition: it repeats an answer-compatible verification but does not include the exact A line. Baseline `accretion_a_after_b=-0.1029`, gossip `-0.0787`, repulsion `-0.2156` over seeds `0 1 2 3 4 5` in `runs/20260520T044944778425Z/results.json`.
- `B_rehearsal` is the positive-control condition. It verifies the scaffold detects expected accretion: baseline `accretion_a_after_b=+0.1514`, gossip `+0.1576`, repulsion `+0.1437` over seeds `0 1 2 3 4 5` in `runs/20260520T021958288119Z/results.json`.
- In the 6-seed rehearsal condition, gossip improved paired-seed `accretion_a_after_b` by `+0.0063` absolute versus baseline, while fixed repulsion changed it by `-0.0077`.
- Compatibility metrics support the A-B adapter-alignment story more than the gradient-alignment story. In the 6-seed `B_related` confirmation, gossip increased paired-seed `lora_cosine_a_b_mean` by `+0.0077` absolute while barely changing `accretion_a_after_b` by `+0.0014`; repulsion lowered A-B LoRA cosine by `-0.0173` and worsened accretion by `-0.0580`. Gradient cosines were noisier and did not cleanly track transfer improvements.
- On the 6-seed `B_related_strong` confirmation, gossip increased paired-seed `lora_cosine_a_b_mean` by `+0.0043` absolute and improved `accretion_a_after_b` by `+0.0243`; repulsion lowered A-B LoRA cosine by `-0.0261` and worsened accretion by `-0.1127`.

This is not adapter routing or compaction. It is only the measurement scaffold needed before those mechanisms are worth implementing.
