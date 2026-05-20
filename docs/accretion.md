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

All three ladder conditions below use matched seeds `0 1 2 3 4 5`, `Qwen/Qwen2.5-0.5B`, `--phase-steps 150`, `--repulsion-weight 2.0`, `--gossip-weight 5.0`, `--gossip-tau 0.5`, `--gossip-k 8`, `--max-gossip-vectors 256`, and compatibility metrics.

| B condition | Run | Variant | `accretion_a_after_b` | paired accretion delta | `lora_cosine_a_b_mean` | paired A-B cosine delta | `retention_a_after_c` | `learning_b` | `learning_c` |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B_related` | `runs/20260520T053747223341Z/results.json` | baseline | `-0.0637` | n/a | `0.0818` | n/a | `0.6494` | `2.0445` | `1.4755` |
| `B_related` | `runs/20260520T053747223341Z/results.json` | gossip | `-0.0623` | `+0.0014` | `0.0895` | `+0.0077` | `0.7171` | `1.9551` | `1.5587` |
| `B_related` | `runs/20260520T053747223341Z/results.json` | repulsion | `-0.1217` | `-0.0580` | `0.0645` | `-0.0173` | `0.6862` | `2.0921` | `1.9322` |
| `B_related_strong` | `runs/20260520T044944778425Z/results.json` | baseline | `-0.1029` | n/a | `0.0841` | n/a | `0.6169` | `1.6873` | `1.4291` |
| `B_related_strong` | `runs/20260520T044944778425Z/results.json` | gossip | `-0.0787` | `+0.0243` | `0.0884` | `+0.0043` | `0.6634` | `1.6553` | `1.5630` |
| `B_related_strong` | `runs/20260520T044944778425Z/results.json` | repulsion | `-0.2156` | `-0.1127` | `0.0580` | `-0.0261` | `0.5400` | `1.7530` | `1.9061` |
| `B_rehearsal` | `runs/20260520T063040455096Z/results.json` | baseline | `+0.1514` | n/a | `0.0270` | n/a | `0.6488` | `1.9557` | `1.9027` |
| `B_rehearsal` | `runs/20260520T063040455096Z/results.json` | gossip | `+0.1576` | `+0.0063` | `0.0334` | `+0.0064` | `0.6519` | `1.9360` | `2.1024` |
| `B_rehearsal` | `runs/20260520T063040455096Z/results.json` | repulsion | `+0.1437` | `-0.0077` | `0.0210` | `-0.0060` | `0.6123` | `2.0641` | `2.2103` |

Interpretation:

- `B_related` is near-neutral rather than positive. Gossip raises A-B LoRA cosine but barely changes mean accretion; fixed repulsion lowers A-B LoRA cosine and worsens accretion.
- `B_related_strong` is the middle condition. Gossip improves both accretion and A-B LoRA cosine; fixed repulsion worsens both.
- `B_rehearsal` is the positive control. It verifies the scaffold detects expected accretion, and gossip modestly improves both accretion and A-B LoRA cosine.
- Across the matched ladder, A-B LoRA cosine is more consistent with accretion behavior than optional gradient cosine metrics, which remain noisy and do not cleanly track transfer improvements.

This is not adapter routing or compaction. It is only the measurement scaffold needed before those mechanisms are worth implementing.
