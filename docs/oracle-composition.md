# Oracle Composition

`stt-oracle-compose` is an unfair upper-bound experiment for LoRA accretion routing. It trains one A-to-B-to-C adapter sequence, snapshots trainable LoRA parameter states at phase boundaries, then evaluates post-hoc compositions without retraining.

The goal is not to build the final router. The goal is to test whether routing would be useful if a compatibility signal existed.

The scaffold snapshots:

```text
state_initial
state_a
state_ab
state_abc
```

It computes parameter-space updates:

```text
delta_b = state_ab - state_a
delta_c = state_abc - state_ab
```

Then it evaluates scalar compositions:

```text
A only
A + alpha * delta_b
A + best_B + beta * delta_c
```

The oracle uses A/B/C eval losses to classify candidates:

```text
shared: old-task loss improves and new-task loss improves
private: old-task loss is approximately unchanged and new-task loss improves
conflict_private: old-task loss worsens and new-task loss improves
reject_or_downweight: new-task loss does not improve
```

This is intentionally unfair. Behavioral labels from old tasks are allowed because this is a routing upper bound.

Smoke test:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 poetry run stt-oracle-compose \
  --model sshleifer/tiny-gpt2 \
  --device cpu \
  --phase-steps 2 \
  --max-length 64 \
  --batch-size 1 \
  --eval-batches 2 \
  --grad-accum 1 \
  --learning-rate 2e-4 \
  --variant gossip \
  --gossip-weight 1.0 \
  --gossip-tau 0.5 \
  --gossip-k 4 \
  --max-gossip-vectors 64 \
  --b-scales 0 0.5 1.0 \
  --c-scales 0 0.5 1.0 \
  --seeds 0 \
  --task-a-file data/accretion_task_a.txt \
  --task-b-file data/accretion_task_b_related.txt \
  --task-c-file data/accretion_task_c_conflict.txt \
  --output-dir runs
```

First Qwen run:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 poetry run stt-oracle-compose \
  --model Qwen/Qwen2.5-0.5B \
  --device auto \
  --phase-steps 150 \
  --max-length 128 \
  --batch-size 1 \
  --eval-batches 8 \
  --compat-batches 1 \
  --grad-accum 4 \
  --learning-rate 2e-4 \
  --variant gossip \
  --gossip-weight 12.5 \
  --gossip-tau 0.5 \
  --gossip-k 8 \
  --max-gossip-vectors 256 \
  --b-scales 0 0.25 0.5 0.75 1.0 \
  --c-scales 0 0.25 0.5 0.75 1.0 \
  --seeds 0 1 2 \
  --task-a-file data/accretion_task_a.txt \
  --task-b-file data/accretion_task_b_related.txt \
  --task-c-file data/accretion_task_c_conflict.txt \
  --output-dir runs
```

Primary readout:

- `selected_b_scale`: oracle-selected B sharing scale.
- `selected_c_scale`: oracle-selected C sharing scale; `0` means C should stay private or rejected.
- `oracle_accretion_a`: A preservation or improvement after routed composition.
- `oracle_learning_b`: B learning retained by the routed composition.
- `oracle_learning_c`: C learning retained by the routed composition.
- `oracle_interference_a`: A damage from selected C relative to selected B-only composition.
- `oracle_interference_b`: B damage from selected C relative to selected B-only composition.

Interpretation rules:

- If partial B improves or preserves A while learning B, there is reusable B capacity worth routing.
- If C candidates usually route as `conflict_private` or select `c_scale=0`, C updates contain damaging directions that should be isolated.
- If oracle routing cannot beat blind sequential A/B/C, learned routing is premature.
- If scalar routing works, the next experiment is layerwise or modulewise routing.
