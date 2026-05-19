# Continual Learning Task Pairs

This repo currently includes three A/B task pairs for `stt-continual`.

## WikiText Split

Files:

- `data/wikitext2_task_a.txt`
- `data/wikitext2_task_b.txt`

These are deterministic halves of the cleaned WikiText-2 corpus. They are useful for checking broad domain adaptation, but they are homogeneous enough that training on task B can improve task A. That means they may not create measurable forgetting.

## Conflicting Facts

Files:

- `data/conflict_task_a.txt`
- `data/conflict_task_b.txt`

These files are synthetic. They reuse the same entity identifiers across both tasks but assign different colors, stations, roles, and token objects in task B.

Example structure:

```text
Task A: Entity-0007 has color orange, station Helio, role engineer, object engine.
Task B: Entity-0007 has color cyan, station Kestrel, role judge, object gavel.
```

This is designed to create direct interference. A baseline LoRA adapter should be more likely to overwrite task-A mappings while learning task-B mappings.

For this task pair, lower `backward_transfer_a` is better, but only if `learning_b` remains close to baseline. Negative `backward_transfer_a` means task B improved task A. If a regularizer improves task A only by preventing task-B learning, that is stability without enough plasticity.

Initial 5-seed confirmation on this pair showed a modest signal around `repulsion=2.0` to `2.25`:

```text
repulsion=2.0:  backward_transfer_a about -2.3% vs baseline, learning_b about -0.6%
repulsion=2.25: backward_transfer_a about -3.9% vs baseline, learning_b about -0.8%
```

The effect is small relative to seed variance, so paired seed deltas from `stt-analyze` should be preferred over unpaired mean comparisons.

Use multi-file analysis when a seed set is split across runs:

```bash
poetry run stt-analyze \
  runs/<seeds-0-2>/results.json \
  runs/<seeds-3-5>/results.json
```

## Suggested First Conflict Run

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 poetry run stt-continual \
  --model Qwen/Qwen2.5-0.5B \
  --device auto \
  --phase-steps 150 \
  --max-length 128 \
  --batch-size 1 \
  --eval-batches 16 \
  --grad-accum 4 \
  --learning-rate 2e-4 \
  --variants baseline repulsion \
  --sweep repulsion=1.0,1.5,2.0 \
  --seeds 0 1 2 \
  --task-a-file data/conflict_task_a.txt \
  --task-b-file data/conflict_task_b.txt \
  --output-dir runs \
  | tee runs/qwen-continual-conflict.log
```

For a narrower confirmation around the current best region:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 poetry run stt-continual \
  --model Qwen/Qwen2.5-0.5B \
  --device auto \
  --phase-steps 150 \
  --max-length 128 \
  --batch-size 1 \
  --eval-batches 16 \
  --grad-accum 4 \
  --learning-rate 2e-4 \
  --variants baseline repulsion \
  --sweep repulsion=1.75,2.0,2.25 \
  --seeds 0 1 2 3 4 \
  --task-a-file data/conflict_task_a.txt \
  --task-b-file data/conflict_task_b.txt \
  --output-dir runs \
  | tee runs/qwen-continual-conflict-confirm.log
```

For the current best gossip comparison on the conflict task:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 poetry run stt-continual \
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
  --task-a-file data/conflict_task_a.txt \
  --task-b-file data/conflict_task_b.txt \
  --output-dir runs \
  | tee runs/qwen-continual-conflict-gossip.log
```

This compares baseline, fixed repulsion, and thresholded gossip under the same task split. The desired gossip behavior is similar or better `backward_transfer_a` than fixed repulsion with a smaller `eval_b_after_b` penalty.

The 100-step, 3-seed short run showed exactly that pattern:

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

Paired seed deltas showed gossip improved `backward_transfer_a` on every seed while hurting B-task learning less than fixed repulsion.

The longer 150-step, 3-seed confirmation strengthened the result:

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

For the 150-step run, gossip improved paired `backward_transfer_a` on every seed while fixed repulsion was mixed.

Repeating the same command on seeds `3 4 5` replicated the result direction. The combined 6-seed aggregate is:

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

An initial `gossip_tau=0.4`, `gossip_weight=5` check on seeds `0 1 2` was weaker than `tau=0.5`: `backward_transfer_a -9.34%`, `learning_b -0.18%`, and `eval_b_after_b +1.56%` vs baseline. Keep `tau=0.5` as the current best threshold.

An initial stronger-weight check, `gossip_tau=0.5`, `gossip_weight=7`, was also weaker than `weight=5` on `backward_transfer_a`: `-10.67%` vs baseline with `learning_b +0.04%` and `eval_b_after_b -0.33%`. Keep `gossip_weight=5` as the current best weight.

## Conflicting Rules

Files:

- `data/conflict2_task_a.txt`
- `data/conflict2_task_b.txt`

These files reuse the same unit identifiers across tasks, but assign conflicting numeric quotas, route names, permissions, time windows, and cause/action mappings. This gives a second conflict family whose surface form differs from the color/station/role/object profile task.

Example structure:

```text
Task A: Unit-0007 route=north loop; quota=59; permission=signal; cause=static; action=inspect.
Task B: Unit-0007 route=black spur; quota=61; permission=launch; cause=frost; action=depart.
```

Use this task pair to test whether the gossip stability-plasticity result transfers beyond the original conflicting-facts template.

Recommended first replication command:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 poetry run stt-continual \
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
  --task-a-file data/conflict2_task_a.txt \
  --task-b-file data/conflict2_task_b.txt \
  --output-dir runs
```

The 6-seed result on this task family shows partial transfer. Both gossip and fixed repulsion improved `backward_transfer_a` versus baseline:

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

Paired seed deltas were less decisive than the first conflict task. Gossip deltas were `[-0.0722, +0.0096, -0.0562, +0.0008, -0.0066, -0.0502]`. Fixed repulsion deltas were `[-0.0646, -0.0395, -0.0027, +0.0483, -0.0542, -0.0274]`. This supports cross-template transfer for gossip, but not a clean win over repulsion on the second task family.
