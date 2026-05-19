# Continual Learning Task Pairs

This repo currently includes two A/B task pairs for `stt-continual`.

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
