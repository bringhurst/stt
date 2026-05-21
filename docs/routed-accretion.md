# Routed Accretion

`stt-routed-accretion` is the deployed fixed-route version of the oracle composition experiment. It trains one A-to-B-to-C LoRA sequence, snapshots trainable adapter state at phase boundaries, then publishes a final adapter formed by a predeclared update route instead of the blind sequential C state.

Default route:

```text
state_routed = state_a + 0.9 * (state_b - state_a) + 0.25 * (state_c - state_b)
```

This is intentionally not an oracle. The scales are fixed before the run and are evaluated against blind sequential metrics.

Qwen ladder command template:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 poetry run stt-routed-accretion \
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
  --route-b-scale 0.9 \
  --route-c-scale 0.25 \
  --seeds 0 1 2 \
  --task-a-file data/accretion_task_a.txt \
  --task-b-file data/accretion_task_b_related.txt \
  --task-c-file data/accretion_task_c_conflict.txt \
  --output-dir runs
```

Primary readout:

- `sequential_*`: blind A-to-B-to-C adapter metrics.
- `routed_*`: metrics after applying the fixed routed final state.
- `routed_accretion_win_count`: seeds where routed A accretion beats blind sequential.
- `routed_interference_a_win_count`: seeds where routed C causes less A interference.
- `routed_interference_b_win_count`: seeds where routed C causes less B interference.
- `routed_learning_c_preserved_count`: seeds where routed C learning is at least blind sequential C learning.

Analyze one or more routed runs:

```bash
poetry run stt-analyze runs/<timestamp>/results.json
```

Interpretation:

- If fixed routing beats blind sequential across the ladder, `A + 0.9B + 0.25C` becomes the baseline for learned routing.
- If it fails on specific regimes, use the failure to decide whether layerwise routing, adaptive C scale selection, or a less conservative route objective is warranted.

## First Qwen Ladder

Runs:

```text
B_related: runs/20260521T144308396837Z/results.json
B_related_strong: runs/20260521T144942693397Z/results.json
B_rehearsal: runs/20260521T145625954856Z/results.json
```

Shared condition:

```text
Qwen/Qwen2.5-0.5B
gossip_weight=12.5
route_b_scale=0.9
route_c_scale=0.25
seeds=0 1 2
```

Summary:

| Condition | Method | `accretion_a` | `interference_a` | `interference_b` | `learning_b` | `learning_c` | `eval_c` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `B_related` | Sequential | `+0.0071` | `+0.1267` | `+0.4364` | `+1.9102` | `+1.6740` | `0.1679` |
| `B_related` | Routed | `+0.1207` | `-0.1135` | `+0.0201` | `+1.8901` | `+3.0337` | `0.4865` |
| `B_related_strong` | Sequential | `+0.0076` | `+0.2370` | `+0.3477` | `+1.6814` | `+1.6244` | `0.1637` |
| `B_related_strong` | Routed | `+0.1101` | `-0.1025` | `+0.0121` | `+1.6694` | `+3.0534` | `0.4668` |
| `B_rehearsal` | Sequential | `+0.1892` | `+0.4191` | `+0.5163` | `+1.9242` | `+2.1740` | `0.1658` |
| `B_rehearsal` | Routed | `+0.1831` | `+0.0061` | `+0.0157` | `+1.9085` | `+2.7097` | `0.8105` |

Win counts:

| Condition | Accretion wins | A-interference wins | B-interference wins | C-learning preserved |
| --- | ---: | ---: | ---: | ---: |
| `B_related` | `3/3` | `3/3` | `3/3` | `3/3` |
| `B_related_strong` | `3/3` | `3/3` | `3/3` | `3/3` |
| `B_rehearsal` | `0/3` | `3/3` | `3/3` | `3/3` |

Interpretation:

- The fixed route is a strong deployed baseline. It preserves A/B far better than blind sequential while keeping C learning above the blind sequential C-learning increment on all nine seeds.
- The related conditions are clean wins: routed composition improves accretion and sharply reduces C interference.
- The rehearsal condition trades a small amount of already-strong A accretion for a large reduction in C interference and better C-learning preservation. This is acceptable for interference control but suggests rehearsal may prefer a slightly larger B scale or adaptive B scaling.
- Learned or layerwise routing should now be required to beat this fixed route, not just blind sequential.

Reproduce the summary:

```bash
poetry run stt-analyze \
  runs/20260521T144308396837Z/results.json \
  runs/20260521T144942693397Z/results.json \
  runs/20260521T145625954856Z/results.json
```
