# Routed Accretion

`stt-routed-accretion` is the deployed fixed-route version of the oracle composition experiment. It trains one A-to-B-to-C LoRA sequence, snapshots trainable adapter state at phase boundaries, then publishes a final adapter formed by a predeclared update route instead of the blind sequential C state.

Default route:

```text
state_routed = state_a + 0.9 * (state_b - state_a) + 0.25 * (state_c - state_b)
```

This is intentionally not an oracle. The scales are fixed before the run and are evaluated against blind sequential metrics.

Route sweeps evaluate several fixed pairs after one A-to-B-to-C training pass:

```bash
--route-pairs 0.9:0.15 0.9:0.25 0.9:0.35 1.0:0.15 1.0:0.25 1.0:0.35
```

Each pair is emitted as a separate variant such as `gossip_b0.9_c0.25`.

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
- `frontier_score`: balanced improvement over blind sequential: `accretion_delta + A_interference_reduction + B_interference_reduction + C_learning_delta + 0.25 * B_learning_delta`.

Analyze one or more routed runs:

```bash
poetry run stt-analyze runs/<timestamp>/results.json
```

Interpretation:

- If fixed routing beats blind sequential across the ladder, `A + 0.9B + 0.25C` becomes the baseline for learned routing.
- If it fails on specific regimes, use the failure to decide whether layerwise routing, adaptive C scale selection, or a less conservative route objective is warranted.
- For scale sweeps, choose the fixed pair by `frontier_score`, not accretion alone. This prevents choosing a route that improves A while silently giving up C learning or B retention.

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

## Frontier Sweep Check

Before the 6-seed gauntlet, a small rehearsal-only route sweep checked whether selecting by a balanced frontier score differs from selecting by accretion alone.

Run:

```text
runs/20260521T224630759252Z/results.json
```

Route pairs:

```text
0.9:0.15 0.9:0.25 0.9:0.35 1.0:0.15 1.0:0.25 1.0:0.35
```

Summary:

| Route | `accretion_a` | `interference_a` | `interference_b` | `learning_b` | `learning_c` | `frontier_score` | Wins |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `0.9B+0.15C` | `+0.1925` | `-0.0033` | `+0.0069` | `+1.9172` | `+2.2431` | `+1.0023` | accretion `3/3`, C preserved `2/3` |
| `0.9B+0.25C` | `+0.1831` | `+0.0061` | `+0.0157` | `+1.9085` | `+2.7097` | `+1.4394` | C preserved `3/3` |
| `0.9B+0.35C` | `+0.1573` | `+0.0319` | `+0.0446` | `+1.8796` | `+2.9883` | `+1.6302` | C preserved `3/3` |
| `1.0B+0.15C` | `+0.1904` | `-0.0011` | `-0.0010` | `+1.9251` | `+2.2612` | `+1.0261` | accretion `2/3`, C preserved `2/3` |
| `1.0B+0.25C` | `+0.1800` | `+0.0092` | `+0.0100` | `+1.9142` | `+2.7271` | `+1.4576` | C preserved `3/3` |
| `1.0B+0.35C` | `+0.1550` | `+0.0342` | `+0.0360` | `+1.8881` | `+3.0060` | `+1.6540` | C preserved `3/3` |

Interpretation:

- Accretion-only selection would prefer smaller C scales such as `0.9B+0.15C`, but that under-preserves C learning on one seed.
- Frontier scoring prefers `1.0B+0.35C` in this rehearsal-only sweep because it values C learning and interference reduction, not just A accretion.
- This supports using `frontier_score` for later local calibration, while keeping the immediate gauntlet predeclared at `0.9B+0.25C`.

## Six-Seed Fixed-Route Gauntlet

Predeclared route:

```text
A + 0.9B + 0.25C
```

Runs:

```text
B_related: runs/20260521T230138841306Z/results.json
B_related_strong: runs/20260521T234032906888Z/results.json
B_rehearsal: runs/20260522T000208301357Z/results.json
```

Shared condition:

```text
Qwen/Qwen2.5-0.5B
gossip_weight=12.5
eval_batches=16
seeds=0 1 2 3 4 5
```

Summary:

| Condition | Method | `accretion_a` | `interference_a` | `interference_b` | `learning_b` | `learning_c` | `eval_c` | `frontier_score` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B_related` | Sequential | `-0.0246` | `+0.0771` | `+0.3976` | `+1.9221` | `+1.5906` | `0.1707` | n/a |
| `B_related` | Routed | `+0.1009` | `-0.1255` | `+0.0126` | `+1.9095` | `+3.0704` | `0.4764` | `+2.1898` |
| `B_related_strong` | Sequential | `-0.0285` | `+0.1423` | `+0.2928` | `+1.6782` | `+1.5572` | `0.1742` | n/a |
| `B_related_strong` | Routed | `+0.0926` | `-0.1210` | `+0.0115` | `+1.6667` | `+3.0834` | `0.4633` | `+2.1890` |
| `B_rehearsal` | Sequential | `+0.1778` | `+0.3728` | `+0.5095` | `+1.8758` | `+2.1052` | `0.1751` | n/a |
| `B_rehearsal` | Routed | `+0.1726` | `+0.0052` | `+0.0160` | `+1.8598` | `+2.7299` | `0.8168` | `+1.4767` |

Win counts:

| Condition | Accretion wins | A-interference wins | B-interference wins | C-learning preserved |
| --- | ---: | ---: | ---: | ---: |
| `B_related` | `6/6` | `6/6` | `6/6` | `6/6` |
| `B_related_strong` | `6/6` | `6/6` | `6/6` | `6/6` |
| `B_rehearsal` | `2/6` | `6/6` | `6/6` | `6/6` |

Interpretation:

- The predeclared route passes the strongest current gauntlet for related and strongly related B tasks: it converts negative mean A accretion under blind sequential into positive routed accretion while reducing C-phase A/B interference and preserving C learning on every seed.
- The rehearsal positive control shows the expected tradeoff. Blind sequential already has high A accretion from rehearsal; routed composition gives up a small amount of that accretion but still preserves most of it while massively reducing C interference and improving C learning.
- This supports the limited claim that sequential LoRA updates contain separable compatible and interfering components, and that a simple fixed high-B/low-C composition rule can implement a compact toy continual-learning mechanism.

Reproduce the gauntlet summary:

```bash
poetry run stt-analyze \
  runs/20260521T230138841306Z/results.json \
  runs/20260521T234032906888Z/results.json \
  runs/20260522T000208301357Z/results.json
```
