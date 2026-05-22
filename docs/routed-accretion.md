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

## Corrected Six-Seed Qwen Ladder

The routed C-learning metric is phase-local:

```text
routed_learning_c = eval_c_after_b - routed_eval_c
```

Earlier routed runs used `eval_c_before - routed_eval_c` and overstated C learning. Treat the earlier stored C-learning, frontier-score, and C-preservation claims as stale unless the metrics are recomputed from the raw eval fields.

Runs:

```text
B_related: runs/20260522T030324434757Z/results.json
B_related_strong: runs/20260522T031752104235Z/results.json
B_rehearsal: runs/20260522T033529504809Z/results.json
```

Shared condition:

```text
Qwen/Qwen2.5-0.5B
gossip_weight=12.5
route_b_scale=0.9
route_c_scale=0.25
eval_batches=16
seeds=0 1 2 3 4 5
```

Summary:

| Condition | Method | `accretion_a` | `interference_a` | `interference_b` | `learning_b` | `learning_c` | `eval_c` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `B_related` | Sequential | `-0.0246` | `+0.0771` | `+0.3975` | `+1.9221` | `+1.5906` | `0.1707` |
| `B_related` | Routed | `+0.1009` | `-0.1255` | `+0.0126` | `+1.9095` | `+1.2850` | `0.4764` |
| `B_related_strong` | Sequential | `-0.0285` | `+0.1423` | `+0.2928` | `+1.6782` | `+1.5572` | `0.1742` |
| `B_related_strong` | Routed | `+0.0926` | `-0.1210` | `+0.0115` | `+1.6667` | `+1.2681` | `0.4633` |
| `B_rehearsal` | Sequential | `+0.1778` | `+0.3728` | `+0.5096` | `+1.8758` | `+2.1052` | `0.1750` |
| `B_rehearsal` | Routed | `+0.1726` | `+0.0052` | `+0.0160` | `+1.8598` | `+1.4634` | `0.8168` |

Win counts:

| Condition | Accretion wins | A-interference wins | B-interference wins | C-learning preserved |
| --- | ---: | ---: | ---: | ---: |
| `B_related` | `6/6` | `6/6` | `6/6` | `0/6` |
| `B_related_strong` | `6/6` | `6/6` | `6/6` | `0/6` |
| `B_rehearsal` | `2/6` | `6/6` | `6/6` | `0/6` |

Interpretation:

- The fixed route is a strong A/B retention baseline, not a Pareto win. It sharply reduces C-phase A/B interference across the ladder.
- The related conditions are clean A-retention wins: routed composition converts negative mean A accretion under blind sequential into positive routed accretion and wins accretion on every seed.
- The route under-learns C relative to blind sequential in every corrected seed because the published adapter only applies `0.25C`. This is visible in both lower `learning_c` and higher `eval_c`.
- The rehearsal condition shows the tradeoff most clearly: blind sequential already has high A accretion from rehearsal, so the fixed route gives up a small mean amount of accretion while massively reducing A/B interference.
- Learned, layerwise, or adaptive routing must beat this fixed A/B retention route while recovering more C learning.

Reproduce the summary:

```bash
poetry run stt-analyze \
  runs/20260522T030324434757Z/results.json \
  runs/20260522T031752104235Z/results.json \
  runs/20260522T033529504809Z/results.json
```

## Frontier Sweep Check

Before the C-learning metric correction, a small rehearsal-only route sweep checked whether selecting by a balanced frontier score differs from selecting by accretion alone. The stored C-learning and frontier-score fields in this run are stale because they used `eval_c_before - routed_eval_c`. Recompute or rerun before using it to choose route scales.

Run:

```text
runs/20260521T224630759252Z/results.json
```

Route pairs:

```text
0.9:0.15 0.9:0.25 0.9:0.35 1.0:0.15 1.0:0.25 1.0:0.35
```

Current interpretation:

- Accretion-only selection still risks choosing very small C scales that protect A while under-learning C.
- `frontier_score` remains the right selection rule because it penalizes C-learning loss, but all pre-correction frontier numbers should be discarded.
- The next local calibration should rerun the route grid with the corrected phase-local `routed_learning_c` definition.

## Corrected Local Route Sweep

Runs:

```text
B_related: runs/20260522T040104524297Z/results.json
B_related_strong: runs/20260522T042339078093Z/results.json
B_rehearsal: runs/20260522T044601500876Z/results.json
```

Route grid:

```text
B scales: 0.85 0.90 0.95 1.00
C scales: 0.20 0.25 0.30
```

Best corrected frontier routes:

| Condition | Best route | `accretion_a` | `interference_a` | `interference_b` | `learning_b` | `learning_c` | `eval_c` | `frontier_score` | C preserved |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B_related` | `0.90B+0.30C` | `+0.1061` | `-0.1307` | `+0.0191` | `+1.9030` | `+1.3788` | `0.3825` | `+0.5004` | `0/6` |
| `B_related_strong` | `0.85B+0.30C` | `+0.1105` | `-0.1390` | `+0.0275` | `+1.6508` | `+1.3469` | `0.3845` | `+0.4685` | `0/6` |
| `B_rehearsal` | `1.00B+0.30C` | `+0.1626` | `+0.0152` | `+0.0163` | `+1.8595` | `+1.6453` | `0.6350` | `+0.3716` | `0/6` |

Sequential references:

| Condition | `accretion_a` | `interference_a` | `interference_b` | `learning_b` | `learning_c` | `eval_c` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `B_related` | `-0.0246` | `+0.0771` | `+0.3976` | `+1.9221` | `+1.5906` | `0.1707` |
| `B_related_strong` | `-0.0285` | `+0.1423` | `+0.2928` | `+1.6782` | `+1.5572` | `0.1742` |
| `B_rehearsal` | `+0.1778` | `+0.3728` | `+0.5096` | `+1.8758` | `+2.1052` | `0.1751` |

Interpretation:

- Corrected frontier selection consistently pushes to the largest tested C scale, `0.30`, because the earlier fixed `0.25C` route under-learns C.
- The local grid still does not preserve C learning on any seed. Increasing C from `0.25` to `0.30` improves `learning_c` and `eval_c`, but not enough to match blind sequential C learning.
- Lower B scales help the related tasks keep A accretion while allowing slightly more C. The rehearsal condition prefers `1.00B+0.30C` because B already rehearses A facts.
- The next calibration should test larger C scales, for example `0.35..0.60`, and should keep `frontier_score` as the primary selector.

Reproduce the corrected sweep summary:

```bash
poetry run stt-analyze \
  runs/20260522T040104524297Z/results.json \
  runs/20260522T042339078093Z/results.json \
  runs/20260522T044601500876Z/results.json
```
