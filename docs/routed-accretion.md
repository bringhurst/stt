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

Grouped C sweeps scale LoRA tensor families separately after the same training pass:

```bash
--group-route-pairs 0.9:0.6:0.4 0.9:0.4:0.8
```

The grouped format is `B:C_A:C_B`, where `C_A` scales LoRA `lora_A` tensors and `C_B` scales LoRA `lora_B` tensors. Each triple is emitted as a variant such as `gossip_b0.9_ca0.6_cb0.4`.

Layer-band C sweeps scale early, middle, and late transformer layers separately:

```bash
--layer-route-pairs 0.9:0.25:0.4:0.6 0.9:0.25:0.4:0.8
```

The layer format is `B:C_E:C_M:C_L`, where `C_E`, `C_M`, and `C_L` scale C updates in early, middle, and late layer thirds. Each quadruple is emitted as a variant such as `gossip_b0.9_ce0.25_cm0.4_cl0.6`.

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

## B-Related High-C Calibration

Runs:

```text
C=0.35..0.60: runs/20260522T052712608650Z/results.json
C=0.65..1.00: runs/20260522T060205346069Z/results.json
```

Best frontier routes:

| Route | `accretion_a` | `interference_b` | `learning_b` | `learning_c` | `eval_c` | `frontier_score` | C preserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.90B+0.40C` | `+0.1054` | `+0.0477` | `+1.8744` | `+1.4847` | `0.2766` | `+0.5691` | `0/6` |
| `0.95B+0.40C` | `+0.0908` | `+0.0346` | `+1.8875` | `+1.4934` | `0.2679` | `+0.5650` | `0/6` |
| `0.95B+0.45C` | `+0.0884` | `+0.0549` | `+1.8671` | `+1.5211` | `0.2402` | `+0.5626` | `0/6` |

Best C-learning routes:

| Route | `accretion_a` | `interference_b` | `learning_b` | `learning_c` | `eval_c` | `frontier_score` | C preserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `1.00B+1.00C` | `-0.1017` | `+0.3976` | `+1.5245` | `+1.5906` | `0.1707` | `-0.1765` | `3/6` |
| `1.00B+0.90C` | `-0.0324` | `+0.3288` | `+1.5933` | `+1.5893` | `0.1720` | `+0.0468` | `2/6` |
| `0.95B+0.90C` | `-0.0153` | `+0.3404` | `+1.5816` | `+1.5871` | `0.1742` | `+0.0642` | `0/6` |

Sequential reference:

```text
accretion_a=-0.0246
interference_b=+0.3976
learning_b=+1.9221
learning_c=+1.5906
eval_c=0.1707
```

Interpretation:

- The best frontier route for `B_related` is around `0.90B+0.40C`. It keeps the A/B retention benefit and improves C learning over `0.25C`, but still does not preserve C learning on any seed.
- C preservation only appears near `C=0.90..1.00`. At that point the fixed global route largely collapses back toward blind sequential: B interference rises sharply and A accretion becomes weak or negative.
- More scalar C tuning is unlikely to produce a clean Pareto route. The next C-focused step should change route form, for example layerwise or tensor-group C scaling, rather than only increasing the global C coefficient.

## B-Related Grouped C Calibration

Run:

```text
runs/20260522T070738579040Z/results.json
```

Grid:

```text
B scales: 0.90 0.95
C_A scales: 0.40 0.60 0.80 1.00
C_B scales: 0.40 0.60 0.80 1.00
```

Best frontier routes:

| Route | `accretion_a` | `interference_b` | `learning_b` | `learning_c` | `eval_c` | `frontier_score` | C preserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.90B+0.60C_A+0.40C_B` | `+0.1066` | `+0.0581` | `+1.8639` | `+1.4993` | `0.2620` | `+0.5731` | `0/6` |
| `0.95B+0.60C_A+0.40C_B` | `+0.0935` | `+0.0444` | `+1.8777` | `+1.5070` | `0.2543` | `+0.5718` | `0/6` |
| `0.95B+0.80C_A+0.40C_B` | `+0.0950` | `+0.0555` | `+1.8665` | `+1.5172` | `0.2441` | `+0.5711` | `0/6` |

Best C-learning routes:

| Route | `accretion_a` | `interference_b` | `learning_b` | `learning_c` | `eval_c` | `frontier_score` | C preserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.95B+1.00C_A+1.00C_B` | `-0.0792` | `+0.4105` | `+1.5115` | `+1.5888` | `0.1726` | `-0.1496` | `0/6` |
| `0.95B+0.80C_A+1.00C_B` | `-0.0631` | `+0.3937` | `+1.5283` | `+1.5878` | `0.1735` | `-0.0974` | `0/6` |
| `0.95B+0.60C_A+1.00C_B` | `-0.0493` | `+0.3776` | `+1.5445` | `+1.5862` | `0.1751` | `-0.0510` | `0/6` |

Interpretation:

- Splitting LoRA A/B C scales gives only a small frontier improvement over the best scalar route: `+0.5731` versus `+0.5691`.
- The best frontier routes prefer higher `C_A` and lower `C_B`, but they still preserve C learning on `0/6` seeds.
- Routes that approach sequential C learning require high `C_B`; those routes lose accretion and B retention, recreating the scalar tradeoff.
- LoRA tensor-family routing is not expressive enough to separate C learning from A/B interference. The next useful route form is layerwise routing or module/block-group routing.

## B-Related Layer-Band C Calibration

Run:

```text
runs/20260522T074030315081Z/results.json
```

Grid:

```text
B scales: 0.90 0.95
C_E/C_M/C_L selected around low early/middle C and higher late C
```

Best frontier route:

| Route | `accretion_a` | `interference_b` | `learning_b` | `learning_c` | `eval_c` | `frontier_score` | C preserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.90B+0.25C_E+0.40C_M+0.60C_L` | `+0.1134` | `+0.0783` | `+1.8437` | `+1.5174` | `0.2439` | `+0.5796` | `0/6` |

Higher-C comparison routes:

| Route | `accretion_a` | `interference_b` | `learning_b` | `learning_c` | `eval_c` | `frontier_score` | C preserved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `0.90B+0.25C_E+0.40C_M+0.80C_L` | `+0.1007` | `+0.1391` | `+1.7829` | `+1.5360` | `0.2253` | `+0.4968` | `0/6` |
| `0.90B+0.40C_E+0.80C_M+1.00C_L` | `-0.0003` | `+0.3405` | `+1.5816` | `+1.5727` | `0.1887` | `+0.0797` | `0/6` |

Comparison against previous `B_related` route forms:

| Route form | Best route | `frontier_score` | C preserved |
| --- | --- | ---: | ---: |
| Scalar C | `0.90B+0.40C` | `+0.5691` | `0/6` |
| LoRA tensor family C | `0.90B+0.60C_A+0.40C_B` | `+0.5731` | `0/6` |
| Layer-band C | `0.90B+0.25C_E+0.40C_M+0.60C_L` | `+0.5796` | `0/6` |

Interpretation:

- Layer-band routing is the best fixed-route frontier result so far on `B_related`, but the improvement over scalar and LoRA-family C routing is small.
- Late-layer C scaling helps recover some C learning, but increasing late or middle C quickly raises B interference and reduces A accretion.
- None of the layer-band routes preserve blind sequential C learning on any seed. The current fixed-route family still exposes an A/B-retention versus C-learning tradeoff rather than a clean Pareto solution.
- Further work should move to module/block-group routing or learned route selection only if the goal is to diagnose whether a more expressive route can escape this tradeoff.

## Calibration-Selected Held-Out Routes

Calibration-selected routing chooses one scalar route per seed using calibration probes only, then reports metrics on the held-out eval split. This avoids selecting routes from the final reported probes.

Shared Qwen condition:

```text
Qwen/Qwen2.5-0.5B
phase_steps=150
eval_batches=16
gossip_weight=12.5
gossip_tau=0.5
gossip_k=8
max_gossip_vectors=256
seeds=0 1 2 3 4 5
route grid: B={0.85,0.90,0.95}, C={0.35,0.40,0.45,0.50}
selection_c_retention_min=0.9
selection_learning_b_tolerance=0.05
```

Runs:

```text
B_related: runs/20260529T004957043486Z/results.json
B_related_strong: runs/20260529T014120007099Z/results.json
B_rehearsal: runs/20260529T015930070845Z/results.json
```

Held-out summary:

| Condition | Mean selected route | Constraint passes | `delta_accretion_a` | `delta_interference_a` | `delta_interference_b` | `delta_learning_b` | `delta_learning_c` | `frontier_score` | Mean C retention |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `B_related` | `0.883B+0.375C` | `5/6` | `+0.1373` | `+0.2118` | `+0.3255` | `-0.0372` | `-0.1276` | `+0.5377` | `0.919` |
| `B_related_strong` | `0.875B+0.358C` | `6/6` | `+0.1416` | `+0.2964` | `+0.2653` | `-0.0350` | `-0.1327` | `+0.5619` | `0.916` |
| `B_rehearsal` | `0.942B+0.442C` | `3/6` | `-0.0419` | `+0.3463` | `+0.5241` | `-0.0600` | `-0.2027` | `+0.6108` | `0.903` |

Held-out win counts:

| Condition | Accretion wins | A-interference wins | B-interference wins | Strict C-learning preserved | Frontier wins |
| --- | ---: | ---: | ---: | ---: | ---: |
| `B_related` | `6/6` | `6/6` | `6/6` | `0/6` | `6/6` |
| `B_related_strong` | `6/6` | `6/6` | `6/6` | `0/6` | `6/6` |
| `B_rehearsal` | `1/6` | `6/6` | `6/6` | `0/6` | `6/6` |

Interpretation:

- Calibration-selected scalar routing reproduces the fixed-route result on the related conditions without using held-out probes for selection: it consistently improves A accretion and reduces A/B interference while retaining about `92%` of sequential C learning.
- The strict `C-learning preserved` count remains `0/6` because the route intentionally down-scales C; the calibrated constraint is about retaining at least `90%` of sequential C learning on calibration probes, not matching or exceeding sequential C learning.
- `B_rehearsal` is not a clean accretion win. Blind sequential already benefits from A rehearsal inside B, and the calibrated route trades some A accretion for much lower A/B interference.
- Rehearsal constraint failures are mostly B-learning tolerance failures after satisfying C retention. In failing seeds, C-retaining candidates exist, but their B-learning drop is above `0.05`.

Reproduce the calibrated summary:

```bash
poetry run stt-analyze \
  runs/20260529T004957043486Z/results.json \
  runs/20260529T014120007099Z/results.json \
  runs/20260529T015930070845Z/results.json
```
