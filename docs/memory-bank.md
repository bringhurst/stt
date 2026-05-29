# Contextual Routed Memory Bank

This experiment is the next step after calibrated scalar routed accretion. Instead of
publishing one global sleep adapter, it snapshots every sequential LoRA phase and
stores post-A phase deltas as a small memory bank.

The memory bank composes prompt-scoped routes such as:

```text
A
A+B
A+0.9B+0.4C
A+C
A+D
A+E
```

The stable adapter is the post-A adapter. Later names are adjacent phase deltas:

```text
delta_B = adapter_after_B - adapter_after_A
delta_C = adapter_after_C - adapter_after_B
delta_D = adapter_after_D - adapter_after_C
delta_E = adapter_after_E - adapter_after_D
```

## CLI

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 poetry run stt-memory-bank \
  --model Qwen/Qwen2.5-0.5B \
  --device auto \
  --phase-steps 150 \
  --max-length 128 \
  --batch-size 1 \
  --eval-batches 16 \
  --grad-accum 4 \
  --learning-rate 2e-4 \
  --variant gossip \
  --gossip-weight 12.5 \
  --gossip-tau 0.5 \
  --gossip-k 8 \
  --max-gossip-vectors 256 \
  --task-files \
    data/memory_task_a.txt \
    data/memory_task_b_related.txt \
    data/memory_task_c_conflict.txt \
    data/memory_task_d_unrelated.txt \
    data/memory_task_e_conflict.txt \
  --phase-names A B C D E \
  --snapshot-each-phase \
  --emit-deltas \
  --route-expr "A" \
  --route-expr "A+B" \
  --route-expr "A+0.9B+0.4C" \
  --route-expr "A+C" \
  --route-expr "A+D" \
  --route-expr "A+E" \
  --audit-route-expr "A+0.9B+0.4C+0.4D+0.4E" \
  --contextual-route \
  --route-selection loss_probe \
  --seeds 0 1 2 3 4 5 \
  --output-dir runs
```

Route-selection modes:

- `oracle`: uses the known prompt/domain label as an upper bound.
- `loss_probe`: scores candidate routes per held-out example and chooses the lowest-loss route.
- `calibration`: picks one route per domain on calibration probes, then reports held-out loss.

`--ambiguity-margin` marks a prompt as `uncertain` when the top two loss-probe routes are too close.
The run still reports the best route loss for comparability, while route counts expose uncertainty.

`--route-expr` defines the contextual router candidates. `--audit-route-expr` adds broader routes for
optimality audits and `--global-route-baseline` without letting those routes become contextual candidates.
This keeps clean semantic route labels such as `A+C` separate from scalar compromise audit routes such as
`A+0.9B+0.4C+0.4D+0.4E`.

Eval-only boundary probes can be added with aligned probe lists:

```bash
--probe-files \
  data/memory_probe_beta_boundary.txt \
  data/memory_probe_gamma_boundary.txt \
  data/memory_probe_epsilon_boundary.txt \
  data/memory_probe_ambiguous_scope.txt \
--probe-names beta_boundary gamma_boundary epsilon_boundary ambiguous_scope \
--probe-routes "A+B" "A+C" "A+E" "A"
```

The probes are never used for phase training. In `calibration` mode, each probe corpus is split into
selection and held-out report halves, matching the domain calibration flow.

## Output

Each seed emits:

- `candidate_routes`: route expressions evaluated by the router.
- `audit_routes`: candidate routes plus any extra routes used only for audits/global baselines.
- `per_domain`: selected route counts, expected route, route accuracy, eval loss, retention, and interference.
- `per_probe`: eval-only boundary probe route counts and optimality audit metrics, when probes are provided.
- `route_accuracy`: weighted route accuracy across prompt domains.
- `probe_route_accuracy`: weighted route accuracy across eval-only probes.
- `ambiguous_rate`: weighted uncertainty rate.
- `frontier_score`: `sequential_eval_loss - contextual_eval_loss`.

Analyze runs with:

```bash
poetry run stt-analyze runs/<timestamp>/results.json
```

For multiple memory-bank records, `stt-analyze` prints condition-level aggregate rows and per-domain
route-choice rows.

## Claim Discipline

This experiment does not prove solved continual learning or chatbot memory. A safe positive result is:

```text
Contextual routed memory improves over a single global sleep adapter on scoped conflict tasks.
Sequential LoRA deltas can be stored as a memory bank and composed per prompt.
```

The first target comparison is whether contextual routing retains more C/D/E learning than fixed scalar
sleep without increasing A/B interference.

Boundary probe claims should use `calibration` selection, not `loss_probe`, when claiming non-oracle
held-out behavior. `loss_probe` remains useful as a diagnostic because it reads the reported examples
while selecting a route.

The current boundary probe fixtures are intentionally scoped:

| Probe file | Expected route | Purpose |
| --- | --- | --- |
| `data/memory_probe_beta_boundary.txt` | `A+B` | Related Beta cache facts should not activate Gamma/Epsilon conflicts. |
| `data/memory_probe_gamma_boundary.txt` | `A+C` | Gamma SQLite-WAL conflict should suppress Redis/Beta facts. |
| `data/memory_probe_epsilon_boundary.txt` | `A+E` | Epsilon Aerospike/risk facts should stay scoped to Epsilon. |
| `data/memory_probe_ambiguous_scope.txt` | `A` | Unscoped conflict prompts should prefer stable memory/clarification over a conflict route. |

## First Qwen A/B/C/D/E Runs

Shared condition:

```text
Qwen/Qwen2.5-0.5B
phase_steps=150
eval_batches=16
gossip_weight=12.5
gossip_tau=0.5
gossip_k=8
max_gossip_vectors=256
seeds=0 1 2 3 4 5
routes: A, A+B, A+0.9B+0.4C, A+C, A+D, A+E
```

Runs:

```text
loss_probe: runs/20260529T024814939771Z/results.json
oracle: runs/20260529T030855287079Z/results.json
calibration: runs/20260529T033054619820Z/results.json
```

Aggregate summary:

| Selection | Eval loss | Sequential loss | Frontier score | Wins | Route accuracy | Ambiguous rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `loss_probe` | `4.7416` | `6.0904` | `+1.3488` | `6/6` | `0.5000` | `0.0000` |
| `oracle` | `5.2588` | `6.0964` | `+0.8376` | `6/6` | `1.0000` | `0.0000` |
| `calibration` | `4.8736` | `6.1030` | `+1.2293` | `6/6` | `0.4000` | `0.0000` |

Per-domain route table:

| Selection | Domain | Most selected route | Accuracy | Eval loss | Sequential loss | Learning retained | Interference |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `loss_probe` | A | `A` | `0.0000` | `4.1538` | `6.8855` | `1.0437` | `-0.1647` |
| `loss_probe` | B | `A+0.9B+0.4C` | `0.2500` | `4.2505` | `7.1538` | `1.1469` | `-0.4160` |
| `loss_probe` | C | `A+0.9B+0.4C` | `0.3333` | `4.3018` | `6.0956` | `1.2433` | `-0.6873` |
| `loss_probe` | D | `A+D` | `0.9167` | `5.8086` | `6.1465` | `0.6107` | `+1.6036` |
| `loss_probe` | E | `A+E` | `1.0000` | `5.1935` | `4.1707` | `0.6760` | `+1.0228` |
| `oracle` | A | `A+B` | `1.0000` | `5.5988` | `6.8825` | `0.6425` | `+1.2803` |
| `oracle` | B | `A+B` | `1.0000` | `4.6666` | `7.1576` | `1.0000` | `-0.0000` |
| `oracle` | C | `A+C` | `1.0000` | `5.0150` | `6.1072` | `1.0387` | `+0.0253` |
| `oracle` | D | `A+D` | `1.0000` | `5.8200` | `6.1662` | `0.6081` | `+1.6123` |
| `oracle` | E | `A+E` | `1.0000` | `5.1934` | `4.1687` | `0.6755` | `+1.0247` |
| `calibration` | A | `A+C` | `0.0000` | `4.5272` | `6.8737` | `0.9474` | `+0.2085` |
| `calibration` | B | `A+0.9B+0.4C` | `0.0000` | `4.3099` | `7.1627` | `1.1321` | `-0.3566` |
| `calibration` | C | `A+0.9B+0.4C` | `0.0000` | `4.4770` | `6.1448` | `1.1643` | `-0.5125` |
| `calibration` | D | `A+D` | `1.0000` | `5.8216` | `6.1703` | `0.6076` | `+1.6147` |
| `calibration` | E | `A+E` | `1.0000` | `5.2323` | `4.1632` | `0.6643` | `+1.0691` |

Interpretation:

- Contextual memory-bank routing beats blind sequential on aggregate for all three route-selection modes, with `6/6` positive frontier-score seeds.
- The leakage-safe `calibration` run is close to `loss_probe`: `+1.2293` versus `+1.3488` frontier score.
- Route-label accuracy is not the main success metric yet. The lowest-loss routes often disagree with the hand-authored expected labels, especially for A/B/C, where mixed `A+0.9B+0.4C` routes frequently win.
- D and E route labels are stable (`A+D`, `A+E`), but D/E retention remains weak. E is worse than blind sequential on held-out loss despite selecting `A+E`.
- Oracle domain routing is not an upper bound on eval loss here because the hand-authored domain route is not always the best-loss composition.
- This is a positive first memory-bank result against blind sequential, not yet a completed win against a generalized five-phase scalar-sleep baseline.

## Global-Route Baseline And Optimality Audit

The next run added `--global-route-baseline` and route-optimality audit fields. It trains each seed once,
then evaluates the calibration contextual route plus every route expression as a fixed global route.

Run:

```text
runs/20260529T041750942128Z/results.json
```

Additional route candidates:

```text
A+0.9B+0.4C+0.4D+0.4E
A+0.9B+0.4C+0.6D+0.6E
A+0.9B+0.4C+D+E
A+B+C+D+E
```

Top aggregate rows:

| Variant | Selection | Eval loss | Sequential loss | Frontier | Wins | Label accuracy | Optimal rate | Selected gap | Expected gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `contextual_calibration` | per-domain calibration | `4.2938` | `6.0952` | `+1.8015` | `6/6` | `0.0333` | `0.4333` | `0.2517` | `1.2208` |
| `global_0.4D_0.4E` | fixed `A+0.9B+0.4C+0.4D+0.4E` | `4.4177` | `6.0952` | `+1.6775` | `6/6` | `0.0000` | `0.3833` | `0.3757` | `1.2208` |
| `global_0.6D_0.6E` | fixed `A+0.9B+0.4C+0.6D+0.6E` | `4.6330` | `6.0952` | `+1.4622` | `6/6` | `0.0000` | `0.2000` | `0.5910` | `1.2208` |
| `global_0.4C` | fixed `A+0.9B+0.4C` | `5.6149` | `6.0952` | `+0.4803` | `4/6` | `0.0000` | `0.1667` | `1.5729` | `1.2208` |

Best domain routes by held-out probe loss in this expanded grid:

| Domain | Most frequent best route | Contextual selected route | Contextual selected gap | Expected route gap |
| --- | --- | --- | ---: | ---: |
| A | `A+D` | `A+C` | `0.5274` | `1.5445` |
| B | `A+0.9B+0.4C+0.4D+0.4E` | `A+0.9B+0.4C+0.4D+0.4E` | `0.0434` | `0.7253` |
| C | `A+0.9B+0.4C+0.4D+0.4E` | `A+0.9B+0.4C` | `0.3322` | `0.8700` |
| D | `A+0.9B+0.4C+0.4D+0.4E` | `A+0.9B+0.4C+0.4D+0.4E` | `0.2044` | `1.3013` |
| E | `A+0.9B+0.4C+0.6D+0.6E` | `A+0.9B+0.4C+0.6D+0.6E` | `0.1513` | `1.6630` |

Interpretation:

- Calibration contextual routing beats the best fixed global route in this grid: `+1.8015` versus `+1.6775` frontier score.
- The best global route is itself a scalar sleep-style compromise: `A+0.9B+0.4C+0.4D+0.4E`.
- The route-accuracy oddity is mostly a metric-label problem. Label accuracy is low because the hand-authored expected route is usually not the best-loss route. The selected route gap (`0.2517`) is much smaller than the expected route gap (`1.2208`).
- The audit suggests the benchmark has strong cross-domain transfer or lexical overlap: mixed routes improve A/B/C/D/E losses more than clean domain-isolated routes.
- The next benchmark revision should include stricter conflict probes and ambiguity probes if the goal is testing clean route labels rather than best-loss routing.
