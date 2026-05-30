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
- `distilled`: labels calibration examples with the loss-probe oracle once, trains a cheap token selector,
  then routes held-out examples without scoring every candidate route for selection.
- `micro_probe`: builds a short answer-stripped prefix probe from each held-out example, scores routes on
  that tiny selector probe, then evaluates the chosen route on the full held-out example.

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
--probe-select-files \
  data/memory_probe_select_beta_boundary.txt \
  data/memory_probe_select_gamma_boundary.txt \
  data/memory_probe_select_epsilon_boundary.txt \
  data/memory_probe_select_ambiguous_scope.txt \
--probe-names beta_boundary gamma_boundary epsilon_boundary ambiguous_scope \
--probe-routes "A+B" "A+C" "A+E" "A"
```

The probes are never used for phase training. In `calibration` mode, each probe corpus is split into
selection and held-out report halves unless `--probe-select-files` is provided. With explicit select files,
route choice uses only the select corpora and `--probe-files` remain held-out report probes.

`--route-top-k` controls the route-optimality diagnostic cutoff, defaulting to `3`. Results include selected
top-k rate, expected top-k rate, top-k loss spread, boundary margin, low-margin rate, abstention rate, and
false-confident route rate. These are more useful than raw semantic label accuracy for residual routing.

Distilled selector flags:

```bash
--route-selection distilled \
--distilled-selector-method centroid \
--distilled-selector-margin 0.0
```

or:

```bash
--route-selection distilled \
--distilled-selector-method knn \
--distilled-knn-k 1 \
--distilled-selector-margin 0.0
```

The selector margin is separate from `--residual-ambiguity-margin`. The default `0.0` is intentionally
decisive: it tests whether cheap distillation can pick useful routes without converting difficult boundary
prompts into blanket abstention.

Micro-probe selector flags:

```bash
--route-selection micro_probe \
--micro-probe-prefix-words 6 \
--micro-probe-max-length 24 \
--micro-probe-template "Route selector probe: {prefix}" \
--micro-probe-margin 0.0
```

The micro-probe builder strips suffixes beginning with `answer should` or `expected semantic route` before
taking the prefix. Templates can use `{prefix}` and `{scope}`. `{scope}` is inferred from the stripped prompt
text, for example `Gamma boundary probe answer should say ...` becomes `Project Gamma`, while ambiguous or
unscoped prompts become `scope unclear`. This keeps route selection prompt/scope-conditioned without using
the answer facts in the held-out report line. `--micro-probe-margin` is separate from
`--residual-ambiguity-margin`; the default is decisive so active probing can be tested before adding
abstention.

## Residual Contextual Sleep Routing

Residual routing treats the best global sleep route as a resting state and generates prompt-conditioned
deformations around it:

```text
base_sleep = A+0.9B+0.4C+0.4D+0.4E
adapter = A + (0.9+dB)B + (0.4+dC)C + (0.4+dD)D + (0.4+dE)E
```

CLI flags:

```bash
--residual-contextual-route \
--residual-route-base-expr "A+0.9B+0.4C+0.4D+0.4E" \
--residual-route-phases B C D E \
--residual-route-grid -0.4 -0.2 0.0 0.2 0.4 \
--residual-route-mode full \
--residual-route-min-scale 0.0 \
--residual-route-max-scale 1.5 \
--residual-ambiguity-margin 0.02
```

Modes:

| Mode | Candidate set |
| --- | --- |
| `full` | all grid combinations across residual phases; `5^4 = 625` for B/C/D/E with five offsets |
| `axis` | base route plus one phase varied at a time |
| `pairs` | axis routes plus B/C and D/E pair interactions |

Residual routes are emitted as normal route expressions, with metadata:

```json
{
  "route_expr": "A+1.1B+0.2C+0.4D+0.4E",
  "base_route_expr": "A+0.9B+0.4C+0.4D+0.4E",
  "residual": {"B": 0.2, "C": -0.2, "D": 0.0, "E": 0.0}
}
```

`stt-analyze` reports residual candidate count, selected residual, best residual, route rank, top-k rates,
top-k margins, entropy, and selected gap. `--residual-ambiguity-margin` marks low-margin selections as
`uncertain`; the result still evaluates the best-loss route for comparability, while abstention and
false-confidence metrics expose low-confidence routing.

Use `full` for oracle/audit route-space checks. Use `axis` or a smaller grid for faster calibration runs.

### First Reduced-Grid Qwen Residual Run

Run:

```text
runs/20260529T161339361467Z/results.json
```

Condition:

```text
Qwen/Qwen2.5-0.5B
phase_steps=150
eval_batches=16
gossip_weight=12.5
gossip_tau=0.5
gossip_k=8
max_gossip_vectors=256
seeds=0 1 2 3 4 5
residual base: A+0.9B+0.4C+0.4D+0.4E
residual grid: [-0.2, 0.0, 0.2]
residual mode: full
residual candidates: 81
selection: calibration
residual ambiguity margin: 0.02
```

Top aggregate rows:

| Variant | Selection | Eval loss | Sequential loss | Frontier | Wins | Probe eval | Probe selected gap | Probe optimal rate | Ambiguous rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_calibration_81` | residual calibration | `3.9050` | `6.3643` | `+2.4593` | `6/6` | `7.6932` | `0.4536` | `0.0833` | `0.2667` |
| `global_0.4D_0.4E` | fixed `A+0.9B+0.4C+0.4D+0.4E` | `4.4638` | `6.3643` | `+1.9005` | `6/6` | `7.9452` | `0.7057` | `0.0000` | `0.0000` |
| `global_0.6D_0.6E` | fixed `A+0.9B+0.4C+0.6D+0.6E` | `4.7684` | `6.3643` | `+1.5959` | `6/6` | `8.0220` | `0.7824` | `0.0000` | `0.0000` |

Residual per-domain route summary:

| Domain | Most selected | Most frequent best | Eval loss | Best eval loss | Selected gap | Learning retained | Interference |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| A | `uncertain` | `A` | `4.3243` | `3.9862` | `0.3380` | `0.9821` | `+0.0038` |
| B | `uncertain` | `A+0.7B+0.2C+0.2D+0.2E` | `3.8538` | `3.6685` | `0.1853` | `1.2971` | `-0.8516` |
| C | `uncertain` | `A+0.7B+0.6C+0.2D+0.2E` | `3.9758` | `3.6295` | `0.3463` | `1.3510` | `-1.0167` |
| D | `A+0.9B+0.6C+0.6D+0.2E` | `A+0.7B+0.4C+0.6D+0.2E` | `3.6873` | `3.5728` | `0.1145` | `1.1700` | `-0.6291` |
| E | `A+0.7B+0.6C+0.6D+0.6E` | `A+0.7B+0.6C+0.6D+0.6E` | `3.6837` | `3.5666` | `0.1171` | `1.2353` | `-0.5672` |

Interpretation:

- This is the first positive result for the residual-contextual framing. A reduced 81-route residual grid beats the best fixed global route on both aggregate domain eval and boundary-probe eval.
- It also improves selected-gap versus the best fixed global route: probe selected gap `0.4536` versus `0.7057`.
- Exact probe route-label accuracy is still not meaningful here: many selected labels are `uncertain` because `--residual-ambiguity-margin 0.02` intentionally abstains on low-margin choices.
- The residual route space contains much better domain solutions than the scalar global baseline. The next check should separate route-space oracle from selector quality by running `loss_probe` over the same residual grid, then tune calibration/ambiguity margins.

### Reduced-Grid Residual Loss-Probe Oracle Check

Run:

```text
runs/20260529T180730714702Z/results.json
```

Same 81-candidate grid as above, but with `--route-selection loss_probe`. This is an oracle/diagnostic
run because it selects routes using the reported examples.

Aggregate comparison:

| Variant | Selection | Eval loss | Frontier | Probe eval | Probe selected gap | Probe optimal rate | Ambiguous rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_loss_probe_81` | leakage diagnostic | `3.7128` | `+2.6625` | `7.3933` | `0.1491` | `0.6042` | `0.2833` |
| `residual_calibration_81` | held-out calibration | `3.9050` | `+2.4593` | `7.6932` | `0.4536` | `0.0833` | `0.2667` |
| `global_0.4D_0.4E` | fixed global | `4.4576` | `+1.9177` | `7.9486` | `0.7043` | `0.0000` | `0.0000` |

Interpretation:

- The residual route space clearly contains better solutions than the fixed global baseline.
- The gap between `loss_probe` and calibration is now the main problem: route selection/calibration is weaker than the route space.
- Probe optimal rate jumps from `0.0833` in calibration to `0.6042` with loss-probe selection, and probe selected gap drops from `0.4536` to `0.1491`.
- The next experiment should improve the selector rather than expand the route space first. Best candidates are boundary-shaped calibration examples, lower/diagnosed ambiguity thresholds, or a contrastive route-selection objective.

### Probe-Select Residual Calibration Run

Run:

```text
runs/20260529T225159956798Z/results.json
```

Same 81-candidate residual grid as above, but calibration used explicit boundary-shaped select files via
`--probe-select-files`; held-out probe reporting still used the original `--probe-files`.

Aggregate comparison:

| Variant | Selection | Eval loss | Frontier | Probe eval | Probe selected gap | Probe optimal rate | Probe top-3 rate | Probe abstention | Probe false-confident |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_probe_select_81` | held-out probe-select calibration | `3.9861` | `+2.1645` | `7.6328` | `0.4922` | `0.0260` | `0.1510` | `0.7500` | `0.2500` |
| `residual_calibration_81` | split-probe calibration | `3.9050` | `+2.4593` | `7.6932` | `0.4536` | `0.0833` | n/a | `0.3750` | n/a |
| `residual_loss_probe_81` | leakage diagnostic | `3.7128` | `+2.6625` | `7.3933` | `0.1491` | `0.6042` | n/a | `0.3438` | n/a |
| `global_0.4D_0.4E` | fixed global in probe-select run | `4.5687` | `+1.5820` | `7.8578` | `0.7171` | `0.0000` | `0.0000` | `0.0000` | `1.0000` |

Residual probe rows:

| Probe | Expected route | Most selected | Most frequent best | Eval loss | Selected gap | Optimal rate | Selected top-3 | Expected top-3 | Abstention | False-confident |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ambiguous_scope` | `A` | `uncertain` | `A+C` | `7.1030` | `0.4038` | `0.0417` | `0.1875` | `0.2083` | `0.8333` | `0.1667` |
| `beta_boundary` | `A+B` | `uncertain` | `A+0.7B+0.6C+0.6D+0.2E` | `8.0450` | `0.4945` | `0.0000` | `0.0833` | `0.0000` | `0.8333` | `0.1667` |
| `epsilon_boundary` | `A+E` | `uncertain` | `A+B+C+D+E` | `7.1702` | `0.4914` | `0.0208` | `0.1667` | `0.1458` | `0.8333` | `0.1667` |
| `gamma_boundary` | `A+C` | `uncertain` | `A+C` | `8.2131` | `0.5790` | `0.0417` | `0.1667` | `0.3125` | `0.5000` | `0.5000` |

Interpretation:

- Explicit boundary-select calibration is leakage-safe and still beats all fixed global baselines in this run, but it does not close the selector gap to `loss_probe`.
- Probe eval improved slightly versus the earlier split-probe calibration (`7.6328` versus `7.6932`), but selected gap and optimal-route rate got worse.
- The new abstention metrics are useful: the router now mostly reports `uncertain` on held-out probes, especially ambiguous, Beta, and Epsilon. That is safer than confident wrong conflict routing, but it is not good route selection.
- Expected semantic routes are often not even top-3 by LM loss, especially Beta (`0.0000`) and only partially Gamma (`0.3125`). This points away from simply adding more boundary-select examples and toward a stronger selection objective or answer-level contrastive probe design.

### Distilled Residual Selector Runs

Runs:

```text
centroid: runs/20260530T024807529024Z/results.json
knn_k1:  runs/20260530T054921802315Z/results.json
```

Both runs used explicit boundary-shaped select files and `--distilled-selector-margin 0.0`, so the selector
was forced to choose a route instead of abstaining. The centroid run averages oracle-labeled calibration
examples by route; the kNN run chooses from the nearest oracle-labeled calibration example.

Aggregate comparison:

| Variant | Selection | Eval loss | Frontier | Selected gap | Optimal rate | Top-3 rate | Abstention | False-confident | Probe eval | Probe selected gap | Probe optimal rate | Probe top-3 rate | Probe abstention | Probe false-confident |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `residual_knn_distilled_81` | kNN cheap selector | `4.0523` | `+2.1055` | `0.3275` | `0.1500` | `0.2833` | `0.0000` | `1.0000` | `7.6193` | `0.4794` | `0.0417` | `0.1823` | `0.0000` | `1.0000` |
| `residual_centroid_distilled_81` | centroid cheap selector | `4.0739` | `+2.0893` | `0.3485` | `0.1500` | `0.2833` | `0.0000` | `1.0000` | `7.6305` | `0.4898` | `0.0312` | `0.1615` | `0.0000` | `1.0000` |
| `residual_probe_select_81` | held-out calibration with abstention | `3.9861` | `+2.1645` | `0.2652` | `0.1333` | `0.3000` | `0.3000` | `0.7000` | `7.6328` | `0.4922` | `0.0260` | `0.1510` | `0.7500` | `0.2500` |
| `residual_loss_probe_81` | leakage diagnostic | `3.7128` | `+2.6625` | `0.0324` | `0.9000` | n/a | `0.2833` | n/a | `7.3933` | `0.1491` | `0.6042` | n/a | `0.3438` | n/a |

Interpretation:

- Cheap distillation did avoid blanket abstention: both distilled runs had `0.0000` probe abstention.
- It did not solve boundary routing. Probe false-confident rate was `1.0000` for both cheap selectors.
- kNN improved slightly over centroids and slightly beat probe-select calibration on probe eval, but it remained far from the loss-probe oracle on selected gap and optimal-route rate.
- This is a negative result for naive token-level oracle distillation. The bottleneck is not just abstention policy; the selector target itself is unstable/noisy under LM-loss labels.
- Next selector work should use a stronger target, likely answer-level contrastive probes or a tiny learned router trained against route-loss vectors, not just token centroids/kNN over oracle top-1 labels.

### Active Micro-Probe Residual Runs

Run:

```text
prefix smoke:       runs/20260530T155515026205Z/results.json
scope-answer smoke: runs/20260530T160843119790Z/results.json
scope-answer seed0: runs/20260530T162104522554Z/results.json
```

Smoke condition:

```text
Qwen/Qwen2.5-0.5B
phase_steps=10
eval_batches=2
seed=0
residual base: A+0.9B+0.4C+0.4D+0.4E
residual grid: [-0.2, 0.0, 0.2]
residual mode: full
residual candidates: 81
selection: micro_probe
micro_probe_prefix_words=6
micro_probe_max_length=24
micro_probe_margin=0.0
```

The scope-answer smoke used this template with `micro_probe_max_length=32`:

```text
Route selector question: {prefix}. Answer scope: {scope}.
```

The full-step seed-0 run used the same scope-answer template with `phase_steps=150` and `eval_batches=16`.

Aggregate rows:

| Variant | Selection | Eval loss | Sequential loss | Frontier | Selected gap | Optimal rate | Selected top-3 | Probe eval | Probe selected gap | Probe optimal rate | Probe top-3 | Probe false-confident |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `prefix_smoke_10step` | active prefix probe | `6.6095` | `6.1139` | `-0.4956` | `0.0456` | `0.4000` | `0.6000` | `7.8443` | `0.0723` | `0.0000` | `0.0000` | `1.0000` |
| `scope_answer_smoke_10step` | scope-answer probe | `6.5672` | `6.1139` | `-0.4533` | `0.0033` | `0.6000` | `1.0000` | `7.8020` | `0.0300` | `0.2500` | `0.2500` | `1.0000` |
| `scope_answer_seed0_150step` | scope-answer probe | `4.3469` | `5.1343` | `+0.7874` | `0.8739` | `0.0000` | `0.2000` | `7.6805` | `0.6254` | `0.0000` | `0.0625` | `1.0000` |

Full-step seed-0 probe rows:

| Probe | Expected route | Most selected | Most frequent best | Selected gap | Rank | Low-margin | False-confident |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| `ambiguous_scope` | `A` | `A+1.1B+0.6C+0.2D+0.6E` | `A+D` | `0.6329` | `49.75` | `0.2500` | `1.0000` |
| `beta_boundary` | `A+B` | `A+0.7B+0.2C+0.2D+0.6E` | `A+E` | `0.6772` | `24.75` | `0.2500` | `1.0000` |
| `epsilon_boundary` | `A+E` | `A+1.1B+0.2C+0.4D+0.6E` | `A+E` | `0.7672` | `16.50` | `0.0000` | `1.0000` |
| `gamma_boundary` | `A+C` | `A+0.9B+0.2C+0.4D+0.4E` | `A+C` | `0.4241` | `44.50` | `0.7500` | `1.0000` |

Interpretation:

- The active micro-probe path works end-to-end on the real residual route space and avoids answer-suffix leakage.
- Adding a scope-answer target helped the short smoke substantially: probe selected gap dropped from `0.0723` to `0.0300`, and probe optimal/top-3 improved from `0.0000` to `0.2500`.
- The improvement did not survive the full-step seed-0 run. The run beat sequential on aggregate, but selected gaps were poor and probe false-confident remained `1.0000`.
- Do not run a full six-seed Qwen comparison for this scope-answer template. It is still a confidently wrong selector on held-out boundary probes.
- The next selector should stop relying on a single active probe text. Stronger options are candidate-specific contrastive probes, explicit abstention training for ambiguous scope, or the learned prompt+route loss-surface scorer.

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

Boundary-shaped calibration fixtures are separate from held-out report probes:

| Select file | Intended route-selection signal |
| --- | --- |
| `data/memory_probe_select_beta_boundary.txt` | Select `A+B` for scoped Beta prompts while rejecting Gamma/Epsilon facts. |
| `data/memory_probe_select_gamma_boundary.txt` | Select `A+C` for scoped Gamma prompts while rejecting Redis/Aerospike leakage. |
| `data/memory_probe_select_epsilon_boundary.txt` | Select `A+E` for scoped Epsilon prompts and risk paragraphs. |
| `data/memory_probe_select_ambiguous_scope.txt` | Encourage abstention or stable memory for unscoped conflict prompts. |

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

## Boundary Probe Qwen Run

Run:

```text
runs/20260529T055806486128Z/results.json
```

Condition:

```text
Qwen/Qwen2.5-0.5B
phase_steps=150
eval_batches=16
gossip_weight=12.5
gossip_tau=0.5
gossip_k=8
max_gossip_vectors=256
seeds=0 1 2 3 4 5
contextual candidates: A, A+B, A+C, A+D, A+E
audit/global routes: candidates plus scalar compromise routes
selection: calibration
boundary probes: beta_boundary, gamma_boundary, epsilon_boundary, ambiguous_scope
```

Aggregate rows:

| Variant | Selection | Eval loss | Sequential loss | Frontier | Wins | Label accuracy | Optimal rate | Selected gap | Probe eval | Probe accuracy | Probe optimal rate | Probe selected gap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `contextual_clean` | per-domain/probe calibration over clean routes | `5.1075` | `6.3769` | `+1.2694` | `6/6` | `0.8000` | `0.0667` | `1.0203` | `7.8526` | `0.1667` | `0.1667` | `0.5398` |
| `global_0.4D_0.4E` | fixed `A+0.9B+0.4C+0.4D+0.4E` | `4.4593` | `6.3769` | `+1.9176` | `6/6` | `0.0000` | `0.3500` | `0.3721` | `7.9489` | `0.0000` | `0.1250` | `0.6362` |
| `global_0.6D_0.6E` | fixed `A+0.9B+0.4C+0.6D+0.6E` | `4.7653` | `6.3769` | `+1.6117` | `6/6` | `0.0000` | `0.1667` | `0.6781` | `8.0279` | `0.0000` | `0.0729` | `0.7152` |

Probe rows for the contextual clean route:

| Probe | Expected route | Most selected | Most frequent best | Accuracy | Eval loss | Best eval loss | Selected gap | Expected gap |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `ambiguous_scope` | `A` | `A+C` | `A+C` | `0.1667` | `7.3636` | `6.9884` | `0.3752` | `0.3881` |
| `beta_boundary` | `A+B` | `A+C` | `A+D` | `0.0000` | `8.3263` | `7.6969` | `0.6294` | `1.2996` |
| `epsilon_boundary` | `A+E` | `A+C` | `A+B+C+D+E` | `0.0000` | `7.3826` | `6.7493` | `0.6333` | `0.7743` |
| `gamma_boundary` | `A+C` | `A+C` | `A+C` | `0.5000` | `8.3376` | `7.8163` | `0.5213` | `0.4103` |

Interpretation:

- This is a negative result for the strict semantic-boundary claim. Clean contextual calibration still beats blind sequential, but the best scalar/global compromise is stronger on normal domain loss.
- Boundary probes do not currently validate route semantics. The calibration router often chooses `A+C`, including for Beta, Epsilon, and ambiguous-scope probes.
- The optimality audit says the expected semantic routes are not usually the best LM-loss routes on these probe fixtures. Gamma is the only probe where `A+C` is both expected and most frequently best.
- The next benchmark revision should make probes more answer-shaped and discriminative, or add an explicit route classifier/contrastive selection signal. Plain LM loss over short factual probe lines is still too weak a proxy for scoped route semantics.
