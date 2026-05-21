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

This is intentionally unfair. Behavioral labels from old tasks are allowed because this is a routing upper bound. Scale selection uses one eval split and final metrics are reported on a held-out eval split when enough eval examples are available. The record field `heldout_report` states whether held-out reporting was used.

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
  --fixed-compositions 0.9:0.25 \
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
  --fixed-compositions 0.9:0.25 1.0:0.25 \
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
- `sequential_learning_c`: blind sequential C learning, used to verify routing did not preserve A/B by refusing C.
- `fixed_*`: metrics for dumb fixed composers such as `A + 0.9B + 0.25C`.

Summary win counts report whether oracle and fixed composition beat blind sequential per seed:

- `oracle_accretion_win_count`
- `oracle_interference_a_win_count`
- `oracle_interference_b_win_count`
- `oracle_learning_c_preserved_count`
- `fixed_accretion_win_count`
- `fixed_interference_a_win_count`
- `fixed_interference_b_win_count`
- `fixed_learning_c_preserved_count`

Interpretation rules:

- If partial B improves or preserves A while learning B, there is reusable B capacity worth routing.
- If C candidates usually route as `conflict_private` or select `c_scale=0`, C updates contain damaging directions that should be isolated.
- If oracle routing cannot beat blind sequential A/B/C, learned routing is premature.
- If scalar routing works, the next experiment is layerwise or modulewise routing.
- If `A + 0.9B + 0.25C` gets most of the oracle benefit, a simple global composition rule may be enough for this regime. If per-seed oracle selection beats fixed scaling, that supports real routing.

## First Qwen Result

First run:

```text
runs/20260521T052232177558Z/results.json
```

Condition:

```text
B_related
gossip_weight=12.5
seeds=0 1 2
b_scales=0,0.25,0.5,0.75,1.0
c_scales=0,0.25,0.5,0.75,1.0
```

Summary:

| Metric | Blind Sequential | Oracle Routed |
| --- | ---: | ---: |
| `accretion_a` | `+0.0071` | `+0.1080` |
| `interference_a_after_c` | `+0.1267` | `-0.0421` |
| `interference_b_after_c` | `+0.4364` | `+0.0139` |
| `learning_b` | not summarized | `+3.2883` |
| `learning_c` | not summarized | `+2.9733` |
| `selected_b_scale` | n/a | `0.9167` |
| `selected_c_scale` | n/a | `0.2500` |

Per-seed route choices:

| Seed | `selected_b_scale` | B route | `selected_c_scale` | C route |
| ---: | ---: | --- | ---: | --- |
| `0` | `1.0` | `shared` | `0.25` | `private` |
| `1` | `0.75` | `shared` | `0.25` | `private` |
| `2` | `1.0` | `shared` | `0.25` | `shared` |

Interpretation:

- This is a positive oracle-routing result. Scalar post-hoc composition found B directions that were safe to share with A and a small C component that learned C while dramatically reducing A/B interference versus blind sequential C.
- The result supports the core hypothesis that later LoRA updates contain routeable components.
- This does not yet prove a learned router is available. The route labels are behavioral oracle labels from eval losses. This first run predated held-out reporting and fixed composer baselines; newer runs should use the upgraded metrics above.
- The next check is whether layerwise routing beats scalar routing and whether fixed global composition remains stable beyond the current task ladder.

## Held-Out Reporting Rerun

Upgraded run:

```text
runs/20260521T054314830375Z/results.json
```

Condition:

```text
B_related
gossip_weight=12.5
seeds=0 1 2
heldout_report=true
fixed_compositions=0.9:0.25,1.0:0.25
```

Summary:

| Metric | Blind Sequential | Fixed `0.9B+0.25C` | Oracle Routed |
| --- | ---: | ---: | ---: |
| `accretion_a` | `-0.0137` | `+0.0914` | `+0.0957` |
| `interference_a_after_c` | `+0.2038` | `-0.0468` | `-0.0269` |
| `interference_b_after_c` | `+0.3996` | `+0.0005` | `+0.0108` |
| `learning_b` | `+3.2743` | `+3.2641` | `+3.2252` |
| `learning_c` | `+1.6573` | `+3.0872` | `+3.0124` |
| `eval_c` | `0.1629` | `0.4671` | `0.5419` |
| `selected_b_scale` | n/a | `0.9000` | `0.8333` |
| `selected_c_scale` | n/a | `0.2500` | `0.2500` |

Win counts over seeds `0 1 2`:

| Method | Accretion wins | A-interference wins | B-interference wins | C-learning preserved |
| --- | ---: | ---: | ---: | ---: |
| Fixed `0.9B+0.25C` | `3/3` | `3/3` | `3/3` | `3/3` |
| Oracle routed | `3/3` | `3/3` | `3/3` | `3/3` |

Interpretation:

- The reviewer objection that routing preserves A/B by refusing C does not hold here. Both fixed and oracle composition improve C learning versus blind sequential on held-out reporting.
- The dumb fixed composer captures most of the oracle gain. That is good news for architecture simplicity: in this regime, a global rule close to `A + 0.9B + 0.25C` may be enough.
- Oracle routing still has slightly higher mean accretion, but fixed composition has slightly lower A/B interference and stronger C learning. This argues for comparing fixed global rules before investing in a learned router.

## Held-Out Ladder Replication

The same held-out protocol was repeated on the stronger related task and the rehearsal positive control.

Runs:

```text
B_related_strong: runs/20260521T055856973515Z/results.json
B_rehearsal: runs/20260521T061141201409Z/results.json
```

Shared condition:

```text
gossip_weight=12.5
seeds=0 1 2
heldout_report=true
fixed_compositions=0.9:0.25,1.0:0.25
```

`B_related_strong` summary:

| Metric | Blind Sequential | Fixed `0.9B+0.25C` | Oracle Routed |
| --- | ---: | ---: | ---: |
| `accretion_a` | `+0.0006` | `+0.0973` | `+0.1062` |
| `interference_a_after_c` | `+0.2186` | `-0.0430` | `-0.0228` |
| `interference_b_after_c` | `+0.3069` | `+0.0030` | `+0.0204` |
| `learning_b` | `+2.6200` | `+2.6101` | `+2.5718` |
| `learning_c` | `+1.6472` | `+3.0830` | `+2.7071` |
| `eval_c` | `0.1569` | `0.4713` | `0.8472` |
| `selected_b_scale` | n/a | `0.9000` | `0.8333` |
| `selected_c_scale` | n/a | `0.2500` | `0.2500` |

`B_related_strong` win counts over seeds `0 1 2`:

| Method | Accretion wins | A-interference wins | B-interference wins | C-learning preserved |
| --- | ---: | ---: | ---: | ---: |
| Fixed `0.9B+0.25C` | `3/3` | `3/3` | `3/3` | `3/3` |
| Oracle routed | `3/3` | `3/3` | `3/3` | `3/3` |

`B_rehearsal` summary:

| Metric | Blind Sequential | Fixed `0.9B+0.25C` | Oracle Routed |
| --- | ---: | ---: | ---: |
| `accretion_a` | `+0.1926` | `+0.1887` | `+0.1936` |
| `interference_a_after_c` | `+0.4344` | `+0.0015` | `-0.0010` |
| `interference_b_after_c` | `+0.5814` | `+0.0053` | `+0.0004` |
| `learning_b` | `+3.4683` | `+3.4517` | `+3.4679` |
| `learning_c` | `+2.1112` | `+2.7069` | `+1.7143` |
| `eval_c` | `0.1697` | `0.8474` | `1.8400` |
| `selected_b_scale` | n/a | `0.9000` | `1.0000` |
| `selected_c_scale` | n/a | `0.2500` | `0.0833` |

`B_rehearsal` win counts over seeds `0 1 2`:

| Method | Accretion wins | A-interference wins | B-interference wins | C-learning preserved |
| --- | ---: | ---: | ---: | ---: |
| Fixed `0.9B+0.25C` | `2/3` | `3/3` | `3/3` | `3/3` |
| Oracle routed | `3/3` | `3/3` | `3/3` | `1/3` |

Interpretation:

- `B_related_strong` reproduces the `B_related` result: scalar fixed composition and oracle routing both reduce A/B interference while improving held-out C learning versus blind sequential.
- `B_rehearsal` is different. Because B already rehearses A strongly, blind sequential keeps high A accretion and B learning, but C still damages A/B heavily. Both fixed and oracle composition remove most C interference.
- The fixed `0.9B+0.25C` rule is stronger than the scalar oracle on `B_rehearsal` for C learning because the oracle selection rule rejects C on two seeds. This is evidence that the current oracle objective is too conservative for positive-control rehearsal regimes.
- Across the held-out ladder so far, the simple fixed composer is a serious baseline, not a throwaway comparator. A learned router should beat `A + 0.9B + 0.25C`, not just blind sequential.

Reproduce the ladder summary from persisted JSON:

```bash
poetry run stt-analyze \
  runs/20260521T054314830375Z/results.json \
  runs/20260521T055856973515Z/results.json \
  runs/20260521T061141201409Z/results.json
```
