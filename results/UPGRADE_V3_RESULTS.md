# UPGRADE V3 — learning *how* to improve: a meta-learned improver, tested against a frozen twin

**Bottom line: the mechanism is real but not yet ready. The pre-registered go/no-go said NO_GO, so the confirmatory
seeds 7001–7300 were not touched and H2 is not confirmed.**

v3 set out to fix v2's failed H2 (*adapting the improver beats a frozen improver*). The adaptive improver
now learns something measurable, and that learning reaches the final external holdout. The one part of the
mechanism that makes it *recursive in the strong sense*, knowledge carried from one improvement problem to the
next, did not clear its pre-registered bar.

| Pre-registered go/no-go condition (seeds 3101–3200, n = 100, identical 4.4 M-execution cap per run) | Result | |
|---|---|---|
| ADAPTIVE − FROZEN on the final external holdout (`ext`), one-sided p < 0.05 | **+1.35 tasks** [+0.05, +2.68], p = 0.023; 58 wins / 4 ties / 38 losses | pass |
| ADAPTIVE − ADAPTIVE_NOCARRY on `ext`, one-sided p < 0.10 | **+0.53** [−0.79, +1.87], p = 0.22; 51/3/46 | **fail** |
| ADAPTIVE − FROZEN on meta-level quality, one-sided p < 0.05 | +0.0044 [+0.0025, +0.0063], p = 0.0001 | pass |
| The per-world advantage persists or grows (second-half mean ≥ first-half mean) | worlds 6–10 **+0.21** vs worlds 1–5 +0.06 | pass |
| **Verdict (all four required)** | **NO_GO**: 7001–7300 untouched; development continues on dev seeds only | |

**What the go/no-go data support.** These are dev-check evidence: one pre-registered look, not a confirmatory
test.

* **The improver learns how to improve, and in this look it pays off.** ADAPTIVE beats its frozen twin on unseen families:
  +1.35 of 240 final-holdout tasks, about +2.3% relative. The twin has identical code, tasks, streams and budget.
  ADAPTIVE also wins on the in-family holdout (+1.51, p = 0.003) and at the meta level.
* **Its learned ranking predicts real future gain.** The predictor scores candidates *before* they are tested.
  Those scores rank-correlate with the gain the candidates then realize on fresh cross-family META-VAL probes:
  Spearman +0.066, p < 1e-4, positive in 91 of 100 runs. The correlation rises from 0.03 in world 1 to 0.10 in
  world 10.
* **The advantage grows over the sequence of problems.** Against FROZEN it is +0.06 per world in worlds 1–5
  and +0.21 in worlds 6–10. Against NOCARRY it is −0.09 and +0.20 (per-seed trend +1.47, two-sided p = 0.02).
* **Every recursive arm beats the compute-matched single round** by 12–14 tasks (98–96 wins of 100).

**What they do not support (yet).**

* **Cross-world carry-over is not shown on the final metric.** At the meta level it is clear: ADAPTIVE − NOCARRY
  meta-quality is +0.0036, p = 0.0008. On `ext`, most of ADAPTIVE's benefit over 10 worlds is already captured by
  learning *within* each world (NOCARRY − FROZEN = +0.82). The carry-over increment (+0.53) is small and
  back-loaded.
* **The diagnosis is inert on the final metric.** NODIAG_META matches ADAPTIVE on `ext` (+0.34 for NODIAG, n.s.).
  It is still a real channel: it changes the generated pools, and ADAPTIVE beats NODIAG on the in-family holdout,
  +1.20, p = 0.016. But the final-metric gain comes from the learned ranking, not from the diagnosis.
* **The effect is small.** A few percent of the final metric. The noise analysis in §4 explains why a larger
  effect is hard to get in this domain.

---

## 1. Why v2's H2 failed (the starting diagnosis)

v2's adaptive improver (RECURSIVE_FULL) lost to its frozen twin by −0.08 tasks (n = 300). Post-hoc analysis
(`results/V2_MECHANISM_ANALYSIS.md`) pointed at the *learning problem*, not the tuning:

* **~20 sparse outcomes per run:** one adopt/reject bit per strategy per round.
* **Categorical, imbalanced failure labels:** one failure mode dominated. FULL and its context-free ablation made the
  same decisions in 297/300 seeds.
* **A Q-table** with nothing to generalise across cells.
* **Wrong reward:** the probe reward came from *training* families, while the target was transfer to unseen families.

## 2. The v3 design (`src/rsi_v3`)

**One run = a sequence of 10 independent improvement problems ("worlds").** Each world has a fresh 3-way family
split drawn from 24 families: 12 TRAIN, 6 META-VALIDATION and 6 FINAL, behaviour-disjoint across the three sets. It
also has fresh tasks (40 train, 96 meta-val probes, 16 + 24 final-holdout) and a fresh base solver. The only thing
that may carry from one world to the next is the meta-predictor. So "improving the improver" has a precise meaning
here: what was learned while improving solvers in worlds 1…w−1 should make the improvement in world w better.

Each of the 5 rounds per world runs **attempt → diagnose → generate → rank → counterfactual test → adopt → learn**:

1. **Attempt** the training tasks (metered search).
2. **Diagnose:** a continuous 15-dim state (missing-operator, bad-order, low-exploration, over-specialisation,
   composition-failure and residual/near-miss scores, solved fraction, entropy, dead-macro ratio, macro count,
   mean best fitness, diversity, composition depth, round, incumbent rate). Every score is kept; nothing is argmaxed.
3. **Generate** a pool of 24 candidates from six structurally different actions. The mix is a softmax over the
   diagnosed needs, so **the diagnosis changes which candidates exist**. The six actions:
   * `MINE`: behaviour-level frequent-fragment mining.
   * `RESID`: residual-guided mining. It searches short repairs that turn near-miss programs into exact solutions;
     the junction fragments become operator candidates.
   * `COMPOSE`: hierarchical macro-of-macro composition.
   * `PRIOR`: prior/order repair.
   * `EXPLORE`: exploration repair.
   * `PRUNE`: removes dead macros, and is generated **only when the evidence says the vocabulary is
     over-specialised**.
4. **Rank** with the shared ridge meta-predictor: one model for every action. Features are the action, the
   candidate's concrete deltas, the generator's preference, the action × its matched failure score, and the
   incumbent's progress on the probe (25 dims). ADAPTIVE picks the top k−1 plus one exploration slot. FROZEN uses a
   shared random order, and the first round is provably identical for both.
5. **Counterfactual test:** each of the k = 8 ranked candidates and the incumbent are searched on the same 6
   cross-family META-VAL probes, with the same streams. The finalist is re-tested on 6 fresh probes.
6. **Adopt** the screening winner among candidates that solve no fewer screen probes than the incumbent. This rule
   was chosen because it made **FROZEN** strongest (dev-06).
7. **Learn:** every (candidate, probe) pair is one training row with a *paired* target:
   y = progress(candidate) − progress(incumbent), where progress = 1 if solved, else 0.5 × best partial fitness.
   That is ~50–60 rows per round, not one bit. The ridge is refit in closed form after each round.

**Arms** share one code path, and every arm has the same cap: 5 × (40 × 1,000 + 48,000) = 440,000 program executions
per world, metered per world so no arm can shift compute between worlds. Total: 4.4 M per run.

| arm | what differs |
|---|---|
| `ADAPTIVE_META` | predictor learns online and **carries across worlds** |
| `FROZEN_META` | predictor frozen at its uninformative initial state (shared random order); everything else identical |
| `ADAPTIVE_NOCARRY` | learns within a world, **reset at every world**: isolates *cross-problem* meta-learning |
| `NODIAG_META` | learns and carries, but diagnosis removed from generation (uniform mix) and from the predictor |
| `SINGLE_COMPUTE_MATCHED` | one round per world with 5× attempt and 5× gate compute, a larger pool (36) and staged halving (6 → 12 → 42 probes) |
| `COLD` | no improvement (noise-floor replicate on a second eval stream) |

The meta-controller never executes a program. Its own cost (rows, refits, predictions, flops, wall seconds) is
logged per unit, and every program execution goes through the meter.

## 3. Protocol and safeguards

| Safeguard | Mechanism | Executable check (`tests/test_v3_anticheat.py`) |
|---|---|---|
| Equal compute | identical per-world child meters; metered spend = process-global counter; diagnostic arms can never be contenders | `test_equal_caps_and_metered`, `test_hidden_compute_detected`, `test_diagnostic_arms_are_not_contenders` |
| Common random numbers | every stream keyed by (seed, purpose, indices), never by arm or policy | `test_streams_never_use_arm` (AST), `test_frozen_and_adaptive_identical_first_round`, `test_final_eval_identical_for_identical_config`, `test_cross_process_determinism` |
| No final-holdout leakage | FINAL rows unreachable from the improver (sealed at runtime and absent from the improver's view); learning rows only from META-VAL probes | `test_three_way_family_split_and_behaviour_disjoint`, `test_final_holdout_sealed_metaval_allowed`, `test_improver_has_no_final_holdout_path`, `test_improver_view_contains_no_final_rows`, `test_learning_rows_use_metaval_probes_only` |
| The control is really frozen / the treatment really learns | frozen ridge raises on data; every candidate × probe row reaches ADAPTIVE; NOCARRY resets; meta-compute counts all worlds | `test_frozen_predictor_never_learns`, `test_adaptive_trains_on_every_candidate_x_probe_row`, `test_nocarry_resets_but_carry_persists` |
| Diagnosis is not decorative | pool mix differs from the NODIAG mix in ≥ 75% of real round-1 states (measured 15/16); matched failure scores enter the predictor; PRUNE only on evidence | `test_diagnosis_changes_generated_candidates`, `test_diagnosis_enters_the_predictor`, `test_pruning_requires_evidence` |
| Seed hygiene | dev 3001–3040; go/no-go 3101–3200 (spent), next go/no-go 3201–3300; confirm 7001–7300; v1/v2 seeds (1–40, 101–200, 1001–1300) refused; a dev check needs a criterion recorded after the previous look | `test_seed_ranges_never_touch_v2_or_each_other`, `test_dev_runs_refuse_forbidden_seeds`, `test_devcheck_needs_fresh_criterion_and_unspent_seeds` |
| Frozen protocol | one-shot `PREREG_FREEZE`: prereg sha, hp sha, code hash of every rsi_v3 + rsi_v2 file, evaluator hash, task-library and confirm-manifest digests | `test_frozen_protocol_if_frozen` |
| Decision rule | primary alone at α; secondaries only if the primary passes (Holm among themselves) | `test_secondaries_blocked_when_primary_fails`, `test_primary_full_alpha_secondaries_holm` |
| Append-only record | hash-chained ledger; locked, re-verifying appends | `test_stale_writer_cannot_fork_the_chain` |

## 4. Development history (dev seeds 3001–3040 only; every run is in the ledger)

Every dev iteration was recorded as `DEV_START` / `DEV_END` in `results/ledger/rsi_v3_ledger.jsonl`, with its per-unit
records in `results/logs/v3_dev_units.jsonl` (full detail: `v3_dev_detail.jsonl.gz`). The failed designs are listed as
prominently as the kept one. p-values in this table are two-sided (as logged in `DEV_END`); `meta` = `meta_quality`, the
mean realized paired gain on cross-family META-VAL probes of the candidates an arm chose to test, from world 2 on.

| run | design change | seeds × worlds | ADAPTIVE − FROZEN (ext) | outcome / decision |
|---|---|---|---|---|
| dev-01 | first v3 build (1 world, strict v2 gate) | 16 × 1 | −0.13 (p = 0.90) | null; predictor has ~50 rows per run |
| dev-02 | ORACLE_RANK headroom probe (sees true screen outcomes; not compute-matched) | 16 × 1 | ORACLE +1.0 vs FROZEN | there is headroom *if* the screen signal were reliable |
| dev-03 | multi-world runs (predictor carries across 4 worlds) | 12 × 4 | +1.08 (p = 0.36); meta +0.012 (p = 0.09) | ORACLE_RANK −0.25 at 4 worlds → screen-probe gains do not transfer |
| dev-04 | empirical-Bayes adoption gate | 12 × 4 | +1.75 (p = 0.12) | **rejected**: the new gate *weakened the control* (FROZEN 19.67 vs 23.42 with the old gate, p = 0.007); the apparent gain was a hobbled FROZEN |
| dev-05 | predictor-dominated finalist choice | 12 × 4 | **−2.50** (p = 0.12) | **rejected** |
| (note) | reliability study | 3001–3004 | — | split-half r of a candidate's paired META-VAL gain = 0.47 at 48 probes/half, ≈0.1 at the 6 probes a round screens; per-probe noise SD ≈0.23 vs true gain SD ≈0.03. A deterministic enumeration solver prototype did not reduce noise (it is task heterogeneity, not search luck) and was discarded |
| dev-06 | gate sweep **on FROZEN_META only** (arm-agnostic knob) | 12 × 4 | strict 23.42 · non-inferior 25.17 · **always 26.25** | "always adopt the screening winner" is the *strongest control* → used for every arm from dev-07 on |
| dev-07 | ADAPTIVE with the strongest-control gate | 12 × 4 | +0.17 (p = 0.95) | null |
| dev-08 | pooled study of 3,837 randomly screened candidates (FROZEN) | 24 × 4 | — | candidate value *is* learnable (out-of-seed ridge R² ≈ 0.017 ≈ 60% of the non-noise variance) but needs ≈800 candidates before R² > 0; a 4-world run saw ≈800 in total → **data starvation**. Full action × diagnosis block overfits → compact predictor |
| dev-09 | compact predictor + 10 worlds per run | 8 × 10 | +4.0 (p = 0.26) | promising, n too small |
| **dev-10** | same, 24 seeds | 24 × 10 | **+1.67** (p = 0.20); `in` +2.33 (p = 0.02); **meta +0.0064 (p = 0.003)**; vs NOCARRY ext +2.04, meta +0.0058 (p = 0.005) | first credible *causal* signal at the meta level; final-metric effect positive but not significant |
| (note) | alignment diagnostic | 3033–3038 | — | corr(candidate's META-VAL gain, its FINAL-holdout gain) = **−0.045** over 86 candidates (reliabilities would allow ≈0.56): *which* candidate helps does not transfer between disjoint family sets; only coarse regularities can |
| dev-11 | diagnosis-parameterised proposals, pool 32, k 12, screen 4 | 24 × 10 | +1.38 (p = 0.33); meta **−0.0011** | worse than dev-10 on the meta level → by the pre-written rule the go/no-go uses dev-10 |

Two design decisions deserve emphasis because they are where a positive result could have been manufactured:

* **The control was never weakened.** Every arm-agnostic knob (adoption gate, k, screening sizes) was tuned on
  FROZEN_META alone, and the setting that made *FROZEN* strongest was adopted for all arms (dev-06). The one design
  that improved ADAPTIVE − FROZEN by weakening FROZEN (dev-04) was rejected and is documented.
* **Predictor knobs were judged on the meta level, not on the final holdout.** The final-holdout metric was only ever
  read on dev seeds, and the choice between dev-10 and dev-11 for the go/no-go was made by the rule recorded in
  `GO_NO_GO_CRITERION` *before* dev-11's results were read.

### Ledger incident (repaired transparently)

During dev-11 a long-running background process appended its `DEV_END` from a stale in-memory chain head after two
notes had been appended by another process, forking the chain at record 31. The fork was never committed. It was
repaired by preserving the orphan verbatim (`results/ledger/rsi_v3_ledger.orphans.jsonl`), appending a
`LEDGER_REPAIR` record that names its hash and cause, and re-chaining the `DEV_END`. `rsi_v3.ledger.SafeLedger`
(exclusive OS lock + re-read + re-verify before every append) is used from then on, and a test
(`test_stale_writer_cannot_fork_the_chain`) reproduces the stale-writer race. `rsi_v2.ledger` is unchanged
(it is hashed into the frozen v2 protocol).

## 5. The go/no-go look (seeds 3101–3200)

**Pre-registration.** The criterion (ledger `GO_NO_GO_CRITERION`, seq 32) and the rule for choosing the
configuration were recorded *before* dev-11's results were read and before any seed in 3101–3200 ran. By that rule
the dev-10 configuration was used (`DEVCHECK_DESIGN`, seq 35), fixed by `hp_sha 87b366a0…` and
`code_hash f9b0393b…`. The verdict was computed mechanically by `experiments/v3_gonogo.py` and appended as
`GO_NO_GO_RESULT` (seq 38). Arms: FROZEN_META, ADAPTIVE_META, ADAPTIVE_NOCARRY, NODIAG_META,
SINGLE_COMPUTE_MATCHED. Each run is 10 worlds; 500 units in total.

### 5.1 Means per run (n = 100; `ext` is out of 240 final-holdout tasks, `in` out of 160)

| arm | ext | in | meta_quality | macros / world | EXPLORE share of adoptions |
|---|---|---|---|---|---|
| ADAPTIVE_META | **60.33** | **45.50** | **0.0085** | 3.58 | 10.6% |
| NODIAG_META | 60.67 | 44.30 | 0.0073 | 3.51 | 9.1% |
| ADAPTIVE_NOCARRY | 59.80 | 44.26 | 0.0050 | 3.44 | 14.1% |
| FROZEN_META | 58.98 | 43.99 | 0.0041 | 3.41 | 17.5% |
| SINGLE_COMPUTE_MATCHED | 46.72 | 32.85 | — | 0.75 | — |

Every recursive arm adopts about 49 candidates per run. The learning arms adopt fewer EXPLORE edits (which rarely
transfer) and more PRIOR/order repairs: 38.6% for ADAPTIVE vs 31.6% for FROZEN. The adaptive improver does **not**
win by making more macros (3.58 vs 3.41 per world).

### 5.2 All contrasts (one-sided paired sign-flip permutation p; bootstrap 95% CI; read-only analysis `results/v3_devcheck_report.json`)

| contrast | ext | in | meta_quality |
|---|---|---|---|
| ADAPTIVE − FROZEN | **+1.35** [+0.09, +2.68], p = 0.022 | +1.51 [+0.44, +2.58], p = 0.003 | +0.0044, p < 1e-4 |
| ADAPTIVE − NOCARRY | +0.53 [−0.82, +1.89], p = 0.23 | +1.24 [+0.16, +2.38], p = 0.015 | +0.0036, p = 0.0008 |
| ADAPTIVE − NODIAG | −0.34 [−1.82, +1.19], p = 0.67 | +1.20 [+0.15, +2.29], p = 0.016 | +0.0012, p = 0.17 |
| NODIAG − FROZEN | +1.69 [+0.41, +3.05], p = 0.007 | +0.31, p = 0.31 | +0.0032, p = 0.002 |
| ADAPTIVE − SINGLE_COMPUTE_MATCHED | +13.61 [+12.19, +15.18] | +12.65 | — |
| FROZEN − SINGLE_COMPUTE_MATCHED | +12.26 [+11.06, +13.50] | +11.14 | — |

(The two CIs for ADAPTIVE − FROZEN, [+0.05, +2.68] in the ledgered verdict and [+0.09, +2.68] here, come from
different bootstrap streams. The ledgered one decides.)

### 5.3 Per-world trajectories (mean over 100 runs, ADAPTIVE minus control, final-holdout tasks per world)

| world | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| vs FROZEN | +0.13 | −0.01 | −0.08 | +0.06 | +0.21 | +0.18 | +0.01 | +0.39 | +0.27 | +0.19 |
| vs NOCARRY | 0.00 | +0.15 | −0.01 | −0.46 | −0.15 | +0.20 | −0.12 | +0.39 | +0.36 | +0.17 |
| rank validity (Spearman) | 0.029 | 0.024 | 0.068 | 0.060 | 0.071 | 0.052 | 0.081 | 0.054 | 0.079 | 0.103 |

World 1 is identical by construction for ADAPTIVE and NOCARRY (both start from an empty predictor), and its first
round is identical to FROZEN (tested). The ADAPTIVE − FROZEN world-1 gap comes from learning inside world 1.

### 5.4 Verification checks (from the v3 validation brief)

| check | result |
|---|---|
| ADAPTIVE > FROZEN | final holdout +1.35 (p = 0.022), in-family +1.51, meta level +0.0044 |
| ADAPTIVE > NOCARRY | meta level yes (p = 0.0008) and in-family yes (p = 0.015); **final holdout not shown** (+0.53, p = 0.23), the failed go/no-go condition |
| advantage persists or grows across later worlds | yes: vs FROZEN 0.06 → 0.21 per world; vs NOCARRY −0.09 → +0.20 |
| the learned ranking predicts real future cross-family gain | yes: Spearman(prediction made before testing, realized gain on fresh META-VAL probes from families disjoint from TRAIN) = +0.066, p < 1e-4, 91/100 runs positive, rising over worlds |
| identical task / RNG / compute conditions | identical caps (4.4 M per run, 440 k per world) for every unit; metered spend = process-global counter in all 500 units; streams arm-independent (AST test); FROZEN and ADAPTIVE round 1 bit-identical (test); identical evaluation for identical configs (test) |
| meta-controller compute logged | ADAPTIVE per run: 2,686 training rows, 50 closed-form refits, 820 predictions, 0.82 Mflop, 0.038 s wall. 0 program executions (asserted), against 3.58 M metered program executions for the same run. NOCARRY's counters in this battery cover only its last world; that logging bug was fixed after the battery started, and a bit-identical re-run shows no behavioural change |
| no final-holdout leakage | the FINAL rows are sealed at runtime while any improver runs, and are absent from the improver's view; learning rows come only from META-VAL probes; the improver has no code path to the final holdout (AST). All tested |

**Spend vs cap.** Every arm has the same cap. The recursive arms spend 3.58–3.59 M of 4.4 M; SINGLE spends 3.22 M,
because its attempts stop once the tasks are solved. The unspent part of SINGLE's cap is budget its design did not
use, not budget it was denied.

## 6. Why this is NO_GO and not a confirmatory run

The rule required evidence that the adaptive mechanism transfers *across improvement problems*. That is the part
that makes the improver itself improve over time, rather than merely adapt within one problem. The rule also said
that failing any condition keeps 7001–7300 sealed. So they stay sealed:

* **No confirmatory unit was run.** The ledger has no `PREREG_FREEZE` and no confirm-phase record.
  `python3 -m rsi_v3 verify` reports the protocol as not frozen.
* **v3's H2 (ADAPTIVE_META > FROZEN_META) is therefore neither confirmed nor refuted.** The go/no-go evidence
  (p = 0.022, one look, n = 100) is suggestive and consistent across three metrics. It is not a confirmatory
  result, and it is not presented as one.
* **Seeds 3101–3200 are now spent.** A `SEED_PLAN_AMENDMENT` reserves 3201–3300 for any future go/no-go. The runner
  refuses spent seeds, and refuses a dev check without a *new* criterion recorded after the previous look (both
  tested).

What would make the mechanism "genuinely ready": the carry-over increment on the final metric has to become
detectable. The dev-check SDs give the scale. The paired SD of ADAPTIVE − NOCARRY on `ext` is 6.96, so +0.53 would
need about 1,100 runs for 80% power (one-sided α = 0.05), while ADAPTIVE − FROZEN (+1.35, SD 6.69) needs about 150. Candidate directions
(dev seeds only):
* longer problem sequences, where the back-loaded carry-over dominates;
* letting the carried predictor shape *generation* as well as ranking;
* features that describe a candidate's generality rather than its family-specific fit (the alignment diagnostic
  in §4 shows family-specific fit does not transfer).

## 7. Limitations and honest boundaries

* **Small effects in a synthetic domain.** +1.35 of 240 tasks. The domain has 24 procedurally generated task
  families on a stack substrate. Transfer is to unseen *families* of the same substrate, not to other domains.
* **Measurement noise bounds what any meta-learner can learn here.** A candidate's true paired gain has SD ≈0.03,
  against per-probe noise SD ≈0.23. Screening reliability is ≈0.1, and candidate-level META-VAL gain does not
  predict candidate-level FINAL gain (r = −0.045). The predictor can only exploit coarse, family-independent
  regularities: fewer EXPLORE edits, more order repair.
* **The diagnosis does not help the final metric.** It changes what is generated (15/16 real states) and helps
  in-family. The v3 request required diagnosis to *materially change* the candidates; it does. It did not require
  diagnosis to help, and it doesn't.
* **The adoption rule adopts the screening winner every round.** It was chosen because it makes the *control*
  strongest (dev-06). A gate that rejects more often could change every arm's absolute numbers.
* **Many dev iterations preceded the look (11 batteries on 3001–3040).** That is why the look used fresh seeds,
  a pre-written criterion and a pre-written configuration-selection rule.
## 8. Reproduction

```bash
python3 -m unittest discover -s tests -v                    # v2 + v3 anti-cheat suites
(cd src && python3 -m rsi_v3 verify)                        # v3 ledger chain; reports "not frozen"
python3 experiments/v3_gonogo.py v3devcheck-01-dev10config  # re-applies the criterion (appends to ledger)
python3 experiments/v3_devcheck_analysis.py v3devcheck-01-dev10config   # read-only tables (§5)
# re-run any logged go/no-go unit from scratch with the current code and compare every behavioural
# field (no ledger writes; ~1 min per unit)
python3 experiments/v3_repro_check.py v3devcheck-01-dev10config ADAPTIVE_META:3101 FROZEN_META:3101
```

(`v3_gonogo.py` appends another `GO_NO_GO_RESULT` record each time it runs; the verdict is deterministic. The
runner refuses to start a new dev check on the spent seeds 3101–3200.)

Files:
* `src/rsi_v3/`: `tasks.py` (3-way family split, manifests), `evaluate.py` (sealed final holdout),
  `metamodel.py` (shared ridge predictor), `improver.py` (loop, actions, arms), `ledger.py` (locked ledger),
  `runner.py` (dev / devcheck / freeze / confirm / report / verify).
* `tests/test_v3_anticheat.py`: 25 executable checks (§3).
* `results/ledger/rsi_v3_ledger.jsonl`: every dev battery, note, criterion, dev check and verdict, hash-chained.
  `results/ledger/rsi_v3_ledger.orphans.jsonl` holds the preserved fork record.
* `results/logs/v3_dev_units.jsonl` (compact per-unit records) and `results/logs/v3_dev_detail.jsonl.gz` (full
  per-round detail: pools, screened candidates, features, predictions).
* `results/v3_devcheck_report.json`: the §5 analysis.
