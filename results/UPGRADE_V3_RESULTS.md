# UPGRADE V3 — learning *how* to improve: a meta-learned improver, tested against a frozen twin

> **Final status (Part II, pre-registered confirmation on seeds 7001–7300, n = 300):** a memory-conditioned *process*
> controller whose improvement memory persists across problems beats the identical learner with memory wiped
> between problems: **+1.52 final-holdout tasks, p = 0.0011 (H1 supported)**. It also beats the frozen controller
> (+1.78, H2 supported). Within-problem learning alone does not beat the frozen controller (H3 null), and the
> advantage does not grow with more experience (H4 null). See **Part II** below. Part I documents the earlier
> ranking-only design, whose pre-registered go/no-go was NO_GO; it left the confirmatory seeds untouched.

## Part I — the ranking-only meta-predictor (NO_GO)

**Bottom line of Part I: the mechanism was real but not ready. The pre-registered go/no-go said NO_GO, so the
confirmatory seeds 7001–7300 were not touched and H2 was not confirmed.**

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

---

# Part II — memory-conditioned adaptive PROCESS controller (confirmed)

**Bottom line.** Persistent improvement memory makes the improver improve itself better on later, unseen problems.
The pre-registered confirmatory battery ran once on the untouched seeds 7001–7300 (n = 300, identical tasks,
streams and 4.976 M-execution cap per run, sealed final holdout). It **supports the primary hypothesis**:

| pre-registered hypothesis (frozen report `results/v3_confirm_report.json`) | result |
|---|---|
| **H1 (primary): MEMORY_CARRY > NO_CARRY** — the *only* difference is whether the improvement memory survives the world boundary | **Supported.** **+1.52 tasks** of ~64 (95% CI [+0.55, +2.48]); 174 wins / 16 ties / 110 losses; one-sided p = 0.0011 |
| **H2: MEMORY_CARRY > FROZEN** | **Supported.** **+1.78** [+0.85, +2.73]; Holm p = 0.0006 |
| **H3: NO_CARRY > FROZEN** | **Null.** +0.25 [−0.67, +1.15]; Holm p = 0.60 |
| **H4: the MEMORY_CARRY advantage grows over worlds** | **Null.** +0.05 [−0.84, +0.95]; Holm p = 0.60 |

What the evidence supports, and what it does not:

* **Past self-improvement experience causes better future self-improvement decisions on unseen problems.** The
  learner is identical in MEMORY_CARRY and NO_CARRY, and the two are bit-identical throughout world 1. Only carried
  memory separates them. The gain appears on the final external holdout, whose task families no improver ever saw.
  It also appears in the improver's own unbiased signal, cross-family transfer on the tracking probes
  (+0.0041/probe, p = 0.0001).
* **Learning within one problem is not enough.** NO_CARRY runs the same learner but wipes its memory at every world,
  and it does not beat FROZEN (H3 null). The whole gain needs experience accumulated *across* problems.
* **The ordering MEMORY_CARRY > NO_CARRY > FROZEN is only partly confirmed.** MEMORY_CARRY beats both; NO_CARRY ≈
  FROZEN.
* **The advantage does not keep growing.** It appears once one world of experience exists and then holds steady:
  +0.15 per world in worlds 1–5 (world 1 is 0 by construction), +0.16 in worlds 6–10. The memory's useful knowledge
  saturates early.
* **Memory → process control beyond memory → ranking is not established on the final metric.** Persistent memory
  used *only for ranking* (MEMORY_RANKONLY) gains +1.17 over FROZEN. MEMORY_CARRY beats it by only +0.60 (n.s.). On
  the meta level MEMORY_CARRY's transfer is higher (+0.0023, p = 0.013), but that did not reach the final holdout
  detectably. The round-level options (attempt mode, screening shape, exploration, adoption rule) add nothing
  detectable on the final metric: MEMORY_CARRY − MEMORY_GENONLY = +0.26 (n.s.).
* **Not "more macros", not "more rounds".** Every arm runs exactly 5 rounds per world. NO_CARRY builds almost as many
  macros as MEMORY_CARRY (6.5 vs 6.9 per world) with no gain over FROZEN. MEMORY_RANKONLY builds few (5.2) and
  still gains.
* **The effect is small**, about +2.4% of the final metric.

This result replaces the NO_GO of the ranking-only design in Part I. The confirmatory seeds 7001–7300 had never
been run before this battery.

## II.1 What changed: memory → process control

The v3 ranking design used memory to *rank* candidates that a fixed generator had already produced. Its cross-world
carry-over was null on the final metric. This phase keeps the rsi_v3 system (same tasks, solver, evaluator, ledger,
seed discipline). It adds `src/rsi_v3/controller.py`: an **improvement memory** plus a **process controller** that
uses the memory to change *how the system improves itself* on later problems.

**Improvement memory.**

* *Episodic:* one record per tested candidate and per round. Each record holds:
  * state (a compact context vector) and intervention (the edit variant and its parameters);
  * resource allocation (the round's plan);
  * immediate gain (paired screen gain) and delayed same-world gain (the adopted edit's effect on tracking probes,
    and the return-to-go to the world's end);
  * cross-world transfer gain (world-end cross-family gain) and compute cost.
* *Generalised meta-memory:* three components.
  * A Bayesian ridge value model over edit-variant features × context, fitted on all three horizons.
  * One Bayesian ridge per round-level process option, with return-to-go as the target.
  * A retrieval weight that mixes episodic k-nearest-neighbour estimates with the generalised model. It is adapted
    online from their prediction errors.
* **Records hold numbers and labels only.** They contain no program, macro, token or task. A test walks the whole
  controller object after a multi-world run and fails on any solver content. Only experience about *how to improve*
  can cross a world boundary.

**What the controller decides**, each round, from the memory:

| process dimension | options |
|---|---|
| proposal generation | which variants of each strategy are generated, from a parameterised grid: fragment bundle size, prior step 0.2–0.9, flattening, search depth +1/+2, depth-3 composition, pruning |
| pool allocation across strategies | diagnosis prior × learned value of each strategy's best variant |
| compute allocation | attempt mode: base / focus (redistribute the round's attempt budget over unsolved tasks) / deep (the same, one token deeper) |
| probe / evaluation allocation | screening shape 12×4 / 8×6 / 6×8 at roughly constant cost |
| exploration rate | exploratory ranking slots 0/1/2 plus the Thompson noise scale for generation |
| plasticity | adoption rule (always / non-inferior); edit intensity through the generated variants |
| memory retrieval | episodic-vs-generalised mixing weight, adapted from prediction errors |

**Feedback horizons.** Each world reserves a *tracking set* of 16 fresh META-VAL probes that is never used for any
selection. The incumbent is measured on it before the first round and after every change. That gives two unbiased,
paired signals:
* the delayed effect of each adopted edit (T_r − T_{r−1}), and its return-to-go (T_R − T_{r−1});
* the world's cross-family transfer (T_R − T_0).

Budgets are allotted per round and cannot be banked. So maximising gain under the allotted budget is the same as
maximising gain per unit of allotted compute. The actual compute is recorded with every decision.

**Arms** (identical tasks, streams and caps: 4.976 M executions per run, 497,600 per world):

| arm | memory written | plan reads memory | ranking reads memory | memory survives the world boundary |
|---|---|---|---|---|
| `FROZEN` | no | – | – | – |
| `NO_CARRY` | yes | yes | yes | **no (wiped)** |
| `MEMORY_CARRY` | yes | yes | yes | **yes** |
| `MEMORY_GENONLY` (ablation) | yes | generation and allocation only; round-level options default | yes | yes |
| `MEMORY_RANKONLY` (ablation: the old mechanism) | yes | no | yes | yes |

The only difference between `MEMORY_CARRY` and `NO_CARRY` is whether the improvement memory persists. Two tests
check this. The two arms are bit-identical throughout world 1. `NO_CARRY`'s memory holds exactly one world's
records at every world end.

## II.2 Safeguards added for this phase (`tests/test_v3_anticheat.py`, `TestProcessController`)

| check | test |
|---|---|
| The process machinery is the v3 process: with the v3 variant list as the prior, `FROZEN` makes exactly `FROZEN_META`'s decisions (also verified at full budgets on 16 dev seeds: identical final solvers and scores) | `test_frozen_reproduces_v3_frozen_decisions`, `test_default_grid_equals_v3_generator` |
| The uninformed prior draws the same pool for every arm that has no memory | `test_default_prior_is_arm_independent` |
| `MEMORY_CARRY` ≡ `NO_CARRY` in world 1; `NO_CARRY` wipes memory; `MEMORY_CARRY` accumulates it | `test_carry_and_nocarry_identical_in_world_one`, `test_nocarry_wipes_memory_carry_keeps_it` |
| Memory holds no solver content (no config, candidate, macro token or primitive) | `test_memory_holds_no_solver_content` |
| `FROZEN`'s controller never learns and always uses the default plan | `test_frozen_controller_never_learns` |
| Identical caps across the process arms; exact metering; zero program executions by the controller; its cost counted across resets | `test_caps_identical_and_metered`, `test_equal_caps_and_metered`, `test_meta_compute_counted_across_resets` |
| Learned memory changes real process decisions (pools and round-level options). The ablations change only what they are allowed to change | `test_learned_memory_changes_process_decisions` |
| Tracking probes are META-VAL only and disjoint from the round probes; the final holdout stays sealed and unreachable (AST checks extended to `controller.py`) | `test_tracking_probes_are_metaval_and_disjoint`, `test_improver_has_no_final_holdout_path`, `test_streams_never_use_arm` |
| A new dev check needs a new criterion; spent go/no-go seeds are refused | `test_devcheck_needs_fresh_criterion_and_unspent_seeds` |

## II.3 Development history (dev seeds 3001–3100; every battery is in the ledger)

| battery | what was run | result | decision |
|---|---|---|---|
| (analysis) | observational look at the spent v3 go/no-go battery (1,000 FROZEN worlds) | At the level of **edit variants**, the improver's META-VAL signal agrees with the sealed final holdout (r = 0.63 over 26 variants); at the level of individual candidates it does not (r = −0.045). Variant-level differences need hundreds of observations. | Target process knowledge that is learnable only across worlds: *how to generate and allocate*, not which candidate to rank first |
| pc-01 | every new round-level option as a **fixed** setting on FROZEN only (n = 24) | focus −1.00, deep +1.29, broad screen +0.04, deep screen −0.71 (all n.s.) | defaults unchanged |
| pc-02 | the four arms, first build (n = 24) | MEMORY_CARRY − NO_CARRY +0.75 (n.s.). Transfer +0.0096/probe (p = 0.003). Round-level option draws near-random: 60% non-default | add a switching margin; calibrate the option noise from data |
| pc-03 | fresh seeds 3041–3088 (n = 48) | MEMORY_CARRY 64.96 > NO_CARRY 62.77 > FROZEN 58.10. Carry +2.19 (p = 0.053). Round-1 carry effect p = 0.021. Process control beats rank-only memory by +4.13 (p = 0.002) | check whether the control is the strongest fixed one |
| pc-04a | **control-strength check**: FROZEN with a fixed uninformed prior over the *whole* variant grid (n = 48) | **+3.21 over FROZEN with the v3 list (p = 0.007).** Against it, NO_CARRY is only +1.46 and MEMORY_CARRY +3.65 (p = 0.006) | **the v3-list FROZEN was a weak control**; the whole-grid prior becomes the default for every arm |
| pc-04b | ablation MEMORY_GENONLY (round-level options at default) | +0.35 below MEMORY_CARRY (n.s.) | the carry effect lives mainly in memory → generation |
| pc-05 | all five arms under the stronger control (n = 48) | MEMORY_CARRY 63.46 > NO_CARRY 61.94 ≈ MEMORY_RANKONLY 61.90 ≈ MEMORY_GENONLY 61.83 > FROZEN 61.31. Carry +1.52 (p = 0.13). Candidate quality rises with experience only under persistent memory (0.0053 → 0.0062 → 0.0102) | stop tuning; run the reserved go/no-go |

Three points deserve emphasis, because each is where a positive result could have been manufactured:

1. **The control was strengthened, not weakened.** Under the original v3 prior, NO_CARRY beat FROZEN by +4.7.
   pc-04a showed that most of that came from the richer variant grid, which a *fixed* uninformed policy also gets.
   The stronger prior was adopted for every arm (FROZEN and every learner's no-memory start). That cut
   MEMORY_CARRY's margin over FROZEN from +6.9 to +2.1. The go/no-go uses the stronger control.
2. **Round-level process options did not earn their keep.** No fixed alternative beat the default (pc-01). The
   learned option models are data-starved: return-to-go has SD 0.051 against option differences of about 0.005,
   which would need about 400 rounds, and a run has 50. With the switching margin, their draws still deviate from
   the default about 50% of the time. The ablation (MEMORY_GENONLY) attributes the carry effect mainly to
   memory → generation.
3. **Where the carry-over value is lost.** Persistent memory produces measurably better candidates, and more so
   with experience. The carried memory also predicts screen outcomes far better than a within-world memory
   (corr 0.35 vs 0.12). But the adopted edits' unbiased effect on the tracking probes improves only about 15%.
   Choosing one winner from 8 candidates on 6 noisy probes dilutes the advantage. Memory cannot fix that
   selection step: its predictions do not correlate with an adopted edit's unbiased effect (≈0), so
   memory-driven adoption was not attempted.

## II.4 The go/no-go look (seeds 3201–3300, reserved, never run before)

The criterion (ledger `GO_NO_GO_CRITERION`, phase `process-controller`) was recorded **before** any seed in
3201–3300 ran. It fixed:
* the configuration (the pc-05 code and HP, by hash);
* the arms;
* five conditions, all required.

`experiments/pc_gonogo.py` was committed beforehand and computed the verdict mechanically. The ledger shows the code
and HP hashes unchanged from the criterion through the end of the look.

| condition (n = 100, 10 worlds per run, 4.976 M executions per run for every arm) | result | |
|---|---|---|
| MEMORY_CARRY − NO_CARRY, final external holdout, one-sided p < 0.05 | **+1.29** [−0.17, +2.70], p = 0.046; 56 wins / 3 ties / 41 losses | pass (narrowly) |
| MEMORY_CARRY − FROZEN, final external holdout, p < 0.05 | **+2.08** [+0.45, +3.75], p = 0.007 | pass |
| MEMORY_CARRY − NO_CARRY, world-end cross-family transfer on the tracking probes, p < 0.10 | +0.0046 per probe [+0.0009, +0.0084], p = 0.010 | pass |
| persistence: MEMORY_CARRY − NO_CARRY per world over worlds 6–10 > 0 | +0.22 per world (worlds 1–5: +0.04) | pass |
| decisions change: MEMORY_CARRY's pool overlap with the default pool (worlds ≥ 2) < 0.8 | 0.39 | pass |
| **verdict** | **GO** | |

Informational (from the same look): NO_CARRY − FROZEN was +0.79 (p = 0.15). MEMORY_CARRY − MEMORY_RANKONLY was
**+0.08** (n.s.). On this look, persistent memory used for ranking alone did as well as persistent memory used for
process control, unlike on dev (+4.13 under the old prior, +1.56 under the stronger prior). Per world,
MEMORY_CARRY − NO_CARRY was 0.00, −0.11, −0.22, +0.18, +0.36, −0.10, +0.46, +0.30, +0.36, +0.06. The GO triggered
the pre-registration and freeze **before** any confirmatory seed ran.

## II.5 Confirmatory battery (seeds 7001–7300, run exactly once after `PREREG_FREEZE`)

**Protocol.** `results/PREREGISTRATION_V3.json` was written from the unchanged go/no-go code and HP and committed
first. The one-shot `PREREG_FREEZE` then hashed:
* the pre-registration;
* the HP;
* every `rsi_v3` and `rsi_v2` source file;
* the evaluator;
* the task library;
* the confirm-seed manifests.

With the protocol frozen, `test_frozen_protocol_if_frozen` runs and passes. All 1,500 units (5 arms × 300 seeds)
ran once: 0 missing, 0 abandoned, 0 restarts. The frozen runner then produced `results/v3_confirm_report.json` and
the ledger's `REPORT` record. `python3 -m rsi_v3 verify` reports that the chain and all hashes match.

### II.5.1 Means per run (final holdout `ext` out of 240 tasks over 10 worlds; `in` out of 160)

| arm | ext | in | macros / world | executions spent (cap 4,976,000) |
|---|---|---|---|---|
| MEMORY_CARRY | **65.89** | **49.85** | 6.91 | 4,236,838 |
| MEMORY_GENONLY | 65.63 | 49.75 | 6.98 | 4,112,962 |
| MEMORY_RANKONLY | 65.28 | 49.34 | 5.20 | 4,118,668 |
| NO_CARRY | 64.36 | 48.17 | 6.47 | 4,131,922 |
| FROZEN | 64.11 | 47.37 | 4.89 | 4,133,244 |

MEMORY_CARRY spends about 2.5% more of the identical cap: its learned attempt modes use attempt budget that the
default leaves idle. That is not the source of the gain. MEMORY_GENONLY spends *less* than FROZEN and gains the same
+1.52.

### II.5.2 Confirmatory tests (one-sided paired sign-flip permutation; fixed-sequence gatekeeping, Holm among H2–H4; α = 0.05)

| hypothesis | field | mean | 95% CI | w / t / l | p | adjusted p | verdict |
|---|---|---|---|---|---|---|---|
| H1 MEMORY_CARRY − NO_CARRY | ext | **+1.523** | [+0.55, +2.48] | 174 / 16 / 110 | 0.0011 | 0.0011 | **supported** |
| H2 MEMORY_CARRY − FROZEN | ext | **+1.777** | [+0.85, +2.73] | 176 / 14 / 110 | 0.0002 | 0.0006 | **supported** |
| H3 NO_CARRY − FROZEN | ext | +0.253 | [−0.67, +1.15] | 132 / 15 / 153 | 0.30 | 0.60 | null |
| H4 growth (MEMORY_CARRY − NO_CARRY, worlds 6–10 minus 1–5) | ext_late_minus_early | +0.050 | [−0.84, +0.95] | 136 / 17 / 147 | 0.46 | 0.60 | null |

### II.5.3 Pre-registered exploratory contrasts (two-sided, descriptive)

| contrast | mean | 95% CI | p |
|---|---|---|---|
| MEMORY_CARRY − MEMORY_RANKONLY (process control beyond ranking) | +0.60 | [−0.24, +1.47] | 0.19 |
| MEMORY_CARRY − MEMORY_GENONLY (round-level options) | +0.26 | [−0.60, +1.14] | 0.56 |
| MEMORY_GENONLY − NO_CARRY (persistent memory → generation) | +1.27 | [+0.37, +2.15] | 0.006 |
| MEMORY_RANKONLY − FROZEN (persistent memory → ranking) | +1.17 | [+0.28, +2.06] | 0.011 |
| MEMORY_CARRY − NO_CARRY, in-family holdout | +1.68 | [+0.89, +2.48] | 0.0001 |
| MEMORY_CARRY − FROZEN, in-family holdout | +2.48 | [+1.66, +3.30] | < 1e-4 |

## II.6 Verification checks (in the frozen report)

| check | result |
|---|---|
| **Does the advantage grow as experience accumulates?** | Per world, MEMORY_CARRY − NO_CARRY: 0.00, +0.30, +0.26, −0.13, +0.30, +0.19, +0.07, +0.05, +0.29, +0.19. That is +0.15 in worlds 1–5 and +0.16 in worlds 6–10; the per-seed trend is n.s. **The advantage appears from world 2 on and persists, but does not grow.** |
| **Does memory improve future self-improvement on unseen families** (world-end cross-family transfer, tracking probes never used for selection)? | MEMORY_CARRY − NO_CARRY **+0.0041/probe [+0.0022, +0.0060], p = 0.0001**. Against FROZEN +0.0066 (p < 1e-4), against MEMORY_GENONLY +0.0022 (p = 0.006), against MEMORY_RANKONLY +0.0023 (p = 0.013). |
| **Does the controller change actual process decisions?** | Mean pool overlap with the default pool by world: MEMORY_CARRY 0.51 → 0.39 → 0.40; NO_CARRY about 0.50 flat; FROZEN and MEMORY_RANKONLY 1.00 by construction. MEMORY_CARRY's round-level options are non-default in about 50% of rounds from world 3 (the options model first has ≥ 10 rows at world 3). **Yes: memory changes what is generated and how compute and probes are allocated, not just internal scores.** |
| Does the learned value predict real future cross-family gain? | Within-round rank validity for MEMORY_CARRY is +0.015 (p = 0.0001): positive but tiny. The learned generation has already filled the pool with similar-valued variants, which restricts the range. |
| Identical task / RNG / compute conditions | Identical caps (4,976,000 per run, 497,600 per world) for all 1,500 units. Metered spend equals the process-global counter in every unit. Streams are arm-independent (AST test). |
| Meta-controller compute | MEMORY_CARRY per run: 4,395 training rows, 100 closed-form refits, 2,552 predictions, 9,157 episodic retrievals, 5.6 Mflop, 0.26 s wall. **0 program executions** (asserted), against 4.24 M metered program executions in the same run. |
| No final-holdout leakage | Final rows are sealed at runtime while any improver runs and absent from its view. Learning and tracking rows come from META-VAL probes only. No code path from improver or controller to the final holdout (AST). The memory holds no solver content. All tested. |

## II.7 Causal reading: what the carried memory does

The arms form a ladder that isolates each channel. On the confirmatory battery:

| step | final-holdout effect | reading |
|---|---|---|
| FROZEN → NO_CARRY (the learner, memory wiped every world) | +0.25 (n.s.) | within one 5-round problem the learner has too little experience to beat an uninformed prior |
| NO_CARRY → MEMORY_GENONLY (memory persists; it drives generation, allocation and ranking) | **+1.27** (p = 0.006) | carried experience about *which kinds of edits to generate* is the main channel |
| MEMORY_GENONLY → MEMORY_CARRY (plus round-level options) | +0.26 (n.s.) | compute / probe / exploration / adoption options add nothing detectable |
| FROZEN → MEMORY_RANKONLY (memory persists, used for ranking only) | **+1.17** (p = 0.011) | carried experience also helps when it only reorders a fixed generator's pool |
| MEMORY_RANKONLY → MEMORY_CARRY | +0.60 (n.s.); transfer +0.0023 (p = 0.013) | process control beyond ranking: positive, not established on the final metric |

**What the memory learns** (dev diagnostics, confirmed by the adopted edits in the confirmatory battery):
* Avoid flattening the search prior. Share of adoptions that are EXPLORE edits in the confirmatory battery:
  FROZEN 16.4%, MEMORY_RANKONLY 10.9%, NO_CARRY 9.1%, MEMORY_CARRY 6.1%, MEMORY_GENONLY 5.8%.
* Avoid pruning (2.3% of FROZEN's adoptions, 0.7% of MEMORY_CARRY's).
* Favour mined macro bundles (MINE 50.5% of MEMORY_CARRY's adoptions vs 35.9% of FROZEN's), often combined with a
  prior refit, and residual-guided repairs (RESID 8.4% vs 5.9%).

This knowledge is at the level of *edit variants*, where the improver's META-VAL signal agrees with the final
holdout. It needs far more observations than one 5-round problem supplies, which is why only carried memory
exploits it. It also saturates within a world or two, which is why the advantage does not keep growing.

## II.8 Limitations and honest boundaries

* **Small effect in a synthetic domain.** +1.5 of about 64 final-holdout tasks per run (+2.4%). Transfer is to
  unseen *families* of the same stack substrate, not to other domains.
* **No growth.** The pre-registered growth hypothesis is null. A memory that kept improving the improver would show
  a widening gap; this one reaches its useful level after about one world.
* **The strong ordering is not confirmed.** NO_CARRY ≈ FROZEN. The benefit exists only with carried memory.
* **Process control vs ranking.** On the final metric the pre-registered contrast between full process control and
  ranking-only memory is +0.60 (n.s.). The round-level options (compute allocation, probe allocation, exploration,
  adoption rule) are demonstrably *used*: about 50% non-default. But they are data-starved: return-to-go has SD ≈
  0.05 against option effects of about 0.005, and a run has only 50 decisions. Their contribution is not detectable.
* **The first control was too weak, and it was caught.** Under the v3 variant list, NO_CARRY − FROZEN was +4.7 on
  dev. It fell to +0.6 once a fixed whole-grid prior was used, and that stronger prior is what every confirmatory
  arm uses. Other untested fixed policies might be stronger still.
* **The go/no-go passed narrowly** (p = 0.046 on the primary). The confirmatory effect (+1.52) matches the
  go/no-go estimate (+1.29) and the dev estimates (+0.75 to +2.19). The confirmation is what carries the claim.
* **Dev iterations.** Five dev batteries plus a sweep and a control check on dev seeds 3001–3100 preceded the look.
  That is why the look used fresh reserved seeds, a pre-written criterion and a hash-pinned configuration, and why
  the confirmation used never-run seeds under a one-shot freeze.

## II.9 Reproduction

```bash
python3 -m unittest discover -s tests -v                      # 76 tests (v2 40 + v3 36)
(cd src && python3 -m rsi_v3 verify)                          # chain + frozen hashes; 0 missing / 0 abandoned
(cd src && python3 -m rsi_v3 report --out /tmp/rep.json)      # recomputes the frozen report (appends a REPORT record)
python3 experiments/pc_analysis.py pc-devcheck-01             # go/no-go look tables (read-only)
python3 experiments/pc_analysis.py pc-05-griddefault          # last dev battery (read-only)
# re-run any confirmatory unit from scratch and compare every behavioural field
python3 - <<'PY'
import sys, json; sys.path.insert(0, 'src')
from rsi_v3 import runner as R, improver as I
u = [json.loads(l) for l in open('results/logs/v3_confirm_units.jsonl')][0]
new = R.compact(R.run_unit((u['arm'], u['seed'], I.HP)))
print(all(new[k] == u[k] for k in ('ext', 'ext_w', 'final_cfg_shas', 'spent', 'plans_w')))
PY
```

Files for this part:
* `src/rsi_v3/controller.py` (memory + controller).
* `src/rsi_v3/improver.py`: `run_process`, `build_grid`, and the arms.
* `experiments/pc_analysis.py`, `experiments/pc_gonogo.py`, `experiments/pc_make_prereg.py`.
* `results/PREREGISTRATION_V3.json` and `results/v3_confirm_report.json`.
* Logs: `results/logs/v3_confirm_units.jsonl`, `v3_confirm_detail.jsonl.gz`, `v3_dev_units.jsonl` (labels `pc-*`).
* The ledger: phase `process-controller` records, the `GO_NO_GO_CRITERION` / `GO_NO_GO_RESULT`, and
  `PREREG_FREEZE`, `UNIT_*` and `REPORT`.
