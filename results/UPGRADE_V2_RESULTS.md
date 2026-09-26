# UPGRADE V2 — adapting the improvement process itself, under compute-matched, leak-proof controls

**Bottom line: honest and mixed.**

| Preregistered hypothesis (n = 300 fresh seeds, identical 610,000-execution cap, external held-out families) | Result |
|---|---|
| **H1 (primary): RECURSIVE_FULL > SINGLE_5X** | **Supported, but small.** **+0.40** tasks of 24 (95% CI [+0.10, +0.70]); 131 wins / 64 ties / 105 losses; one-sided p = 0.0049, **Holm p = 0.0098 < 0.05**. |
| **H2 (secondary): RECURSIVE_FULL > RECURSIVE_FROZEN** | **Null.** **−0.08** (CI [−0.35, +0.19]); 124/56/120; p = 0.73. |

* **What the evidence supports:** five rounds of recursive self-improvement beat one round with 5× the compute
  (same tasks, same attempt streams, same total executions, an equally strong gate). The gain appears on tasks the
  system never trained on, from task families it never saw. Exploratory: the *frozen-policy* recursive arm beats
  SINGLE_5X by the same margin (+0.48, CI [+0.20, +0.75]). So the H1 gain comes from **recursion on a
  difficulty ladder** (vocabulary built on vocabulary: 13.8 vs 7.8 training tasks solved, 2.7 vs 1.0 hierarchical
  macros, gains concentrated on 5–12-primitive tasks). It does **not** come from adapting the improver.
* **What it does not support:** *adapting the improvement process itself* produced no measurable capability gain.
  The adaptive improver learned something: it screens and adopts differently, mines more boldly and ends with
  more macros. None of that reached the external holdout. Gains larger than +0.19 tasks are excluded at 95%.
  Its failure-mode context was inert: the context-free ablation gave the identical result in 297/300 seeds.
* **The v1 compounding headline is retracted** (Part A). It was produced by arm-keyed evaluation randomness plus a
  data confound. Under fair comparison, the v1 chain is below the untrained baseline.
* Noise floor (COLD re-scored on a second eval stream): −0.03, p = 0.73. Every arm spent ≤ its cap, and in all
  1,500 units the metered spend equals the process-global execution counter.


---

## Part A — Audit of the v1 headline (before changing anything)

### A1. Checklist

| Requirement | v1 status | Evidence / fix |
|---|---|---|
| Equal total candidate-evaluation compute | **Partly.** R5PLUS, ROUND5 and ROUND1_5X share a 62,500-evaluation cap. **GATED5 did not:** its gate spent up to 5 × 2 × 3 × 800 = 24,000 extra evaluations that were never counted (mean 51,081, max 70,592; over the cap in 4/100 seeds). | `xv_build_chains` now records `gated_gate_evals` / `gated_total_evals` / `gated5_compute_matched`. |
| Identical evaluation tasks | Yes (`xv_eval_tasks(seed)` is shared by every arm). | — |
| **Identical RNG streams across arms** | **No.** Every held-out search was seeded `'ev|<cond>|seed|lvl|j'`. The **arm name was in the key**, so counterfactual arms were scored under different search randomness. The same pattern was in `xvi_run_unit`, `gx_run_unit`, `up_eval_unit`, `rsi_upgrade.eval_unit`, `sdt_layer` evaluation, and `transferforge.try_targets` (`'T-<arm>|…'`). | New `omniforge.lf_eval_prng(cond, seed, lvl, j)`: the key is `'ev|s<stream>|seed|lvl|j'`, with stream 0 for every arm. Only the COLD2 noise-floor replicate uses stream 1, on purpose. `transferforge` now uses `'T-solve|name|seed'`. Test: `test_v1_audit_fix_eval_stream_is_arm_independent`. |
| Isolated train / probe / holdout | Yes at the behaviour-pool level: source/target halves per level, probes from the source half. | — |
| Frozen confirmatory seeds | Yes (dev 1–12; holdout 101–140; Kaggle 141–200). | — |
| Immutable scoring / evaluation | Not enforced. The evaluator lived in the same file as the improver, with no hash freeze. | v2 hashes the evaluator and the whole package into a one-shot ledger freeze. |
| **Data matching of the 1-round control** | **No.** R5PLUS attempts **25 distinct** source tasks (5 rounds × 5). ROUND1_5X attempts **5** tasks with 5× budget each. "Recursion beats one round at 5× compute" was confounded with "saw 5× more distinct tasks". | New control R1PLUS_DM: the *same* 25 tasks, the same 62,500 budget, and one presence fit with the cumulative trust-region step 1 − (1 − λ)⁵. |

### A2. Rerun of the v1 XV2 battery (seeds 101–200, n = 100)

`python3 experiments/audit_v1_rerun.py run 101 200 4 && python3 experiments/audit_v1_rerun.py report`
(raw: `results/logs/audit_v1_rerun.jsonl`). Both scorings run on *identical* priors per seed. The arm-keyed scoring reproduces
the published numbers exactly (e.g. local +1.550, Kaggle +1.467).

| Contrast (combined n = 100) | arm-keyed streams (as published) | **shared streams (fixed)** |
|---|---|---|
| R5PLUS − ROUND1_5X (the v1 "compounding" headline) | +1.500 [+1.09, +1.89], p < 1e-4 | **+1.300 [+0.90, +1.71], p < 1e-4** |
| R5PLUS − COLD (absolute lift) | −0.180, p = 0.36 (reported "ns") | **−0.700 [−1.06, −0.34], p = 0.0005** |
| ROUND1_5X − COLD | −1.680 | −2.000 |
| ROUND5 − COLD | −2.050 | −2.140 |
| **R5PLUS − R1PLUS_DM (data-matched single round)** | — | **−0.590 [−0.94, −0.24], p = 0.0018** |
| R1PLUS_DM − ROUND1_5X | — | +1.890, p < 1e-4 |
| R1PLUS_DM − COLD | — | −0.110, p = 0.62 |
| R5PLUS − R5PLUS_NG (gate ablation) | — | −0.150, p = 0.42 |
| GATED5 − ROUND5 | — | +0.630, p = 0.007 (GATED5 is **not** compute-matched, see A1) |
| COLD2 − COLD (noise floor, deliberate second stream) | — | −0.330, p = 0.07 |

### A3. Corrected v1 claims

1. **The v1 "compounding RSI" headline does not survive.** Its +1.3 advantage over ROUND1_5X is explained by
   data diversity: a *single* round over the same 25 tasks, at the same budget, is **better** than five recursive
   rounds (−0.59, p = 0.002).
2. **The repaired chain is significantly below the untrained baseline** once arms share streams (−0.70,
   p = 0.0005). v1 had reported this as "+0.10 ns / −0.37 ns".
3. Nothing in the v1 prior-refit family beats COLD on held-out tasks.

README, `results/UPGRADE_RESULTS.md`, `results/FINAL_RESULTS.md` and `wiki/Results.md` now carry these
corrections.

### A4. The same RNG bug in other v1 experiments: fixed and rerun

| v1 experiment | Arm-keyed stream? | Rerun with shared streams | Verdict |
|---|---|---|---|
| **Cross-substrate transfer** (`transferforge.py`) | yes: target solves seeded `'T-<arm>|…'` | seeds 1–11 (the claimed range): B_skill − B_alone **+2.00** (p = 0.001); learning premium B_skill − B_rand **+1.00** (p = 0.008). Extension seeds 1–60: +1.98 and **+1.28** (both p < 1e-4); STRUCT control ties 60/60. Raw: `results/logs/transfer_log_shared_stream.jsonl`. | **Survives, and replicates at n = 60.** Also found: the committed raw log held only seeds 1–3 of the claimed n = 11. All 60 are committed now. |
| **SDT gate integrity** (`sdt_layer.py`) | yes: eval seeded `'ev|<arm>|…'` | Collapse findings unchanged (they do not use eval streams): CLOSED 40/40 at round 3, WIRE 9/40. **Eval competence:** ARB − FULL = −0.18 (p = 0.65); WIRE − FULL = −0.70 (p = 0.04); CLOSED − FULL = −0.60 (p = 0.04). Raw: `results/logs/sdt_log_shared_stream.jsonl`. | Collapse-order claim survives. **"SDT_ARB has the worst eval competence" does not**; WIRE and CLOSED are the worst. |
| MetaForge adaptive vs frozen (omniforge `rsi` section) | no: streams keyed by (wave, task id) | — | Unaffected (still one deterministic task stream, no seed variance). |
| tforge / openforge | eval stream shared (tforge); openforge's metric is certified novelty, which the CLOSED arm cannot produce by construction | — | Unaffected. |


---

## Part B — The v2 system (`src/rsi_v2`)

v1 improved the *solver* with one fixed rule. **v2 also improves the improver.** Each round runs:

```
attempt ─► diagnose failure ─► choose strategy ─► build candidate solver ─► counterfactual test ─► adopt / rollback ─► learn ─┐
   ▲                                                                                                                        │
   └──────────────── next round uses the adopted solver AND the updated improvement policy ◄────────────────────────────────┘
```

* **Object level (what gets improved):** a token-level evolutionary program searcher over the 20 omniforge list
  primitives. Its `SolverConfig` holds learned **macros** (multi-primitive operators, content-addressed), a
  sampling **prior**, a length bound and a random-restart rate. The search code itself never changes.
* **Failure diagnosis** comes from the round's own search traces (never holdout). The five modes are MISSING_OP,
  BAD_ORDER, LOW_EXPLORE, OVERSPEC and COMPOSE_FAIL.
* **Improvement strategies:**
  * MINE: behaviour-level mining of frequent fragments from Occam-simplified solutions.
  * COMPOSE: macro-of-macro composition from adjacent tokens in solutions and near-misses. This is the reuse and
    composition channel.
  * REFIT: trust-region prior refit.
  * EXPLORE: flatten the prior and raise the random-restart rate.
  * PRUNE: drop the least-used macros.
  * Four stacked combos (e.g. MINE+COMPOSE+REFIT).
* **Counterfactual test:**
  1. The policy may *screen* at most 4 candidates on 6 fresh probe tasks.
  2. The screening winner is *confirmed* on 6 more fresh probes.
  3. It is adopted only if it is **not worse on the confirmation probes** and strictly better over the batch.
     Otherwise the incumbent stays (rollback).

  Every probe search uses the same stream for the incumbent and every candidate. An unsolved probe costs the full
  budget, so finishing early with a train-fitting-but-wrong program earns nothing.
* **What "learning to improve" means here** (the only thing RECURSIVE_FULL has that RECURSIVE_FROZEN lacks):
  1. **Contextual strategy values** Q[failure mode][action]. They are partially pooled toward a context-free value
     and ranked with a UCB bonus, and they decide which ≤ 4 of the 9 actions get the scarce test budget.
  2. **Self-tuned strategy intensities** (how many macros to mine or compose, the prior step size, the flattening,
     the prune fraction). An adoption makes the strategy bolder; negative evidence makes it more conservative.
  3. **Delayed credit:** after an adopted vocabulary change, the *next* round's new solves are checked for
     whether they use the new macros, and that is fed back into the strategy's value.
* **Arms.** All share one cap of **610,000 program executions**. Every execution is metered, and the metered total
  must equal the process-global execution counter.

| arm | improvement phase |
|---|---|
| COLD | none (reference) |
| SINGLE_5X | one round: every train task gets the **same 5 attempt streams** the recursive rounds use (as restarts); all 9 actions plus 3 bold 3× variants are tested by staged successive halving (6 → 12 probes); the finalist is confirmed on the remaining 42 probes. 5× attempt and 5× gate budget. |
| RECURSIVE_FROZEN | 5 rounds; the improvement policy (Q, intensities) stays at its initial, uninformed value |
| RECURSIVE_FULL | 5 rounds; the improvement policy adapts |
| RECURSIVE_NODIAG | (ablation) adaptive, but context-free: ignores the diagnosis |

* **Tasks** (`tasks.py`, frozen and hashed) are written by the experimenter, never by the system. A hidden library
  of 8 two-primitive "atoms" and 6 four-primitive "compounds" feeds 16 families (3 motifs + 2 glue primitives each).
  A task chains 1–3 motifs, so programs run 2–12 primitives. Per seed, 10 families train and 6 are **external**.
  * **train (40):** training tasks.
  * **probe (60):** the improver's validation data.
  * **holdout_in (16):** new behaviours from the training families.
  * **holdout_ext (24):** tasks from the 6 external families only.

  Every task has a distinct behaviour across all splits. Train inputs have length 6–9; the hidden test inputs are
  longer (length 10–14).
* **Primary metric:** `holdout_ext` tasks solved (of 24) by the final solver. It is scored by the frozen evaluator
  with 3,000 executions per task and a stream keyed only by (seed, split, task index).

---

## Part C — Research discipline

* **Dev iterations (all in the ledger, seeds 1–12 only).** The design was iterated with arm contrasts visible
  (they are recorded). That makes dev estimates optimistic; only the untouched confirmatory seeds can be trusted.

| ledger | label | what changed and why | FULL − SINGLE_5X | FULL − FROZEN |
|---|---|---|---|---|
| 1 | dev-00 (retroactive) | first smoke test: probes were almost all long tasks, so gates never discriminated | — | — |
| 2 | dev-01 (retroactive) | initial design: the gate fit ~1 candidate per round | −1.12 | 0.00 |
| 4 | dev-02 | stacked combos, racing gate | +1.12 | 0.00 (racing tested all 9 actions, so the policy was irrelevant) |
| 6 | dev-03 | policy may test only k = 3 per round; 60 probes | −1.38 | +0.38 |
| 8 | dev-04 | screen → confirm gate (winner's curse); more single-motif footholds | −0.33 | −0.42 |
| 10 | dev-05 | behaviour-level mining, Occam simplification, bigger budgets | −1.67 | −1.33 |
| 11 | stopping rule | at most 3 more iterations; freeze chosen on arm-agnostic grounds only | | |
| 13 | dev-06 | fixed the cost-tiebreak loophole; intensity shrinks only on *negative* evidence; delayed credit; SINGLE_5X gets a confirmation stage | +1.83 | +0.42 |
| 15 | dev-07 (**frozen**) | *strengthened the control*: SINGLE_5X gets bold 3× macro variants | +1.08 | +0.42 |

* **Preregistration:** `results/PREREGISTRATION_V2.json` was frozen as ledger record 16 (`PREREG_FREEZE`). It
  commits to:
  * the prereg sha, the code sha of every `rsi_v2` file, the evaluator sha and the hyperparameter sha;
  * the task-library digest and the confirm-manifest digest;
  * seeds 1001–1300.

  It was committed and pushed as `62cbfe2` **before any confirmatory unit ran**. The runner refuses to run or report
  if any hash drifts, and the freeze is one-shot.
* **Primary test:** RECURSIVE_FULL − SINGLE_5X on `ext`, a one-sided paired sign-flip permutation test with 20,000
  permutations. **Secondary:** RECURSIVE_FULL − RECURSIVE_FROZEN. Holm correction over the two, α = 0.05.
* **Stopping:** each (arm, seed) runs exactly once with no interim looks; the report states whatever comes out.

---

## Part D — Confirmatory results (frozen report: `results/v2_confirm_report.json`)

Seeds 1001–1300 (n = 300), 5 arms, 1,500 units. All ran exactly once after the freeze: 0 missing, 0 abandoned,
0 restarts. The ledger is intact (verify: `python3 -m rsi_v2 verify`), and the report is recorded as a `REPORT` ledger record.

### D1. Means per seed

| arm | external holdout (of 24) | in-family holdout (of 16) | train solved (of 40) | final macros | spent (cap 610,000) |
|---|---|---|---|---|---|
| COLD | 4.39 | 2.34 | 0 | 0 | 0 |
| SINGLE_5X | 6.80 | 4.30 | 7.76 | 3.89 | 503,714 |
| RECURSIVE_FROZEN | **7.27** | **4.62** | 13.80 | 5.35 | 501,200 |
| RECURSIVE_FULL | 7.19 | 4.37 | 13.42 | 6.27 | 503,056 |
| RECURSIVE_NODIAG | 7.18 | 4.38 | 13.42 | 6.25 | 503,062 |

### D2. Confirmatory tests (one-sided paired sign-flip permutation, Holm over H1 and H2, α = 0.05)

| | contrast | mean | 95% CI | W/T/L | p | Holm p | verdict |
|---|---|---|---|---|---|---|---|
| H1 | RECURSIVE_FULL − SINGLE_5X | **+0.397** | [+0.10, +0.70] | 131/64/105 | 0.0049 | **0.0098** | **supported** |
| H2 | RECURSIVE_FULL − RECURSIVE_FROZEN | −0.080 | [−0.35, +0.19] | 124/56/120 | 0.728 | 0.728 | **null** |

The effect is far smaller than the dev estimate (+1.08). That was expected, because dev estimates survived design
iteration. H1's standardized effect is dz = 0.15.

### D3. Preregistered exploratory contrasts (two-sided; no multiplicity correction; descriptive)

| contrast | metric | mean | 95% CI | p |
|---|---|---|---|---|
| RECURSIVE_FROZEN − SINGLE_5X | ext | +0.477 | [+0.20, +0.76] | 0.0012 |
| RECURSIVE_NODIAG − RECURSIVE_FULL | ext | −0.013 | [−0.05, +0.02] | 0.75 (identical in 297/300 seeds) |
| RECURSIVE_FULL − COLD | ext | +2.80 | [+2.52, +3.10] | < 1e-4 |
| SINGLE_5X − COLD | ext | +2.41 | [+2.16, +2.67] | < 1e-4 |
| RECURSIVE_FROZEN − COLD | ext | +2.88 | [+2.61, +3.16] | < 1e-4 |
| RECURSIVE_FULL − SINGLE_5X | in-family | +0.08 | [−0.16, +0.33] | 0.56 |
| RECURSIVE_FULL − RECURSIVE_FROZEN | in-family | −0.25 | [−0.46, −0.04] | 0.02 |
| COLD replicate stream − COLD | ext | −0.03 | [−0.16, +0.11] | 0.73 (noise floor) |
| RECURSIVE_FULL − COLD | self-generated tasks | −0.62 | [−0.81, −0.43] | < 1e-4 |
| RECURSIVE_FULL − SINGLE_5X | train solved | +5.66 | [+5.24, +6.10] | < 1e-4 |

Reading:

* **Recursion compounds.** It holds against a data-matched, stream-matched, compute-matched one-shot control that
  was deliberately strengthened during dev (confirmation stage, bold 3× variants).
* **The adaptive policy does not help,** and on the in-family holdout it is slightly worse than the frozen policy
  (exploratory, uncorrected).
* **Self-generated-task scores go *down*** for improvers (their own vocabularies generate harder tasks) while
  external scores go up. This is why they are never counted.


---

## Part E — Mechanism analysis (post hoc, exploratory; full tables in `results/V2_MECHANISM_ANALYSIS.md`)

**Compounding is visible in the training trajectory.** Recursive arms re-attempt unsolved tasks with the
improved solver each round. Training solves climb 5.3 → 8.2 → 10.5 → 12.4 → 13.8 (FROZEN). SINGLE_5X spends
the same attempt compute as five restarts with the base solver and reaches 7.8.

| arm | final macros | hierarchical macros (contain another learned macro) | max macro length |
|---|---|---|---|
| SINGLE_5X | 3.89 | 1.00 | 2.49 |
| RECURSIVE_FROZEN | 5.35 | 2.71 | 3.92 |
| RECURSIVE_FULL | 6.27 | 3.61 | 3.84 |

**Composition and reuse.** Later rounds build macros on earlier ones. After an adopted vocabulary change, the
next round's newly solved tasks use the new macros in 2.6 tasks on average (85% of 525 credit events). The one-shot
arm can only mine what the base solver reaches.

**Where the external gain lives** (external holdout solve rate by hidden-program length):

| arm | len 2–4 | len 5–6 | len 7–8 | len 9–12 |
|---|---|---|---|---|
| COLD | 0.681 | 0.072 | 0.025 | 0.002 |
| SINGLE_5X | **0.755** | 0.262 | 0.094 | 0.022 |
| RECURSIVE_FROZEN | 0.710 | **0.306** | **0.134** | **0.050** |
| RECURSIVE_FULL | 0.744 | 0.280 | 0.129 | 0.042 |

This is the ladder signature. Recursion helps on long compositions and costs a little on short tasks, because a
bigger vocabulary dilutes the search.

**What the adaptive improver learned, and why it did not pay off:**

* **Final intensities:** MINE 3.53 (start 3.0), COMPOSE 2.39 (2.0), REFIT 0.43 (0.4), EXPLORE 0.23 (0.3),
  PRUNE 0.28 (0.3). It became bolder at vocabulary growth and more conservative at flattening and pruning, which
  gave more and longer macros than FROZEN.
* **Screening changed:** in rounds 3–5 under MISSING_OP, FULL screened PRUNE (295), REFIT (257) and PRUNE+MINE
  (254) most. FROZEN screened uniformly. The learned preferences follow the noisy screening rewards, not a
  designer's mapping ("missing operator → mine").
* **The context was inert.** Diagnoses were MISSING_OP 58.6%, LOW_EXPLORE 22.7%, BAD_ORDER 18.3% and OVERSPEC
  0.4%; COMPOSE_FAIL never fired. With about 20 outcomes per run spread over (mode, action) cells, the partially
  pooled contextual values never ranked differently from the context-free ones: NODIAG = FULL in 297/300 seeds.
* **Net:** the adaptive policy is a working learner inside a 5-round, 12-probe-per-round signal budget, but that
  budget is too noisy and too short for strategy learning to beat a random strategy order. This is the concrete,
  measured reason H2 is null.


---

## Part F — Anti-cheat audit (executable: `python3 -m unittest discover -s tests -v`)

| Threat | How it would fake a result | Defense | Test(s) |
|---|---|---|---|
| Extra hidden compute | An arm runs programs outside its budget | Every execution goes through `substrate.execute` (global counter). Arms may only execute through a `Meter`; `run_arm` asserts global delta == metered spend. The evaluator does the same. | `test_hidden_compute_in_improver_is_detected`, `…_in_evaluator_is_detected`, `test_equal_caps_and_within_cap`, `test_phase_budgets_sum_to_same_cap`, `test_meter_is_a_hard_cap`, `test_search_respects_budget` |
| Condition-specific RNG | Arms get luckier search streams | Every stream key is (seed, purpose, indices), with no arm; the evaluator stream is (seed, split, k) | `test_eval_stream_has_no_arm`, `test_improvement_streams_never_use_arm` (AST), `test_round1_identical_across_recursive_arms` (bit-identical round 1), `test_single_uses_the_recursive_attempt_streams`, `test_v1_audit_fix_eval_stream_is_arm_independent` |
| Holdout leakage | The improver sees holdout tasks | Behaviour-disjoint splits; external families never in train or probe; `improver_view` strips holdout; a `sealed()` tripwire around the whole improvement phase; no AST path from improver or solver to holdout loaders | `test_behaviour_disjoint_splits`, `test_improver_view_has_no_holdout`, `test_improvement_phase_never_reads_holdout`, `test_improver_and_solver_have_no_holdout_code_path` |
| Seed / task-ID memorisation | A lookup keyed by id or seed | The solver gets `PublicTask(train)` only (no id, family, program or test); its signature has no seed; the final config is macros, weights and two scalars only | `test_public_task_exposes_train_only`, `test_task_ids_do_not_affect_results` (relabel ids ⇒ identical results), `test_solver_api_receives_no_seed_or_id`, `test_final_config_carries_no_answers` |
| Hard-coded answers | Outputs baked into code or config | Source scan for every holdout test output (dev and confirm seeds); a shuffled-label control shows the scorer checks the real hidden outputs | `test_no_holdout_outputs_hardcoded_in_source`, `test_shuffled_labels_are_not_solved` |
| Evaluator modification | Edit the scorer after seeing results | sha256 of every `rsi_v2` file plus a separate evaluator sha, frozen in the ledger; `check_frozen` runs before confirm and report | `test_protocol_hashes_match_freeze`, `test_evaluator_tampering_is_detected`, `test_hp_changes_are_detected` |
| Weakening success criteria | Raise α, switch metric or contrast | The prereg sha is frozen; the report asserts the primary metric is the external holdout | `test_weakened_criteria_are_detected`, `test_prereg_criteria_are_strict` |
| Selective deletion of failed runs | Drop bad seeds or rerun until lucky | Hash-chained append-only ledger (delete, edit and reorder break it); START before END (abandoned runs listed); the report refuses a verdict with missing units; freeze is one-shot; completed units are never re-run; dev seeds and confirm seeds are disjoint | `test_deleted_record_breaks_chain`, `test_edited_record_breaks_chain`, `test_reordered_records_break_chain`, `test_deleted_tail_is_visible_as_abandoned`, `test_missing_units_block_the_verdict`, `test_no_confirm_unit_before_freeze`, `test_end_to_end` |
| Self-generated-task gains | Claim progress on tasks the system made up | The self-generated score is logged as a diagnostic only; the verdict reads only the external metric; a "self-gamer" demo | `test_self_score_can_rise_without_external_gain`, `test_report_primary_uses_external_metric_only` |
| Substrate drift | A quietly easier primitive set | rsi_v2 primitives are asserted identical to omniforge's | `test_primitives_match_omniforge`, `test_determinism` |

---

## Part G — Limitations and honest boundaries

1. **Scope of the positive result.** H1 shows compounding in a synthetic program-synthesis domain whose task
   generator has a difficulty ladder by construction. Compounding is impossible without one, so the ladder was
   stated as a design requirement before any confirmatory run. The external families are *novel compositions of
   building blocks shared with the training families*. That is transfer across families, not across unrelated
   domains. Without shared structure, no transfer would be possible by construction.
2. **The effect is small.** +0.40 of 24 tasks (≈ +6% relative to SINGLE_5X). All improvers are far below an
   oracle holding the hidden motif library (≈ 13–15 of 24 on dev seeds).
3. **The qualitative mechanism did not prove itself.** "Adapting the improvement process" (H2) is null, and its
   context sensitivity was inert. This experiment does **not** show that learning *how* to improve beats a frozen
   improvement policy. That would need a richer or longer improvement signal (more rounds, larger probe sets, or
   improvement-policy transfer across runs, whose compute would then have to be counted).
4. **Dev iteration saw arm contrasts.** Eight dev iterations on seeds 1–12, all ledgered. Several changes
   (screen→confirm gating, delayed credit, strengthening the control) were made with dev contrasts visible.
   Freezing, and running the untouched seeds once, is the protection. The dev → confirm shrinkage (+1.08 →
   +0.40; +0.42 → −0.08) shows why that protection was needed.
5. **One frozen configuration.** Budgets, round count and gate sizes were not varied in the confirmatory battery.
   Whether compounding grows with more rounds is untested.
6. **Compute unit.** "Equal compute" means equal program executions, not equal wall-clock time. The meters cover
   every execution in the improvement phase, including gate probes, macro-signature checks and Occam
   simplification.
7. **v1 legacy.** Only the experiments listed in A4 were re-audited. MetaForge remains a single deterministic
   stream with no seed variance.


---

## Reproduction

```bash
# anti-cheat suite (≈1 min)
python3 -m unittest discover -s tests -v

# v1 audit rerun (≈4 min on 4 cores) and report
python3 experiments/audit_v1_rerun.py run 101 200 4
python3 experiments/audit_v1_rerun.py report

# v2: verify ledger + frozen hashes, then (re)produce the confirmatory report
cd src
python3 -m rsi_v2 verify
python3 -m rsi_v2 report --out ../results/v2_confirm_report.json

# v2 battery from scratch (≈45 min on 4 cores): start a fresh ledger path
#   (the committed ledger refuses a second freeze and never re-runs units)
python3 -m rsi_v2 dev --label my-dev --seeds 1-12      # dev seeds only
python3 -m rsi_v2 confirm --workers 4                  # needs PREREG_FREEZE

# post-hoc mechanism tables
cd .. && python3 experiments/v2_analysis.py confirm
```

Raw data:
* `results/ledger/rsi_v2_ledger.jsonl`: every dev iteration, the freeze, and every confirmatory unit START/END
  with full per-task outcomes, hash-chained.
* `results/logs/v2_confirm.jsonl` and `results/logs/v2_dev.jsonl`: the same unit records, one JSON per line.
* `results/v2_confirm_report.json`: the frozen report.
* `results/logs/audit_v1_rerun.jsonl`: the v1 audit rerun.
