# FINAL CONSOLIDATED RESULTS — omniforge RSI program

> **AUDIT CORRECTION (v2 upgrade, see `results/UPGRADE_V2_RESULTS.md` Part A).** The numbers below were
> produced with held-out evaluation streams keyed on the **arm name** (`'ev|<cond>|…'`), and ROUND1_5X saw 5
> distinct source tasks where R5PLUS saw 25. Rerun on the same seeds 101–200 with shared streams and a
> data-matched single-round control (R1PLUS_DM): R5PLUS − ROUND1_5X = +1.30 (p < 1e-4), but
> **R5PLUS − COLD = −0.70 (p = 0.0005)** and **R5PLUS − R1PLUS_DM = −0.59 (p = 0.002)**. The "compounding"
> advantage is a data-diversity confound, not recursion; the repaired chain is below the untrained baseline.
> The original text is kept below unchanged for the record.

### Local sandbox (n=40 holdout) + Kaggle full-budget battery (independent environment, 42 min wall)

Kernel: kaggle.com/code/sunghunkwag/omniforge-rsi-full-battery (private, complete)
Raw artifacts: `kaggle_results/` (KAGGLE_REPORT.md, kaggle_log.jsonl, XIX ledgers)

## 1. Recursive prior self-improvement (XV2, upgraded mechanism) — REPLICATED

The pre-registered compounding criterion (5 recursive rounds vs 1 round at 5×
compute, equal total budget, paired seeds, held-out eval):

| Contrast | Local holdout (n=40, seeds 101–140) | **Kaggle (n=60, fresh seeds 141–200)** |
|---|---|---|
| R5PLUS − ROUND1_5X (compounding) | +1.550, p < 0.0001 | **+1.467, p = 0.00005** |
| R5PLUS − ROUND5 (vs original mechanism) | +2.175, p < 0.0001 | **+1.667, p = 0.00005** |
| ROUND5 − COLD (original chain harm) | −2.075, p < 0.0001 | **−2.033, p = 0.00005** |
| R5PLUS − COLD (absolute lift) | +0.100, ns | −0.367, p = 0.14 |

Combined evidence across n = 100 seeds in two independent environments: the
upgraded improver **compounds** (the file's own pre-registered RSI signal) and
repairs the original mechanism's self-harm — but does not exceed the untrained
baseline (the unigram container's ceiling; the gate ablation is also not
robust across environments: +0.300 ns local, −0.150 ns Kaggle).

## 2. MetaForge adaptive vs frozen counterfactual — FULL BUDGET, POSITIVE

The experiment the local 45 s cap made impossible; full knobs (33 sealed
tasks, RESTARTS 6, GATE_TRIALS 8, 8 waves), matched frozen arm:

```
adaptive: 19 → 21 → 22 → 23 → 23 → 23 → 23 → 23   searcher v0 → v3 (3 adoptions)
frozen  : 19 → 19 → 19 → 19 → 19 → 19 → 19 → 19   searcher v0 (never changes)
```

**Adaptive +4 designer-task solves (+21%) over the frozen counterfactual**, with
each solve jump following a meta-gated searcher self-improvement adoption
(v1, v2, v3), while the frozen arm with identical tasks/budgets/seeds stays
flat for all 8 waves. This is the system's own headline design — searcher
improves searcher, validated against a frozen counterfactual — and at full
budget it produces a positive result. (Single deterministic task-stream by
design; no seed-variance estimate.)

## 3. Expedition XIX — machine-certified structural self-modification (C1)

Both seeds produced hash-chained ADMISSION certificates. Example (seed 1,
gen 1, from the tamper-evident ledger):

```
kind: ADMISSION
ir:   ZIP(MAX(V, A), 1)          ← operator SYNTHESIZED by the engine
op:   o01   op_sha: 4b5312…      certificate_sha: 71cfa0…
parent_vocab_sha: 4f53cd… → new_vocab_sha: 8051e0…
solved_task_id: d05              ← previously unsolvable discovery task
```

OPEN arm (admission live): vocabulary 0→2 synthesized operators, discovery
solves 3→7/24, occasional solves of *certified-inexpressible* frozen tasks
(1/12). **OPEN_OFF control (admission disabled, same budgets): 0/24 across
all 30 generations.** The capability difference is attributable to the
self-modification mechanism.

## 4. Meta-RL grid — definitive null at full config

Full CFG (96 episodes, 3000 eval budget), n = 40: every contrast within the
noise floor (HYBRID−COLD −0.002 ns; floor −0.008 ns). Together with the local
n = 40 fast-config null: the plasticity/controller mechanisms do not lift
ladder score at any tested scale. Reported as measured.

## 5. SDT reflective-endorsement layer (local, holdout n = 40)

Which of the three conditions collapses first when broken: **closed
satisfaction path first** (40/40 seeds, round 3, anchor-vacuity signature),
wireheadable anchor second (9/40, classic ramp 0.75→0.97), arbitrary anchor
drifts without formal collapse (worst eval competence), full triple survives
all rounds with working endorsement. Details: SDT_RESULTS.md.

## Bottom line

On this substrate, with counterfactual controls and full budgets, three
distinct RSI mechanisms now show positive, machine-checkable evidence —
compounding prior self-improvement (replicated, n=100, p≈1e-4 scale),
meta-gated searcher self-improvement (+21% vs frozen counterfactual), and
certified expressivity expansion (2/2 seeds) — while two honest nulls stand
(no absolute lift over untrained priors; meta-RL grid flat). The claim
"meaningful recursive self-improvement" is supported in the specific,
pre-registered senses above, and only in those senses.

**Housekeeping:** results are fetched; you can now revoke the Kaggle token
(kaggle.com → Settings → API → Expire API Token).
