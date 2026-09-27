# Engineered Recursive Self-Improvement (RSI Upgrade)

> **AUDIT CORRECTION (v2 upgrade, see `results/UPGRADE_V2_RESULTS.md` Part A).** The numbers below were
> produced with held-out evaluation streams keyed on the **arm name** (`'ev|<cond>|…'`), and ROUND1_5X saw 5
> distinct source tasks where R5PLUS saw 25. Rerun on the same seeds 101–200 with shared streams and a
> data-matched single-round control (R1PLUS_DM): R5PLUS − ROUND1_5X = +1.30 (p < 1e-4), but
> **R5PLUS − COLD = −0.70 (p = 0.0005)** and **R5PLUS − R1PLUS_DM = −0.59 (p = 0.002)**. The "compounding"
> advantage is a data-diversity confound, not recursion; the repaired chain is below the untrained baseline.
> The original text is kept below unchanged for the record.


The original Expedition-XV chain was measured to be *harmful*
(ROUND5 − COLD = −2.075, p<1e-4). Diagnosed (noisy partial-fitness pool;
occurrence counts collapse entropy; no step control) and repaired: M1
solved-only deduped credit, M2 presence-based level-stratified target, M3
trust-region + entropy floor, M4 generalization-gated acceptance + rollback.

## Confirmatory holdout (n=40 local + n=60 Kaggle = n=100), frozen params

| Contrast | Local n=40 | Kaggle n=60 |
|---|---|---|
| R5PLUS − ROUND1_5X (compounding) | +1.55, p<1e-4 | +1.47, p=5e-5 |
| R5PLUS − ROUND5 (vs original) | +2.18, p<1e-4 | +1.67, p=5e-5 |
| ROUND5 − COLD (original harm) | −2.08, p<1e-4 | −2.03, p=5e-5 |
| R5PLUS − COLD (absolute) | +0.10, ns | −0.37, ns |

Compounding recovered (pre-registered criterion, p≈1e-4, two environments);
the original self-harm eliminated. Not claimed: beating the untrained
baseline (unigram container ceiling). Meta-RL grid: null at full config.
Reproduce: `python3 src/rsi_upgrade.py report2`.
