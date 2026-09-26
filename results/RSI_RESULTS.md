# Baseline RSI Audit (before the upgrade)

> **AUDIT CORRECTION (v2 upgrade, see `results/UPGRADE_V2_RESULTS.md` Part A).** The numbers below were
> produced with held-out evaluation streams keyed on the **arm name** (`'ev|<cond>|…'`), and ROUND1_5X saw 5
> distinct source tasks where R5PLUS saw 25. Rerun on the same seeds 101–200 with shared streams and a
> data-matched single-round control (R1PLUS_DM): R5PLUS − ROUND1_5X = +1.30 (p < 1e-4), but
> **R5PLUS − COLD = −0.70 (p = 0.0005)** and **R5PLUS − R1PLUS_DM = −0.59 (p = 0.002)**. The "compounding"
> advantage is a data-diversity confound, not recursion; the repaired chain is below the untrained baseline.
> The original text is kept below unchanged for the record.


Running the six original files' own pre-registered designs with
counterfactual controls, the answer to "is there meaningful recursive
self-improvement?" was, at first: **mostly no.**

- Expedition XV (n=12): ROUND5 − ROUND1_5X = +1.5 (p=0.054) but
  ROUND5 − COLD = **−1.5 (p=0.039)** — recursive prior-fitting lands BELOW
  no-learning (overfits source stats).
- Expedition XV-RL (n=20): every mechanism contrast smaller than the noise
  floor.
- MetaForge wave-1 counterfactual (capped): adaptive = frozen, delta 0.

This honest null motivated the diagnosis-and-repair in UPGRADE_RESULTS.md
and the full-budget re-runs in FINAL_RESULTS.md (where MetaForge, given full
budget, shows +21% adaptive vs frozen). Reproduce: `python3 experiments/rsi_experiments.py report`.
