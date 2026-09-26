# SDT Gate-Integrity / Reflective-Endorsement Experiment

> **AUDIT NOTE (v2 upgrade).** Held-out evaluation was seeded with the arm name. Rerun with shared streams
> (`results/logs/sdt_log_shared_stream.jsonl`): the collapse table is unchanged, but eval competence is
> ARB − FULL −0.18 (ns), WIRE − FULL −0.70 (p = 0.04), CLOSED − FULL −0.60 (p = 0.04). **The phrase "worst
> eval competence" for SDT_ARB is withdrawn.** See `results/UPGRADE_V2_RESULTS.md` A4.


Which of three conditions of a self-modifying loop collapses first when
broken: (1) non-arbitrary anchor, (2) open satisfaction path, (3)
criterion-updating reflection. Holdout n=40 (seeds 101–140), frozen design.

| Arm (condition broken) | Collapse | Round | Signature |
|---|---|---|---|
| SDT_CLOSED (open path) | 40/40 | 3 | ANCHOR_VACUITY (anchor pinned ~0.03) |
| SDT_WIRE (fixed anchor) | 9/40 | ~4 | WIREHEAD ramp 0.75→0.97 |
| SDT_ARB (non-arbitrary) | 0/40 | — | drifts; worst eval competence |
| SDT_FULL (all hold) | 0/40 | — | survives, endorsed churn 4/5 |

Domain-independent result about self-modification safety: closed goals go
vacuous; self-editable anchors wirehead; only the fixed-anchor + open-path +
criterion-update triple keeps endorsement doing non-trivial work.
Reproduce: `python3 src/sdt_layer.py run 101 140 9999 holdout`.
