## Post-hoc mechanism analysis (confirm, n = 300 seeds)

### Compute actually spent (cap is identical for every arm)

| arm | cap | mean spent | max spent | global counter == metered |
|---|---|---|---|---|
| COLD | 610000 | 0 | 0 | True |
| SINGLE_5X | 610000 | 503714 | 550919 | True |
| RECURSIVE_FROZEN | 610000 | 501200 | 578740 | True |
| RECURSIVE_FULL | 610000 | 503056 | 572556 | True |
| RECURSIVE_NODIAG | 610000 | 503062 | 572556 | True |

### All pairwise contrasts on `ext` (two-sided p, exploratory)

| contrast | mean | 95% CI | W/T/L | p |
|---|---|---|---|---|
| RECURSIVE_FULL - SINGLE_5X | +0.397 | [+0.10, +0.70] | 131/64/105 | 0.0098 |
| RECURSIVE_FULL - RECURSIVE_FROZEN | -0.080 | [-0.35, +0.19] | 124/56/120 | 0.5823 |
| RECURSIVE_FROZEN - SINGLE_5X | +0.477 | [+0.20, +0.75] | 140/53/107 | 0.0008 |
| RECURSIVE_NODIAG - RECURSIVE_FULL | -0.013 | [-0.05, +0.01] | 1/297/2 | 0.7549 |
| RECURSIVE_NODIAG - RECURSIVE_FROZEN | -0.093 | [-0.36, +0.17] | 123/56/121 | 0.5136 |
| SINGLE_5X - COLD | +2.407 | [+2.15, +2.66] | 236/50/14 | 0.0000 |
| RECURSIVE_FULL - COLD | +2.803 | [+2.52, +3.08] | 238/40/22 | 0.0000 |

### All pairwise contrasts on `in` (two-sided p, exploratory)

| contrast | mean | 95% CI | W/T/L | p |
|---|---|---|---|---|
| RECURSIVE_FULL - SINGLE_5X | +0.077 | [-0.17, +0.32] | 124/60/116 | 0.5633 |
| RECURSIVE_FULL - RECURSIVE_FROZEN | -0.250 | [-0.46, -0.05] | 103/78/119 | 0.0189 |
| RECURSIVE_FROZEN - SINGLE_5X | +0.327 | [+0.08, +0.57] | 144/62/94 | 0.0078 |
| RECURSIVE_NODIAG - RECURSIVE_FULL | +0.003 | [-0.01, +0.01] | 2/297/1 | 1.0000 |
| RECURSIVE_NODIAG - RECURSIVE_FROZEN | -0.247 | [-0.45, -0.04] | 104/78/118 | 0.0203 |
| SINGLE_5X - COLD | +1.960 | [+1.75, +2.18] | 218/66/16 | 0.0000 |
| RECURSIVE_FULL - COLD | +2.037 | [+1.81, +2.26] | 227/55/18 | 0.0000 |

### Training tasks solved after each round (of 40)

| arm | r1 | r2 | r3 | r4 | r5 | final macros | rounds adopted |
|---|---|---|---|---|---|---|---|
| RECURSIVE_FROZEN | 5.3 | 8.2 | 10.5 | 12.4 | 13.8 | 5.35 | 2.71 |
| RECURSIVE_FULL | 5.3 | 8.2 | 10.4 | 12.1 | 13.4 | 6.27 | 2.75 |
| RECURSIVE_NODIAG | 5.3 | 8.2 | 10.4 | 12.1 | 13.4 | 6.25 | 2.76 |
| SINGLE_5X | (one round, 5 restarts/task) 7.8 | | | | | 3.89 | 0.91 |

### Composition / reuse of discovered operators

| arm | macros | hierarchical macros (contain another macro) | max macro length |
|---|---|---|---|
| SINGLE_5X | 3.89 | 1.00 | 2.49 |
| RECURSIVE_FROZEN | 5.35 | 2.71 | 3.92 |
| RECURSIVE_FULL | 6.27 | 3.61 | 3.84 |
| RECURSIVE_NODIAG | 6.25 | 3.58 | 3.85 |

### What the adaptive improver learned (RECURSIVE_FULL)

Most-screened actions in rounds 3-5, by diagnosed failure mode (FULL vs FROZEN):

| failure mode | FULL top-3 screened | FROZEN top-3 screened |
|---|---|---|
| BAD_ORDER | PRUNE (107), REFIT (97), MINE (95) | MINE (99), COMPOSE+REFIT (94), EXPLORE (94) |
| LOW_EXPLORE | COMPOSE+REFIT (74), REFIT (73), MINE+COMPOSE+REFIT (64) | REFIT (74), MINE+COMPOSE+REFIT (70), COMPOSE+REFIT (68) |
| MISSING_OP | PRUNE (295), REFIT (257), PRUNE+MINE (254) | COMPOSE+REFIT (264), REFIT (255), MINE+COMPOSE+REFIT (254) |
| OVERSPEC | PRUNE (3), REFIT (3), MINE+REFIT (2) | COMPOSE+REFIT (1), MINE (1), PRUNE (1) |

FULL adoptions by failure mode:

- BAD_ORDER: MINE+REFIT 25, REFIT 20, MINE 19, MINE+COMPOSE+REFIT 18, PRUNE+MINE 16, COMPOSE+REFIT 16, PRUNE 11, COMPOSE 8, EXPLORE 5
- LOW_EXPLORE: COMPOSE+REFIT 36, MINE+REFIT 32, MINE+COMPOSE+REFIT 30, REFIT 25, PRUNE+MINE 21, MINE 20, COMPOSE 15, PRUNE 5, EXPLORE 5
- MISSING_OP: MINE+COMPOSE+REFIT 75, MINE+REFIT 75, REFIT 73, COMPOSE+REFIT 72, PRUNE+MINE 68, MINE 50, COMPOSE 36, PRUNE 31, EXPLORE 15
- OVERSPEC: MINE 1, MINE+REFIT 1, PRUNE+MINE 1, MINE+COMPOSE+REFIT 1

Final strategy intensities (FULL; FROZEN keeps the initial values): COMPOSE 2.39, EXPLORE 0.23, MINE 3.53, PRUNE 0.28, REFIT 0.43

Delayed credit: after an adopted vocabulary change, the next round's new solves used the new macros in 2.62 tasks on average (525 credit events; 85% non-zero).

### External holdout solve rate by task length (primitives in the hidden program)

| arm | len 2-4 | len 5-6 | len 7-8 | len 9-12 |
|---|---|---|---|---|
| COLD | 0.681 | 0.072 | 0.025 | 0.002 |
| SINGLE_5X | 0.755 | 0.262 | 0.094 | 0.022 |
| RECURSIVE_FROZEN | 0.710 | 0.306 | 0.134 | 0.050 |
| RECURSIVE_FULL | 0.744 | 0.280 | 0.129 | 0.042 |
| RECURSIVE_NODIAG | 0.742 | 0.279 | 0.130 | 0.042 |

### Self-generated curriculum vs external holdout (self score is a diagnostic only)

| arm | self-generated solved (of 8) | external solved (of 24) |
|---|---|---|
| COLD | 6.22 | 4.39 |
| SINGLE_5X | 5.80 | 6.80 |
| RECURSIVE_FROZEN | 5.83 | 7.27 |
| RECURSIVE_FULL | 5.60 | 7.19 |
| RECURSIVE_NODIAG | 5.61 | 7.18 |
