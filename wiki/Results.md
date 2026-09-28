# Results

> Updated by the v2 upgrade: see `results/UPGRADE_V2_RESULTS.md` for the audit, the preregistration and the raw
> ledger.

## v3 Part II: persistent improvement memory → process control (pre-registered, n = 300 untouched seeds)
See `results/UPGRADE_V3_RESULTS.md`, Part II.
- **H1 supported:** the same learner with its improvement memory carried across problems beats the same learner
  with memory wiped between problems by **+1.52** final-holdout tasks (95% CI [+0.55, +2.48], p = 0.0011) on
  task families no improver saw.
- **H2 supported:** +1.78 over the frozen controller. The frozen controller uses the strongest fixed prior found on
  dev; the first, weaker control was caught and replaced.
- **H3 null:** learning within one problem does not beat the frozen controller (+0.25).
- **H4 null:** the advantage appears after one problem of experience and then stays flat (+0.15 → +0.16 per world).
- **Ablations:**
  - Persistent memory helps through generation (+1.27 over no-carry, p = 0.006) and through ranking alone (+1.17
    over frozen, p = 0.011).
  - Full process control beyond ranking-only memory: +0.60, n.s.
  - The round-level process options: +0.26, n.s.

## v3 Part I: ranking-only meta-predictor (go/no-go NO_GO, seeds 3101–3200)
- Beat its frozen twin (+1.35, p = 0.023), but its cross-problem carry-over missed the bar (+0.53, p = 0.22) →
  NO_GO; the confirmatory seeds were not used by it.

## Preregistered v2 battery (n = 300 fresh seeds, identical 610k-execution cap, external task families)
- **H1 supported (small):** RECURSIVE_FULL − SINGLE_5X = **+0.40** tasks of 24 (95% CI [+0.10, +0.70]),
  Holm p = 0.0098.
- **H2 null:** RECURSIVE_FULL − RECURSIVE_FROZEN = **−0.08** (CI [−0.35, +0.19]). Adapting the improvement
  policy did not help.
- Exploratory: RECURSIVE_FROZEN − SINGLE_5X = +0.48, so the gain comes from recursion on a difficulty ladder
  (13.8 vs 7.8 training tasks solved, hierarchical macros, gains on long tasks).

## Positive (counterfactually controlled)
- **MetaForge:** adaptive 19→23 vs frozen flat-19 over 8 waves (+21%), searcher v0→v3 (arms share streams; one
  deterministic stream).
- **Open-ended:** 189 certified novel behaviours (OPEN) vs 0 (CLOSED) over 4,444 generations.
- **Turing-complete:** 15 gated macros (OPEN) vs 0 (CLOSED), 8 seeds; held-out `reverse` solved.
- **Cross-substrate transfer (re-audited):** +2.00 tasks (p = 0.001) and learning premium +1.00 (p = 0.008) with
  shared streams; the n = 60 extension gives +1.98 / +1.28 (p < 1e-4).
- **Real-file repair:** frozen 0.204 → adaptive 1.000 on file-world families (incl. repo_repair).

## Retracted / null / negative (reported, not hidden)
- **v1 compounding claim retracted:** with arm-independent eval streams, R5PLUS − COLD = −0.70 (p = 0.0005), and
  a data-matched single round beats R5PLUS by 0.59 (p = 0.002).
- Original chain was **harmful** (ROUND5 − COLD ≈ −2.1).
- Meta-RL grid: flat at full config (all contrasts within the noise floor).
- Open-ended growth is linear, not accelerating; it hits a search-dilution wall.
- Compositional transfer is limited by the skill I/O interface.
- SDT: "ARB has the worst eval competence" withdrawn (shared-stream rerun); the collapse order stands.

See the per-experiment files in `results/`.
