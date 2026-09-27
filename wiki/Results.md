# Results

> Updated by the v2 upgrade: see `results/UPGRADE_V2_RESULTS.md` for the audit, the preregistration and the raw
> ledger.

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
