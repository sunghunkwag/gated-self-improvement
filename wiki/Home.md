# gated-selfimprovement — Wiki

A measurement-first study of recursive self-improvement (RSI). Every gain is
**counterfactually gated**: credited only if a matched control, denied the one
mechanism under test, does worse at equal budget. LLM-free, pure-Python stdlib,
deterministic.

## Pages
- [Architecture](Architecture) — the substrates, engines, and the unified model
- [Methodology](Methodology) — counterfactual gates, held-out eval, certificates
- [Experiments](Experiments) — what each battery tests and how to run it
- [Results](Results) — headline numbers and the honest nulls
- [Reproducing](Reproducing) — commands, seeds, Kaggle kernels
- [FAQ](FAQ) — "is this AGI?", "why no LLM?", "is it a toy?"

## 30-second version
Two arms run the same loop; one may admit self-discovered building blocks (GATED), one may not (FROZEN). The gap
is the evidence.

- **v2 (preregistered, n = 300, identical compute, unseen task families):** recursive self-improvement beats
  one-shot improvement with 5× compute by +0.40 tasks (Holm p = 0.0098). Adapting the improvement policy itself
  is a null (−0.08).
- **Other positives:** searcher-improves-searcher (+21% vs frozen), certified open-ended expansion (189 vs 0),
  Turing-complete port (15 vs 0 macros), cross-substrate transfer (+2.00, p = 0.001 after re-audit).
- **Retracted:** the v1 "compounding" headline (arm-keyed eval RNG + data confound).
- **Nulls:** meta-RL grid flat; growth linear, not accelerating.
