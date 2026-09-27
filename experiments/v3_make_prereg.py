#!/usr/bin/env python3
"""Writes results/PREREGISTRATION_V3.json from the CURRENT rsi_v3 code and
HP. Run once, immediately before `python3 -m rsi_v3 freeze`; the freeze
then hashes this file together with every rsi_v3/rsi_v2 source file."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from rsi_v2 import substrate as S      # noqa: E402
from rsi_v3 import improver as I       # noqa: E402
from rsi_v3 import tasks as T          # noqa: E402
from rsi_v3 import evaluate as E       # noqa: E402

confirm = sys.argv[1] if len(sys.argv) > 1 else "7001-7300"
power = sys.argv[2] if len(sys.argv) > 2 else ""
hp = I.HP
wcap = I.world_cap(hp, dict(T.SPLITS)["train"])
prereg = {
    "title": "rsi_v3 confirmatory battery: does learning HOW to improve "
             "(a shared contextual meta-predictor trained on paired "
             "counterfactual outcomes, carried across improvement problems) "
             "beat the same improver with the meta-predictor frozen, at "
             "identical execution budgets?",
    "written_before_confirmatory_runs": True,
    "algorithm": {
        "package": "src/rsi_v3 (+ the rsi_v2 substrate/solver/ledger/stats "
                   "it imports); every file hashed at freeze",
        "improver_version": I.VERSION,
        "run_structure": "each run = %d independent improvement problems "
                         "('worlds': fresh TRAIN/META-VAL/FINAL family split,"
                         " fresh tasks, fresh base solver); only the "
                         "meta-predictor may carry across worlds"
                         % hp["n_worlds"],
        "loop": "attempt -> continuous diagnosis -> diagnosis-conditioned "
                "proposal generation (MINE, RESID residual-guided mining, "
                "COMPOSE hierarchical composition, PRIOR repair, EXPLORE "
                "repair, PRUNE only on evidence) -> meta-predictor ranks the "
                "pool -> k candidates screened on cross-family META-VAL "
                "probes (paired streams) -> adopt the screening winner -> "
                "every candidate x probe paired outcome trains the "
                "predictor",
        "actions": list(I.ACTIONS),
        "feature_dim": I.feat_dim()},
    "hp": hp, "hp_sha": S.sha256_text(S.canon(hp)),
    "tasks": {"version": T.TASKS_VERSION, "splits": dict(T.SPLITS),
              "families": [T.N_FAMILIES, T.N_TRAIN_FAMILIES,
                           T.N_METAVAL_FAMILIES],
              "library_digest": T.library_digest()},
    "compute": {"unit": "one program execution on one input, metered and "
                        "cross-checked against a process-global counter",
                "cap_per_world": wcap,
                "cap_per_arm": hp["n_worlds"] * wcap,
                "per_world_meters": "every arm gets an identical child meter "
                                    "per world (no shifting between worlds)",
                "evaluation": "identical for all arms: %d executions per "
                              "final-holdout task, stream keyed by (world, "
                              "split, task index) only" % E.EVAL_BUDGET},
    "seeds": {"dev_iterate": "3001-3040", "dev_check": "3101-3200",
              "confirm": confirm,
              "never_used": "1-40, 101-200, 1001-1300 (v1/v2)"},
    "arms": list(I.ARMS),
    "primary_metric": {
        "field": "ext", "split": E.PRIMARY_SPLIT,
        "definition": "per run: total number of FINAL-HOLDOUT tasks (24 per "
                      "world, from the world's FINAL families, never seen by "
                      "any improver) solved exactly on train and hidden test "
                      "examples, summed over all %d worlds" % hp["n_worlds"]},
    "confirmatory_contrasts": [
        {"name": "H1_adaptive_vs_frozen", "a": "ADAPTIVE_META",
         "b": "FROZEN_META", "role": "primary"},
        {"name": "H2_adaptive_vs_single", "a": "ADAPTIVE_META",
         "b": "SINGLE_COMPUTE_MATCHED", "role": "secondary"}],
    "statistics": {
        "test": "paired (by run seed) sign-flip permutation test, one-sided "
                "H_a: mean(a - b) > 0, 20000 permutations from a fixed "
                "deterministic stream",
        "multiplicity": "Holm over H1 and H2", "alpha": 0.05,
        "decision_rule": "supported iff mean(a - b) > 0 and Holm-adjusted "
                         "p < alpha; otherwise reported as null"},
    "exploratory_contrasts": [
        ["ADAPTIVE_META", "ADAPTIVE_NOCARRY", "ext"],
        ["ADAPTIVE_META", "NODIAG_META", "ext"],
        ["ADAPTIVE_META", "HEURISTIC_META", "ext"],
        ["FROZEN_META", "SINGLE_COMPUTE_MATCHED", "ext"],
        ["ADAPTIVE_META", "FROZEN_META", "meta_quality"],
        ["ADAPTIVE_META", "FROZEN_META", "in"],
        ["COLD", "COLD", "replicate_ext"],
        ["ADAPTIVE_META", "COLD", "ext"]],
    "power": power,
    "stopping_and_integrity": "every (arm, seed) runs exactly once after the "
                              "PREREG_FREEZE ledger record; no interim looks; "
                              "a crashed unit may only be re-run "
                              "deterministically (UNIT_RESTART); the frozen "
                              "runner produces the report whatever it says",
    "dev_history": "v3 ledger: every dev iteration, dev-check and note, "
                   "including the failed designs"}
path = os.path.join(ROOT, "results", "PREREGISTRATION_V3.json")
with open(path, "w", encoding="utf-8") as f:
    f.write(json.dumps(prereg, indent=1, sort_keys=True) + "\n")
print("wrote", path, prereg["hp_sha"])
