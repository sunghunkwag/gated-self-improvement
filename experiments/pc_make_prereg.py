#!/usr/bin/env python3
"""Writes results/PREREGISTRATION_V3.json for the PROCESS-CONTROLLER
confirmatory battery from the CURRENT rsi_v3 code and HP. Run once,
immediately before `python3 -m rsi_v3 freeze` (and only after a GO); the
freeze then hashes this file together with every rsi_v3/rsi_v2 source file.

  python3 experiments/pc_make_prereg.py [CONFIRM_SEEDS] [POWER_TEXT]
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from rsi_v2 import substrate as S      # noqa: E402
from rsi_v3 import improver as I       # noqa: E402
from rsi_v3 import controller as C     # noqa: E402
from rsi_v3 import tasks as T          # noqa: E402
from rsi_v3 import evaluate as E       # noqa: E402

confirm = sys.argv[1] if len(sys.argv) > 1 else "7001-7300"
power = sys.argv[2] if len(sys.argv) > 2 else ""
hp = I.HP
wcap = I.world_cap(hp, dict(T.SPLITS)["train"]) + I.track_cap(hp)
prereg = {
    "title": "rsi_v3 process controller, confirmatory battery: does "
             "persistent improvement memory (memory -> process control) "
             "make an improver improve itself better on later, unseen "
             "improvement problems, at identical tasks, streams and "
             "execution budgets?",
    "written_before_confirmatory_runs": True,
    "algorithm": {
        "package": "src/rsi_v3 (+ the rsi_v2 substrate/solver/ledger/stats "
                   "it imports); every file hashed at freeze",
        "improver_version": I.VERSION,
        "controller": "rsi_v3.controller.ProcessController",
        "run_structure": "each run = %d independent improvement problems "
                         "('worlds'): fresh TRAIN/META-VAL/FINAL family "
                         "split, fresh tasks, fresh base solver; only the "
                         "controller's improvement memory may carry across "
                         "worlds (MEMORY_CARRY) and it holds numbers and "
                         "labels only" % hp["n_worlds"],
        "process_dimensions": [
            "proposal generation over a parameterised variant grid",
            "pool allocation across the six strategies",
            "attempt compute allocation (base / focus / deep)",
            "screening shape (probe allocation)",
            "exploration (ranking slots + Thompson scale)",
            "adoption rule (plasticity)",
            "memory retrieval weight (episodic vs generalised)"],
        "feedback_horizons": "immediate (paired screen gain), delayed "
                             "(tracking-probe effect of the adopted edit "
                             "and return-to-go), transfer (world-end "
                             "cross-family gain on the tracking probes)",
        "value_dim": C.value_dim()},
    "hp": hp, "hp_sha": S.sha256_text(S.canon(hp)),
    "tasks": {"version": T.TASKS_VERSION, "splits": dict(T.SPLITS),
              "families": [T.N_FAMILIES, T.N_TRAIN_FAMILIES,
                           T.N_METAVAL_FAMILIES],
              "library_digest": T.library_digest()},
    "compute": {"unit": "one program execution on one input, metered and "
                        "cross-checked against a process-global counter",
                "cap_per_world": wcap,
                "cap_per_arm": hp["n_worlds"] * wcap,
                "per_world_meters": "identical child meter per world for "
                                    "every arm (v3 rounds + tracking)",
                "evaluation": "identical for all arms: %d executions per "
                              "final-holdout task, stream keyed by (world, "
                              "split, task index) only" % E.EVAL_BUDGET},
    "seeds": {"dev_iterate": "3001-3100",
              "dev_check": "3101-3200 spent (v3 ranking design, NO_GO); "
                           "3201-3300 (process-controller go/no-go)",
              "confirm": confirm,
              "never_used": "1-40, 101-200, 1001-1300 (v1/v2)"},
    "arms": ["FROZEN", "NO_CARRY", "MEMORY_CARRY", "MEMORY_RANKONLY"],
    "primary_metric": {
        "field": "ext", "split": E.PRIMARY_SPLIT,
        "definition": "per run: total number of FINAL-HOLDOUT tasks (24 "
                      "per world, from the world's FINAL families, never "
                      "seen by any improver) solved exactly on train and "
                      "hidden test examples, summed over all %d worlds"
                      % hp["n_worlds"]},
    "confirmatory_contrasts": [
        {"name": "H1_memory_carry_vs_no_carry", "a": "MEMORY_CARRY",
         "b": "NO_CARRY", "role": "primary"},
        {"name": "H2_memory_carry_vs_frozen", "a": "MEMORY_CARRY",
         "b": "FROZEN", "role": "secondary"},
        {"name": "H3_no_carry_vs_frozen", "a": "NO_CARRY", "b": "FROZEN",
         "role": "secondary"}],
    "statistics": {
        "test": "paired (by run seed) sign-flip permutation test, one-sided "
                "H_a: mean(a - b) > 0, 20000 permutations from a fixed "
                "deterministic stream",
        "multiplicity": "fixed-sequence gatekeeping: H1 (primary) is "
                        "tested alone at alpha; only if H1 is supported are "
                        "H2 and H3 tested, Holm-adjusted between themselves "
                        "at alpha (family-wise error <= alpha)",
        "alpha": 0.05,
        "decision_rule": "supported iff mean(a - b) > 0 and the adjusted "
                         "p < alpha; otherwise reported as null"},
    "verification_spec": {"treatment": "MEMORY_CARRY",
                          "baselines": ["NO_CARRY", "FROZEN",
                                        "MEMORY_RANKONLY"]},
    "verification_checks": [
        "growth: per-world MEMORY_CARRY - NO_CARRY, second-half mean vs "
        "first-half mean, paired test of the per-seed difference",
        "transfer: per-seed mean world-end cross-family gain on the "
        "tracking probes, MEMORY_CARRY vs each baseline",
        "decisions change: pool overlap with the default pool and share of "
        "non-default round-level options, per world, per arm",
        "meta-controller compute logged (rows/refits/predictions/flops/"
        "retrievals/seconds); zero program executions asserted",
        "identical caps and per-world caps across arms; spent <= cap; "
        "metered spend == global execution counter for every unit"],
    "exploratory_contrasts": [
        ["MEMORY_CARRY", "MEMORY_RANKONLY", "ext"],
        ["MEMORY_RANKONLY", "FROZEN", "ext"],
        ["MEMORY_CARRY", "NO_CARRY", "in"],
        ["MEMORY_CARRY", "FROZEN", "in"]],
    "power": power,
    "stopping_and_integrity": "every (arm, seed) runs exactly once after the "
                              "PREREG_FREEZE ledger record; no interim looks; "
                              "a crashed unit may only be re-run "
                              "deterministically (UNIT_RESTART); the frozen "
                              "runner produces the report whatever it says",
    "dev_history": "v3 ledger: every dev battery, sweep, note, criterion, "
                   "dev check and verdict, including the failed designs"}
path = os.environ.get("RSI_V3_PREREG_OUT",
                      os.path.join(ROOT, "results", "PREREGISTRATION_V3.json"))
with open(path, "w", encoding="utf-8") as f:
    f.write(json.dumps(prereg, indent=1, sort_keys=True) + "\n")
print("wrote", path, prereg["hp_sha"])
