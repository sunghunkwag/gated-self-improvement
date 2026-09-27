#!/usr/bin/env python3
"""
Re-runs logged units with the CURRENT code and checks that every
behavioural field is bit-identical (timing fields excluded). Used before the
freeze to prove that post-dev-check edits (logging, report, comments) did
not change what the frozen protocol computes.

  python3 experiments/v3_repro_check.py <label> ARM:SEED [ARM:SEED ...]
"""
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from rsi_v3 import runner as R          # noqa: E402
from rsi_v3 import improver as I        # noqa: E402

FIELDS = ("ext", "ext_w", "in", "in_w", "per_task", "spent", "world_spent",
          "global_delta", "cap", "world_cap", "train_solved", "n_macros",
          "model_w", "model_n", "mq_w", "adopt_w", "adopted_actions",
          "final_cfg_shas", "macros_w", "rank_validity", "self_solved",
          "self_made")


def main(label, pairs):
    logged = {}
    for line in open(R.DEV_LOG):
        u = json.loads(line)
        if u.get("label") == label:
            logged[(u["arm"], u["seed"])] = u
    ok = True
    for p in pairs:
        arm, seed = p.split(":")
        old = logged[(arm, int(seed))]
        new = R.compact(R.run_unit((arm, int(seed), copy.deepcopy(I.HP))))
        bad = [f for f in FIELDS if f in old and old[f] != new[f]]
        print(arm, seed, "IDENTICAL" if not bad else "DIFFERS: %s" % bad,
              flush=True)
        ok = ok and not bad
    print("ALL IDENTICAL" if ok else "MISMATCH")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2:]))
