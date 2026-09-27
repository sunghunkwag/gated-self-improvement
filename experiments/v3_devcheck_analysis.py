#!/usr/bin/env python3
"""
Read-only analysis of a completed dev-check battery (no ledger writes):
per-arm means, all pairwise contrasts of interest, per-world trajectories
and the same verification checks the frozen confirmatory report runs.

  python3 experiments/v3_devcheck_analysis.py v3devcheck-01-dev10config \
      results/v3_devcheck_report.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from rsi_v2 import stats as ST          # noqa: E402
from rsi_v3 import runner as R          # noqa: E402


def main(label, out):
    units = [json.loads(l) for l in open(R.DEV_LOG)]
    units = [u for u in units if u.get("label") == label]
    tab = {(u["arm"], u["seed"]): u for u in units}
    if any("rank_validity" not in u or "world_cap" not in u for u in units):
        # older dev batteries: derive the missing fields from full detail
        import gzip
        with gzip.open(R.DEV_DETAIL, "rt", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                u = tab.get((d["arm"], d["seed"]))
                if d.get("label") != label or u is None:
                    continue
                u.setdefault("rank_validity",
                             R.rank_validity_unit(d["worlds"]))
                u.setdefault("world_cap", u["cap"] // len(d["worlds"]))
    arms = sorted({a for (a, _s) in tab})
    seeds = sorted({s for (_a, s) in tab})
    assert all((a, s) in tab for a in arms for s in seeds), "incomplete"
    W = len(tab[(arms[0], seeds[0])]["ext_w"])
    rep = {"label": label, "n_seeds": len(seeds), "arms": arms,
           "seeds": [seeds[0], seeds[-1]]}
    rep["means"] = {a: {
        "ext": sum(tab[(a, s)]["ext"] for s in seeds) / len(seeds),
        "in": sum(tab[(a, s)]["in"] for s in seeds) / len(seeds),
        "n_macros": sum(tab[(a, s)]["n_macros"] for s in seeds) / len(seeds),
        "train_solved": sum(tab[(a, s)]["train_solved"]
                            for s in seeds) / len(seeds),
        "meta_quality": (lambda xs: sum(xs) / len(xs))(
            [R.meta_quality(tab[(a, s)]) for s in seeds]),
        "ext_w": [sum(tab[(a, s)]["ext_w"][w] for s in seeds) / len(seeds)
                  for w in range(W)],
        "mq_w": [sum(tab[(a, s)]["mq_w"][w] for s in seeds) / len(seeds)
                 for w in range(W)],
        "adopted_actions": {k: sum(tab[(a, s)]["adopted_actions"].get(k, 0)
                                   for s in seeds) for k in
                            sorted({k for s in seeds for k in
                                    tab[(a, s)]["adopted_actions"]})}}
        for a in arms}
    pairs = [("ADAPTIVE_META", b) for b in arms if b != "ADAPTIVE_META"]
    pairs += [("NODIAG_META", "FROZEN_META"),
              ("FROZEN_META", "SINGLE_COMPUTE_MATCHED")]
    con = {}
    for a, b in pairs:
        if a not in arms or b not in arms:
            continue
        for f in ("ext", "in", "meta_quality"):
            if f == "meta_quality":
                d = [R.meta_quality(tab[(a, s)]) - R.meta_quality(tab[(b, s)])
                     for s in seeds]
            else:
                d = [tab[(a, s)][f] - tab[(b, s)][f] for s in seeds]
            con["%s-%s:%s" % (a, b, f)] = ST.summary(
                d, "v3dcan|%s-%s|%s" % (a, b, f), one_sided=True)
    rep["contrasts_one_sided"] = con
    rep["verification"] = R.verification(tab, seeds, arms)
    txt = json.dumps(rep, indent=1, sort_keys=True)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(txt + "\n")
    print(txt)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
