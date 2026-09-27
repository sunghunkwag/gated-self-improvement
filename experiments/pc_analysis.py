#!/usr/bin/env python3
"""
Read-only analysis of a process-controller battery (no ledger writes).

  python3 experiments/pc_analysis.py LABEL [OUT.json]

Per arm: final-holdout means (ext, in), per-world trajectories, and the
process-decision record (pool overlap with the default v3 pool, share of
non-default round-level options, allocation shift, transfer on the
tracking probes). Contrasts: MEMORY_CARRY vs NO_CARRY (persistent memory),
vs FROZEN, NO_CARRY vs FROZEN, MEMORY_CARRY vs MEMORY_RANKONLY (process
control beyond ranking). Growth: per-seed (second half - first half) of the
per-world advantage.
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from rsi_v2 import stats as ST          # noqa: E402
from rsi_v3 import runner as R          # noqa: E402

PAIRS = (("MEMORY_CARRY", "NO_CARRY"), ("MEMORY_CARRY", "FROZEN"),
         ("NO_CARRY", "FROZEN"), ("MEMORY_CARRY", "MEMORY_RANKONLY"),
         ("MEMORY_RANKONLY", "FROZEN"), ("MEMORY_RANKONLY", "NO_CARRY"))
KN = ("attempt", "shape", "explore", "gate")


def load(label, path=None):
    units = [json.loads(l) for l in open(path or R.DEV_LOG)]
    units = [u for u in units if u.get("label") == label]
    return {(u["arm"], u["seed"]): u for u in units}


def analyse(tab):
    arms = sorted({a for (a, _s) in tab})
    seeds = sorted({s for (_a, s) in tab})
    seeds = [s for s in seeds if all((a, s) in tab for a in arms)]
    W = len(tab[(arms[0], seeds[0])]["ext_w"])
    n = float(len(seeds))
    rep = {"arms": arms, "n_seeds": len(seeds), "worlds": W}
    means = {}
    for a in arms:
        us = [tab[(a, s)] for s in seeds]
        m = {"ext": sum(u["ext"] for u in us) / n,
             "in": sum(u["in"] for u in us) / n,
             "ext_w": [round(sum(u["ext_w"][w] for u in us) / n, 3)
                       for w in range(W)],
             "macros": sum(u["n_macros"] for u in us) / n,
             "spent": sum(u["spent"] for u in us) / n}
        if "pool_jaccard_w" in us[0]:
            for f in ("pool_jaccard_w", "knob_nondefault_w", "alloc_l1_w",
                      "omega_w"):
                m[f] = [round(sum(u[f][w] for u in us) / n, 4)
                        for w in range(W)]
            tr = [[u["transfer_w"][w] for u in us
                   if u["transfer_w"][w] is not None] for w in range(W)]
            m["transfer_w"] = [round(sum(x) / len(x), 5) if x else None
                               for x in tr]
            use = {k: collections.Counter() for k in KN}
            for u in us:
                for pw in u["plans_w"]:
                    for plan in pw:
                        for k, o in zip(KN, plan):
                            use[k][o] += 1
            m["option_use"] = {k: dict(sorted(use[k].items())) for k in KN}
            acts = collections.Counter()
            for u in us:
                for aw in u["adopted_w"]:
                    for lb in aw:
                        if lb:
                            acts[lb.split("[")[0]] += 1
            tot = float(max(1, sum(acts.values())))
            m["adopted_share"] = {k: round(v / tot, 3)
                                  for k, v in sorted(acts.items())}
        means[a] = m
    rep["means"] = means
    con = {}
    for a, b in PAIRS:
        if a not in arms or b not in arms:
            continue
        d = [tab[(a, s)]["ext"] - tab[(b, s)]["ext"] for s in seeds]
        sm = ST.summary(d, "pcan|%s-%s|ext" % (a, b), one_sided=True)
        di = [tab[(a, s)]["in"] - tab[(b, s)]["in"] for s in seeds]
        si = ST.summary(di, "pcan|%s-%s|in" % (a, b), one_sided=True)
        pw = [sum(tab[(a, s)]["ext_w"][w] - tab[(b, s)]["ext_w"][w]
                  for s in seeds) / n for w in range(W)]
        gr = [sum(tab[(a, s)]["ext_w"][w] - tab[(b, s)]["ext_w"][w]
                  for w in range(W // 2, W))
              - sum(tab[(a, s)]["ext_w"][w] - tab[(b, s)]["ext_w"][w]
                    for w in range(W // 2)) for s in seeds]
        sg = ST.summary(gr, "pcan|%s-%s|growth" % (a, b), one_sided=True)
        con["%s-%s" % (a, b)] = {
            "ext": _short(sm), "in": _short(si),
            "per_world": [round(x, 3) for x in pw],
            "first_half": round(sum(pw[:W // 2]) / (W // 2), 3),
            "second_half": round(sum(pw[W // 2:]) / (W - W // 2), 3),
            "growth": _short(sg)}
    rep["contrasts"] = con
    return rep


def _short(sm):
    return {"mean": round(sm["mean"], 4), "ci95": [round(x, 3) for x in
                                                   sm["ci95"]],
            "p": round(sm["p"], 4),
            "wtl": [sm["wins"], sm["ties"], sm["losses"]]}


def show(rep):
    print("n_seeds %d, worlds %d" % (rep["n_seeds"], rep["worlds"]))
    for a, m in rep["means"].items():
        print("%-16s ext %6.2f in %6.2f macros %.2f spent %.0f  ext_w %s" % (
            a, m["ext"], m["in"], m["macros"], m["spent"], m["ext_w"]))
        if "pool_jaccard_w" in m:
            print("   pool~default %s" % m["pool_jaccard_w"])
            print("   knobs!=dflt  %s" % m["knob_nondefault_w"])
            print("   transfer     %s" % m["transfer_w"])
            print("   options %s  adopted %s" % (m["option_use"],
                                                 m["adopted_share"]))
    for k, c in rep["contrasts"].items():
        print("%-32s ext %+.3f %s p=%.4f wtl=%s | in %+.3f p=%.4f" % (
            k, c["ext"]["mean"], c["ext"]["ci95"], c["ext"]["p"],
            c["ext"]["wtl"], c["in"]["mean"], c["in"]["p"]))
        print("   per-world %s  1st %.3f 2nd %.3f growth p=%.4f" % (
            c["per_world"], c["first_half"], c["second_half"],
            c["growth"]["p"]))


if __name__ == "__main__":
    rep = analyse(load(sys.argv[1]))
    show(rep)
    if len(sys.argv) > 2:
        with open(sys.argv[2], "w", encoding="utf-8") as f:
            f.write(json.dumps(rep, indent=1, sort_keys=True) + "\n")
