#!/usr/bin/env python3
"""
POST-HOC (exploratory) mechanism analysis of the rsi_v2 confirmatory
battery. Nothing here decides a confirmatory hypothesis -- the frozen
verdict comes only from `python3 -m rsi_v2 report`. This script reads the
raw per-unit records (results/logs/v2_confirm.jsonl, identical to the
ledger's UNIT_END bodies) and describes HOW the arms differ.

  python3 experiments/v2_analysis.py [confirm|dev] [--label L] > out.md
"""
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from rsi_v2 import tasks as T          # noqa: E402
from rsi_v2 import solver as SV        # noqa: E402
from rsi_v2 import stats as ST         # noqa: E402

ARMS = ("COLD", "SINGLE_5X", "RECURSIVE_FROZEN", "RECURSIVE_FULL",
        "RECURSIVE_NODIAG")


def load(phase, label=None):
    path = os.path.join(ROOT, "results", "logs", "v2_%s.jsonl" % phase)
    recs = [json.loads(l) for l in open(path)]
    if label:
        recs = [r for r in recs if r.get("label") == label]
    tab = {}
    for r in recs:
        tab[(r["arm"], r["seed"])] = r
    seeds = sorted({s for (_a, s) in tab})
    return tab, seeds


def mean(xs):
    xs = list(xs)
    return sum(xs) / float(len(xs)) if xs else float("nan")


def trajectories(tab, seeds):
    print("\n### Training tasks solved after each round (of 40)\n")
    print("| arm | r1 | r2 | r3 | r4 | r5 | final macros | rounds adopted |")
    print("|---|---|---|---|---|---|---|---|")
    for a in ("RECURSIVE_FROZEN", "RECURSIVE_FULL", "RECURSIVE_NODIAG"):
        cols = []
        for k in range(5):
            cols.append("%.1f" % mean(tab[(a, s)]["rounds"][k]["train_solved"]
                                      for s in seeds))
        adopt = mean(sum(1 for r in tab[(a, s)]["rounds"] if r["adopted"])
                     for s in seeds)
        print("| %s | %s | %.2f | %.2f |" % (
            a, " | ".join(cols),
            mean(tab[(a, s)]["n_macros"] for s in seeds), adopt))
    a = "SINGLE_5X"
    print("| SINGLE_5X | (one round, 5 restarts/task) %.1f | | | | | %.2f | "
          "%.2f |" % (mean(tab[(a, s)]["train_solved"] for s in seeds),
                      mean(tab[(a, s)]["n_macros"] for s in seeds),
                      mean(1 if tab[(a, s)]["rounds"][0]["adopted"] else 0
                           for s in seeds)))


def hierarchy(tab, seeds):
    """Macro-of-macro reuse: a final macro whose expansion strictly
    contains another final macro's expansion."""
    print("\n### Composition / reuse of discovered operators\n")
    print("| arm | macros | hierarchical macros (contain another macro) |"
          " max macro length |")
    print("|---|---|---|---|")
    for a in ARMS[1:]:
        nh, nm, mx = [], [], []
        for s in seeds:
            ms = [SV.expansion(t) for t in tab[(a, s)]["macros"]]
            nm.append(len(ms))
            h = 0
            for m in ms:
                if any(o != m and len(o) < len(m) and any(
                        tuple(m[i:i + len(o)]) == o
                        for i in range(len(m) - len(o) + 1)) for o in ms):
                    h += 1
            nh.append(h)
            mx.append(max([len(m) for m in ms] or [0]))
        print("| %s | %.2f | %.2f | %.2f |" % (a, mean(nm), mean(nh),
                                               mean(mx)))


def learned_policy(tab, seeds):
    print("\n### What the adaptive improver learned (RECURSIVE_FULL)\n")
    by_mode = collections.defaultdict(collections.Counter)
    adopt_mode = collections.defaultdict(collections.Counter)
    frozen_mode = collections.defaultdict(collections.Counter)
    for s in seeds:
        for arm, dst in (("RECURSIVE_FULL", by_mode),
                         ("RECURSIVE_FROZEN", frozen_mode)):
            for r in tab[(arm, s)]["rounds"][2:]:     # rounds 3-5
                for t in r["tested"]:
                    if "screen" in t:
                        dst[r["mode"]][t["s"]] += 1
        for r in tab[("RECURSIVE_FULL", s)]["rounds"]:
            if r["adopted"]:
                adopt_mode[r["mode"]][r["adopted"]] += 1
    print("Most-screened actions in rounds 3-5, by diagnosed failure mode "
          "(FULL vs FROZEN):\n")
    print("| failure mode | FULL top-3 screened | FROZEN top-3 screened |")
    print("|---|---|---|")
    for m in sorted(set(by_mode) | set(frozen_mode)):
        f = ", ".join("%s (%d)" % kv for kv in by_mode[m].most_common(3))
        z = ", ".join("%s (%d)" % kv for kv in frozen_mode[m].most_common(3))
        print("| %s | %s | %s |" % (m, f, z))
    print("\nFULL adoptions by failure mode:\n")
    for m in sorted(adopt_mode):
        print("- %s: %s" % (m, ", ".join("%s %d" % kv for kv in
                                          adopt_mode[m].most_common())))
    th = collections.defaultdict(list)
    for s in seeds:
        for k, v in tab[("RECURSIVE_FULL", s)]["theta_final"].items():
            th[k].append(v)
    print("\nFinal strategy intensities (FULL; FROZEN keeps the initial "
          "values): " + ", ".join("%s %.2f" % (k, mean(v))
                                  for k, v in sorted(th.items())))
    dc = [r["delayed_credit"]["new_solves_using_it"]
          for s in seeds for r in tab[("RECURSIVE_FULL", s)]["rounds"]
          if r.get("delayed_credit")]
    if dc:
        print("\nDelayed credit: after an adopted vocabulary change, the "
              "next round's new solves used the new macros in %.2f tasks "
              "on average (%d credit events; %.0f%% non-zero)."
              % (mean(dc), len(dc),
                 100.0 * sum(1 for x in dc if x > 0) / len(dc)))


def by_length(tab, seeds):
    print("\n### External holdout solve rate by task length "
          "(primitives in the hidden program)\n")
    bins = ((2, 4), (5, 6), (7, 8), (9, 12))
    print("| arm | " + " | ".join("len %d-%d" % b for b in bins) + " |")
    print("|---|" + "---|" * len(bins))
    rows = {}
    for a in ARMS:
        cnt = collections.Counter()
        tot = collections.Counter()
        for s in seeds:
            man = T.seed_manifest(s)["holdout_ext"]
            flags = tab[(a, s)]["per_task"]["holdout_ext"]
            for row, f in zip(man, flags):
                for b in bins:
                    if b[0] <= row["length"] <= b[1]:
                        tot[b] += 1
                        cnt[b] += f
        rows[a] = {b: cnt[b] / float(tot[b]) for b in bins}
        print("| %s | %s |" % (a, " | ".join("%.3f" % rows[a][b]
                                             for b in bins)))
    return rows


def self_vs_external(tab, seeds):
    print("\n### Self-generated curriculum vs external holdout "
          "(self score is a diagnostic only)\n")
    print("| arm | self-generated solved (of 8) | external solved (of 24) |")
    print("|---|---|---|")
    for a in ARMS:
        print("| %s | %.2f | %.2f |" % (
            a, mean(tab[(a, s)]["self_solved"] for s in seeds),
            mean(tab[(a, s)]["ext"] for s in seeds)))


def compute(tab, seeds):
    print("\n### Compute actually spent (cap is identical for every arm)\n")
    print("| arm | cap | mean spent | max spent | "
          "global counter == metered |")
    print("|---|---|---|---|---|")
    for a in ARMS:
        rs = [tab[(a, s)] for s in seeds]
        print("| %s | %d | %.0f | %d | %s |" % (
            a, rs[0]["cap"], mean(r["spent"] for r in rs),
            max(r["spent"] for r in rs),
            all(r["spent"] == r["global_delta"] for r in rs)))


def paired_table(tab, seeds, field="ext"):
    print("\n### All pairwise contrasts on `%s` (two-sided p, exploratory)\n"
          % field)
    print("| contrast | mean | 95% CI | W/T/L | p |")
    print("|---|---|---|---|---|")
    for a, b in (("RECURSIVE_FULL", "SINGLE_5X"),
                 ("RECURSIVE_FULL", "RECURSIVE_FROZEN"),
                 ("RECURSIVE_FROZEN", "SINGLE_5X"),
                 ("RECURSIVE_NODIAG", "RECURSIVE_FULL"),
                 ("RECURSIVE_NODIAG", "RECURSIVE_FROZEN"),
                 ("SINGLE_5X", "COLD"), ("RECURSIVE_FULL", "COLD")):
        d = [tab[(a, s)][field] - tab[(b, s)][field] for s in seeds]
        sm = ST.summary(d, "post|%s|%s-%s" % (field, a, b))
        print("| %s - %s | %+.3f | [%+.2f, %+.2f] | %d/%d/%d | %.4f |" % (
            a, b, sm["mean"], sm["ci95"][0], sm["ci95"][1], sm["wins"],
            sm["ties"], sm["losses"], sm["p"]))


def main():
    phase = sys.argv[1] if len(sys.argv) > 1 else "confirm"
    label = None
    if "--label" in sys.argv:
        label = sys.argv[sys.argv.index("--label") + 1]
    tab, seeds = load(phase, label)
    seeds = [s for s in seeds if all((a, s) in tab for a in ARMS)]
    print("## Post-hoc mechanism analysis (%s, n = %d seeds)" % (phase,
                                                               len(seeds)))
    compute(tab, seeds)
    paired_table(tab, seeds, "ext")
    paired_table(tab, seeds, "in")
    trajectories(tab, seeds)
    hierarchy(tab, seeds)
    learned_policy(tab, seeds)
    by_length(tab, seeds)
    self_vs_external(tab, seeds)


if __name__ == "__main__":
    main()
