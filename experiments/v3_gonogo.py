#!/usr/bin/env python3
"""
Applies the PRE-REGISTERED go/no-go criterion (v3 ledger record
GO_NO_GO_CRITERION, written before dev-11 results and before any seed in
3101-3200 was run) to the dev-check battery, and appends the verdict to the
v3 ledger. No judgement calls: all four conditions are computed here.

  python3 experiments/v3_gonogo.py v3devcheck-01-dev10config
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))

from rsi_v2 import stats as ST          # noqa: E402
from rsi_v3 import runner as R          # noqa: E402


def main(label):
    led = R.Ledger(R.LEDGER_PATH)
    crit = led.find("GO_NO_GO_CRITERION")
    assert len(crit) == 1
    end = [r for r in led.find("DEVCHECK_END") if r["body"]["label"] == label]
    assert len(end) == 1, "dev-check not finished"
    assert crit[0]["seq"] < min(r["seq"] for r in led.find("DEVCHECK_START"))
    units = [json.loads(l) for l in open(R.DEV_LOG)
             if json.loads(l).get("label") == label]
    tab = {(u["arm"], u["seed"]): u for u in units}
    seeds = sorted({s for (_a, s) in tab})
    assert seeds == list(range(3101, 3201)), "incomplete dev-check"

    def diffs(a, b, f):
        if f == "meta_quality":
            return [R.meta_quality(tab[(a, s)]) - R.meta_quality(tab[(b, s)])
                    for s in seeds]
        return [tab[(a, s)][f] - tab[(b, s)][f] for s in seeds]
    out = {}
    for name, a, b, f, thr in (
            ("ext_vs_frozen", "ADAPTIVE_META", "FROZEN_META", "ext", 0.05),
            ("ext_vs_nocarry", "ADAPTIVE_META", "ADAPTIVE_NOCARRY", "ext",
             0.10),
            ("meta_vs_frozen", "ADAPTIVE_META", "FROZEN_META",
             "meta_quality", 0.05)):
        d = diffs(a, b, f)
        sm = ST.summary(d, "gonogo|%s" % name, one_sided=True)
        sm["threshold"] = thr
        sm["pass"] = bool(sm["mean"] > 0 and sm["p"] < thr)
        out[name] = sm
    W = len(tab[("ADAPTIVE_META", seeds[0])]["ext_w"])
    per_w = [sum(tab[("ADAPTIVE_META", s)]["ext_w"][w]
                 - tab[("FROZEN_META", s)]["ext_w"][w] for s in seeds)
             / float(len(seeds)) for w in range(W)]
    first, second = per_w[:W // 2], per_w[W // 2:]
    out["persistence"] = {
        "per_world_adaptive_minus_frozen": [round(x, 3) for x in per_w],
        "first_half_mean": sum(first) / len(first),
        "second_half_mean": sum(second) / len(second),
        "pass": bool(sum(second) / len(second) >= sum(first) / len(first))}
    for a, b in (("ADAPTIVE_META", "NODIAG_META"),
                 ("ADAPTIVE_META", "SINGLE_COMPUTE_MATCHED"),
                 ("FROZEN_META", "SINGLE_COMPUTE_MATCHED")):
        out["info_%s-%s" % (a, b)] = ST.summary(
            diffs(a, b, "ext"), "gonogo|info|%s-%s" % (a, b))
    go = all(out[k]["pass"] for k in ("ext_vs_frozen", "ext_vs_nocarry",
                                      "meta_vs_frozen", "persistence"))
    out["verdict"] = "GO" if go else "NO_GO"
    led.append("GO_NO_GO_RESULT", {"label": label, "result": out})
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
