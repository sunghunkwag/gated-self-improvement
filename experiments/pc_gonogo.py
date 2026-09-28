#!/usr/bin/env python3
"""
Applies the go/no-go criterion for the process-controller phase to a
DEVCHECK battery and appends the verdict to the v3 ledger. The criterion
itself is recorded in the ledger (GO_NO_GO_CRITERION, phase
"process-controller") BEFORE the battery is run; this script only computes
it -- no judgement calls.

  python3 experiments/pc_gonogo.py LABEL
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)

from rsi_v2 import stats as ST          # noqa: E402
from rsi_v3 import runner as R          # noqa: E402

SEEDS = list(range(3201, 3301))


def criterion(led):
    crit = [r for r in led.find("GO_NO_GO_CRITERION")
            if r["body"].get("phase") == "process-controller"]
    assert len(crit) == 1, "exactly one process-controller criterion"
    return crit[0]


def mean(x):
    return sum(x) / float(len(x))


def main(label):
    led = R.Ledger(R.LEDGER_PATH)
    crit = criterion(led)
    c = crit["body"]["params"]
    starts = [r for r in led.find("DEVCHECK_START")
              if r["body"]["label"] == label]
    ends = [r for r in led.find("DEVCHECK_END")
            if r["body"]["label"] == label]
    assert len(starts) == 1 and len(ends) == 1, "dev-check not finished"
    assert crit["seq"] < starts[0]["seq"], "criterion must precede the look"
    tab = {}
    for line in open(R.DEV_LOG):
        u = json.loads(line)
        if u.get("label") == label:
            tab[(u["arm"], u["seed"])] = u
    T, N, F = "MEMORY_CARRY", "NO_CARRY", "FROZEN"
    assert all((a, s) in tab for a in (T, N, F) for s in SEEDS)
    out = {}

    def test(name, d, thr):
        sm = ST.summary(d, "pcgonogo|%s" % name, one_sided=True)
        sm["threshold"] = thr
        sm["pass"] = bool(sm["mean"] > 0 and sm["p"] < thr)
        out[name] = sm

    test("ext_carry_vs_nocarry",
         [tab[(T, s)]["ext"] - tab[(N, s)]["ext"] for s in SEEDS],
         c["alpha_primary"])
    test("ext_carry_vs_frozen",
         [tab[(T, s)]["ext"] - tab[(F, s)]["ext"] for s in SEEDS],
         c["alpha_frozen"])

    def transfer(u):
        xs = [x for x in u["transfer_w"] if x is not None]
        return mean(xs) if xs else 0.0
    test("transfer_carry_vs_nocarry",
         [transfer(tab[(T, s)]) - transfer(tab[(N, s)]) for s in SEEDS],
         c["alpha_transfer"])
    W = len(tab[(T, SEEDS[0])]["ext_w"])
    per_w = [mean([tab[(T, s)]["ext_w"][w] - tab[(N, s)]["ext_w"][w]
                   for s in SEEDS]) for w in range(W)]
    h = W // 2
    out["persistence"] = {
        "per_world_carry_minus_nocarry": [round(x, 3) for x in per_w],
        "first_half": mean(per_w[:h]), "second_half": mean(per_w[h:]),
        "pass": bool(mean(per_w[h:]) > 0)}
    jac = mean([mean(tab[(T, s)]["pool_jaccard_w"][1:]) for s in SEEDS])
    r1 = []
    for s in SEEDS:
        a, b = tab[(T, s)], tab[(N, s)]
        # round-1 decisions of worlds >= 2: NO_CARRY has no memory there
        r1 += [1.0 if pa[0] != pb[0] else 0.0
               for pa, pb in zip(a["plans_w"][1:], b["plans_w"][1:])]
    out["decisions_change"] = {
        "carry_pool_jaccard_vs_default_worlds_ge2": jac,
        "pass": bool(jac < c["max_pool_jaccard"])}
    out["info_round1_plan_differs_share"] = mean(r1) if r1 else None
    for a, b in (("NO_CARRY", "FROZEN"), ("MEMORY_CARRY", "MEMORY_RANKONLY")):
        if all((a, s) in tab and (b, s) in tab for s in SEEDS):
            out["info_%s-%s" % (a, b)] = ST.summary(
                [tab[(a, s)]["ext"] - tab[(b, s)]["ext"] for s in SEEDS],
                "pcgonogo|info|%s-%s" % (a, b), one_sided=True)
    keys = ("ext_carry_vs_nocarry", "ext_carry_vs_frozen",
            "transfer_carry_vs_nocarry", "persistence", "decisions_change")
    out["verdict"] = "GO" if all(out[k]["pass"] for k in keys) else "NO_GO"
    led.append("GO_NO_GO_RESULT", {"label": label,
                                   "phase": "process-controller",
                                   "criterion_seq": crit["seq"],
                                   "result": out})
    print(json.dumps(out, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(sys.argv[1])
