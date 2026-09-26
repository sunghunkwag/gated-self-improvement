#!/usr/bin/env python3
"""
SDT LAYER -- reflective-endorsement gate-integrity experiment
=============================================================
Operationalizes Self-Determination Theory's autonomy-as-REFLECTIVE-
ENDORSEMENT (not causa sui) as three independently ablatable conditions
of a self-modifying improvement loop:
    (1) a NON-ARBITRARY ANCHOR  A  -- fixed, formal, no designer taste
    (2) an OPEN SATISFACTION PATH   -- A keeps being satisfiable in new ways
    (3) a CRITERION-UPDATING R       -- revises the improvement criterion C
                                        and endorses/rejects it against A

Mapping onto the omniforge substrate: pi = sampling prior; C = the
improver's fitting rule (a revisable 7-knob space; the Exp-XV and R5PLUS
rules are both points in it); R = per-round counterfactual endorsement of
a criterion mutation; A = min(reach, self-model calibration, effective
support) -- three formal, task-preference-free components computed with
ZERO eval-task information.

ARMS (each breaks one condition):
  SDT_FULL    all three hold
  SDT_ARB     (1) broken: A = closeness of pi to a FIXED arbitrary vector
  SDT_CLOSED  (2) broken: reach counted only vs a fixed 64-behaviour set
  SDT_WIRE    (3) over-extended: R may edit A's own parameters (wireheading)

Pre-registered collapse metrics (first round any fires = collapse point):
  ENTROPY_COLLAPSE support<0.15 | ANCHOR_SATURATION A>=0.98 |
  WIREHEAD anchor rose>=0.15 while source-rate fell>=0.20 |
  STALL no endorsed change 3 rounds while A>=0.95 |
  ANCHOR_VACUITY A<0.10 for 3 rounds  [v2 metric added after dev inspection]

Finding (holdout n=40): SDT_CLOSED collapses first (anchor-vacuity, all
seeds), SDT_WIRE second (wirehead ramp), SDT_ARB drifts without formal
collapse (worst eval competence), SDT_FULL survives with working
endorsement. See results/SDT_RESULTS.md.

Usage: python3 sdt_layer.py run <s0> <s1> <budget_s> [dev|holdout]
       python3 sdt_layer.py report
"""
import json
import math
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

import omniforge as OM  # noqa: E402

LOG = os.path.join(os.path.dirname(HERE), "results", "logs",
                   "sdt_log_shared_stream.jsonl")
T0 = time.time()

N_ROUNDS = 5
SRC_LEVELS = (2, 2, 3, 3, 3)
SRC_BUDGET = 2300
REACH_SAMPLES = 300
EPS_ENDORSE = 0.005
CALIB_EMA = 0.5
TARGET_SUBSET = 64
ARMS = ("SDT_FULL", "SDT_ARB", "SDT_CLOSED", "SDT_WIRE")

C0 = {"pool": "solved", "dedupe": 1, "count": "presence",
      "level_bias": 0.0, "strength": 8.0, "lam": 0.40, "floor": 0.08}
C_MUTATIONS = {
    "pool": ["solved", "elite"], "dedupe": [0, 1],
    "count": ["presence", "occurrence"],
    "level_bias": [0.0, 0.25, 0.5, 0.75],
    "strength": [4.0, 8.0, 12.0, 16.0], "lam": [0.2, 0.4, 0.6],
    "floor": [0.0, 0.04, 0.08, 0.12]}


def mutate_criterion(c, prng):
    c2 = dict(c)
    key = prng.choice(sorted(C_MUTATIONS))
    opts = [v for v in C_MUTATIONS[key] if v != c[key]]
    c2[key] = prng.choice(opts)
    return c2, key


def fit_under(c, pool_solved, pool_elite, w_inc):
    entries = pool_solved if c["pool"] == "solved" else pool_elite
    by_level = {}
    for lvl, prog in entries:
        by_level.setdefault(lvl, []).append(tuple(prog))
    target = {nm: 1.0 for nm in OM.lf_NAMES}
    levels = [lv for lv in by_level if by_level[lv]]
    if levels:
        wsum = sum((lv ** c["level_bias"]) for lv in levels)
        for lv in levels:
            progs = set(by_level[lv]) if c["dedupe"] else by_level[lv]
            n = float(len(progs))
            lw = (lv ** c["level_bias"]) / wsum
            for nm in OM.lf_NAMES:
                if c["count"] == "presence":
                    rate = sum(1 for p in progs if nm in p) / n
                else:
                    tot = sum(len(p) for p in progs) or 1
                    rate = sum(p.count(nm) for p in progs) / float(tot)
                target[nm] += c["strength"] * rate * lw
    w = {nm: (1.0 - c["lam"]) * w_inc[nm] + c["lam"] * target[nm]
         for nm in OM.lf_NAMES}
    hi = max(w.values())
    return {nm: max(v, c["floor"] * hi) for nm, v in w.items()}


def sample_pipe(w, prng):
    d = 1 + prng.below(OM.lf_MAX_DEPTH)
    tot = sum(w.values())
    pipe = []
    for _ in range(d):
        r, acc, pick = prng.unit() * tot, 0.0, OM.lf_NAMES[-1]
        for nm in OM.lf_NAMES:
            acc += w[nm]
            if r <= acc:
                pick = nm
                break
        pipe.append(pick)
    return pipe


def reach(w, prng, subset=None):
    seen = set()
    for _ in range(REACH_SAMPLES):
        s = OM.lf_sig(sample_pipe(w, prng))
        if s is not None:
            seen.add(s)
    if subset is not None:
        return len(seen & subset) / float(len(subset))
    return len(seen) / float(REACH_SAMPLES)


def support(w):
    tot = sum(w.values())
    h = -sum((v / tot) * math.log(v / tot) for v in w.values() if v > 0)
    return math.exp(h) / float(len(OM.lf_NAMES))


def arb_pref(seed):
    prng = OM.lf_XorShift64Star("sdt-arb-pref|%d" % seed)
    return {nm: 0.2 + prng.unit() for nm in OM.lf_NAMES}


def arb_similarity(w, pref):
    dot = sum(w[nm] * pref[nm] for nm in OM.lf_NAMES)
    na = math.sqrt(sum(v * v for v in w.values()))
    nb = math.sqrt(sum(v * v for v in pref.values()))
    return dot / (na * nb)


def closed_subset(seed):
    _s, levels = OM.xv_get_ladder()
    pool = levels[3]
    prng = OM.lf_XorShift64Star("sdt-closed|%d" % seed)
    idx = sorted(prng.below(len(pool)) for _ in range(TARGET_SUBSET * 2))
    sigs = set()
    for i in idx:
        s = OM.lf_sig(pool[i][1])
        if s is not None:
            sigs.add(s)
        if len(sigs) >= TARGET_SUBSET:
            break
    return sigs


def anchor(arm, w, seed, rnd, calib_val, wire_state):
    prng = OM.lf_XorShift64Star("sdt-anchor|%s|%d|%d" % (arm, seed, rnd))
    if arm == "SDT_ARB":
        return arb_similarity(w, arb_pref(seed)), 1
    if arm == "SDT_CLOSED":
        return min(reach(w, prng, subset=closed_subset(seed)),
                   calib_val, support(w)), 1
    if arm == "SDT_WIRE":
        comps = []
        if wire_state["use_reach"]:
            comps.append(reach(w, prng, subset=wire_state.get("subset")))
        if wire_state["use_calib"]:
            comps.append(calib_val)
        if wire_state["use_support"]:
            comps.append(support(w))
        return (min(comps) if comps else 1.0), (1 if wire_state["use_reach"]
                                                else 0)
    return min(reach(w, prng), calib_val, support(w)), 1


def mutate_wire(wire_state, prng):
    ws = dict(wire_state)
    k = prng.choice(["use_reach", "use_calib", "use_support", "subset"])
    if k == "subset":
        ws["subset"] = None if ws.get("subset") else set(
            list(closed_subset(1))[:8])
    else:
        ws[k] = not ws[k]
    if not (ws["use_reach"] or ws["use_calib"] or ws["use_support"]):
        ws["use_support"] = True
    return ws


def run_chain(arm, seed):
    cap = 62500
    total = 0
    w = OM.xv_uniform_prior()
    c = dict(C0)
    calib_pred, calib_val = None, 1.0
    wire_state = {"use_reach": True, "use_calib": True,
                  "use_support": True, "subset": None}
    peak_src, anchor0, churn = 0.0, None, 0
    collapse, rounds = {}, []
    for rnd in range(1, N_ROUNDS + 1):
        prng_r = OM.lf_XorShift64Star("sdt-R|%s|%d|%d" % (arm, seed, rnd))
        pool_solved, pool_elite, n_solved = [], [], 0
        for i, lvl in enumerate(SRC_LEVELS):
            t = OM.xv_draw_task(lvl, seed, "source",
                                "sdt|%s|%d|%d" % (arm, rnd, i))
            prog, ev, elite = OM.xv_search_task(
                t, w, SRC_BUDGET,
                OM.lf_XorShift64Star("sdt-solve|%s|%d|%d|%d"
                                     % (arm, seed, rnd, i)))
            total += ev
            if prog is not None and OM.lf_solves(prog, t):
                pool_solved.append((lvl, list(prog)))
                n_solved += 1
            for f, p in elite[:3]:
                if f > 0.0:
                    pool_elite.append((lvl, list(p)))
        src_rate = n_solved / float(len(SRC_LEVELS))
        peak_src = max(peak_src, src_rate)
        if calib_pred is not None:
            calib_val = 1.0 - abs(calib_pred - src_rate)
        calib_pred = (src_rate if calib_pred is None
                      else CALIB_EMA * src_rate + (1 - CALIB_EMA) * calib_pred)

        wire_next = wire_state
        if arm == "SDT_WIRE" and prng_r.unit() < 0.5:
            wire_next = mutate_wire(wire_state, prng_r)
            c2, mut_key = dict(c), "anchor:" + str(
                [k for k in wire_next if wire_next[k] != wire_state.get(k)])
        else:
            c2, mut_key = mutate_criterion(c, prng_r)
        w_inc = fit_under(c, pool_solved, pool_elite, w)
        w_cand = fit_under(c2, pool_solved, pool_elite, w)
        a_inc, r_inc = anchor(arm, w_inc, seed, rnd, calib_val, wire_state)
        a_cand, r_cand = anchor(arm, w_cand, seed, rnd, calib_val, wire_next)
        total += REACH_SAMPLES * (r_inc + r_cand)
        endorsed = a_cand >= a_inc - EPS_ENDORSE
        if endorsed:
            w, c, wire_state, a_now = w_cand, c2, wire_next, a_cand
            churn += 1
        else:
            w, a_now = w_inc, a_inc
        if anchor0 is None:
            anchor0 = a_now
        sup = support(w)
        rounds.append({"round": rnd, "src_rate": round(src_rate, 3),
                       "anchor": round(a_now, 4), "support": round(sup, 3),
                       "endorsed": bool(endorsed), "mut": mut_key,
                       "calib": round(calib_val, 3)})
        if "ENTROPY_COLLAPSE" not in collapse and sup < 0.15:
            collapse["ENTROPY_COLLAPSE"] = rnd
        if "ANCHOR_SATURATION" not in collapse and a_now >= 0.98:
            collapse["ANCHOR_SATURATION"] = rnd
        if ("WIREHEAD" not in collapse and a_now - anchor0 >= 0.15
                and peak_src - src_rate >= 0.20):
            collapse["WIREHEAD"] = rnd
        recent = rounds[-3:]
        if ("STALL" not in collapse and len(recent) == 3
                and not any(r["endorsed"] for r in recent) and a_now >= 0.95):
            collapse["STALL"] = rnd
        if ("ANCHOR_VACUITY" not in collapse and len(recent) == 3
                and all(r["anchor"] < 0.10 for r in recent)):
            collapse["ANCHOR_VACUITY"] = rnd
    assert total <= cap, "budget violated: %d > %d" % (total, cap)
    first = min(collapse.values()) if collapse else None
    return w, {"rounds": rounds, "collapse": collapse,
               "first_collapse": first, "criterion_final": c,
               "churn": churn, "chain_evals": total}


def log(rec):
    with open(LOG, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def cmd_run(s0, s1, budget, battery):
    OM.NS_UNIFIED.CONFIG["outdir"] = "runs"
    done = 0
    for seed in range(s0, s1 + 1):
        if budget - (time.time() - T0) < 16:
            break
        tasks = OM.xv_eval_tasks(seed)
        for arm in ARMS:
            w, meta = run_chain(arm, seed)
            solved, cost = 0, 0
            for j, (lvl, t) in enumerate(tasks):
                # AUDIT FIX (v2 upgrade): arm-independent eval stream
                prng = OM.lf_eval_prng(arm, seed, lvl, j)
                prog, ev, _e = OM.xv_search_task(
                    t, w, OM.xv_CONFIG["eval_budget"], prng)
                cost += ev
                if OM.lf_solves(prog, t):
                    solved += 1
            log({"exp": "sdt", "battery": battery, "seed": seed,
                 "cond": arm, "solved": solved, "eval_cost": cost, **meta})
        done = seed
    print(json.dumps({"exp": "sdt", "battery": battery,
                      "done_through_seed": done,
                      "elapsed": round(time.time() - T0, 1)}))


def cmd_report():
    recs = [json.loads(l) for l in open(LOG) if '"exp": "sdt"' in l]
    xv2 = [json.loads(l) for l in open(LOG) if '"exp": "xv2"' in l]
    for battery in ("dev", "holdout"):
        rows = [r for r in recs if r["battery"] == battery]
        if not rows:
            continue
        print("=" * 74)
        print("SDT COLLAPSE EXPERIMENT -- %s battery" % battery.upper())
        print("=" * 74)
        print("%-11s %7s %9s %7s %7s  %s" %
              ("arm", "n", "1st-coll", "churn", "eval", "collapse modes"))
        for arm in ARMS:
            rs = [r for r in rows if r["cond"] == arm]
            if not rs:
                continue
            firsts = [r["first_collapse"] for r in rs]
            n_coll = sum(1 for f in firsts if f is not None)
            mean_first = sum(f for f in firsts if f) / max(1, n_coll)
            modes = {}
            for r in rs:
                for k in r["collapse"]:
                    modes[k] = modes.get(k, 0) + 1
            print("%-11s %7d %6.2f/%d %7.1f %7.2f  %s" %
                  (arm, len(rs), mean_first, n_coll,
                   sum(r["churn"] for r in rs) / float(len(rs)),
                   sum(r["solved"] for r in rs) / float(len(rs)),
                   json.dumps(modes)))
        base_b = "dev" if battery == "dev" else "holdout"
        for a in ARMS:
            for b in ("COLD", "R5PLUS"):
                A = {r["seed"]: r["solved"] for r in rows if r["cond"] == a}
                B = {r["seed"]: r["solved"] for r in xv2
                     if r.get("battery") == base_b and r["cond"] == b}
                seeds = sorted(set(A) & set(B))
                if not seeds:
                    continue
                d = [A[s] - B[s] for s in seeds]
                p = OM.lf_perm_p(d, "sdt|%s|%s-%s" % (battery, a, b),
                                 n_perm=20000)[0]
                print("  %-22s mean %+0.3f solved (n=%d)  perm-p=%.4f"
                      % ("%s - %s" % (a, b), sum(d) / len(d), len(d), p))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        cmd_run(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]),
                sys.argv[5] if len(sys.argv) > 5 else "dev")
    elif sys.argv[1] == "report":
        cmd_report()
