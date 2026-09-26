#!/usr/bin/env python3
"""
RSI UPGRADE -- engineered recursive self-improvement for omniforge
==================================================================
Diagnosis of the ORIGINAL Expedition-XV mechanism's failure (see
results/RSI_RESULTS.md): ROUND5 - COLD = -1.50 solved (p=0.039) -- five
recursive prior-fitting rounds land BELOW the untrained baseline.
Autopsy: (P1) noisy partial-fitness elite pollute the fit pool; (P2)
occurrence counts over a cumulative pool compound early biases (entropy
collapses); (P3) no step-size control or self-correction.

UPGRADED MECHANISM (condition R5PLUS; ablation R5PLUS_NG = no gate):
  M1 solved-only, deduped credit
  M2 presence-based, level-stratified target (kills length bias)
  M3 trust-region update + entropy floor (exploration never collapses)
  M4 generalization-gated acceptance + rollback decay (self-correction)

Fairness: identical held-out eval tasks / budget / PRNG streams as every
condition; total chain compute <= the same 62,500 candidate budget as
ROUND5 / ROUND1_5X, asserted at runtime; probes from the SOURCE half,
eval from the TARGET half. Params frozen after 3 dev iterations on seeds
1-12; the confirmatory battery runs on untouched holdout seeds 101-140.

This is the standalone dev script; the same mechanism is embedded in
omniforge.py as the `up_*` section (python3 omniforge.py upgrade ...).

Usage:
  python3 rsi_upgrade.py xv2 <s0> <s1> <budget_s> [dev|holdout]
  python3 rsi_upgrade.py mrl2 <s0> <s1> <budget_s> [dev|holdout]
  python3 rsi_upgrade.py report2
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

import omniforge as OM  # noqa: E402

LOG = os.path.join(os.path.dirname(HERE), "results", "logs", "experiments_log.jsonl")
T0 = time.time()

# frozen after dev-seed tuning (seeds 1-12); holdout never touched these
LAM = 0.40
STRENGTH = 8.0
FLOOR = 0.08
DECAY = 0.15
SRC_LEVELS_PLUS = (2, 2, 3, 3, 3)
SRC_BUDGET_G = 2000
SRC_BUDGET_NG = 2500
PROBE_LEVELS = (3, 3, 4)
PROBE_BUDGET = 400


def log(rec):
    with open(LOG, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def left(budget):
    return budget - (time.time() - T0)


# ---------------------------------------------------------------------------
def fit_presence(solved_by_level):
    """M1+M2: presence-rate target from solved-only, deduped programs."""
    target = {nm: 1.0 for nm in OM.lf_NAMES}
    levels = [lv for lv, progs in solved_by_level.items() if progs]
    for lv in levels:
        progs = {tuple(p) for p in solved_by_level[lv]}
        n = float(len(progs))
        for nm in OM.lf_NAMES:
            rate = sum(1 for p in progs if nm in p) / n
            target[nm] += STRENGTH * rate / len(levels)
    return target


def trust_step(w_inc, target):
    """M3: bounded step + entropy floor."""
    w = {nm: (1.0 - LAM) * w_inc[nm] + LAM * target[nm]
         for nm in OM.lf_NAMES}
    hi = max(w.values())
    return {nm: max(v, FLOOR * hi) for nm, v in w.items()}


def decay_to_uniform(w, lam=DECAY):
    return {nm: (1.0 - lam) * v + lam * 1.0 for nm, v in w.items()}


def build_chain_plus(seed, gated=True):
    cfg = OM.xv_CONFIG
    tagp = "p" if gated else "png"
    w = OM.xv_uniform_prior()
    total = 0
    budget_cap = cfg["n_rounds"] * len(cfg["src_levels"]) * cfg["src_budget"]
    src_budget = SRC_BUDGET_G if gated else SRC_BUDGET_NG
    gate_log = []
    for rnd in range(1, cfg["n_rounds"] + 1):
        solved_by_level = {}
        for i, lvl in enumerate(SRC_LEVELS_PLUS):
            t = OM.xv_draw_task(lvl, seed, "source", "%s|%d|%d" % (tagp,
                                                                   rnd, i))
            prog, ev, _elite = OM.xv_search_task(
                t, w, src_budget,
                OM.lf_XorShift64Star("%s-solve|%s|%d|%d"
                                     % (tagp, seed, rnd, i)))
            total += ev
            if prog is not None and OM.lf_solves(prog, t):
                solved_by_level.setdefault(lvl, []).append(list(prog))
        target = fit_presence(solved_by_level)
        cand = trust_step(w, target)
        if not gated:
            w = cand
            continue
        probes = [OM.xv_draw_task(lv, seed, "source",
                                  "%s-probe|%d|%d" % (tagp, rnd, j))
                  for j, lv in enumerate(PROBE_LEVELS)]
        inc_s, inc_c = OM.xv_trial(w, probes, PROBE_BUDGET,
                                   "%s-gate-i|%s|%d" % (tagp, seed, rnd))
        can_s, can_c = OM.xv_trial(cand, probes, PROBE_BUDGET,
                                   "%s-gate-c|%s|%d" % (tagp, seed, rnd))
        total += inc_c + can_c
        accept = can_s > inc_s or (can_s == inc_s and can_c <= inc_c)
        gate_log.append({"round": rnd, "inc": inc_s, "cand": can_s,
                         "accepted": bool(accept)})
        w = cand if accept else decay_to_uniform(w)
    assert total <= budget_cap, "budget violated: %d > %d" % (total,
                                                              budget_cap)
    return w, total, gate_log


def eval_unit(cond, w, seed, tasks):
    """Mirror of xv_run_unit's evaluation (same streams, same budget)."""
    cfg = OM.xv_CONFIG
    by_level, solved, cost = {}, 0, 0
    for j, (lvl, t) in enumerate(tasks):
        # AUDIT FIX: arm-independent stream (was keyed on `cond`)
        prng = OM.lf_eval_prng(cond, seed, lvl, j)
        prog, ev, _e = OM.xv_search_task(t, w, cfg["eval_budget"], prng)
        cost += ev
        if OM.lf_solves(prog, t):
            solved += 1
            by_level[str(lvl)] = by_level.get(str(lvl), 0) + 1
    return {"cond": cond, "seed": seed, "solved": solved,
            "by_level": by_level, "cost": cost}


def cmd_xv2(s0, s1, budget, battery):
    OM.NS_UNIFIED.CONFIG["outdir"] = "runs"
    baselines = ["COLD", "ROUND5", "ROUND1_5X", "GATED5"]
    done = 0
    for seed in range(s0, s1 + 1):
        if left(budget) < 11:
            break
        ch = OM.xv_build_chains(seed)
        tasks = OM.xv_eval_tasks(seed)
        for cond in baselines:
            r = OM.xv_run_unit(cond, seed, ch, tasks)
            log({"exp": "xv2", "battery": battery, "seed": seed,
                 "cond": cond, "solved": r["solved"],
                 "by_level": r["by_level"], "cost": r["cost"]})
        for cond, gated in (("R5PLUS", True), ("R5PLUS_NG", False)):
            w, used, glog = build_chain_plus(seed, gated=gated)
            r = eval_unit(cond, w, seed, tasks)
            log({"exp": "xv2", "battery": battery, "seed": seed,
                 "cond": cond, "solved": r["solved"],
                 "by_level": r["by_level"], "cost": r["cost"],
                 "chain_evals": used,
                 "gate_accepts": sum(1 for g in glog if g["accepted"])
                 if gated else None})
        done = seed
    print(json.dumps({"exp": "xv2", "battery": battery,
                      "done_through_seed": done,
                      "elapsed": round(time.time() - T0, 1)}))


def mrl2_cfg():
    cfg = OM.mrl_CFG().fast()
    cfg.EVAL_TASKS = 8
    cfg.PARTIAL = 0.0
    return cfg


def cmd_mrl2(s0, s1, budget, battery):
    cfg = mrl2_cfg()
    conds = ["COLD", "COLD2", "LEARN", "PLAST", "CTRL", "HYBRID"]
    done = 0
    for seed in range(s0, s1 + 1):
        if left(budget) < 8:
            break
        for cond in conds:
            r = OM.mrl_run_unit(cond, seed, cfg, write=False)
            log({"exp": "mrl2", "battery": battery, "seed": seed,
                 "cond": cond, "ladder_score": r["ladder_score"],
                 "frontier": r["frontier"],
                 "train_solved": r["train_solved"]})
        done = seed
    print(json.dumps({"exp": "mrl2", "battery": battery,
                      "done_through_seed": done,
                      "elapsed": round(time.time() - T0, 1)}))


def _paired(recs, exp, battery, a, b, field):
    A, B = {}, {}
    for r in recs:
        if r.get("exp") == exp and r.get("battery") == battery:
            if r["cond"] == a:
                A[r["seed"]] = r[field]
            if r["cond"] == b:
                B[r["seed"]] = r[field]
    seeds = sorted(set(A) & set(B))
    return [A[s] - B[s] for s in seeds]


def cmd_report2():
    recs = [json.loads(l) for l in open(LOG)] if os.path.exists(LOG) else []
    for battery in ("dev", "holdout"):
        if not any(r.get("battery") == battery for r in recs):
            continue
        print("=" * 74)
        print("UPGRADED-MECHANISM BATTERY: %s" % battery.upper())
        print("=" * 74)
        print("[XV2] recursive prior self-improvement, upgraded")
        for a, b, note in (
                ("R5PLUS", "COLD", "upgraded 5-round chain vs no learning"),
                ("R5PLUS", "ROUND1_5X",
                 "upgraded chain vs 1 round at 5x compute (compounding)"),
                ("R5PLUS", "ROUND5", "upgraded vs original chain"),
                ("R5PLUS", "R5PLUS_NG", "generalization gate ablation"),
                ("R5PLUS_NG", "COLD", "M1-M3 alone vs no learning"),
                ("ROUND5", "COLD", "original chain vs no learning (ref)")):
            d = _paired(recs, "xv2", battery, a, b, "solved")
            if not d:
                continue
            p = OM.lf_perm_p(d, "%s|%s-%s" % (battery, a, b),
                             n_perm=20000)[0]
            print("  %-22s mean %+0.3f solved (n=%d)  perm-p=%.4f  %s"
                  % ("%s - %s" % (a, b), sum(d) / len(d), len(d), p, note))
        print("[MRL2] meta-RL grid, upgraded yardstick + solved-only credit")
        for a, b, note in (
                ("HYBRID", "COLD", "full stack vs none"),
                ("LEARN", "COLD", "learning vs none"),
                ("PLAST", "LEARN", "metaplasticity increment"),
                ("CTRL", "LEARN", "controller increment"),
                ("COLD2", "COLD", "noise floor (should be ~0)")):
            d = _paired(recs, "mrl2", battery, a, b, "ladder_score")
            if not d:
                continue
            p = OM.lf_perm_p(d, "m2|%s|%s-%s" % (battery, a, b),
                             n_perm=20000)[0]
            print("  %-22s mean %+0.4f ladder (n=%d)  perm-p=%.4f  %s"
                  % ("%s - %s" % (a, b), sum(d) / len(d), len(d), p, note))


def cli():
    cmd = sys.argv[1]
    if cmd == "xv2":
        cmd_xv2(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]),
                sys.argv[5] if len(sys.argv) > 5 else "dev")
    elif cmd == "mrl2":
        cmd_mrl2(int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]),
                 sys.argv[5] if len(sys.argv) > 5 else "dev")
    elif cmd == "report2":
        cmd_report2()
    else:
        sys.exit("unknown command")


if __name__ == "__main__":
    cli()
