#!/usr/bin/env python3
"""
AUDIT RERUN of the v1 compounding-RSI headline (XV2, R5PLUS vs ROUND1_5X)
=========================================================================
Audit findings this script tests (see results/UPGRADE_V2_RESULTS.md, Part A):

  F1  Evaluation RNG depended on the arm name: every held-out search was
      seeded 'ev|<cond>|seed|lvl|j', so counterfactual arms were scored
      under DIFFERENT randomness. Fixed in omniforge.lf_eval_prng (shared
      stream for every arm; only the COLD2 noise-floor replicate uses a
      second stream on purpose).
  F2  GATED5's counterfactual gate spent up to 5 x 2 x 3 x 800 = 24,000
      candidate evaluations that were never counted against its 62,500
      budget. Now recorded in the chain meta ('gated_total_evals').
  F3  Data confound: R5PLUS attempts 25 distinct source tasks, ROUND1_5X
      only 5 (with 5x budget each). A data-matched single-round control
      (R1PLUS_DM: the SAME 25 tasks, same 62,500 budget, one presence fit
      with the cumulative trust step 1-(1-lam)^5) separates "recursion"
      from "saw 5x more distinct tasks".

For every seed both scorings are computed on identical priors:
  *_OLD : the original arm-keyed stream  (reproduces the published numbers)
  plain : the fixed shared stream        (the corrected numbers)

Usage:
  python3 experiments/audit_v1_rerun.py run 101 200 [workers]
  python3 experiments/audit_v1_rerun.py report
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)
LOG = os.path.join(ROOT, "results", "logs", "audit_v1_rerun.jsonl")


def _om():
    os.chdir(SRC)  # ladder cache lives in src/runs (as rsi_upgrade.py uses)
    import rsi_upgrade as RU  # noqa: E402  (imports omniforge as OM)
    return RU, RU.OM


def eval_with(OM, w, seed, tasks, stream_key):
    cfg = OM.xv_CONFIG
    solved, cost, by_level = 0, 0, {}
    for j, (lvl, t) in enumerate(tasks):
        prng = stream_key(lvl, j)
        prog, ev, _e = OM.xv_search_task(t, w, cfg["eval_budget"], prng)
        cost += ev
        if OM.lf_solves(prog, t):
            solved += 1
            by_level[str(lvl)] = by_level.get(str(lvl), 0) + 1
    return solved, by_level, cost


def build_r1plus_dm(RU, OM, seed):
    """One round over the SAME 25 source tasks R5PLUS attempts, same total
    budget (25 x 2500 = 62,500), a single presence fit with the cumulative
    trust-region step of five rounds. No recursion, no gate."""
    cfg = OM.xv_CONFIG
    lam_eff = 1.0 - (1.0 - RU.LAM) ** cfg["n_rounds"]
    w0 = OM.xv_uniform_prior()
    total, solved_by_level = 0, {}
    for rnd in range(1, cfg["n_rounds"] + 1):
        for i, lvl in enumerate(RU.SRC_LEVELS_PLUS):
            t = OM.xv_draw_task(lvl, seed, "source", "p|%d|%d" % (rnd, i))
            prog, ev, _e = OM.xv_search_task(
                t, w0, RU.SRC_BUDGET_NG,
                OM.lf_XorShift64Star("dm-solve|%s|%d|%d" % (seed, rnd, i)))
            total += ev
            if prog is not None and OM.lf_solves(prog, t):
                solved_by_level.setdefault(lvl, []).append(list(prog))
    target = RU.fit_presence(solved_by_level)
    w = {nm: (1.0 - lam_eff) * w0[nm] + lam_eff * target[nm]
         for nm in OM.lf_NAMES}
    hi = max(w.values())
    w = {nm: max(v, RU.FLOOR * hi) for nm, v in w.items()}
    cap = cfg["n_rounds"] * len(cfg["src_levels"]) * cfg["src_budget"]
    assert total <= cap
    return w, total


def run_seed(seed):
    RU, OM = _om()
    t0 = time.time()
    ch = OM.xv_build_chains(seed)
    tasks = OM.xv_eval_tasks(seed)
    priors = {
        "COLD": OM.xv_prior_for("COLD", ch),
        "ROUND5": OM.xv_prior_for("ROUND5", ch),
        "ROUND1_5X": OM.xv_prior_for("ROUND1_5X", ch),
        "GATED5": OM.xv_prior_for("GATED5", ch),
    }
    chain_evals = {"ROUND5": ch["meta"]["chain_evals"],
                   "ROUND1_5X": ch["meta"]["fivex_evals"],
                   "GATED5": ch["meta"]["gated_total_evals"], "COLD": 0}
    w, used, glog = RU.build_chain_plus(seed, gated=True)
    priors["R5PLUS"] = w
    chain_evals["R5PLUS"] = used
    w, used, _g = RU.build_chain_plus(seed, gated=False)
    priors["R5PLUS_NG"] = w
    chain_evals["R5PLUS_NG"] = used
    w, used = build_r1plus_dm(RU, OM, seed)
    priors["R1PLUS_DM"] = w
    chain_evals["R1PLUS_DM"] = used
    recs = []
    for cond, w in sorted(priors.items()):
        s, bl, c = eval_with(OM, w, seed, tasks,
                             lambda lvl, j: OM.lf_eval_prng(cond, seed,
                                                            lvl, j))
        recs.append({"exp": "audit_v1_rerun", "stream": "shared",
                     "seed": seed, "cond": cond, "solved": s,
                     "by_level": bl, "eval_cost": c,
                     "chain_evals": chain_evals[cond],
                     "chain_cap": 62500})
    # COLD2: the deliberate noise-floor replicate (stream 1)
    s, bl, c = eval_with(OM, priors["COLD"], seed, tasks,
                         lambda lvl, j: OM.lf_eval_prng("COLD2", seed,
                                                        lvl, j))
    recs.append({"exp": "audit_v1_rerun", "stream": "shared", "seed": seed,
                 "cond": "COLD2", "solved": s, "by_level": bl,
                 "eval_cost": c, "chain_evals": 0, "chain_cap": 62500})
    # the ORIGINAL arm-keyed scoring, same priors (reproduces published)
    for cond in ("COLD", "ROUND5", "ROUND1_5X", "R5PLUS"):
        s, bl, c = eval_with(
            OM, priors[cond], seed, tasks,
            lambda lvl, j: OM.lf_XorShift64Star(
                "ev|%s|%s|%d|%d" % (cond, seed, lvl, j)))
        recs.append({"exp": "audit_v1_rerun", "stream": "arm_keyed_OLD",
                     "seed": seed, "cond": cond, "solved": s,
                     "by_level": bl, "eval_cost": c})
    for r in recs:
        r["gated5_gate_evals"] = ch["meta"]["gated_gate_evals"]
        r["elapsed_s"] = round(time.time() - t0, 1)
    return recs


def cmd_run(s0, s1, workers):
    done = set()
    if os.path.exists(LOG):
        for line in open(LOG):
            r = json.loads(line)
            done.add(r["seed"])
    seeds = [s for s in range(s0, s1 + 1) if s not in done]
    RU, OM = _om()
    OM.xv_get_ladder()  # build/load the cache once before forking
    if workers > 1:
        import multiprocessing as mp  # runner-level only; engine untouched
        with mp.Pool(workers) as pool:
            for recs in pool.imap_unordered(run_seed, seeds):
                with open(LOG, "a") as f:
                    for r in recs:
                        f.write(json.dumps(r, sort_keys=True) + "\n")
                print("seed %d done (%.0fs)" % (recs[0]["seed"],
                                               recs[0]["elapsed_s"]),
                      flush=True)
    else:
        for s in seeds:
            recs = run_seed(s)
            with open(LOG, "a") as f:
                for r in recs:
                    f.write(json.dumps(r, sort_keys=True) + "\n")
            print("seed %d done" % s, flush=True)


def cmd_report():
    RU, OM = _om()
    recs = [json.loads(l) for l in open(LOG)]
    tab = {}
    for r in recs:
        tab[(r["stream"], r["cond"], r["seed"])] = r

    def diffs(stream, a, b, seeds):
        return [tab[(stream, a, s)]["solved"] - tab[(stream, b, s)]["solved"]
                for s in seeds
                if (stream, a, s) in tab and (stream, b, s) in tab]
    for label, seeds in (("local holdout 101-140", range(101, 141)),
                         ("kaggle seeds 141-200", range(141, 201)),
                         ("combined 101-200", range(101, 201))):
        print("=" * 72)
        print(label)
        for stream in ("arm_keyed_OLD", "shared"):
            print("  [%s eval stream]" % stream)
            for a, b in (("R5PLUS", "ROUND1_5X"), ("R5PLUS", "COLD"),
                         ("ROUND1_5X", "COLD"), ("ROUND5", "COLD"),
                         ("R5PLUS", "ROUND5"), ("R5PLUS", "R1PLUS_DM"),
                         ("R1PLUS_DM", "ROUND1_5X"), ("R1PLUS_DM", "COLD"),
                         ("R5PLUS", "R5PLUS_NG"), ("GATED5", "ROUND5"),
                         ("COLD2", "COLD")):
                d = diffs(stream, a, b, seeds)
                if not d:
                    continue
                p = OM.lf_perm_p(d, "audit|%s|%s-%s" % (label, a, b))[0]
                lo, hi = OM.lf_boot_ci(d, "audit|%s|%s-%s" % (label, a, b))
                print("    %-22s %+0.3f  [%+.2f, %+.2f]  n=%3d  p=%.4f"
                      % ("%s - %s" % (a, b), sum(d) / len(d), lo, hi,
                         len(d), p))
    g = [tab[("shared", "GATED5", s)]["chain_evals"] for s in range(101, 201)
         if ("shared", "GATED5", s) in tab]
    if g:
        print("GATED5 total improvement compute (source + gate): mean %.0f, "
              "max %d vs 62,500 cap -> over cap in %d/%d seeds"
              % (sum(g) / len(g), max(g), sum(1 for x in g if x > 62500),
                 len(g)))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        cmd_run(int(sys.argv[2]), int(sys.argv[3]),
                int(sys.argv[4]) if len(sys.argv) > 4 else 1)
    elif sys.argv[1] == "report":
        cmd_report()
    else:
        sys.exit(__doc__)
