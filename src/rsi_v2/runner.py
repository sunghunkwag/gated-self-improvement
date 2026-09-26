"""
rsi_v2.runner -- dev iterations, preregistration freeze, confirmatory
battery, report. Every action is appended to the hash-chained ledger.

  python3 -m rsi_v2 dev --label NAME --seeds 1-8 [--hp JSON] [--workers 4]
  python3 -m rsi_v2 freeze            # writes PREREG_FREEZE (once)
  python3 -m rsi_v2 confirm [--workers 4]
  python3 -m rsi_v2 report [--out results/v2_confirm_report.json]
  python3 -m rsi_v2 verify            # ledger + hashes + completeness
"""
import argparse
import glob
import json
import os
import sys
import time

from . import substrate as S
from . import tasks as T
from . import improver as I
from . import evaluate as E
from . import selfgen as SG
from . import stats as ST
from .ledger import Ledger

PKG = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(PKG))
LEDGER_PATH = os.path.join(ROOT, "results", "ledger", "rsi_v2_ledger.jsonl")
PREREG_PATH = os.path.join(ROOT, "results", "PREREGISTRATION_V2.json")
DEV_LOG = os.path.join(ROOT, "results", "logs", "v2_dev.jsonl")
CONFIRM_LOG = os.path.join(ROOT, "results", "logs", "v2_confirm.jsonl")
DEV_SEEDS = tuple(range(1, 41))
EVALUATOR_FILES = ("substrate.py", "tasks.py", "solver.py", "evaluate.py")


def file_hashes():
    out = {}
    for p in sorted(glob.glob(os.path.join(PKG, "*.py"))):
        with open(p, "rb") as f:
            out[os.path.basename(p)] = S.sha256_text(
                f.read().decode("utf-8"))
    return out


def code_hash():
    return S.sha256_text(S.canon(file_hashes()))


def evaluator_hash():
    h = file_hashes()
    return S.sha256_text(S.canon({k: h[k] for k in EVALUATOR_FILES}))


def parse_seeds(arg):
    out = []
    for part in str(arg).split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def _compact_rounds(rounds):
    keep = ("round", "mode", "scores", "adopted", "train_solved",
            "newly_solved", "n_macros", "order", "tested", "incumbent",
            "theta_after", "attempt_spent", "gate_spent", "batches_done",
            "candidates", "inc_cum", "confirm", "delayed_credit")
    return [{k: r[k] for k in keep if k in r} for r in rounds]


def run_unit(job):
    """One (arm, seed): improvement phase (holdout SEALED) + evaluation."""
    arm, seed, hp = job
    t0 = time.time()
    T.improver_view(seed)            # build manifest before sealing
    with E.sealed():
        cfg, rec = I.run_arm(arm, seed, hp)
    ev = E.evaluate(cfg, seed, stream=0)
    out = {"arm": arm, "seed": seed,
           "ext": ev["by_split"]["holdout_ext"],
           "in": ev["by_split"]["holdout_in"],
           "per_task": ev["per_task"],
           "eval_search_execs": ev["search_execs"],
           "cap": rec["cap"], "spent": rec["spent"],
           "global_delta": rec["global_delta"],
           "train_solved": rec["train_solved"],
           "n_macros": rec["n_macros"], "macros": rec["macros"],
           "final_cfg_sha": rec["final_cfg_sha"],
           "final_cfg": cfg.to_json(),
           "theta_final": rec["theta_final"],
           "rounds": _compact_rounds(rec["rounds"])}
    out.update(SG.self_score(cfg, seed))
    if arm == "COLD":   # deliberate noise-floor replicate (eval stream 1)
        rep = E.evaluate(cfg, seed, stream=1)
        out["replicate_ext"] = rep["by_split"]["holdout_ext"]
        out["replicate_in"] = rep["by_split"]["holdout_in"]
    out["elapsed_s"] = round(time.time() - t0, 2)
    return out


def _pool_map(jobs, workers, on_result):
    if workers <= 1:
        for j in jobs:
            on_result(run_unit(j))
        return
    import multiprocessing as mp   # runner-level only; units independent
    with mp.Pool(workers) as pool:
        for r in pool.imap_unordered(run_unit, jobs):
            on_result(r)


def _append_jsonl(path, rec):
    d = os.path.dirname(path)
    if not os.path.isdir(d):
        os.makedirs(d)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def _summ(results, arms):
    by = {}
    for r in results:
        by.setdefault(r["arm"], {})[r["seed"]] = r
    out = {}
    for a in arms:
        rs = list(by.get(a, {}).values())
        if rs:
            n = float(len(rs))
            out[a] = {"n": len(rs),
                      "ext": sum(r["ext"] for r in rs) / n,
                      "in": sum(r["in"] for r in rs) / n,
                      "train_solved": sum(r["train_solved"] for r in rs) / n,
                      "n_macros": sum(r["n_macros"] for r in rs) / n,
                      "spent": sum(r["spent"] for r in rs) / n,
                      "self": sum(r["self_solved"] for r in rs) / n}
    return by, out


def cmd_dev(label, seeds, hp_over, workers, arms):
    for s in seeds:
        if s not in DEV_SEEDS:
            sys.exit("dev runs are restricted to dev seeds %d-%d"
                     % (DEV_SEEDS[0], DEV_SEEDS[-1]))
    led = Ledger(LEDGER_PATH)
    if led.find("PREREG_FREEZE"):
        print("NOTE: preregistration already frozen; this dev run is "
              "recorded but cannot change the frozen protocol.")
    hp = json.loads(json.dumps(I.HP))
    for k, v in hp_over.items():
        hp[k] = v
    led.append("DEV_START", {"label": label, "seeds": seeds,
                             "hp_overrides": hp_over, "arms": arms,
                             "code_hash": code_hash()})
    results = []

    def on_result(r):
        r["phase"] = "dev"
        r["label"] = label
        _append_jsonl(DEV_LOG, r)
        results.append(r)
        print("  %-17s seed %3d ext %2d in %2d train %2d macros %2d "
              "(%.1fs)" % (r["arm"], r["seed"], r["ext"], r["in"],
                           r["train_solved"], r["n_macros"],
                           r["elapsed_s"]), flush=True)
    jobs = [(a, s, hp) for s in seeds for a in arms]
    _pool_map(jobs, workers, on_result)
    by, summ = _summ(results, arms)
    contrasts = {}
    for a, b in (("RECURSIVE_FULL", "SINGLE_5X"),
                 ("RECURSIVE_FULL", "RECURSIVE_FROZEN"),
                 ("RECURSIVE_FROZEN", "SINGLE_5X"),
                 ("RECURSIVE_NODIAG", "RECURSIVE_FULL"),
                 ("SINGLE_5X", "COLD"), ("RECURSIVE_FULL", "COLD")):
        if a in by and b in by:
            d = [by[a][s]["ext"] - by[b][s]["ext"] for s in seeds]
            contrasts["%s-%s" % (a, b)] = {
                "mean": sum(d) / float(len(d)),
                "wins": sum(1 for x in d if x > 0),
                "losses": sum(1 for x in d if x < 0)}
    led.append("DEV_END", {"label": label, "seeds": seeds,
                           "code_hash": code_hash(), "summary": summ,
                           "contrasts_ext": contrasts,
                           "per_unit": [[r["arm"], r["seed"], r["ext"],
                                         r["in"], r["train_solved"],
                                         r["spent"]] for r in results]})
    for a in arms:
        if a in summ:
            m = summ[a]
            print("%-17s ext %.2f  in %.2f  train %.1f  macros %.1f  "
                  "self %.1f  spent %.0f" % (a, m["ext"], m["in"],
                                             m["train_solved"],
                                             m["n_macros"], m["self"],
                                             m["spent"]))
    for k, v in contrasts.items():
        print("  %-36s %+0.3f  (w %d / l %d)" % (k, v["mean"], v["wins"],
                                                 v["losses"]))


# --------------------------------------------------------------------------
# preregistration / confirmatory

def load_prereg():
    with open(PREREG_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    return json.loads(text), S.sha256_text(text)


def current_freeze_body(prereg, prereg_sha):
    seeds = parse_seeds(prereg["seeds"]["confirm"])
    return {"prereg_sha": prereg_sha, "code_hash": code_hash(),
            "file_hashes": file_hashes(),
            "evaluator_hash": evaluator_hash(),
            "hp_sha": S.sha256_text(S.canon(I.HP)),
            "library_digest": T.library_digest(),
            "confirm_manifest_digest": T.manifest_digest(seeds),
            "confirm_seeds": [seeds[0], seeds[-1], len(seeds)]}


def cmd_freeze():
    led = Ledger(LEDGER_PATH)
    if led.find("PREREG_FREEZE"):
        sys.exit("already frozen (freeze is one-shot)")
    prereg, sha = load_prereg()
    seeds = parse_seeds(prereg["seeds"]["confirm"])
    if set(seeds) & set(DEV_SEEDS):
        sys.exit("confirm seeds overlap dev seeds")
    if led.find("UNIT_START", phase="confirm"):
        sys.exit("confirmatory units exist before freeze")
    if S.sha256_text(S.canon(I.HP)) != prereg["hp_sha"]:
        sys.exit("improver.HP differs from the preregistered hp_sha")
    body = current_freeze_body(prereg, sha)
    led.append("PREREG_FREEZE", body)
    print(json.dumps(body, indent=1, sort_keys=True))


def check_frozen(led):
    fz = led.find("PREREG_FREEZE")
    if len(fz) != 1:
        raise AssertionError("protocol not frozen exactly once")
    prereg, sha = load_prereg()
    cur = current_freeze_body(prereg, sha)
    frozen = fz[0]["body"]
    bad = [k for k in cur if k != "file_hashes" and cur[k] != frozen.get(k)]
    if bad:
        raise AssertionError("frozen protocol violated: %s" % bad)
    return prereg, fz[0]


def cmd_confirm(workers):
    led = Ledger(LEDGER_PATH)
    prereg, _fz = check_frozen(led)
    seeds = parse_seeds(prereg["seeds"]["confirm"])
    arms = prereg["arms"]
    done = {(r["body"]["arm"], r["body"]["seed"])
            for r in led.find("UNIT_END", phase="confirm")}
    started = {(r["body"]["arm"], r["body"]["seed"])
               for r in led.find("UNIT_START", phase="confirm")}
    jobs = [(a, s, I.HP) for s in seeds for a in arms if (a, s) not in done]
    for a, s, _hp in jobs:
        if (a, s) in started:
            led.append("UNIT_RESTART", {"phase": "confirm", "arm": a,
                                        "seed": s,
                                        "note": "START without END (crash);"
                                        " deterministic re-run"})
        led.append("UNIT_START", {"phase": "confirm", "arm": a, "seed": s,
                                  "code_hash": code_hash()})
    t0 = time.time()

    def on_result(r):
        r["phase"] = "confirm"
        _append_jsonl(CONFIRM_LOG, r)
        led.append("UNIT_END", r)
        print("  %-17s seed %4d ext %2d in %2d  [%.0fs]"
              % (r["arm"], r["seed"], r["ext"], r["in"], time.time() - t0),
              flush=True)
    _pool_map(jobs, workers, on_result)


def collect(led, phase, arms, seeds):
    tab = {}
    for r in led.find("UNIT_END", phase=phase):
        b = r["body"]
        tab[(b["arm"], b["seed"])] = b
    missing = [(a, s) for s in seeds for a in arms if (a, s) not in tab]
    return tab, missing


def build_report(led):
    prereg, fz = check_frozen(led)
    seeds = parse_seeds(prereg["seeds"]["confirm"])
    arms = prereg["arms"]
    tab, missing = collect(led, "confirm", arms, seeds)
    rep = {"prereg_sha": fz["body"]["prereg_sha"],
           "ledger_head": led.head, "n_seeds_planned": len(seeds),
           "missing_units": missing,
           "abandoned_units": [[r["body"]["arm"], r["body"]["seed"]]
                               for r in led.abandoned("confirm")]}
    if missing:
        rep["verdict"] = "INCOMPLETE: %d units missing" % len(missing)
        return rep
    metric = prereg["primary_metric"]["field"]
    assert metric == "ext" and prereg["primary_metric"]["split"] == \
        E.PRIMARY_SPLIT, "primary metric must be the external holdout"
    alpha = prereg["statistics"]["alpha"]
    comps = {}
    pvals = {}
    for c in prereg["confirmatory_contrasts"]:
        a, b = c["a"], c["b"]
        d = [tab[(a, s)][metric] - tab[(b, s)][metric] for s in seeds]
        sm = ST.summary(d, "confirm|%s-%s" % (a, b), one_sided=True)
        comps[c["name"]] = dict(sm, a=a, b=b, role=c["role"])
        pvals[c["name"]] = sm["p"]
    adj = ST.holm(pvals)
    for k in comps:
        comps[k]["p_holm"] = adj[k]
        comps[k]["pass"] = bool(comps[k]["mean"] > 0 and adj[k] < alpha)
    rep["confirmatory"] = comps
    expl = {}
    for a, b, field in prereg["exploratory_contrasts"]:
        if field == "replicate_ext":
            d = [tab[(a, s)]["replicate_ext"] - tab[(a, s)]["ext"]
                 for s in seeds]
        else:
            d = [tab[(a, s)][field] - tab[(b, s)][field] for s in seeds]
        expl["%s-%s:%s" % (a, b, field)] = ST.summary(
            d, "expl|%s-%s|%s" % (a, b, field))
    rep["exploratory"] = expl
    rep["means"] = {a: {f: sum(tab[(a, s)][f] for s in seeds) / float(
        len(seeds)) for f in ("ext", "in", "train_solved", "n_macros",
                              "spent", "self_solved")} for a in arms}
    rep["caps"] = {a: sorted({tab[(a, s)]["cap"] for s in seeds})
                   for a in arms}
    prim = [c for c in comps.values() if c["role"] == "primary"][0]
    rep["verdict"] = ("PRIMARY PASSED" if prim["pass"] else
                      "PRIMARY NOT SUPPORTED (null)")
    return rep


def cmd_report(out):
    led = Ledger(LEDGER_PATH)
    rep = build_report(led)
    txt = json.dumps(rep, indent=1, sort_keys=True)
    if out:
        with open(out, "w", encoding="utf-8") as f:
            f.write(txt + "\n")
    led.append("REPORT", {"verdict": rep["verdict"],
                          "report_sha": S.sha256_text(txt)})
    print(txt)


def cmd_verify():
    led = Ledger(LEDGER_PATH)
    print("ledger OK: %d records, head %s" % (len(led.records), led.head))
    try:
        prereg, _ = check_frozen(led)
        print("protocol frozen and code/manifest hashes match")
        seeds = parse_seeds(prereg["seeds"]["confirm"])
        _tab, missing = collect(led, "confirm", prereg["arms"], seeds)
        print("confirm units missing: %d; abandoned: %d"
              % (len(missing), len(led.abandoned("confirm"))))
    except (AssertionError, IOError) as e:
        print("not frozen / mismatch: %s" % e)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rsi_v2")
    sub = ap.add_subparsers(dest="cmd")
    d = sub.add_parser("dev")
    d.add_argument("--label", required=True)
    d.add_argument("--seeds", required=True)
    d.add_argument("--hp", default="{}")
    d.add_argument("--workers", type=int, default=4)
    d.add_argument("--arms", default=",".join(I.ARMS))
    sub.add_parser("freeze")
    c = sub.add_parser("confirm")
    c.add_argument("--workers", type=int, default=4)
    r = sub.add_parser("report")
    r.add_argument("--out", default=os.path.join(
        ROOT, "results", "v2_confirm_report.json"))
    sub.add_parser("verify")
    a = ap.parse_args(argv)
    if a.cmd == "dev":
        cmd_dev(a.label, parse_seeds(a.seeds), json.loads(a.hp), a.workers,
                a.arms.split(","))
    elif a.cmd == "freeze":
        cmd_freeze()
    elif a.cmd == "confirm":
        cmd_confirm(a.workers)
    elif a.cmd == "report":
        cmd_report(a.out)
    elif a.cmd == "verify":
        cmd_verify()
    else:
        ap.print_help()
