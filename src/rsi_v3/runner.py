"""
rsi_v3.runner -- dev iterations, dev-check looks, preregistration freeze,
confirmatory battery, report. Everything is appended to the v3 ledger.

  python3 -m rsi_v3 dev --label NAME --seeds 3001-3012 [--hp JSON]
  python3 -m rsi_v3 devcheck --label NAME        # seeds 3201-3300, logged;
                                                 # needs a fresh criterion
  python3 -m rsi_v3 freeze
  python3 -m rsi_v3 confirm [--workers 4]
  python3 -m rsi_v3 report
  python3 -m rsi_v3 verify
"""
import argparse
import glob
import json
import os
import sys
import time

from rsi_v2 import substrate as S
from rsi_v2 import selfgen as SG
from rsi_v2 import stats as ST
from .ledger import SafeLedger as Ledger
from . import tasks as T
from . import improver as I
from . import evaluate as E

PKG = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.dirname(PKG)
ROOT = os.path.dirname(SRC)
LEDGER_PATH = os.path.join(ROOT, "results", "ledger", "rsi_v3_ledger.jsonl")
PREREG_PATH = os.path.join(ROOT, "results", "PREREGISTRATION_V3.json")
# compact per-unit summaries (plain JSONL, mirrored in the ledger) and the
# FULL per-unit detail (gzip JSONL; its sha256 is in the summary/ledger)
DEV_LOG = os.path.join(ROOT, "results", "logs", "v3_dev_units.jsonl")
DEV_DETAIL = os.path.join(ROOT, "results", "logs", "v3_dev_detail.jsonl.gz")
CONFIRM_LOG = os.path.join(ROOT, "results", "logs", "v3_confirm_units.jsonl")
CONFIRM_DETAIL = os.path.join(ROOT, "results", "logs",
                              "v3_confirm_detail.jsonl.gz")
DEV_ITERATE = tuple(range(3001, 3101))    # 3041-3100 added for the
                                          # process-controller phase
DEV_CHECK_SPENT = tuple(range(3101, 3201))   # v3devcheck-01: NO_GO
DEV_CHECK = tuple(range(3201, 3301))         # reserved for the next go/no-go
FORBIDDEN = tuple(range(1, 41)) + tuple(range(101, 201)) + tuple(
    range(1001, 1301))
EVALUATOR_FILES = ("rsi_v3/tasks.py", "rsi_v3/evaluate.py",
                   "rsi_v2/substrate.py", "rsi_v2/solver.py",
                   "rsi_v2/tasks.py")


def file_hashes():
    out = {}
    for pat in ("rsi_v3/*.py", "rsi_v2/*.py"):
        for p in sorted(glob.glob(os.path.join(SRC, pat))):
            with open(p, "rb") as f:
                out[os.path.relpath(p, SRC)] = S.sha256_text(
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


def run_unit(job):
    """One (arm, seed): n_worlds improvement problems (final holdouts
    SEALED throughout), then each world's final solver is scored on that
    world's FINAL HOLDOUT. ext = total solved over all worlds."""
    arm, seed, hp = job
    t0 = time.time()
    for w in range(1, hp["n_worlds"] + 1):     # build manifests pre-seal
        T.improver_view(I.world_key(seed, w), hp["n_rounds"])
    with E.sealed():
        cfgs, rec = I.run_arm(arm, seed, hp)
    ext_w, in_w, per_task, rep = [], [], [], []
    for w, cfg in enumerate(cfgs, start=1):
        ev = E.evaluate(cfg, I.world_key(seed, w), stream=0)
        ext_w.append(ev["by_split"]["holdout_ext"])
        in_w.append(ev["by_split"]["holdout_in"])
        per_task.append(ev["per_task"])
        if arm == "COLD":
            rep.append(E.evaluate(cfg, I.world_key(seed, w), stream=1)
                       ["by_split"]["holdout_ext"])
    ws = rec["worlds"]
    out = {"arm": arm, "seed": seed, "ext": sum(ext_w), "ext_w": ext_w,
           "in": sum(in_w), "in_w": in_w, "per_task": per_task,
           "cap": rec["cap"], "world_cap": rec["world_cap"],
           "spent": rec["spent"], "global_delta": rec["global_delta"],
           "world_spent": [x["spent"] for x in ws],
           "train_solved": sum(x["train_solved"] for x in ws) / float(len(ws)),
           "n_macros": sum(x["n_macros"] for x in ws) / float(len(ws)),
           "worlds": ws, "model_w": rec["model_w"],
           "model_n": rec["model_n"], "meta_ops": rec["meta_ops"],
           "meta_program_executions": rec["meta_program_executions"],
           "meta_seconds": round(sum(rd.get("meta_seconds", 0.0)
                                     for x in ws for rd in x["rounds"]), 3),
           "rank_validity": rank_validity_unit(ws)}
    if "memory_resets" in rec:
        out["memory_resets"] = rec["memory_resets"]
    out.update(SG.self_score(cfgs[-1], seed))
    if arm == "COLD":
        out["replicate_ext"] = sum(rep)
    out["elapsed_s"] = round(time.time() - t0, 2)
    return out


def _pool_map(jobs, workers, on_result):
    if workers <= 1:
        for j in jobs:
            on_result(run_unit(j))
        return
    import multiprocessing as mp
    with mp.Pool(workers) as pool:
        for r in pool.imap_unordered(run_unit, jobs):
            on_result(r)


def _append(path, rec):
    d = os.path.dirname(path)
    if not os.path.isdir(d):
        os.makedirs(d)
    line = json.dumps(rec, sort_keys=True) + "\n"
    if path.endswith(".gz"):
        import gzip
        with gzip.open(path, "at", encoding="utf-8") as f:   # new member
            f.write(line)
    else:
        with open(path, "a", encoding="utf-8") as f:
            f.write(line)


def compact(full):
    """Summary kept in the ledger / plain log: every field the report and
    the verification checks need, plus the sha256 binding the full detail
    record (per-round pools, screened candidates, features) to the chain."""
    out = {k: v for k, v in full.items() if k != "worlds"}
    out["detail_sha"] = S.sha256_text(S.canon(full))
    out["mq_w"] = meta_quality_by_world(full)
    out["adopt_w"] = [sum(1 for rd in wd["rounds"] if rd.get("adopted"))
                      for wd in full["worlds"]]
    acts = {}
    for wd in full["worlds"]:
        for rd in wd["rounds"]:
            a = rd.get("adopted_action")
            if a:
                acts[a] = acts.get(a, 0) + 1
    out["adopted_actions"] = acts
    out["final_cfg_shas"] = [wd["final_cfg_sha"] for wd in full["worlds"]]
    out["macros_w"] = [wd["n_macros"] for wd in full["worlds"]]
    if "memory_resets" in full:
        out.update(process_summary(full["worlds"]))
    return out


def process_summary(worlds):
    """Process-control arms: per world, how far the controller's actual
    decisions moved away from the default (v3) process, plus the tracking
    horizon (T_0 .. T_R, cross-family transfer) and the retrieval weight."""
    div, knob_nd, alloc_l1, jac, learned = [], [], [], [], []
    for wd in worlds:
        rs = wd["rounds"]
        nd = [sum(1 for k, dflt in rd["plan_default"].items() if not dflt)
              / float(len(rd["plan_default"])) for rd in rs]
        l1 = []
        for rd in rs:
            a, b = rd.get("alloc") or {}, rd.get("alloc_default") or {}
            keys = set(a) | set(b)
            tot = float(max(1, sum(b.values())))
            l1.append(sum(abs(a.get(x, 0) - b.get(x, 0)) for x in keys)
                      / tot)
        knob_nd.append(sum(nd) / len(nd))
        alloc_l1.append(sum(l1) / len(l1))
        jac.append(sum(rd["pool_jaccard"] for rd in rs) / len(rs))
        learned.append(sum(1 for rd in rs if rd["pool_learned"]) / len(rs))
    return {"knob_nondefault_w": [round(x, 4) for x in knob_nd],
            "alloc_l1_w": [round(x, 4) for x in alloc_l1],
            "pool_jaccard_w": [round(x, 4) for x in jac],
            "pool_learned_w": [round(x, 4) for x in learned],
            "transfer_w": [wd["transfer"] for wd in worlds],
            "track_w": [wd["track"] for wd in worlds],
            "omega_w": [wd["memory"]["omega"] for wd in worlds],
            "plans_w": [[[rd["plan"][k] for k in ("attempt", "shape",
                                                  "explore", "gate")]
                         for rd in wd["rounds"]] for wd in worlds],
            "adopted_w": [[rd.get("adopted") for rd in wd["rounds"]]
                          for wd in worlds]}


def rank_validity_unit(worlds):
    """Does the learned ranking predict REAL FUTURE cross-family gain? For
    every round, the predictor's scores mu were computed BEFORE the screened
    candidates were evaluated; y is their realized paired gain on fresh
    META-VAL probes (families disjoint from the train families). Returns
    per-world mean within-round rank correlation (Spearman) and the pooled
    (mu, y) pairs' Pearson correlation."""
    out = []
    for wd in worlds:
        rhos, pairs = [], []
        for rd in wd["rounds"]:
            sc = [(s["mu"], s["y_mean"]) for s in rd.get("screened", [])
                  if "y_mean" in s]
            pairs += sc
            if len(sc) >= 3 and len({m for m, _y in sc}) > 1:
                rho = _spearman([m for m, _y in sc], [y for _m, y in sc])
                if rho is not None:
                    rhos.append(rho)
        out.append({"spearman_within_round": (sum(rhos) / len(rhos))
                    if rhos else None,
                    "pearson": _pearson([m for m, _y in pairs],
                                        [y for _m, y in pairs])})
    return out


def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0
        i = j + 1
    return r


def _pearson(a, b):
    n = len(a)
    if n < 3:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (va * vb) ** 0.5


def _spearman(a, b):
    return _pearson(_rank(a), _rank(b))


def meta_quality_by_world(r):
    out = []
    for wd in r["worlds"]:
        ys = [s["y_mean"] for rd in wd["rounds"]
              for s in rd.get("screened", []) if "y_mean" in s]
        out.append(sum(ys) / len(ys) if ys else 0.0)
    return out


def meta_quality(r, first_world=2):
    """Mean realized paired gain (candidate - incumbent, cross-family
    META-VAL probes) of the candidates this arm chose to test, over worlds
    >= first_world: the direct measure of how well the improver picks
    improvements once cross-world learning could have happened. (Each
    world screens the same number of candidates, so the mean of per-world
    means equals the pooled mean.)"""
    mq = r["mq_w"] if "mq_w" in r else meta_quality_by_world(r)
    mq = mq[first_world - 1:]
    return sum(mq) / len(mq) if mq else None


def summarize(results, arms, seeds):
    by = {}
    for r in results:
        by.setdefault(r["arm"], {})[r["seed"]] = r
    summ = {}
    for a in arms:
        rs = [by[a][s] for s in seeds if s in by.get(a, {})]
        if not rs:
            continue
        n = float(len(rs))
        mq = [meta_quality(r) for r in rs]
        mq = [x for x in mq if x is not None]
        mqw = [r["mq_w"] for r in rs]
        summ[a] = {"n": len(rs), "ext": sum(r["ext"] for r in rs) / n,
                   "in": sum(r["in"] for r in rs) / n,
                   "train": sum(r["train_solved"] for r in rs) / n,
                   "macros": sum(r["n_macros"] for r in rs) / n,
                   "ext_w": [sum(r["ext_w"][w] for r in rs) / n
                             for w in range(len(rs[0]["ext_w"]))],
                   "adopt": sum(sum(r["adopt_w"]) for r in rs) / n,
                   "meta_quality": (sum(mq) / len(mq)) if mq else None,
                   "mq_w": [sum(x[w] for x in mqw) / n
                            for w in range(len(mqw[0]))],
                   "spent": sum(r["spent"] for r in rs) / n}
        if "pool_jaccard_w" in rs[0]:
            W = len(rs[0]["pool_jaccard_w"])
            for f in ("pool_jaccard_w", "knob_nondefault_w", "alloc_l1_w"):
                summ[a][f] = [round(sum(r[f][w] for r in rs) / n, 4)
                              for w in range(W)]
            tr = [[r["transfer_w"][w] for r in rs
                   if r["transfer_w"][w] is not None] for w in range(W)]
            summ[a]["transfer_w"] = [round(sum(x) / len(x), 5) if x else None
                                     for x in tr]
    con = {}
    for a, b in (("MEMORY_CARRY", "NO_CARRY"), ("MEMORY_CARRY", "FROZEN"),
                 ("NO_CARRY", "FROZEN"), ("MEMORY_CARRY", "MEMORY_RANKONLY"),
                 ("MEMORY_RANKONLY", "FROZEN"),
                 ("ADAPTIVE_META", "FROZEN_META"),
                 ("ADAPTIVE_META", "SINGLE_COMPUTE_MATCHED"),
                 ("ADAPTIVE_META", "NODIAG_META"),
                 ("ADAPTIVE_META", "ADAPTIVE_NOCARRY"),
                 ("ADAPTIVE_META", "HEURISTIC_META"),
                 ("ORACLE_RANK", "FROZEN_META"),
                 ("FROZEN_META", "SINGLE_COMPUTE_MATCHED")):
        if a in by and b in by:
            ss = [s for s in seeds if s in by[a] and s in by[b]]
            for field in ("ext", "in"):
                d = [by[a][s][field] - by[b][s][field] for s in ss]
                sm = ST.summary(d, "dev|%s-%s|%s" % (a, b, field))
                con["%s-%s:%s" % (a, b, field)] = {
                    "mean": sm["mean"], "ci95": sm["ci95"], "p": sm["p"],
                    "w/t/l": [sm["wins"], sm["ties"], sm["losses"]]}
            qa = [meta_quality(by[a][s]) for s in ss]
            qb = [meta_quality(by[b][s]) for s in ss]
            d = [x - y for x, y in zip(qa, qb)
                 if x is not None and y is not None]
            if d:
                sm = ST.summary(d, "dev|%s-%s|mq" % (a, b))
                con["%s-%s:meta_quality" % (a, b)] = {
                    "mean": sm["mean"], "ci95": sm["ci95"], "p": sm["p"],
                    "w/t/l": [sm["wins"], sm["ties"], sm["losses"]]}
    return summ, con


def _print(summ, con):
    for a, m in summ.items():
        print("%-24s metaQ by world %s" % (a, [round(x, 4) for x in
                                               m["mq_w"]]))
        print("%-24s ext %.2f %s in %.2f train %.1f macros %.1f adopt "
              "%.2f metaQ %s spent %.0f" % (
                  a, m["ext"], [round(x, 2) for x in m["ext_w"]], m["in"],
                  m["train"], m["macros"], m["adopt"],
                  "%.4f" % m["meta_quality"] if m["meta_quality"] is not None
                  else "-", m["spent"]))
    for a, m in summ.items():
        if "pool_jaccard_w" in m:
            print("%-24s pool~default %s knobs!=default %s transfer %s" % (
                a, m["pool_jaccard_w"], m["knob_nondefault_w"],
                m["transfer_w"]))
    for k, v in con.items():
        print("  %-52s %+.4f [%+.3f, %+.3f] p=%.4f w/t/l=%s" % (
            k, v["mean"], v["ci95"][0], v["ci95"][1], v["p"], v["w/t/l"]))


def _dev_like(kind, label, seeds, hp_over, workers, arms, allowed):
    for s in seeds:
        if s not in allowed or s in FORBIDDEN or s in DEV_CHECK_SPENT:
            sys.exit("seed %d not allowed for %s" % (s, kind))
    led = Ledger(LEDGER_PATH)
    if kind == "DEVCHECK":
        # a go/no-go look is only valid against a criterion written after
        # the previous look (no re-using an old criterion after peeking)
        crit = [r["seq"] for r in led.find("GO_NO_GO_CRITERION")]
        last = [r["seq"] for r in led.find("DEVCHECK_END")]
        if not crit or (last and max(crit) < max(last)):
            sys.exit("record a new GO_NO_GO_CRITERION before this dev-check")
    if led.find("PREREG_FREEZE"):
        print("NOTE: protocol already frozen; this run cannot change it")
    hp = json.loads(json.dumps(I.HP))
    hp.update(hp_over)
    led.append(kind + "_START", {"label": label, "seeds": seeds,
                                 "hp_overrides": hp_over, "arms": arms,
                                 "code_hash": code_hash()})
    results = []

    def on_result(r):
        r["phase"], r["label"] = kind, label
        _append(DEV_DETAIL, r)
        c = compact(r)
        _append(DEV_LOG, c)
        results.append(c)
    _pool_map([(a, s, hp) for s in seeds for a in arms], workers, on_result)
    summ, con = summarize(results, arms, seeds)
    led.append(kind + "_END", {
        "label": label, "seeds": seeds, "code_hash": code_hash(),
        "summary": summ, "contrasts": con,
        "per_unit": [[r["arm"], r["seed"], r["ext"], r["in"],
                      r["train_solved"], r["spent"]] for r in results]})
    _print(summ, con)


# ------------------------------------------------------------------ protocol

def load_prereg():
    with open(PREREG_PATH, "r", encoding="utf-8") as f:
        text = f.read()
    return json.loads(text), S.sha256_text(text)


def freeze_body(prereg, sha):
    seeds = parse_seeds(prereg["seeds"]["confirm"])
    return {"prereg_sha": sha, "code_hash": code_hash(),
            "file_hashes": file_hashes(), "evaluator_hash": evaluator_hash(),
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
    bad = set(seeds) & (set(FORBIDDEN) | set(DEV_ITERATE) | set(DEV_CHECK)
                        | set(DEV_CHECK_SPENT))
    if bad:
        sys.exit("confirm seeds overlap forbidden/dev seeds: %s"
                 % sorted(bad)[:5])
    if led.find("UNIT_START", phase="confirm"):
        sys.exit("confirmatory units exist before freeze")
    if S.sha256_text(S.canon(I.HP)) != prereg["hp_sha"]:
        sys.exit("improver.HP differs from the preregistered hp_sha")
    body = freeze_body(prereg, sha)
    led.append("PREREG_FREEZE", body)
    print(json.dumps(body, indent=1, sort_keys=True))


def check_frozen(led):
    fz = led.find("PREREG_FREEZE")
    if len(fz) != 1:
        raise AssertionError("protocol not frozen exactly once")
    prereg, sha = load_prereg()
    cur = freeze_body(prereg, sha)
    bad = [k for k in cur if k != "file_hashes"
           and cur[k] != fz[0]["body"].get(k)]
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
    for a, s, _h in jobs:
        if (a, s) in started:
            led.append("UNIT_RESTART", {"phase": "confirm", "arm": a,
                                        "seed": s, "note": "deterministic "
                                        "re-run of a unit with no END"})
        led.append("UNIT_START", {"phase": "confirm", "arm": a, "seed": s,
                                  "code_hash": code_hash()})
    t0 = time.time()

    def on_result(r):
        r["phase"] = "confirm"
        _append(CONFIRM_DETAIL, r)
        c = compact(r)
        _append(CONFIRM_LOG, c)
        led.append("UNIT_END", c)
        print("  %-24s seed %d ext %2d [%.0fs]" % (r["arm"], r["seed"],
                                                   r["ext"],
                                                   time.time() - t0),
              flush=True)
    _pool_map(jobs, workers, on_result)


def collect(led, phase, arms, seeds):
    tab = {}
    for r in led.find("UNIT_END", phase=phase):
        tab[(r["body"]["arm"], r["body"]["seed"])] = r["body"]
    missing = [(a, s) for s in seeds for a in arms if (a, s) not in tab]
    return tab, missing


def gatekeeping(comps, alpha):
    """Preregistered fixed-sequence gatekeeping, in place. The primary
    contrast is tested alone at alpha; only if it is supported are the
    secondary contrasts tested, Holm-adjusted among themselves at alpha.
    Family-wise error <= alpha over all confirmatory contrasts."""
    prim = [k for k in comps if comps[k]["role"] == "primary"]
    assert len(prim) == 1
    prim = prim[0]
    comps[prim]["p_adj"] = comps[prim]["p"]
    comps[prim]["pass"] = bool(comps[prim]["mean"] > 0
                               and comps[prim]["p"] < alpha)
    sec = {k: comps[k]["p"] for k in comps if k != prim}
    adj = ST.holm(sec) if sec else {}
    for k in sec:
        comps[k]["p_adj"] = adj[k]
        comps[k]["tested"] = comps[prim]["pass"]
        comps[k]["pass"] = bool(comps[prim]["pass"] and comps[k]["mean"] > 0
                                and adj[k] < alpha)
    return comps


def build_report(led):
    prereg, fz = check_frozen(led)
    seeds = parse_seeds(prereg["seeds"]["confirm"])
    arms = prereg["arms"]
    tab, missing = collect(led, "confirm", arms, seeds)
    rep = {"prereg_sha": fz["body"]["prereg_sha"], "ledger_head": led.head,
           "n_seeds_planned": len(seeds), "missing_units": missing,
           "abandoned_units": [[r["body"]["arm"], r["body"]["seed"]]
                               for r in led.abandoned("confirm")]}
    if missing:
        rep["verdict"] = "INCOMPLETE: %d units missing" % len(missing)
        return rep
    metric = prereg["primary_metric"]["field"]
    assert metric == "ext" and prereg["primary_metric"]["split"] == \
        E.PRIMARY_SPLIT, "primary metric must be the FINAL external holdout"
    alpha = prereg["statistics"]["alpha"]
    comps = {}
    for c in prereg["confirmatory_contrasts"]:
        d = [tab[(c["a"], s)][metric] - tab[(c["b"], s)][metric]
             for s in seeds]
        sm = ST.summary(d, "v3confirm|%s-%s" % (c["a"], c["b"]),
                        one_sided=True)
        comps[c["name"]] = dict(sm, a=c["a"], b=c["b"], role=c["role"])
    gatekeeping(comps, alpha)
    rep["confirmatory"] = comps
    expl = {}
    for a, b, field in prereg["exploratory_contrasts"]:
        if field == "replicate_ext":
            d = [tab[(a, s)]["replicate_ext"] - tab[(a, s)]["ext"]
                 for s in seeds]
        elif field == "meta_quality":
            d = [meta_quality(tab[(a, s)]) - meta_quality(tab[(b, s)])
                 for s in seeds]
        else:
            d = [tab[(a, s)][field] - tab[(b, s)][field] for s in seeds]
        expl["%s-%s:%s" % (a, b, field)] = ST.summary(
            d, "v3expl|%s-%s|%s" % (a, b, field))
    rep["exploratory"] = expl
    rep["means"] = {a: {f: sum(tab[(a, s)][f] for s in seeds) / float(
        len(seeds)) for f in ("ext", "in", "train_solved", "n_macros",
                              "spent", "self_solved")} for a in arms}
    rep["caps"] = {a: sorted({tab[(a, s)]["cap"] for s in seeds})
                   for a in arms}
    rep["verification"] = verification(tab, seeds, arms)
    prim = [c for c in comps.values() if c["role"] == "primary"][0]
    rep["verdict"] = ("PRIMARY PASSED" if prim["pass"]
                      else "PRIMARY NOT SUPPORTED (null)")
    return rep


def verification(tab, seeds, arms):
    """Pre-registered verification checks reported with the verdict."""
    v = {}
    W = len(tab[(arms[0], seeds[0])]["ext_w"])
    # 1. the advantage persists / grows across later worlds
    for b in ("FROZEN_META", "ADAPTIVE_NOCARRY"):
        if b not in arms:
            continue
        per_w = [sum(tab[("ADAPTIVE_META", s)]["ext_w"][w]
                     - tab[(b, s)]["ext_w"][w] for s in seeds)
                 / float(len(seeds)) for w in range(W)]
        trend = [sum(tab[("ADAPTIVE_META", s)]["ext_w"][w]
                     - tab[(b, s)]["ext_w"][w] for w in range(W // 2, W))
                 - sum(tab[("ADAPTIVE_META", s)]["ext_w"][w]
                       - tab[(b, s)]["ext_w"][w] for w in range(W // 2))
                 for s in seeds]
        sm = ST.summary(trend, "v3verif|trend|%s" % b)
        v["per_world_ADAPTIVE_minus_%s" % b] = {
            "per_world": [round(x, 4) for x in per_w],
            "first_half": sum(per_w[:W // 2]) / (W // 2),
            "second_half": sum(per_w[W // 2:]) / (W - W // 2),
            "second_minus_first_per_seed": sm}
    # 2. the learned ranking predicts REAL FUTURE cross-family gain
    rho = []
    for s in seeds:
        xs = [w["spearman_within_round"] for w in
              tab[("ADAPTIVE_META", s)]["rank_validity"][1:]
              if w["spearman_within_round"] is not None]
        if xs:
            rho.append(sum(xs) / len(xs))
    v["rank_validity_ADAPTIVE"] = dict(
        ST.summary(rho, "v3verif|rho", one_sided=True),
        note="per-seed mean within-round Spearman between the predictor's "
             "score (computed BEFORE evaluation) and the realized paired "
             "gain on fresh cross-family META-VAL probes, worlds >= 2")
    v["rank_validity_by_world"] = [
        (lambda xs: sum(xs) / len(xs) if xs else None)(
            [tab[("ADAPTIVE_META", s)]["rank_validity"][w]
             ["spearman_within_round"] for s in seeds
             if tab[("ADAPTIVE_META", s)]["rank_validity"][w]
             ["spearman_within_round"] is not None]) for w in range(W)]
    # 3. meta-controller compute is logged and uses no program executions
    mc = {}
    for a in arms:
        ops = [tab[(a, s)].get("meta_ops") for s in seeds]
        if ops and ops[0]:
            mc[a] = {k: sum(o[k] for o in ops) / float(len(ops))
                     for k in ops[0]}
            mc[a]["seconds"] = sum(tab[(a, s)].get("meta_seconds", 0.0)
                                   for s in seeds) / float(len(seeds))
        assert all(tab[(a, s)].get("meta_program_executions", 0) == 0
                   for s in seeds)
    v["meta_controller_compute_mean_per_run"] = mc
    v["meta_controller_program_executions"] = 0
    # 4. identical compute conditions, exact metering
    v["caps_identical"] = len({tab[(a, s)]["cap"] for a in arms
                               for s in seeds}) == 1
    v["world_caps_identical"] = len({tab[(a, s)]["world_cap"] for a in arms
                                     for s in seeds}) == 1
    v["all_within_cap"] = all(tab[(a, s)]["spent"] <= tab[(a, s)]["cap"]
                              for a in arms for s in seeds)
    v["all_metered_exactly"] = all(tab[(a, s)]["spent"]
                                   == tab[(a, s)]["global_delta"]
                                   for a in arms for s in seeds)
    return v


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
        seeds = parse_seeds(prereg["seeds"]["confirm"])
        _t, missing = collect(led, "confirm", prereg["arms"], seeds)
        print("protocol frozen, hashes match; confirm units missing: %d; "
              "abandoned: %d" % (len(missing),
                                 len(led.abandoned("confirm"))))
    except (AssertionError, IOError) as e:
        print("not frozen / mismatch: %s" % e)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rsi_v3")
    sub = ap.add_subparsers(dest="cmd")
    for name in ("dev", "devcheck"):
        d = sub.add_parser(name)
        d.add_argument("--label", required=True)
        d.add_argument("--seeds", default=None)
        d.add_argument("--hp", default="{}")
        d.add_argument("--workers", type=int, default=4)
        d.add_argument("--arms", default=",".join(I.ARMS))
    sub.add_parser("freeze")
    c = sub.add_parser("confirm")
    c.add_argument("--workers", type=int, default=4)
    r = sub.add_parser("report")
    r.add_argument("--out", default=os.path.join(
        ROOT, "results", "v3_confirm_report.json"))
    sub.add_parser("verify")
    a = ap.parse_args(argv)
    if a.cmd == "dev":
        _dev_like("DEV", a.label, parse_seeds(a.seeds or "3001-3012"),
                  json.loads(a.hp), a.workers, a.arms.split(","), DEV_ITERATE)
    elif a.cmd == "devcheck":
        _dev_like("DEVCHECK", a.label, parse_seeds(a.seeds or "3201-3300"),
                  json.loads(a.hp), a.workers, a.arms.split(","), DEV_CHECK)
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
