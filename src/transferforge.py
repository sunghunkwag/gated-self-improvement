#!/usr/bin/env python3
"""
TRANSFERFORGE -- cross-substrate skill transfer, counterfactually gated
=======================================================================
Does a skill the system DISCOVERS on one substrate enable solving on a
DIFFERENT substrate whose native vocabulary CANNOT express that skill?

Two vocabularies over the same universal tforge VM (like two languages
compiling to one machine):
  SUBSTRATE A (arithmetic):  has ADD/SUB/MUL/INC/DEC/NEG + map/reduce
  SUBSTRATE B (structure):   comparisons, min/max, stack shuffles,
                             branches, loops, map/reduce -- but NO
                             ADD/SUB/MUL/INC/DEC/NEG (cannot compute 2*v,
                             v+1, or a sum)

PHASE 1  A's searcher solves arithmetic train tasks; each distinct solving
         program is a LEARNED skill (a behaviour A found by search and
         exact-verified -- not designer-authored).
PHASE 2  Three arms attempt HELD-OUT target tasks at equal budget:
           B_alone  : B's vocabulary only
           B_rand   : B + RANDOM A-vocabulary programs (matched count &
                      size) -- control for arithmetic CAPABILITY without
                      the learning
           B_skill  : B + A's LEARNED skills (callable macros; the VM runs
                      the macro body, so B can CALL arithmetic it cannot
                      SAMPLE). That is transfer.

Metrics (paired by seed): B_skill-B_alone = raw transfer; B_skill-B_rand =
value of the LEARNED abstraction. STRUCT tasks (B-native) = control.

CLI: python3 transferforge.py {selftest|run <s0> <s1> <budget_s>|report}
"""
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tforge as TF  # noqa: E402

LOG = os.path.join(os.path.dirname(HERE), "results", "logs",
                   "transfer_log_shared_stream.jsonl")
T0 = time.time()

OPSET_A = ["PUSHx", "PUSHi", "DUP", "OVER", "ADD", "SUB", "MUL", "INC",
           "DEC", "NEG", "MAPE", "FOLD", "EMIT", "ADVANCE", "LEN"]
OPSET_B = ["PUSHx", "PUSHi", "DUP", "SWAP", "OVER", "DROP", "CMP", "MINc",
           "MAXc", "EMIT", "ADVANCE", "LEN", "JZ", "LOOP", "MAPE", "FOLD"]
ARITH_ONLY = {"ADD", "SUB", "MUL", "INC", "DEC", "NEG"}


def _double(xs):
    return [v * 2 for v in xs]


def _plus1(xs):
    return [v + 1 for v in xs]


def _sumall(xs):
    return [sum(xs)]


def _sq(xs):
    return [v * v for v in xs]


def _double_max(xs):
    return [max(v * 2 for v in xs)] if xs else []


def _inc_min(xs):
    return [min(v + 1 for v in xs)] if xs else []


def _maxelem(xs):
    return [max(xs)] if xs else []


def _minelem(xs):
    return [min(xs)] if xs else []


def _last(xs):
    return [xs[-1]] if xs else []


A_TRAIN = {"double": _double, "plus1": _plus1, "sumall": _sumall, "sq": _sq}
TARGET_ARITH = {"double": _double, "sumall": _sumall,
                "double_max": _double_max, "inc_min": _inc_min}
TARGET_STRUCT = {"maxelem": _maxelem, "minelem": _minelem, "last": _last}


def mk_pairs(fn, prng, n, lo, hi):
    out = []
    for _ in range(n):
        ln = lo + prng.below(hi - lo + 1)
        xs = [-6 + prng.below(13) for _ in range(ln)]
        try:
            y = fn(list(xs))
        except Exception:
            y = None
        out.append((xs, y))
    return out


def task_of(name, fn, prng):
    tr = mk_pairs(fn, prng, 5, 2, 5)
    te = mk_pairs(fn, prng, 5, 6, 9)
    if any(y is None for _x, y in tr + te):
        return None
    return {"name": name, "train": tr, "test": te}


def vocab_of(opset, macros):
    return list(opset) + ["MACRO:" + m for m in sorted(macros)]


def rand_tok(opset, macros, prng):
    op = prng.choice(vocab_of(opset, macros))
    if op == "PUSHi":
        return ("PUSHi", prng.below(9) - 2)
    if op in ("JZ", "LOOP", "MAPE", "FOLD"):
        return (op, 1 + prng.below(3))
    if op.startswith("MACRO:"):
        return ("MACRO", op[6:])
    return (op,)


def rand_prog(opset, macros, prng, lo=2, hi=10):
    return [rand_tok(opset, macros, prng)
            for _ in range(lo + prng.below(hi - lo + 1))]


def mutate(prog, opset, macros, prng):
    p = list(prog)
    r = prng.unit()
    if r < 0.30 and len(p) < 22:
        p.insert(prng.below(len(p) + 1), rand_tok(opset, macros, prng))
    elif r < 0.55 and len(p) > 1:
        p.pop(prng.below(len(p)))
    elif r < 0.85 and p:
        p[prng.below(len(p))] = rand_tok(opset, macros, prng)
    else:
        if p:
            i = prng.below(len(p))
            j = min(len(p), i + 1 + prng.below(3))
            seg = p[i:j]
            if len(p) + len(seg) <= 22:
                p[i:i] = seg
    return p


def search(task, opset, macros, budget, prng):
    POP = 40
    guide = task["train"]
    evals = [0]

    def ev(p):
        evals[0] += 1
        return TF.fitness(p, guide, macros)

    pop = []
    while len(pop) < POP and evals[0] < budget:
        p = rand_prog(opset, macros, prng)
        f = ev(p)
        if f >= 1.0 and TF.solves(p, task, macros):
            return p, evals[0], [(f, p)]
        pop.append((f, p))
    while evals[0] < budget:
        pop.sort(key=lambda t: -t[0])
        elite = pop[:max(2, POP // 4)]
        newpop = list(elite)
        while len(newpop) < POP and evals[0] < budget:
            r = prng.unit()
            if r < 0.30:
                c = rand_prog(opset, macros, prng)
            elif r < 0.70:
                c = mutate(elite[prng.below(len(elite))][1], opset,
                           macros, prng)
            else:
                c = TF.crossover(elite[prng.below(len(elite))][1],
                                 pop[prng.below(len(pop))][1], prng)
            f = ev(c)
            if f >= 1.0 and TF.solves(c, task, macros):
                return c, evals[0], [(ff, pp) for ff, pp in elite[:8]]
            newpop.append((f, c))
        pop = newpop
    pop.sort(key=lambda t: -t[0])
    return None, evals[0], [(f, p) for f, p in pop[:8]]


LEARN_BUDGET = 20000


def learn_skills_A(seed):
    skills = {}
    for name, fn in sorted(A_TRAIN.items()):
        task = task_of(name, fn, TF.PRNG("A-task|%s|%d" % (name, seed)))
        if task is None:
            continue
        prog, _ev, _e = search(task, OPSET_A, {}, LEARN_BUDGET,
                               TF.PRNG("A-solve|%s|%d" % (name, seed)))
        if prog is not None and TF.solves(prog, task, {}):
            if {t[0] for t in prog} & ARITH_ONLY:
                skills["skill_" + name] = list(prog)
    return skills


def random_A_programs(skills, seed):
    prng = TF.PRNG("A-rand|%d" % seed)
    rand = {}
    for i, (nm, body) in enumerate(sorted(skills.items())):
        L = len(body)
        rp = rand_prog(OPSET_A, {}, prng, max(2, L - 1), L + 1)
        if not ({t[0] for t in rp} & ARITH_ONLY):
            rp[prng.below(len(rp))] = (prng.choice(sorted(ARITH_ONLY)),)
        rand["rand_%d" % i] = rp
    return rand


SOLVE_BUDGET = 20000


def try_targets(arm, macros, seed, targets):
    solved = {}
    for name, fn in sorted(targets.items()):
        task = task_of(name, fn, TF.PRNG("T-task|%s|%d" % (name, seed)))
        if task is None:
            solved[name] = None
            continue
        # AUDIT FIX (v2 upgrade): the solve stream was keyed on the arm
        # ('T-<arm>|...'); every arm now searches under the same stream.
        prog, _ev, _e = search(task, OPSET_B, macros, SOLVE_BUDGET,
                               TF.PRNG("T-solve|%s|%d" % (name, seed)))
        solved[name] = bool(prog is not None and TF.solves(prog, task,
                                                           macros))
    return solved


def run_seed(seed):
    skills = learn_skills_A(seed)
    rands = random_A_programs(skills, seed)
    arms = {"B_alone": {}, "B_rand": rands, "B_skill": skills}
    rec = {"exp": "transfer", "seed": seed,
           "n_skills": len(skills), "skills": sorted(skills)}
    for cat, targets in (("arith", TARGET_ARITH), ("struct", TARGET_STRUCT)):
        for arm, macros in arms.items():
            sv = try_targets(arm, macros, seed, targets)
            rec["%s_%s" % (cat, arm)] = sum(1 for v in sv.values() if v)
            rec["%s_%s_detail" % (cat, arm)] = sv
    with open(LOG, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def cmd_run(s0, s1, budget):
    done = 0
    for seed in range(s0, s1 + 1):
        if budget - (time.time() - T0) < 8:
            break
        r = run_seed(seed)
        print(json.dumps({"seed": seed, "n_skills": r["n_skills"],
                          "ARITH": {"alone": r["arith_B_alone"],
                                    "rand": r["arith_B_rand"],
                                    "skill": r["arith_B_skill"]},
                          "STRUCT": {"alone": r["struct_B_alone"],
                                     "skill": r["struct_B_skill"]}}))
        done = seed
    print(json.dumps({"done_through": done,
                      "elapsed": round(time.time() - T0, 1)}))


def _perm(diffs, n_perm=20000):
    import hashlib
    obs = sum(diffs) / len(diffs)
    st = int(hashlib.sha256(str(diffs).encode()).hexdigest()[:8], 16) or 1
    hits = 0
    for _ in range(n_perm):
        s = 0.0
        for d in diffs:
            st = (st * 6364136223846793005 + 1) & ((1 << 64) - 1)
            s += d if (st >> 33) & 1 else -d
        if abs(s / len(diffs)) >= abs(obs) - 1e-12:
            hits += 1
    return (hits + 1) / float(n_perm + 1)


def cmd_report():
    recs = [json.loads(l) for l in open(LOG)] if os.path.exists(LOG) else []
    recs = [r for r in recs if r.get("exp") == "transfer"]
    if not recs:
        print("no records")
        return
    n = len(recs)
    print("=" * 70)
    print("CROSS-SUBSTRATE SKILL TRANSFER  (n=%d seeds, 4 arith + 3 struct "
          "held-out tasks)" % n)
    print("=" * 70)

    def mean(k):
        return sum(r[k] for r in recs) / float(n)
    print("\nARITH target tasks (need A-behaviours B cannot express; /4):")
    print("  B_alone : %.2f   (floor -- B has no arithmetic)"
          % mean("arith_B_alone"))
    print("  B_rand  : %.2f   (random A-programs: capability, no learning)"
          % mean("arith_B_rand"))
    print("  B_skill : %.2f   (A's LEARNED skills, transferred)"
          % mean("arith_B_skill"))
    d1 = [r["arith_B_skill"] - r["arith_B_alone"] for r in recs]
    d2 = [r["arith_B_skill"] - r["arith_B_rand"] for r in recs]
    print("  transfer (skill - alone): mean %+0.2f  perm-p=%.4f"
          % (sum(d1) / n, _perm(d1)))
    print("  learning (skill - rand ): mean %+0.2f  perm-p=%.4f"
          % (sum(d2) / n, _perm(d2)))
    print("\nSTRUCT target tasks (B-native control; /3):")
    print("  B_alone : %.2f    B_skill : %.2f   (should be ~equal)"
          % (mean("struct_B_alone"), mean("struct_B_skill")))
    print("\nmean skills learned on A per seed: %.2f" % mean("n_skills"))


def selftest():
    print("TRANSFERFORGE selftest\n")
    task = task_of("double", _double, TF.PRNG("st-d"))
    prog, _ev, _e = search(task, OPSET_B, {}, 8000, TF.PRNG("st-ds"))
    ok_b = prog is not None and TF.solves(prog, task, {})
    print("  [1] B_alone solves 'double': %s  (want False)" % ok_b)
    assert not ok_b
    skills = learn_skills_A(1)
    print("  [2] A learned skills:", sorted(skills))
    assert skills
    assert any(({t[0] for t in b} & ARITH_ONLY) for b in skills.values())
    print("  [3] every skill body uses arithmetic B lacks: True")
    r = try_targets("B_skill", skills, 1, TARGET_ARITH)
    print("  [4] B_skill on ARITH targets:", r)
    assert any(r.values())
    r0 = try_targets("B_alone", {}, 1, TARGET_ARITH)
    print("  [5] B_alone on ARITH targets:", r0)
    print("\nOK -- B cannot natively; A learns; transferred skills unlock B.")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    if cmd == "selftest":
        selftest()
    elif cmd == "run":
        cmd_run(int(sys.argv[2]), int(sys.argv[3]),
                float(sys.argv[4]) if len(sys.argv) > 4 else 60.0)
    elif cmd == "report":
        cmd_report()
    else:
        sys.exit("unknown command")
