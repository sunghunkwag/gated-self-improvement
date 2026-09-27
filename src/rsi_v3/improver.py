"""
rsi_v3.improver -- a META-LEARNED improver.

Why v2's adaptive improver failed (measured, results/UPGRADE_V2_RESULTS.md):
~20 sparse outcomes per run, categorical failure labels dominated by one
mode, a Q-table that never ranked differently from its context-free
ablation, and a probe reward drawn from TRAIN families while the final
objective was cross-family. v3 changes the learning problem itself:

1. SHARED CONTINUOUS CONTEXTUAL PREDICTOR (metamodel.Ridge). Features are
   the full continuous diagnosis vector (all failure scores, search
   progress, diversity, entropy, dead-macro ratio, composition depth...),
   the candidate's action, and the candidate's concrete deltas (macros
   added/removed, their length/support/depth/residual coverage, prior
   shift, exploration change), with action x diagnosis interactions.
2. DENSE PAIRED TRAINING DATA: every (candidate, probe) pair of every
   counterfactual test is one row, y = progress(candidate) -
   progress(incumbent) on the same stream (progress = 1 if solved, else
   0.5 x best partial fitness). ~8 candidates x 6-12 probes per round.
3. CROSS-FAMILY META-VALIDATION: probes come from META-VAL families that
   are neither TRAIN nor FINAL families, so what the predictor learns --
   and what the gate adopts -- is transfer to unseen families.
4. STRUCTURALLY DISTINCT ACTIONS, generated in proportions set by the
   diagnosis (so the diagnosis changes WHICH candidates exist):
     MINE     behaviour-level frequent-fragment mining from solutions
     RESID    residual-guided mining: search short repairs that turn a
              near-miss program into an exact solution; the junction
              fragments become operator candidates, repaired tasks become
              training solutions
     COMPOSE  hierarchical composition of existing macros
     PRIOR    prior / order repair toward solutions or near-misses
     EXPLORE  exploration repair (flatten prior, restart rate, depth + 1)
     PRUNE    removal of dead macros -- generated ONLY when the evidence
              says the vocabulary is over-specialised
5. The predictor RANKS the pool; only k candidates get the scarce
   counterfactual-test budget. Adoption uses the same unbiased
   screen -> confirm rule for every arm.

Arms (identical code path, flags differ; identical execution cap):
  COLD                    no improvement
  SINGLE_COMPUTE_MATCHED  one round, 5x attempt + 5x gate compute, bigger
                          pool, staged halving + untouched confirmation
  FROZEN_META             5 rounds; predictor frozen at its initial state
                          (uninformative -> shared random order)
  ADAPTIVE_META           5 rounds; predictor learns online (k-1 greedy +
                          1 exploration slot)
  NODIAG_META             ablation: learns, but diagnosis removed from
                          both generation (uniform) and predictor features
  HEURISTIC_META          exploratory: fixed hand ranking (generator's own
                          preference x diagnosed need)
"""
import math

from rsi_v2 import substrate as S
from rsi_v2 import solver as SV
from rsi_v2.improver import shortest_parse, entropy_ratio, simplify
from . import tasks as T
from .metamodel import Ridge

VERSION = "rsi_v3-improver-1"
ACTIONS = ("MINE", "RESID", "COMPOSE", "PRIOR", "EXPLORE", "PRUNE")
ARMS = ("COLD", "SINGLE_COMPUTE_MATCHED", "FROZEN_META", "ADAPTIVE_META",
        "NODIAG_META", "HEURISTIC_META", "ADAPTIVE_NOCARRY")
DIAGNOSTIC_ARMS = ("ORACLE_RANK",)   # dev-only headroom probes; never
                                     # compute-matched, never confirmatory
LEARNING_ARMS = ("ADAPTIVE_META", "NODIAG_META", "ADAPTIVE_NOCARRY")
CARRY_ARMS = ("ADAPTIVE_META", "NODIAG_META")   # predictor persists across
                                                # worlds; NOCARRY resets it

HP = {
    "n_worlds": 4,
    "n_rounds": 5,
    "attempt_budget": 2000,
    "gate_probe_budget": 600,
    "gate_round_budget": 48000,
    "k_screen": 8,
    "pool_size": 24,
    "single_pool_size": 36,
    "alloc_temp": 0.35,
    "ridge_lam": 3.0,
    "partial_weight": 0.5,
    "adopt_cost_ratio": 0.9,
    "mine_min_len": 2, "mine_max_len": 6, "macro_len_max": 12,
    "refit_strength": 8.0, "prior_floor": 0.08, "new_macro_boost": 2.0,
    "near_miss_fit": 0.5, "resid_max_tasks": 8, "resid_pair_tasks": 3,
    "p_random_max": 0.6, "prune_dead_ratio": 0.34,
    # per-probe noise variance and prior variance of true paired gains,
    # estimated on dev seeds from split-half reliability (used only by
    # learning arms to denoise the finalist choice)
    "noise_var": 0.055, "prior_var": 0.0015,
    "gate": "strict", "finalist": "measured",
}

N_STATE = 15
DIAG_IDX = (0, 1, 2, 3, 4, 5, 6, 9)    # state dims crossed with the action
N_DELTA = 10


def world_cap(hp, n_train):
    return hp["n_rounds"] * (n_train * hp["attempt_budget"]
                             + hp["gate_round_budget"])


def total_cap(hp, n_train):
    return hp["n_worlds"] * world_cap(hp, n_train)


def world_key(seed, w):
    """Each run is a SEQUENCE of independent improvement problems
    ('worlds'): fresh family split, fresh tasks, fresh base solver."""
    return "%s/w%d" % (seed, w)


def stream(seed, *parts):
    return S.XorShift64Star("rsi_v3|" + "|".join(str(p) for p in
                                                 (seed,) + parts))


# each action's "own" failure score (index into the diagnosis vector): the
# diagnosis enters the predictor through these matched interactions
MATCHED = {"MINE": 0, "RESID": 5, "COMPOSE": 4, "PRIOR": 1, "EXPLORE": 2,
           "PRUNE": 3}


def feat_dim():
    return 1 + len(ACTIONS) + N_DELTA + 1 + len(ACTIONS) + 1


# --------------------------------------------------------------------------
# state, attempts, diagnosis

class State(object):
    def __init__(self):
        self.cfg = SV.base_config()
        self.solved = {}          # train idx -> simplified primitive tuple
        self.sigs = {}            # token -> signature
        self.frag_sigs = {}       # primitive fragment -> signature
        self.inc_rates = []       # incumbent meta-val solve rate per round
        self.resid_log = []       # (idx, junction fragment) repairs found


class Trace(object):
    __slots__ = ("idx", "solved", "spent", "best_fit", "best_prog",
                 "diversity")

    def __init__(self, idx, solved, spent, best_fit, best_prog, diversity):
        self.idx, self.solved, self.spent = idx, solved, spent
        self.best_fit, self.best_prog = best_fit, best_prog
        self.diversity = diversity


def expand(prog):
    out = ()
    for t in prog:
        out += SV.expansion(t)
    return out


def _solves(prims, examples, meter):
    fns = S.fns_of(prims)
    for x, y in examples:
        if meter.run(fns, list(x)) != list(y):
            return False
    return True


def attempt(state, comp, task, idx, budget, prng, meter):
    res = SV.search(comp, task.public(), budget, prng, meter)
    ok = res.program is not None and SV.check(comp, res.program, task.test,
                                              meter)
    if ok:
        prims = expand(res.program)
        try:
            prims = simplify(prims, task.train + task.test, meter)
        except S.BudgetExceeded:
            pass
        state.solved.setdefault(idx, prims)
    return Trace(idx, ok, res.spent, res.best_fit, res.best_prog,
                 res.diversity)


def macro_depth(tok, cfg):
    """How many OTHER current macros this macro's expansion contains."""
    e = SV.expansion(tok)
    n = 0
    for m in cfg.macros:
        f = SV.expansion(m)
        if m != tok and len(f) < len(e) and _contains(e, f):
            n += 1
    return n


def _contains(big, small):
    m = len(small)
    return any(tuple(big[i:i + m]) == tuple(small)
               for i in range(len(big) - m + 1))


def diagnose(state, traces, rnd, hp):
    """Continuous diagnosis vector (all scores kept, not an argmax)."""
    cfg = state.cfg
    uns = [t for t in traces if not t.solved]
    sol = [t for t in traces if t.solved]
    nu = float(max(1, len(uns)))
    missing = sum(1 for t in uns if t.best_prog is not None
                  and len(t.best_prog) >= cfg.max_tokens - 1
                  and t.best_fit >= 0.3) / nu
    order = (sum(t.spent for t in sol) / (len(sol) * float(
        hp["attempt_budget"])) if sol else 0.0)
    explore = sum(1 for t in uns if t.best_fit < 0.3
                  or t.diversity < 0.5) / nu
    if cfg.macros:
        used = set()
        for i in sorted(state.solved):
            used.update(shortest_parse(state.solved[i], cfg))
        dead = sum(1 for m in cfg.macros if m not in used) / float(
            len(cfg.macros))
    else:
        dead = 0.0
    ent = entropy_ratio(cfg)
    regress = 0.0
    if len(state.inc_rates) >= 2:
        regress = max(0.0, state.inc_rates[-2] - state.inc_rates[-1])
    overspec = min(1.0, 0.5 * dead + min(1.0, 2.0 * (1.0 - ent)) * 0.5
                   + regress)
    compose = sum(1 for t in uns if t.best_prog is not None
                  and sum(1 for x in t.best_prog if SV.is_macro(x)) >= 2
                  and t.best_fit >= hp["near_miss_fit"]) / nu
    residual = sum(1 for t in uns if t.best_prog is not None
                   and t.best_fit >= hp["near_miss_fit"]) / nu
    depth = max([macro_depth(m, cfg) for m in cfg.macros] or [0])
    v = [missing, order, explore, overspec, compose, residual,
         len(state.solved) / 40.0, ent, dead,
         min(1.0, len(cfg.macros) / 10.0),
         sum(max(0.0, t.best_fit) for t in uns) / nu,
         sum(t.diversity for t in uns) / nu,
         min(1.0, depth / 3.0), rnd / float(hp["n_rounds"]),
         state.inc_rates[-1] if state.inc_rates else 0.0]
    names = ("missing_op", "bad_order", "low_explore", "overspec",
             "compose_fail", "residual", "solved_frac", "entropy",
             "dead_macros", "n_macros", "mean_bestfit", "diversity",
             "comp_depth", "round", "inc_rate")
    return v, dict(zip(names, [round(x, 4) for x in v])), {
        "dead": dead, "regress": regress}


# --------------------------------------------------------------------------
# config edits

def add_macros(cfg, frags, hp):
    c = cfg.copy()
    mean_w = sum(c.weights[t] for t in c.tokens()) / len(c.tokens())
    for fr in frags:
        tok = SV.macro_token(fr)
        if tok in c.weights:
            continue
        c.macros = c.macros + (tok,)
        c.weights[tok] = hp["new_macro_boost"] * mean_w
    return c


def refit(cfg, progs, lam, hp):
    """Trust-region prior step toward token presence in `progs`."""
    if not progs:
        return None
    parses = [set(p) for p in progs]
    n = float(len(parses))
    c = cfg.copy()
    for t in cfg.tokens():
        rate = sum(1 for p in parses if t in p) / n
        c.weights[t] = (1.0 - lam) * cfg.weights[t] + lam * (
            1.0 + hp["refit_strength"] * rate)
    hi = max(c.weights.values())
    for t in c.tokens():
        c.weights[t] = max(c.weights[t], hp["prior_floor"] * hi)
    return c


def solution_parses(state, cfg):
    return [shortest_parse(state.solved[i], cfg)
            for i in sorted(state.solved)]


def explore_cfg(cfg, eps, depth_plus, hp):
    c = cfg.copy()
    mean_w = sum(cfg.weights[t] for t in cfg.tokens()) / len(cfg.tokens())
    for t in cfg.tokens():
        c.weights[t] = (1.0 - eps) * cfg.weights[t] + eps * mean_w
    c.p_random = min(hp["p_random_max"], cfg.p_random + eps / 2.0)
    c.max_tokens = min(6, cfg.max_tokens + depth_plus)
    return c


def prune_cfg(cfg, drop):
    c = cfg.copy()
    c.macros = tuple(m for m in cfg.macros if m not in drop)
    for m in drop:
        c.weights.pop(m, None)
    c.p_random = (c.p_random + SV.BASE_P_RANDOM) / 2.0
    return c


# --------------------------------------------------------------------------
# operator sources (each returns ranked [(fragment, support)])

def frag_sig(state, fr, meter):
    sg = state.frag_sigs.get(fr)
    if sg is None:
        sg = S.signature(S.fns_of(fr), meter)
        state.frag_sigs[fr] = sg
    return sg


def known_sigs(state, cfg, meter):
    out = {frag_sig(state, (), meter)}
    for t in cfg.tokens():
        out.add(frag_sig(state, SV.expansion(t), meter))
    return out


def mine_sources(state, meter, hp):
    """Behaviour-level frequent fragments of solved programs."""
    cfg = state.cfg
    ks = known_sigs(state, cfg, meter)
    support, rep, whole = {}, {}, set()
    for idx in sorted(state.solved):
        seq = state.solved[idx]
        whole.add(frag_sig(state, tuple(seq), meter))
        seen = []
        for L in range(hp["mine_min_len"], hp["mine_max_len"] + 1):
            for st in range(0, len(seq) - L + 1):
                fr = tuple(seq[st:st + L])
                sg = frag_sig(state, fr, meter)
                if sg in ks:
                    continue
                if sg not in seen:
                    seen.append(sg)
                if sg not in rep or (len(fr), fr) < (len(rep[sg]),
                                                     rep[sg]):
                    rep[sg] = fr
        for sg in seen:
            support[sg] = support.get(sg, 0) + 1
    out = []
    for sg in rep:
        sp = support.get(sg, 0)
        fr = rep[sg]
        if sp >= 2:
            out.append(((1, sp * (len(fr) - 1)), fr, sp))
        elif sg in whole:
            out.append(((0, len(fr) - 1), fr, sp))
    out.sort(key=lambda t: (-t[0][0], -t[0][1], -len(t[1]), t[1]))
    return [(fr, sp) for _s, fr, sp in out]


def resid_sources(state, traces, train, meter, hp):
    """Residual-guided operator mining. For the best near-miss programs of
    unsolved train tasks, search a one-token (or one primitive pair)
    repair appended or prepended that makes the program exact. A repair
    that also passes the task's test examples becomes a solution; the
    junction fragments around the repair are operator candidates."""
    cfg = state.cfg
    near = sorted([t for t in traces if not t.solved and t.best_prog
                   and t.best_fit >= hp["near_miss_fit"]],
                  key=lambda t: (-t.best_fit, t.idx))[:hp["resid_max_tasks"]]
    toks = cfg.tokens()
    events = []
    for rank, t in enumerate(near):
        task = train[t.idx]
        p = expand(t.best_prog)
        found = None
        cands = [(pos, SV.expansion(tok)) for pos in ("post", "pre")
                 for tok in toks]
        if rank < hp["resid_pair_tasks"]:
            cands += [("post", (a, b)) for a in S.NAMES for b in S.NAMES]
        for pos, e in cands:
            q = p + e if pos == "post" else e + p
            if len(q) > SV.MAX_EXPANDED:
                continue
            if _solves(q, task.train, meter):
                found = (pos, e, q)
                break
        if found is None:
            continue
        pos, e, q = found
        if _solves(q, task.test, meter) and t.idx not in state.solved:
            try:
                state.solved[t.idx] = simplify(q, task.train + task.test,
                                               meter)
            except S.BudgetExceeded:
                state.solved[t.idx] = q
        for L in (1, 2):
            if pos == "post":
                j = p[max(0, len(p) - L):] + e
            else:
                j = e + p[:L]
            if 2 <= len(j) <= hp["macro_len_max"]:
                events.append((t.idx, tuple(j)))
    state.resid_log = events
    ks = known_sigs(state, cfg, meter)
    support, rep = {}, {}
    for idx, fr in events:
        sg = frag_sig(state, fr, meter)
        if sg in ks:
            continue
        support.setdefault(sg, [])
        if idx not in support[sg]:
            support[sg].append(idx)
        if sg not in rep or (len(fr), fr) < (len(rep[sg]), rep[sg]):
            rep[sg] = fr
    out = sorted(((len(support[sg]), rep[sg]) for sg in rep),
                 key=lambda t: (-t[0], len(t[1]), t[1]))
    return [(fr, sp) for sp, fr in out]


def compose_sources(state, traces, meter, hp):
    cfg = state.cfg
    if not cfg.macros:
        return []
    sources = [shortest_parse(state.solved[i], cfg)
               for i in sorted(state.solved)]
    for t in traces:
        if (not t.solved and t.best_prog is not None
                and t.best_fit >= hp["near_miss_fit"]):
            sources.append(tuple(t.best_prog))
    support = {}
    for src in sources:
        seen = []
        for a, b in zip(src, src[1:]):
            if not (SV.is_macro(a) or SV.is_macro(b)):
                continue
            if (a, b) not in seen:
                seen.append((a, b))
        for pr in seen:
            support[pr] = support.get(pr, 0) + 1
    existing = {SV.expansion(t) for t in cfg.macros}
    ks = known_sigs(state, cfg, meter)
    out, taken = [], []
    for (a, b), sp in sorted(support.items(),
                             key=lambda kv: (-kv[1], kv[0])):
        if sp < 2:
            continue
        fr = SV.expansion(a) + SV.expansion(b)
        if len(fr) > hp["macro_len_max"] or fr in existing:
            continue
        sg = frag_sig(state, fr, meter)
        if sg in ks or sg in taken:
            continue
        taken.append(sg)
        out.append((fr, sp))
    return out


# --------------------------------------------------------------------------
# candidate pool

class Cand(object):
    __slots__ = ("action", "label", "cfg", "delta", "h", "x", "mu")

    def __init__(self, action, label, cfg, delta, h):
        self.action, self.label, self.cfg = action, label, cfg
        self.delta, self.h = delta, h
        self.x = None
        self.mu = 0.0


MACRO_VARIANTS = ((0,), (1,), (0, 1), (0, 1, 2), (2,), (0, 1, 2, 3, 4),
                  (3,), (1, 2), (0, 2), (4,))
# severe "missing operator"-type failures -> propose BUNDLES of operators
# first; mild ones -> single operators first
BUNDLE_FIRST = ((0, 1, 2), (0, 1), (0, 1, 2, 3, 4), (0,), (1, 2), (0, 2),
                (1,), (2,), (3,), (4,))


def _macro_variants(state, action, ranked, n, hp, allow_prior=True,
                    sev=0.5):
    out = []
    order = BUNDLE_FIRST if sev > 0.5 else MACRO_VARIANTS
    for vi, pick in enumerate(order):
        if len(out) >= n:
            break
        sel = [ranked[i] for i in pick if i < len(ranked)]
        if len(sel) != len(pick):
            continue
        c = add_macros(state.cfg, [fr for fr, _s in sel], hp)
        label = "%s%s" % (action, list(pick))
        out.append((label, c, [sp for _f, sp in sel]))
        if allow_prior and vi in (0, 2) and len(out) < n:
            c2 = refit(c, solution_parses(state, c), 0.4, hp)
            if c2 is not None:
                out.append((label + "+PRIOR", c2, [sp for _f, sp in sel]))
    return out[:n]


def availability(state, traces, diag_on, info, hp):
    near = any(not t.solved and t.best_prog is not None for t in traces)
    ok = {"MINE": bool(state.solved), "RESID": near,
          "COMPOSE": bool(state.cfg.macros),
          "PRIOR": bool(state.solved), "EXPLORE": True,
          "PRUNE": bool(state.cfg.macros)}
    if diag_on and ok["PRUNE"]:   # prune only on evidence
        ok["PRUNE"] = (info["dead"] >= hp["prune_dead_ratio"]
                       or info["regress"] > 0.0)
    return ok


def needs(v):
    missing, order, explore, overspec, compose, residual = v[:6]
    solved_frac, ent, dead, n_mac = v[6], v[7], v[8], v[9]
    return {"MINE": missing + 0.5 * solved_frac,
            "RESID": residual,
            "COMPOSE": compose + 0.5 * n_mac,
            "PRIOR": order + 0.25,
            "EXPLORE": explore + 2.0 * (1.0 - ent),
            "PRUNE": overspec + dead}


def allocate(v, ok, size, diag_on, hp):
    acts = [a for a in ACTIONS if ok[a]]
    if not acts:
        return {}
    if diag_on:
        nd = needs(v)
        mx = max(nd[a] for a in acts)
        w = {a: math.exp((nd[a] - mx) / hp["alloc_temp"]) for a in acts}
    else:
        w = {a: 1.0 for a in acts}
    tot = sum(w.values())
    raw = {a: size * w[a] / tot for a in acts}
    n = {a: max(1, int(math.floor(raw[a]))) for a in acts}
    rem = sorted(acts, key=lambda a: (-(raw[a] - math.floor(raw[a])),
                                      ACTIONS.index(a)))
    i = 0
    while sum(n.values()) < size and rem:
        n[rem[i % len(rem)]] += 1
        i += 1
    return n


def severity(v, diag_on):
    """Per-action failure severity in [0, 1] that PARAMETERISES the
    proposals (bundle vs single operators, prior step size, flattening).
    Without diagnosis every action gets the neutral 0.5."""
    if not diag_on:
        return {a: 0.5 for a in ACTIONS}
    return {a: max(0.0, min(1.0, v[MATCHED[a]])) for a in ACTIONS}


def generate_pool(state, traces, train, v, info, meter, hp, diag_on, size):
    ok = availability(state, traces, diag_on, info, hp)
    alloc = allocate(v, ok, size, diag_on, hp)
    nd = needs(v)
    sev = severity(v, diag_on)
    raw = {}
    for a in ACTIONS:
        n = alloc.get(a, 0)
        if n <= 0:
            raw[a] = []
            continue
        if a == "MINE":
            raw[a] = _macro_variants(state, a, mine_sources(state, meter, hp),
                                     n, hp, sev=sev[a])
        elif a == "RESID":
            raw[a] = _macro_variants(state, a, resid_sources(
                state, traces, train, meter, hp), n, hp, sev=sev[a])
        elif a == "COMPOSE":
            raw[a] = _macro_variants(state, a, compose_sources(
                state, traces, meter, hp), n, hp, sev=sev[a])
        elif a == "PRIOR":
            sp = solution_parses(state, state.cfg)
            nm = [tuple(t.best_prog) for t in traces if not t.solved
                  and t.best_prog is not None
                  and t.best_fit >= hp["near_miss_fit"]]
            lam = round(0.2 + 0.5 * sev[a], 3)    # worse ordering -> bigger
            opts = [("PRIOR[sol%.2f]" % lam, refit(state.cfg, sp, lam, hp)),
                    ("PRIOR[sol%.2f]" % (lam / 2),
                     refit(state.cfg, sp, lam / 2, hp)),
                    ("PRIOR[near%.2f]" % lam, refit(state.cfg, nm, lam, hp)),
                    ("PRIOR[sol%.2f]" % min(0.9, 1.6 * lam),
                     refit(state.cfg, sp, min(0.9, 1.6 * lam), hp)),
                    ("PRIOR[sol+near%.2f]" % lam,
                     refit(state.cfg, sp + nm, lam, hp))]
            raw[a] = [(lb, c, []) for lb, c in opts if c is not None][:n]
        elif a == "EXPLORE":
            eps = round(0.1 + 0.4 * sev[a], 3)    # less traction -> flatter
            opts = [("EXPLORE[%.2f]" % eps, explore_cfg(state.cfg, eps, 0,
                                                        hp)),
                    ("EXPLORE[depth+1]", explore_cfg(state.cfg, 0.0, 1, hp)),
                    ("EXPLORE[%.2f]" % (eps / 2),
                     explore_cfg(state.cfg, eps / 2, 0, hp)),
                    ("EXPLORE[%.2f]" % min(0.7, 1.6 * eps),
                     explore_cfg(state.cfg, min(0.7, 1.6 * eps), 0, hp)),
                    ("EXPLORE[%.2f+depth]" % eps,
                     explore_cfg(state.cfg, eps, 1, hp))]
            raw[a] = [(lb, c, []) for lb, c in opts][:n]
        elif a == "PRUNE":
            use = {m: 0 for m in state.cfg.macros}
            for p in solution_parses(state, state.cfg):
                for t in p:
                    if t in use:
                        use[t] += 1
            order = sorted(state.cfg.macros,
                           key=lambda m: (use[m], -state.cfg.macros.index(m)))
            dead = [m for m in order if use[m] == 0]
            opts = []
            if dead:
                opts.append(("PRUNE[dead]", prune_cfg(state.cfg, dead)))
            opts.append(("PRUNE[1]", prune_cfg(state.cfg, order[:1])))
            if len(order) >= 2:
                opts.append(("PRUNE[2]", prune_cfg(state.cfg, order[:2])))
            raw[a] = [(lb, c, []) for lb, c in opts][:n]
    pool, shas = [], set()
    for a in ACTIONS:
        items = raw.get(a, [])
        for j, (lb, c, sups) in enumerate(items):
            if c is None or c.sha() in shas or c.sha() == state.cfg.sha():
                continue
            shas.add(c.sha())
            h = 1.0 - j / float(max(1, len(items)))
            pool.append(Cand(a, lb, c, deltas(state, c, sups), h))
    return pool, alloc, nd


def _dist(cfg):
    toks = cfg.tokens()
    tot = sum(max(cfg.weights.get(t, 0.0), 0.0) for t in toks)
    return {t: cfg.weights.get(t, 0.0) / tot for t in toks}


def deltas(state, c, sups):
    inc = state.cfg
    added = [m for m in c.macros if m not in inc.macros]
    removed = [m for m in inc.macros if m not in c.macros]
    lens = [len(SV.expansion(m)) for m in added]
    depth = max([macro_depth(m, c) for m in added] or [0])
    p0, p1 = _dist(inc), _dist(c)
    keys = sorted(set(p0) | set(p1))
    shift = 0.5 * sum(abs(p1.get(k, 0.0) - p0.get(k, 0.0)) for k in keys)
    cover = 0.0
    if added and state.resid_log:
        exps = [SV.expansion(m) for m in added]
        hit = sum(1 for _i, fr in state.resid_log
                  if any(_contains(fr, e) or _contains(e, fr)
                         for e in exps))
        cover = hit / float(len(state.resid_log))
    return [len(added) / 5.0, len(removed) / 5.0,
            (sum(lens) / len(lens) / 8.0) if lens else 0.0,
            (sum(sups) / len(sups) / 5.0) if sups else 0.0,
            min(1.0, depth / 3.0), shift,
            (c.p_random - inc.p_random) / 0.3,
            entropy_ratio(c) - entropy_ratio(inc), cover,
            float(c.max_tokens - inc.max_tokens)]


def features(cand, v, diag_on, probe_inc):
    """[bias | action one-hot | candidate deltas | generator preference |
    action x its matched failure score | incumbent progress on the probe].
    Compact on purpose: offline leave-seeds-out CV on dev data showed the
    full action x diagnosis block overfits (ledger DEV_NOTE)."""
    x = [1.0]
    x += [1.0 if cand.action == a else 0.0 for a in ACTIONS]
    x += list(cand.delta)
    x.append(cand.h)
    x += [(v[MATCHED[a]] if (cand.action == a and diag_on) else 0.0)
          for a in ACTIONS]
    x.append(probe_inc)
    return x


# --------------------------------------------------------------------------
# counterfactual tests

class GateExhausted(Exception):
    pass


def progress(solved, best_fit, hp):
    return 1.0 if solved else hp["partial_weight"] * max(0.0, best_fit)


def gate_eval(cfg, probes, seed, meter, hp):
    """probes: [(b_idx, j, task)]. Per probe: (solved, progress, cost)."""
    comp = SV.Compiled(cfg)
    per = []
    for b_idx, j, task in probes:
        if meter.left() <= 0:
            raise GateExhausted()
        res = SV.search(comp, task.public(), hp["gate_probe_budget"],
                        stream(seed, "gate", b_idx, j), meter)
        if res.parent_exhausted:
            raise GateExhausted()
        try:
            ok = res.program is not None and SV.check(comp, res.program,
                                                      task.test, meter)
        except S.BudgetExceeded:
            raise GateExhausted()
        per.append((1 if ok else 0, progress(ok, res.best_fit, hp),
                    res.spent if ok else hp["gate_probe_budget"]))
    return per


def tot(per):
    return (sum(p[0] for p in per), sum(p[2] for p in per),
            sum(p[1] for p in per))


def posterior(ys, mu, hp):
    """Normal-normal posterior over a candidate's true paired gain per
    probe: prior N(mu, prior_var), each paired probe y ~ N(g, noise_var)."""
    prec = 1.0 / hp["prior_var"] + len(ys) / hp["noise_var"]
    mean = (mu / hp["prior_var"] + sum(ys) / hp["noise_var"]) / prec
    return mean, math.sqrt(1.0 / prec)


def p_positive(mean, sd):
    return 0.5 * (1.0 + math.erf(mean / (sd * math.sqrt(2.0))))


def better(c, i, hp):
    return c[0] > i[0] or (c[0] == i[0] and c[1] <= hp["adopt_cost_ratio"]
                           * i[1])


# --------------------------------------------------------------------------
# ranking policies

def rank(pool, policy, model, v, diag_on, inc_mean, nd, prng, k):
    perm = prng.shuffled(range(len(pool)))
    pos = {c: i for i, c in enumerate(perm)}
    idx = list(range(len(pool)))
    if policy == "FROZEN":
        return sorted(idx, key=lambda i: pos[i])[:k]
    if policy == "HEURISTIC":
        return sorted(idx, key=lambda i: (-pool[i].h,
                                          -nd.get(pool[i].action, 0.0),
                                          pos[i]))[:k]
    for c in pool:
        c.x = features(c, v, diag_on, inc_mean)
        c.mu = model.predict(c.x)
    greedy = sorted(idx, key=lambda i: (-pool[i].mu, pos[i]))[:max(0, k - 1)]
    rest = [i for i in sorted(idx, key=lambda i: pos[i]) if i not in greedy]
    return greedy + rest[:1]


# --------------------------------------------------------------------------
# arms

def run_recursive(seed, train, batches, arm_meter, hp, policy, diag_on,
                  learn, log, model):
    state = State()
    half = T.METAVAL_BATCH // 2
    for r in range(1, hp["n_rounds"] + 1):
        att = arm_meter.child(len(train) * hp["attempt_budget"], "attempt")
        comp = SV.Compiled(state.cfg)
        traces = [attempt(state, comp, task, i, hp["attempt_budget"],
                          stream(seed, "attempt", r, i), att)
                  for i, task in enumerate(train) if i not in state.solved]
        v, vd, info = diagnose(state, traces, r, hp)
        gate = arm_meter.child(hp["gate_round_budget"], "gate")
        rec = {"round": r, "attempted": len(traces),
               "newly_solved": sum(1 for t in traces if t.solved),
               "diag": vd, "screened": [], "adopted": None}
        try:
            pool, alloc, nd = generate_pool(state, traces, train, v, info,
                                            gate, hp, diag_on,
                                            hp["pool_size"])
        except S.BudgetExceeded:
            pool, alloc, nd = [], {}, {}
        rec["train_solved"] = len(state.solved)   # incl. residual repairs
        rec["alloc"] = alloc
        rec["pool"] = [c.label for c in pool]
        batch = [(r, j, t) for j, t in enumerate(batches[r - 1])]
        screen, confirm = batch[:half], batch[half:]
        try:
            inc_s = gate_eval(state.cfg, screen, seed, gate, hp)
            inc_c = gate_eval(state.cfg, confirm, seed, gate, hp)
        except GateExhausted:
            inc_s = inc_c = None
        rows, results = [], []
        if inc_s is not None and pool:
            inc_mean = sum(p[1] for p in inc_s) / len(inc_s)
            if policy == "ORACLE":   # DIAGNOSTIC ONLY (not compute-matched)
                orc = arm_meter.child(10 ** 9, "oracle")
                ys = []
                for i, c in enumerate(pool):
                    try:
                        per = gate_eval(c.cfg, screen, seed, orc, hp)
                    except GateExhausted:
                        break
                    ys.append((-sum(pc[1] - pi[1] for pc, pi in
                                    zip(per, inc_s)), i))
                picks = [i for _y, i in sorted(ys)][:hp["k_screen"]]
            else:
                picks = rank(pool, policy, model, v, diag_on, inc_mean, nd,
                             stream(seed, "select", r), hp["k_screen"])
            for i in picks:
                c = pool[i]
                if c.x is None:
                    c.x = features(c, v, diag_on, inc_mean)
                try:
                    per = gate_eval(c.cfg, screen, seed, gate, hp)
                except GateExhausted:
                    rec["screened"].append({"label": c.label,
                                            "incomplete": True})
                    break
                results.append([c, per, None])
                y = [pc[1] - pi[1] for pc, pi in zip(per, inc_s)]
                for (pc, pi) in zip(per, inc_s):
                    rows.append((features(c, v, diag_on, pi[1]),
                                 pc[1] - pi[1]))
                rec["screened"].append({
                    "label": c.label, "action": c.action,
                    "mu": round(c.mu, 5), "y_mean": round(sum(y) / len(y), 5),
                    "solved": tot(per)[0],
                    "x": [round(z, 4) for z in c.x]})
            # ONE adoption gate for every arm (the solve-count gate the v2
            # confirmatory battery used): a finalist must not solve fewer
            # screen probes, must not solve fewer FRESH confirmation probes,
            # and must be strictly better over the whole batch.
            live = [t for t in results if tot(t[1])[0] >= tot(inc_s)[0]]
            if live:
                if policy == "LEARN" and hp["finalist"] == "posterior":
                    # learned meta-knowledge DENOISES the finalist choice:
                    # posterior mean of the paired gain, prior mean = the
                    # predictor, noise/prior variances from dev reliability
                    def score(t):
                        ys = [pc[1] - pi[1] for pc, pi in zip(t[1], inc_s)]
                        return (posterior(ys, t[0].mu, hp)[0],
                                tot(t[1])[0], -tot(t[1])[1])
                else:
                    def score(t):
                        return (tot(t[1])[0], tot(t[1])[2], -tot(t[1])[1])
                fin = max(live, key=score)
                try:
                    fin[2] = gate_eval(fin[0].cfg, confirm, seed, gate, hp)
                except GateExhausted:
                    fin[2] = None
                if fin[2] is not None:
                    for pc, pi in zip(fin[2], inc_c):
                        rows.append((features(fin[0], v, diag_on, pi[1]),
                                     pc[1] - pi[1]))
                    tc = (tot(fin[1])[0] + tot(fin[2])[0],
                          tot(fin[1])[1] + tot(fin[2])[1])
                    ti = (tot(inc_s)[0] + tot(inc_c)[0],
                          tot(inc_s)[1] + tot(inc_c)[1])
                    rec["confirm"] = {"label": fin[0].label,
                                      "cand": list(tc), "inc": list(ti)}
                    if hp["gate"] == "strict":
                        ok = (tot(fin[2])[0] >= tot(inc_c)[0]
                              and better(tc, ti, hp))
                    elif hp["gate"] == "noninferior":
                        ok = (tot(fin[2])[0] >= tot(inc_c)[0]
                              and tc[0] >= ti[0])
                    else:   # "always": adopt the screening winner
                        ok = True
                    if ok:
                        rec["adopted"] = fin[0].label
                        rec["adopted_action"] = fin[0].action
                        state.cfg = fin[0].cfg
        if inc_s is not None:
            state.inc_rates.append((tot(inc_s)[0] + tot(inc_c)[0])
                                   / float(len(batch)))
        if learn and rows:
            for x, y in rows:
                model.add(x, y)
            model.refit()
        rec["n_rows"] = len(rows)
        rec["model_n"] = model.n
        rec["n_macros"] = len(state.cfg.macros)
        rec["macros"] = list(state.cfg.macros)
        rec["attempt_spent"] = att.spent
        rec["gate_spent"] = gate.spent
        log.append(rec)
    return state, model


def run_single(seed, train, batches, arm_meter, hp, log):
    state = State()
    R = hp["n_rounds"]
    att = arm_meter.child(R * len(train) * hp["attempt_budget"], "attempt")
    comp = SV.Compiled(state.cfg)
    traces = []
    for i, task in enumerate(train):
        tr = None
        for r in range(1, R + 1):
            tr = attempt(state, comp, task, i, hp["attempt_budget"],
                         stream(seed, "attempt", r, i), att)
            if tr.solved:
                break
        traces.append(tr)
    v, vd, info = diagnose(state, traces, 1, hp)
    gate = arm_meter.child(R * hp["gate_round_budget"], "gate")
    try:
        pool, alloc, _nd = generate_pool(state, traces, train, v, info, gate,
                                         hp, True, hp["single_pool_size"])
    except S.BudgetExceeded:
        pool, alloc = [], {}
    half = T.METAVAL_BATCH // 2
    flat = [(b, j, t) for b, bt in enumerate(batches, start=1)
            for j, t in enumerate(bt)]
    stages = [(flat[:half], 4), (flat[half:3 * half], 1), (flat[3 * half:], 0)]
    alive = list(range(len(pool)))
    cum = {i: [0, 0, 0.0] for i in alive}
    inc_cum = [0, 0, 0.0]
    conf = None
    rec = {"round": 1, "diag": vd, "alloc": alloc,
           "pool": [c.label for c in pool], "stages": []}
    for probes, keep in stages:
        if not alive:
            break
        try:
            inc = gate_eval(state.cfg, probes, seed, gate, hp)
            res = {i: gate_eval(pool[i].cfg, probes, seed, gate, hp)
                   for i in alive}
        except GateExhausted:
            rec["stages"].append({"exhausted": True})
            break
        ti = tot(inc)
        inc_cum = [inc_cum[0] + ti[0], inc_cum[1] + ti[1], inc_cum[2] + ti[2]]
        for i in alive:
            t = tot(res[i])
            cum[i] = [cum[i][0] + t[0], cum[i][1] + t[1], cum[i][2] + t[2]]
        rec["stages"].append({"n_probes": len(probes),
                              "alive": [pool[i].label for i in alive],
                              "inc": list(ti)})
        if keep == 0:
            conf = (tot(res[alive[0]]), ti)
            break
        alive.sort(key=lambda i: (-cum[i][0], -cum[i][2], cum[i][1], i))
        alive = alive[:keep]
    rec["adopted"] = None
    if conf is not None:
        b = alive[0]
        if conf[0][0] >= conf[1][0] and better(cum[b], inc_cum, hp):
            rec["adopted"] = pool[b].label
            state.cfg = pool[b].cfg
    rec.update({"train_solved": len(state.solved),
                "n_macros": len(state.cfg.macros),
                "macros": list(state.cfg.macros),
                "attempt_spent": att.spent, "gate_spent": gate.spent})
    log.append(rec)
    return state


def run_arm(arm, seed, hp=None):
    """Runs the arm over n_worlds independent worlds. Returns
    ([final cfg per world], record). Every world has its OWN child meter
    with an identical cap, so no arm can shift compute between worlds."""
    hp = hp or HP
    assert arm in ARMS or arm in DIAGNOSTIC_ARMS, arm
    views = [T.improver_view(world_key(seed, w), hp["n_rounds"])
             for w in range(1, hp["n_worlds"] + 1)]
    wcap = world_cap(hp, len(views[0][0]))
    cap = hp["n_worlds"] * wcap
    if arm in DIAGNOSTIC_ARMS:
        wcap += 10 ** 9
        cap += hp["n_worlds"] * 10 ** 9
    arm_meter = S.Meter(cap, "arm")
    g0 = S.EXEC.n
    cfgs, worlds = [], []
    model = None
    if arm not in ("COLD", "SINGLE_COMPUTE_MATCHED"):
        model = Ridge(feat_dim(), hp["ridge_lam"])
        model.frozen = arm not in LEARNING_ARMS
    for w in range(1, hp["n_worlds"] + 1):
        wseed = world_key(seed, w)
        train, batches = views[w - 1]
        wm = arm_meter.child(wcap, "world")
        log = []
        if arm == "COLD":
            state = State()
        elif arm == "SINGLE_COMPUTE_MATCHED":
            state = run_single(wseed, train, batches, wm, hp, log)
        else:
            if arm == "ADAPTIVE_NOCARRY":
                model = Ridge(feat_dim(), hp["ridge_lam"])
            policy = {"FROZEN_META": "FROZEN", "ADAPTIVE_META": "LEARN",
                      "NODIAG_META": "LEARN", "HEURISTIC_META": "HEURISTIC",
                      "ADAPTIVE_NOCARRY": "LEARN",
                      "ORACLE_RANK": "ORACLE"}[arm]
            state, model = run_recursive(
                wseed, train, batches, wm, hp, policy,
                diag_on=arm != "NODIAG_META", learn=arm in LEARNING_ARMS,
                log=log, model=model)
        cfgs.append(state.cfg)
        worlds.append({"world": w, "train_solved": len(state.solved),
                       "n_macros": len(state.cfg.macros),
                       "macros": list(state.cfg.macros),
                       "spent": wm.spent, "rounds": log,
                       "final_cfg_sha": state.cfg.sha()})
    delta = S.EXEC.n - g0
    if delta != arm_meter.spent:
        raise S.HiddenComputeError("%s: %d executions, %d metered"
                                   % (arm, delta, arm_meter.spent))
    assert arm_meter.spent <= cap
    return cfgs, {
        "arm": arm, "seed": seed, "cap": cap, "world_cap": wcap,
        "spent": arm_meter.spent, "global_delta": delta, "worlds": worlds,
        "model_w": [round(x, 6) for x in model.w] if model else None,
        "model_n": model.n if model else 0}
