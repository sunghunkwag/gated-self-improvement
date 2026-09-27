"""
rsi_v2.improver -- the ADAPTIVE IMPROVER (the qualitative upgrade).

v1 improved the SOLVER (a fixed prior-refit rule applied every round).
v2 also improves the IMPROVER: each round runs

    attempt -> diagnose failure -> choose improvement strategy ->
    build candidate solver -> counterfactual test (paired streams) ->
    adopt / rollback -> learn from the outcome -> repeat

and the "learn" step modifies the improvement process itself:

  * a contextual strategy-value table Q[failure_mode][strategy] (partially
    pooled toward a context-free estimate) that decides WHICH improvement
    strategies get the round's scarce counterfactual-test budget;
  * per-strategy intensities theta (how many macros to mine / compose, how
    big a prior step, how much to flatten, how much to prune), adapted by
    a success rule (adopted -> bolder, rejected -> more conservative).

Failure modes (diagnosed from the round's own search traces, never from
holdout data):
  MISSING_OP    unsolved searches pushing the length bound with partial
                progress -> the vocabulary lacks an operator
  BAD_ORDER     solves happen late in the budget -> the prior misorders
  LOW_EXPLORE   unsolved searches with no traction / collapsed population
  OVERSPEC      dead macros and a concentrated prior -> over-specialised
  COMPOSE_FAIL  near-misses that already chain >= 2 macros -> composition

Improvement strategies (each returns a candidate SolverConfig):
  MINE     frequent contiguous fragments of solved programs -> new macros
  COMPOSE  adjacent macro pairs in solutions / near-misses -> macro-of-
           macros (hierarchical reuse of previously discovered operators)
  REFIT    trust-region prior refit toward token presence in solutions
  EXPLORE  flatten the prior, raise the random-restart rate
  PRUNE    drop the least-used macros, relax the prior

Arms (identical code path; only the flags differ):
  COLD              no improvement at all (reference)
  SINGLE_5X         ONE round with 5x attempt and 5x gate compute: every
                    task gets the same 5 attempt streams the recursive arms
                    get (restarts), every strategy + two stacked combos are
                    tested by successive halving over all 5 probe batches
  RECURSIVE_FROZEN  5 rounds; improvement policy (Q, theta) FROZEN at init
  RECURSIVE_FULL    5 rounds; Q and theta adapt from outcomes
  RECURSIVE_NODIAG  (ablation) adaptive but context-free: ignores diagnosis

Compute: every arm gets the SAME cap
    n_rounds * (n_train * ATTEMPT_BUDGET + GATE_ROUND_BUDGET)
executions, enforced by a hierarchical Meter and cross-checked against the
process-global execution counter (hidden-compute detector).
"""
import math

from . import substrate as S
from . import solver as SV
from . import tasks as T

VERSION = "rsi_v2-improver-1"

MODES = ("MISSING_OP", "BAD_ORDER", "LOW_EXPLORE", "OVERSPEC",
         "COMPOSE_FAIL")
STRATEGIES = ("MINE", "COMPOSE", "REFIT", "EXPLORE", "PRUNE")
COMBOS = ("MINE+REFIT", "MINE+COMPOSE+REFIT", "COMPOSE+REFIT", "PRUNE+MINE")
ACTIONS = STRATEGIES + COMBOS
ARMS = ("COLD", "SINGLE_5X", "RECURSIVE_FROZEN", "RECURSIVE_FULL",
        "RECURSIVE_NODIAG")

HP = {
    "n_rounds": 5,
    "attempt_budget": 2000,       # executions per train-task attempt
    "gate_probe_budget": 1000,    # executions per probe search in a gate
    "gate_round_budget": 42000,   # executions per round for gating
    "k_candidates": 4,            # candidates the policy may SCREEN / round
    "theta0": {"MINE": 3.0, "COMPOSE": 2.0, "REFIT": 0.4, "EXPLORE": 0.3,
               "PRUNE": 0.3},
    "theta_lo": {"MINE": 1.0, "COMPOSE": 1.0, "REFIT": 0.1, "EXPLORE": 0.1,
                 "PRUNE": 0.1},
    "theta_hi": {"MINE": 8.0, "COMPOSE": 6.0, "REFIT": 0.9, "EXPLORE": 0.7,
                 "PRUNE": 0.7},
    "theta_up": 1.5, "theta_down": 0.7,
    "q_alpha": 0.5, "pool_kappa": 2.0, "ucb_c": 0.3,
    "adopt_cost_ratio": 0.9,
    "mine_min_len": 2, "mine_max_len": 6, "mine_min_support": 2,
    "macro_len_max": 12, "refit_strength": 8.0, "prior_floor": 0.08,
    "new_macro_boost": 2.0, "near_miss_fit": 0.5,
    "p_random_max": 0.6,
    "single_bold": 3.0,
    "single_bold_actions": ["MINE", "MINE+REFIT", "MINE+COMPOSE+REFIT"],
}


def total_cap(hp, n_train):
    return hp["n_rounds"] * (n_train * hp["attempt_budget"]
                             + hp["gate_round_budget"])


def stream(seed, *parts):
    """All improvement-phase randomness: keyed by seed + purpose + indices.
    No arm name ever enters a key (common random numbers across arms)."""
    return S.XorShift64Star("rsi_v2|" + "|".join(str(p) for p in
                                                 (seed,) + parts))


# --------------------------------------------------------------------------
# helpers on configs / solutions

def shortest_parse(seq, cfg):
    """Fewest-token re-expression of a primitive sequence with cfg's
    vocabulary (syntactic; no execution)."""
    n = len(seq)
    exps = [(t, SV.expansion(t)) for t in cfg.macros]
    best = [None] * (n + 1)
    best[0] = ()
    for i in range(n):
        if best[i] is None:
            continue
        cands = [(seq[i], (seq[i],))] + exps
        for t, e in cands:
            j = i + len(e)
            if j <= n and tuple(seq[i:j]) == e:
                c = best[i] + (t,)
                if best[j] is None or len(c) < len(best[j]) or (
                        len(c) == len(best[j]) and c < best[j]):
                    best[j] = c
    return best[n]


def entropy_ratio(cfg):
    ws = [max(cfg.weights.get(t, 0.0), 1e-12) for t in cfg.tokens()]
    tot = sum(ws)
    h = -sum((w / tot) * math.log(w / tot) for w in ws)
    return h / math.log(len(ws))


class ImproverState(object):
    def __init__(self, hp):
        self.cfg = SV.base_config()
        self.solved = {}        # train idx -> expanded primitive tuple
        self.sigs = {}          # token -> behaviour signature (charged)
        self.frag_sigs = {}     # primitive fragment -> signature (charged)
        self.theta = dict(hp["theta0"])
        self.q = {}             # (ctx, strategy) -> value
        self.n = {}             # (ctx, strategy) -> count
        self.qg = {}            # strategy -> pooled value
        self.ng = {}            # strategy -> pooled count
        self.last_probe = None
        self.history = []


def token_sig(state, tok, meter):
    if tok not in state.sigs:
        state.sigs[tok] = S.signature(S.fns_of(SV.expansion(tok)), meter)
    return state.sigs[tok]


def known_sigs(state, cfg, meter):
    if "" not in state.sigs:
        state.sigs[""] = S.signature((), meter)
    out = {state.sigs[""]}
    for t in cfg.tokens():
        out.add(token_sig(state, t, meter))
    return out


# --------------------------------------------------------------------------
# attempt + diagnose

class Trace(object):
    __slots__ = ("idx", "solved", "spent", "best_fit", "best_prog",
                 "diversity")

    def __init__(self, idx, solved, spent, best_fit, best_prog, diversity):
        self.idx, self.solved, self.spent = idx, solved, spent
        self.best_fit, self.best_prog = best_fit, best_prog
        self.diversity = diversity


def _solves(prims, examples, meter):
    fns = S.fns_of(prims)
    for x, y in examples:
        if meter.run(fns, list(x)) != list(y):
            return False
    return True


def simplify(prims, examples, meter, max_passes=3):
    """Occam step (charged): shrink a solution to a shorter program that
    still fits every train+test example, by deleting one primitive or
    replacing two adjacent primitives with one. Canonical short solutions
    make fragment mining align across tasks."""
    cur = tuple(prims)
    for _ in range(max_passes):
        changed = False
        for i in range(len(cur)):
            c = cur[:i] + cur[i + 1:]
            if c and _solves(c, examples, meter):
                cur, changed = c, True
                break
        if changed:
            continue
        for i in range(len(cur) - 1):
            for p in S.NAMES:
                c = cur[:i] + (p,) + cur[i + 2:]
                if _solves(c, examples, meter):
                    cur, changed = c, True
                    break
            if changed:
                break
        if not changed:
            break
    return cur


def attempt_task(state, comp, task, idx, budget, prng, meter):
    res = SV.search(comp, task.public(), budget, prng, meter)
    ok = res.program is not None and SV.check(comp, res.program,
                                              task.test, meter)
    if ok:
        prims = ()
        for t in res.program:
            prims += SV.expansion(t)
        try:
            prims = simplify(prims, task.train + task.test, meter)
        except S.BudgetExceeded:
            pass
        state.solved.setdefault(idx, prims)
    return Trace(idx, ok, res.spent, res.best_fit, res.best_prog,
                 res.diversity)


def diagnose(state, traces, budget, hp):
    cfg = state.cfg
    uns = [t for t in traces if not t.solved]
    sol = [t for t in traces if t.solved]
    nu = float(max(1, len(uns)))
    missing = sum(1 for t in uns if t.best_prog is not None
                  and len(t.best_prog) >= cfg.max_tokens - 1
                  and t.best_fit >= 0.3) / nu
    order = (sum(t.spent for t in sol) / (len(sol) * float(budget))
             if sol else 0.0)
    explore = sum(1 for t in uns if t.best_fit < 0.3
                  or t.diversity < 0.5) / nu
    if cfg.macros:
        used = set()
        for prims in state.solved.values():
            used.update(shortest_parse(prims, cfg))
        dead = sum(1 for m in cfg.macros if m not in used) / float(
            len(cfg.macros))
    else:
        dead = 0.0
    overspec = 0.5 * dead + 0.5 * min(1.0, 2.0 * (1.0 - entropy_ratio(cfg)))
    compose = sum(1 for t in uns if t.best_prog is not None
                  and sum(1 for x in t.best_prog if SV.is_macro(x)) >= 2
                  and t.best_fit >= hp["near_miss_fit"]) / nu
    scores = {"MISSING_OP": missing, "BAD_ORDER": order,
              "LOW_EXPLORE": explore, "OVERSPEC": overspec,
              "COMPOSE_FAIL": compose}
    mode = max(MODES, key=lambda m: (scores[m], -MODES.index(m)))
    return mode, {k: round(v, 4) for k, v in scores.items()}


# --------------------------------------------------------------------------
# improvement strategies (each: state, traces, theta, meter -> cfg | None)

def _add_macros(cfg, new, hp):
    if not new:
        return None
    c = cfg.copy()
    mean_w = sum(c.weights[t] for t in c.tokens()) / len(c.tokens())
    for prims in new:
        tok = SV.macro_token(prims)
        c.macros = c.macros + (tok,)
        c.weights[tok] = hp["new_macro_boost"] * mean_w
    return c


def frag_sig(state, fr, meter):
    """Behaviour signature of a primitive fragment (cached, charged)."""
    sg = state.frag_sigs.get(fr)
    if sg is None:
        sg = S.signature(S.fns_of(fr), meter)
        state.frag_sigs[fr] = sg
    return sg


def strat_mine(state, traces, theta, meter, hp):
    """Behaviour-level fragment mining: contiguous fragments of solved
    programs are grouped by BEHAVIOUR (observational equivalence on the
    probe inputs), so syntactically different but equivalent pieces of
    different solutions count as the same reusable operator. Each
    behaviour is represented by its shortest fragment."""
    cfg = state.cfg
    k = max(1, int(round(theta)))
    ks = known_sigs(state, cfg, meter)
    support, rep, whole = {}, {}, set()
    for idx in sorted(state.solved):
        seq = state.solved[idx]
        whole.add(frag_sig(state, tuple(seq), meter))
        seen = set()
        for L in range(hp["mine_min_len"], hp["mine_max_len"] + 1):
            for st in range(0, len(seq) - L + 1):
                fr = tuple(seq[st:st + L])
                sg = frag_sig(state, fr, meter)
                if sg in ks:
                    continue
                seen.add(sg)
                if sg not in rep or (len(fr), fr) < (len(rep[sg]),
                                                     rep[sg]):
                    rep[sg] = fr
        for sg in seen:
            support[sg] = support.get(sg, 0) + 1
    cands = []
    for sg, sp in support.items():
        fr = rep[sg]
        if sp >= hp["mine_min_support"]:
            cands.append(((1, sp * (len(fr) - 1)), len(fr), fr, sg))
        elif sg in whole:   # a whole solved behaviour, reused as operator
            cands.append(((0, len(fr) - 1), len(fr), fr, sg))
    cands.sort(key=lambda t: (-t[0][0], -t[0][1], -t[1], t[2]))
    chosen, chosen_sigs = [], set()
    for _score, _L, fr, sg in cands:
        if len(chosen) >= k:
            break
        if sg in chosen_sigs:
            continue
        state.sigs[SV.macro_token(fr)] = sg
        chosen.append(fr)
        chosen_sigs.add(sg)
    return _add_macros(cfg, chosen, hp)


def _contains(big, small):
    n, m = len(big), len(small)
    return any(tuple(big[i:i + m]) == small for i in range(n - m + 1))


def strat_compose(state, traces, theta, meter, hp):
    cfg = state.cfg
    k = max(1, int(round(theta)))
    sources = []
    for idx in sorted(state.solved):
        sources.append(shortest_parse(state.solved[idx], cfg))
    for t in traces:
        if (not t.solved and t.best_prog is not None
                and t.best_fit >= hp["near_miss_fit"]):
            sources.append(tuple(t.best_prog))
    has_macro = bool(cfg.macros)
    support = {}
    for src in sources:
        seen = set()
        for a, b in zip(src, src[1:]):
            if has_macro and not (SV.is_macro(a) or SV.is_macro(b)):
                continue
            seen.add((a, b))
        for pr in seen:
            support[pr] = support.get(pr, 0) + 1
    existing = {SV.expansion(t) for t in cfg.macros}
    cands = []
    for (a, b), sp in support.items():
        if sp < 2:
            continue
        fr = SV.expansion(a) + SV.expansion(b)
        if len(fr) > hp["macro_len_max"] or fr in existing:
            continue
        cands.append((sp * (len(fr) - 1), len(fr), fr))
    cands.sort(key=lambda t: (-t[0], -t[1], t[2]))
    ks = known_sigs(state, cfg, meter)
    chosen, chosen_sigs = [], set()
    for _s, _L, fr in cands:
        if len(chosen) >= k:
            break
        sg = S.signature(S.fns_of(fr), meter)
        if sg in ks or sg in chosen_sigs:
            continue
        state.sigs[SV.macro_token(fr)] = sg
        chosen.append(fr)
        chosen_sigs.add(sg)
    return _add_macros(cfg, chosen, hp)


def strat_refit(state, traces, theta, meter, hp):
    cfg = state.cfg
    if not state.solved:
        return None
    parses = [set(shortest_parse(state.solved[i], cfg))
              for i in sorted(state.solved)]
    n = float(len(parses))
    c = cfg.copy()
    for t in cfg.tokens():
        rate = sum(1 for p in parses if t in p) / n
        target = 1.0 + hp["refit_strength"] * rate
        c.weights[t] = (1.0 - theta) * cfg.weights[t] + theta * target
    hi = max(c.weights.values())
    for t in c.tokens():
        c.weights[t] = max(c.weights[t], hp["prior_floor"] * hi)
    return c


def strat_explore(state, traces, theta, meter, hp):
    cfg = state.cfg
    c = cfg.copy()
    mean_w = sum(cfg.weights[t] for t in cfg.tokens()) / len(cfg.tokens())
    for t in cfg.tokens():
        c.weights[t] = (1.0 - theta) * cfg.weights[t] + theta * mean_w
    c.p_random = min(hp["p_random_max"], cfg.p_random + theta / 2.0)
    return c


def strat_prune(state, traces, theta, meter, hp):
    cfg = state.cfg
    if not cfg.macros:
        return None
    use = {m: 0 for m in cfg.macros}
    for idx in sorted(state.solved):
        for t in shortest_parse(state.solved[idx], cfg):
            if t in use:
                use[t] += 1
    n_drop = max(1, int(math.ceil(theta * len(cfg.macros))))
    order = sorted(cfg.macros, key=lambda m: (use[m], -cfg.macros.index(m)))
    drop = set(order[:n_drop])
    c = cfg.copy()
    c.macros = tuple(m for m in cfg.macros if m not in drop)
    for m in drop:
        del c.weights[m]
    mean_w = sum(c.weights[t] for t in c.tokens()) / len(c.tokens())
    for t in c.tokens():
        c.weights[t] = 0.9 * c.weights[t] + 0.1 * mean_w
    c.p_random = (c.p_random + SV.BASE_P_RANDOM) / 2.0
    return c


STRATEGY_FNS = {"MINE": strat_mine, "COMPOSE": strat_compose,
                "REFIT": strat_refit, "EXPLORE": strat_explore,
                "PRUNE": strat_prune}


def propose(state, name, traces, theta, meter, hp):
    """name may be a stacked combo 'A+B+C' (applied sequentially)."""
    saved = state.cfg
    cfg = None
    try:
        for part in name.split("+"):
            nxt = STRATEGY_FNS[part](state, traces, theta[part], meter, hp)
            if nxt is not None:
                cfg = nxt
                state.cfg = nxt
    finally:
        state.cfg = saved
    return cfg


# --------------------------------------------------------------------------
# counterfactual test (paired streams: incumbent and every candidate search
# probe j of batch b with the SAME stream, identical across arms)

class GateExhausted(Exception):
    pass


def gate_eval(cfg, batch, b_idx, seed, meter, hp, race_vs=None, offset=0):
    """Score a config on one probe batch: (solved, cost, per_probe, raced).
    With race_vs=(inc_solved, inc_cost), stop as soon as the candidate can
    no longer beat the incumbent (racing; identical rule for every arm)."""
    comp = SV.Compiled(cfg)
    solved, cost, per = 0, 0, []
    n = len(batch)
    for j, task in enumerate(batch):
        if meter.left() <= 0:
            raise GateExhausted()
        res = SV.search(comp, task.public(), hp["gate_probe_budget"],
                        stream(seed, "gate", b_idx, j + offset), meter)
        if res.parent_exhausted:
            raise GateExhausted()
        try:
            ok = res.program is not None and SV.check(comp, res.program,
                                                      task.test, meter)
        except S.BudgetExceeded:
            raise GateExhausted()
        # an unsolved probe costs the full budget: finishing early with a
        # train-fitting program that fails the held-out check earns nothing
        cost += res.spent if ok else hp["gate_probe_budget"]
        solved += 1 if ok else 0
        per.append(1 if ok else 0)
        if race_vs is not None:
            rem = n - (j + 1)
            if solved + rem < race_vs[0] or (
                    solved + rem == race_vs[0]
                    and cost > hp["adopt_cost_ratio"] * race_vs[1]):
                return solved, cost, per, True
    return solved, cost, per, False


def gate_eval_flat(cfg, probes, seed, meter, hp):
    """gate_eval over (batch_idx, j, task) triples, same streams as the
    recursive arms' gates for the same probe."""
    comp = SV.Compiled(cfg)
    solved, cost, per = 0, 0, []
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
        cost += res.spent if ok else hp["gate_probe_budget"]
        solved += 1 if ok else 0
        per.append(1 if ok else 0)
    return solved, cost, per, False


def better(cand, inc, hp):
    cs, cc = cand[0], cand[1]
    is_, ic = inc[0], inc[1]
    return cs > is_ or (cs == is_ and cc <= hp["adopt_cost_ratio"] * ic)


def worse(cand, inc):
    return cand[0] < inc[0]


def reward(cand, inc, hp):
    """Paired outcome on the probes the candidate was actually run on."""
    per_c, per_i = cand[2], inc[2]
    n = float(len(per_i))
    d = sum(c - i for c, i in zip(per_c, per_i)) / n
    if not cand[3]:
        d += 0.25 * (inc[1] - cand[1]) / (n * hp["gate_probe_budget"])
    return d


# --------------------------------------------------------------------------
# the improvement policy (what RECURSIVE_FULL learns, FROZEN never changes)

def rank_strategies(state, ctx, prng, adaptive, hp):
    perm = prng.shuffled(ACTIONS)
    if not adaptive:
        return perm
    ntot = sum(state.ng.values())

    def est(s):
        if state.ng.get(s, 0) == 0:
            return float("inf")
        n_c = state.n.get((ctx, s), 0)
        q_c = state.q.get((ctx, s), 0.0)
        k = hp["pool_kappa"]
        v = (n_c * q_c + k * state.qg[s]) / (n_c + k)
        return v + hp["ucb_c"] * math.sqrt(math.log(1.0 + ntot)
                                           / state.ng[s])
    return sorted(perm, key=lambda s: -est(s))


def learn(state, ctx, s, r, adopted, hp):
    a = hp["q_alpha"]
    for key, tab, cnt in (((ctx, s), state.q, state.n),
                          (s, state.qg, state.ng)):
        tab[key] = tab.get(key, 0.0) + a * (r - tab.get(key, 0.0))
        cnt[key] = cnt.get(key, 0) + 1
    for part in s.split("+"):
        th = state.theta[part]
        th = th * hp["theta_up"] if adopted else (th * hp["theta_down"]
                                                  if r < 0 else th)
        state.theta[part] = min(hp["theta_hi"][part],
                                max(hp["theta_lo"][part], th))


# --------------------------------------------------------------------------
# arms

def run_recursive(seed, train, batches, arm_meter, hp, adaptive, contextual,
                  log):
    state = ImproverState(hp)
    pending = None     # (ctx, adopted action, new macro tokens) for credit
    for r in range(1, hp["n_rounds"] + 1):
        att = arm_meter.child(len(train) * hp["attempt_budget"], "attempt")
        comp = SV.Compiled(state.cfg)
        traces = []
        for i, task in enumerate(train):
            if i in state.solved:
                continue
            traces.append(attempt_task(state, comp, task, i,
                                       hp["attempt_budget"],
                                       stream(seed, "attempt", r, i), att))
        if adaptive and pending is not None:
            p_ctx, p_s, p_new = pending
            used = 0
            for t in traces:
                if t.solved and set(shortest_parse(state.solved[t.idx],
                                                   state.cfg)) & p_new:
                    used += 1
            credit = used / float(max(1, len(traces)))
            learn(state, p_ctx, p_s, credit, credit > 0, hp)
            rec_credit = {"action": p_s, "new_solves_using_it": used}
        else:
            rec_credit = None
        pending = None
        mode, scores = diagnose(state, traces, hp["attempt_budget"], hp)
        ctx = mode if contextual else "ANY"
        order = rank_strategies(state, ctx, stream(seed, "select", r),
                                adaptive, hp)
        gate = arm_meter.child(hp["gate_round_budget"], "gate")
        batch = batches[r - 1]
        rec = {"round": r, "attempted": len(traces),
               "newly_solved": sum(1 for t in traces if t.solved),
               "train_solved": len(state.solved), "mode": mode,
               "scores": scores, "order": list(order), "tested": [],
               "theta_before": dict(state.theta)}
        adopted, results = None, []
        rec_macros_before = tuple(state.cfg.macros)
        half = len(batch) // 2
        screen, confirm = batch[:half], batch[half:]
        try:   # incumbent on the whole batch (screen + confirm halves)
            inc_s = gate_eval(state.cfg, screen, r, seed, gate, hp)
            inc_c = gate_eval(state.cfg, confirm, r, seed, gate, hp,
                              offset=half)
        except GateExhausted:
            inc_s = inc_c = None
        if inc_s is not None:
            rec["incumbent"] = [inc_s[0], inc_s[1], inc_c[0], inc_c[1]]
            seen_cfgs = set()
            for s in order:          # SCREEN: policy picks <= k candidates
                if len(results) >= hp["k_candidates"]:
                    break
                try:
                    cand = propose(state, s, traces, state.theta, gate, hp)
                except S.BudgetExceeded:
                    break
                if cand is None or cand.sha() in seen_cfgs:
                    rec["tested"].append({"s": s, "noop": True})
                    continue
                seen_cfgs.add(cand.sha())
                try:
                    sc = gate_eval(cand, screen, r, seed, gate, hp)
                except GateExhausted:
                    rec["tested"].append({"s": s, "incomplete": True})
                    break
                results.append([s, cand, sc, None])
                rec["tested"].append({"s": s, "screen": [sc[0], sc[1]],
                                      "n_macros": len(cand.macros)})
            live = [t for t in results if not worse(t[2], inc_s)]
            if live:             # CONFIRM the screening winner on fresh
                fin = max(live, key=lambda t: (t[2][0], -t[2][1]))
                try:
                    fin[3] = gate_eval(fin[1], confirm, r, seed, gate, hp,
                                       offset=half)
                except GateExhausted:
                    fin[3] = None
                if fin[3] is not None:
                    rec["confirm"] = {"s": fin[0],
                                      "score": [fin[3][0], fin[3][1]]}
                    tot_c = (fin[2][0] + fin[3][0], fin[2][1] + fin[3][1])
                    tot_i = (inc_s[0] + inc_c[0], inc_s[1] + inc_c[1])
                    # fresh confirmation probes must show no harm, and the
                    # whole batch must show a strict improvement
                    if not worse(fin[3], inc_c) and better(tot_c, tot_i, hp):
                        adopted = fin[0]
                        state.cfg = fin[1]
        if adaptive:
            for s, _c, sc, conf in results:
                rw = reward(sc, inc_s, hp)
                if conf is not None:
                    rw = 0.5 * (rw + reward(conf, inc_c, hp))
                learn(state, ctx, s, rw, s == adopted, hp)
        rec["delayed_credit"] = rec_credit
        rec["adopted"] = adopted
        if adopted is not None:
            new_toks = set(state.cfg.macros) - set(
                [m for m in rec_macros_before])
            pending = (ctx, adopted, new_toks) if new_toks else None
        rec["n_macros"] = len(state.cfg.macros)
        rec["macros"] = list(state.cfg.macros)
        rec["theta_after"] = dict(state.theta)
        rec["attempt_spent"] = att.spent
        rec["gate_spent"] = gate.spent
        log.append(rec)
    return state


def run_single(seed, train, batches, arm_meter, hp, log):
    """One improvement round with 5x compute (see module doc)."""
    state = ImproverState(hp)
    R = hp["n_rounds"]
    att = arm_meter.child(R * len(train) * hp["attempt_budget"], "attempt")
    comp = SV.Compiled(state.cfg)
    traces = []
    for i, task in enumerate(train):
        tr = None
        for r in range(1, R + 1):   # the recursive arms' exact streams
            tr = attempt_task(state, comp, task, i, hp["attempt_budget"],
                              stream(seed, "attempt", r, i), att)
            if tr.solved:
                break
        traces.append(tr)
    mode, scores = diagnose(state, traces, hp["attempt_budget"], hp)
    gate = arm_meter.child(R * hp["gate_round_budget"], "gate")
    # every action at the default intensity, plus BOLD (3x) variants of the
    # macro-adding actions: a one-shot improver must be able to make one
    # large change (it gets no later rounds to accumulate small ones)
    bold = dict(state.theta)
    for k in ("MINE", "COMPOSE"):
        bold[k] = state.theta[k] * hp["single_bold"]
    plan = [(nm, state.theta) for nm in ACTIONS] + [
        (nm + "@bold", bold) for nm in hp["single_bold_actions"]]
    names = [nm for nm, _th in plan]
    cands, seen_cfgs = [], set()
    for nm, th in plan:
        try:
            c = propose(state, nm.split("@")[0], traces, th, gate, hp)
        except S.BudgetExceeded:
            break
        if c is not None and c.sha() not in seen_cfgs:
            seen_cfgs.add(c.sha())
            cands.append((nm, c))
    cfg_of = dict(cands)
    # staged successive halving with an unbiased confirmation stage:
    #   stage 1  all candidates, 6 probes      -> keep top 3
    #   stage 2  survivors, 12 more probes     -> keep top 1
    #   stage 3  finalist vs incumbent on the 42 untouched probes
    # (worst case 13*6 + 4*12 + 2*42 = 210 probe searches = 5x the
    #  recursive arms' per-round gate budget)
    half = len(batches[0]) // 2
    flat = [(b_idx, j, t) for b_idx, bt in enumerate(batches, start=1)
            for j, t in enumerate(bt)]
    stages = [(flat[:half], 3), (flat[half:3 * half], 1),
              (flat[3 * half:], 0)]
    alive = [nm for nm, _c in cands]
    cum = {nm: [0, 0] for nm in alive}
    inc_cum = [0, 0]
    conf_c = conf_i = None
    rec = {"round": 1, "mode": mode, "scores": scores,
           "train_solved": len(state.solved), "candidates": list(alive),
           "halving": []}
    stages_done = 0
    for probes, keep in stages:
        if not alive:
            break
        try:
            inc = gate_eval_flat(state.cfg, probes, seed, gate, hp)
            res = {nm: gate_eval_flat(cfg_of[nm], probes, seed, gate, hp)
                   for nm in alive}
        except GateExhausted:
            break
        stages_done += 1
        inc_cum = [inc_cum[0] + inc[0], inc_cum[1] + inc[1]]
        for nm in alive:
            cum[nm] = [cum[nm][0] + res[nm][0], cum[nm][1] + res[nm][1]]
        rec["halving"].append({"n_probes": len(probes), "alive": list(alive),
                               "inc": [inc[0], inc[1]],
                               "res": {k: [v[0], v[1]]
                                       for k, v in res.items()}})
        if keep == 0:
            conf_c, conf_i = res[alive[0]], inc
            break
        alive.sort(key=lambda nm: (-cum[nm][0], cum[nm][1],
                                   names.index(nm)))
        alive = alive[:keep]
    adopted = None
    if conf_c is not None:
        best = alive[0]
        if not worse(conf_c, conf_i) and better(cum[best], inc_cum, hp):
            adopted = best
            state.cfg = cfg_of[best]
    batches_done = stages_done
    rec.update({"batches_done": batches_done, "adopted": adopted,
                "inc_cum": inc_cum, "n_macros": len(state.cfg.macros),
                "macros": list(state.cfg.macros),
                "attempt_spent": att.spent, "gate_spent": gate.spent})
    log.append(rec)
    return state


def run_arm(arm, seed, hp=None):
    """Improvement phase for one arm. Returns (final cfg, record)."""
    hp = hp or HP
    assert arm in ARMS, arm
    train, batches = T.improver_view(seed)  # holdout never loaded here
    cap = total_cap(hp, len(train))
    arm_meter = S.Meter(cap, "arm")
    g0 = S.EXEC.n
    log = []
    if arm == "COLD":
        state = ImproverState(hp)
    elif arm == "SINGLE_5X":
        state = run_single(seed, train, batches, arm_meter, hp, log)
    else:
        state = run_recursive(
            seed, train, batches, arm_meter, hp,
            adaptive=arm in ("RECURSIVE_FULL", "RECURSIVE_NODIAG"),
            contextual=arm == "RECURSIVE_FULL", log=log)
    delta = S.EXEC.n - g0
    if delta != arm_meter.spent:
        raise S.HiddenComputeError("%s: %d executions, %d metered"
                                   % (arm, delta, arm_meter.spent))
    assert arm_meter.spent <= cap
    return state.cfg, {"arm": arm, "seed": seed, "cap": cap,
                       "spent": arm_meter.spent, "global_delta": delta,
                       "train_solved": len(state.solved),
                       "final_cfg_sha": state.cfg.sha(),
                       "n_macros": len(state.cfg.macros),
                       "macros": list(state.cfg.macros),
                       "theta_final": state.theta, "rounds": log}
