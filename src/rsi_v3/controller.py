"""
rsi_v3.controller -- memory-conditioned adaptive PROCESS controller.

v3's first mechanism was memory -> candidate RANKING, and its cross-world
carry-over was null on the final metric. This module changes the target:
memory -> PROCESS CONTROL. Accumulated improvement experience decides:

  proposal generation   which parameterised edit variants exist at all
                        (bundle size, prior step size, flattening,
                        search depth, composition depth, pruning)
  pool allocation       how the pool is split across the six strategies
  compute allocation    attempt mode: base / focus (redistribute the
                        round's attempt budget over unsolved tasks) /
                        deep (the same, with a one-token deeper search)
  probe allocation      screening shape: k candidates x n probes
  exploration rate      exploratory ranking slots + Thompson noise scale
  plasticity            adoption rule (always / non-inferior) and, via the
                        generated variants, edit intensity
  memory retrieval      mixing weight between episodic nearest-neighbour
                        estimates and the generalised model, adapted from
                        their prediction errors

IMPROVEMENT MEMORY
  episodic   one record per tested candidate and per round:
             state -> intervention -> resource allocation -> immediate gain
             -> delayed same-world gain -> cross-world transfer gain
             -> compute cost.
  meta       generalised models fitted on all records: a value model
             (edit variant x context -> gain at three horizons) and one
             model per round-level process option (option x context ->
             return-to-go), plus the retrieval weight.
Records hold NUMBERS AND LABELS ONLY -- never a program, macro, token or
task -- so no solver content can carry from one world to the next; only
experience about HOW to improve can.

FEEDBACK HORIZONS (all paired: same probes, same search streams)
  immediate   candidate - incumbent progress on the round's screen probes
  delayed     the adopted edit's effect on a fixed per-world TRACKING set of
              fresh META-VAL probes never used for any selection
              (T_r - T_{r-1}), and the return-to-go to the world's end
              (T_R - T_{r-1})
  transfer    world-end cross-family gain T_R - T_0 on the tracking set
Budgets are allotted per round and cannot be banked, so maximising gain
under the allotted budget is maximising gain per unit allotted compute;
actual compute per decision is recorded with every record.

All randomness (Thompson draws, exploration) comes from streams keyed by
(world seed, purpose, round, index) -- never by the arm -- so arms differ
only through what their memory contains.
"""
import math

from rsi_v2 import substrate as S
from .metamodel import Ridge, gauss

ACTIONS = ("MINE", "RESID", "COMPOSE", "PRIOR", "EXPLORE", "PRUNE")
MATCHED = {"MINE": 0, "RESID": 5, "COMPOSE": 4, "PRIOR": 1, "EXPLORE": 2,
           "PRUNE": 3}
N_DELTA = 10
N_PARAM = 8
KNOBS = {"attempt": ("base", "focus", "deep"),
         "shape": ("broad", "base", "deep"),     # see shape()
         "explore": (0, 1, 2),
         "gate": ("always", "noninferior")}
KNOB_ORDER = ("attempt", "shape", "explore", "gate")
N_CTX = 5


def shape(opt, hp):
    """Screening shape (candidates screened, probes per candidate) at a
    roughly constant screening cost: broad 1.5k x 2n/3, base k x n,
    deep 3k/4 x 4n/3 (k = hp k_screen, n = hp screen_n; 12x4, 8x6, 6x8)."""
    k, n = hp["k_screen"], hp["screen_n"]
    return ((int(round(1.5 * k)), int(round(2 * n / 3.0))), (k, n),
            (int(round(0.75 * k)), int(round(4 * n / 3.0))))[opt]


def stream(seed, *parts):
    return S.XorShift64Star("rsi_v3|" + "|".join(str(p) for p in
                                                 (seed,) + parts))


def value_dim():
    # bias | action | deltas | h | action x matched failure | probe inc |
    # variant params | row type (track, return) | round
    return 1 + 6 + N_DELTA + 1 + 6 + 1 + N_PARAM + 2 + 1


def knob_dim(knob):
    m = len(KNOBS[knob])
    return 1 + m + m * N_CTX


def context(v, rnd, hp, pre=False):
    """Compact state used by the round-level knob models and episodic
    retrieval: round, solved fraction, mean best fitness of unsolved
    tasks, vocabulary size, incumbent rate."""
    return [rnd / float(hp["n_rounds"]), v[6], 0.0 if pre else v[10], v[9],
            v[14]]


def vfeat(c, v, diag_on, probe_inc, row_type, rnd_frac):
    x = [1.0]
    x += [1.0 if c.action == a else 0.0 for a in ACTIONS]
    x += list(c.delta)
    x.append(c.h)
    x += [(v[MATCHED[a]] if (c.action == a and diag_on) else 0.0)
          for a in ACTIONS]
    x.append(probe_inc)
    x += list(c.params)
    x += [1.0 if row_type == "track" else 0.0,
          1.0 if row_type == "return" else 0.0]
    x.append(rnd_frac)
    return x


def kfeat(knob, opt, ctx):
    m = len(KNOBS[knob])
    x = [1.0] + [1.0 if i == opt else 0.0 for i in range(m)]
    for i in range(m):
        x += [c if i == opt else 0.0 for c in ctx]
    return x


class ProcessController(object):
    """learn:        memory is written (False = FROZEN)
    use_process:  the plan (generation, allocation, knobs) reads memory
    use_rank:     ranking reads memory"""

    def __init__(self, hp, learn, use_process, use_rank, diag_on=True,
                 use_knobs=True):
        self.hp = hp
        self.learn, self.use_process, self.use_rank = (learn, use_process,
                                                       use_rank)
        self.use_knobs = use_knobs
        self.diag_on = diag_on
        self.ops_done = {"rows": 0, "refits": 0, "predictions": 0,
                         "flops": 0, "retrievals": 0}
        self.resets = 0
        self._fresh()

    # ------------------------------------------------------------ memory
    def _fresh(self):
        hp = self.hp
        self.value = Ridge(value_dim(), hp["ridge_lam"])
        self.knob = {k: Ridge(knob_dim(k), hp["knob_lam"]) for k in KNOBS}
        for m in [self.value] + list(self.knob.values()):
            m.frozen = not self.learn
        self.episodes = []        # candidate-level episodic records
        self.rounds = []          # round-level episodic records
        self.worlds = []          # world-level records
        self.log_w = [0.0, 0.0]   # retrieval experts: generalised, episodic
        self.retr = {"ops": 0}

    def reset(self):
        """NO_CARRY: the improvement memory and meta-state are wiped (the
        cost already spent is kept in the accounting)."""
        self._absorb_ops()
        self.resets += 1
        self._fresh()

    def _absorb_ops(self):
        for m in [self.value] + list(self.knob.values()):
            for k in m.ops:
                self.ops_done[k] += m.ops[k]
        self.ops_done["retrievals"] += self.retr["ops"]

    def ops(self):
        out = dict(self.ops_done)
        for m in [self.value] + list(self.knob.values()):
            for k in m.ops:
                out[k] += m.ops[k]
        out["retrievals"] += self.retr["ops"]
        return out

    def omega(self):
        a, b = self.log_w
        mx = max(a, b)
        wa, wb = math.exp(a - mx), math.exp(b - mx)
        w = wb / (wa + wb)
        return min(0.95, max(0.05, w))

    def has_data(self):
        return self.learn and self.value.n > 0

    def snapshot(self):
        """JSON-able summary of what the memory holds (numbers + labels)."""
        return {"episodes": len(self.episodes), "rounds": len(self.rounds),
                "worlds": len(self.worlds), "value_rows": self.value.n,
                "knob_rows": {k: m.n for k, m in self.knob.items()},
                "omega": round(self.omega(), 4)}

    # ----------------------------------------------------- round-level knobs
    def choose(self, knob, ctx, prng):
        """Returns (option index, was_default). Default option until the
        knob model has hp["knob_min_rows"] rows of experience; then a
        Thompson draw from its posterior, with a switching margin: the
        default is left only if the drawn advantage over it exceeds
        hp["knob_margin"] (return-to-go units). With flat evidence the
        posterior tightens around zero differences and deviations become
        rare; a large learned advantage is acted on."""
        d = self.hp["proc_defaults"][knob]
        m = self.knob[knob]
        if not (self.use_process and self.use_knobs and self.learn
                and m.n >= self.hp["knob_min_rows"]):
            return d, True
        w = m.sample_w(self.hp["knob_noise_var"], prng)
        vals = [sum(a * b for a, b in zip(w, kfeat(knob, o, ctx)))
                for o in range(len(KNOBS[knob]))]
        m.ops["predictions"] += len(vals)
        best = max(range(len(vals)), key=lambda o: (vals[o], o == d))
        if best != d and vals[best] - vals[d] < self.hp["knob_margin"]:
            best = d
        return best, best == d

    # ------------------------------------------------------- value estimates
    def _retrieve(self, label, ctx):
        """Episodic k-NN: mean immediate gain of the k most similar past
        tests of the same edit variant."""
        recs = [e for e in self.episodes if e["label"] == label]
        self.retr["ops"] += len(recs)
        if len(recs) < self.hp["knn_min"]:
            return None
        recs.sort(key=lambda e: sum((a - b) ** 2 for a, b in
                                    zip(e["ctx"], ctx)))
        near = recs[:self.hp["knn_k"]]
        return sum(e["y"] for e in near) / len(near)

    def estimate(self, cands, v, ctx, inc_mean, rnd_frac):
        """Posterior mean (generalised model blended with episodic
        retrieval) and sd for every candidate."""
        om = self.omega()
        for c in cands:
            c.x = vfeat(c, v, self.diag_on, inc_mean, "screen", rnd_frac)
            c.mu_r = self.value.predict(c.x)
            c.mu_k = self._retrieve(c.label, ctx)
            c.mu = c.mu_r if c.mu_k is None else ((1.0 - om) * c.mu_r
                                                  + om * c.mu_k)
            c.sd = self.value.pred_sd(c.x, self.hp["noise_var"])

    # ------------------------------------------------------ generation plan
    def plan_pool(self, grid, alloc_default, v, ctx, inc_mean, rnd,
                  explore, seed):
        """grid: {action: [Cand in default order]}. Returns (pool, info).
        Default plan (no memory, FROZEN, rank-only): the first
        alloc_default[a] variants of every action -- exactly the v3
        generator. Learned plan: the allocation across strategies is
        tilted by the value of each strategy's best variant, and within a
        strategy variants are chosen by value + Thompson exploration."""
        hp = self.hp
        size = hp["pool_size"]
        acts = [a for a in ACTIONS if grid.get(a)]
        info = {"alloc_default": dict(alloc_default), "learned": False}
        if hp.get("gen_default", "v3") == "grid":
            # uninformed prior over the WHOLE grid: a shared random subset
            # (stream keyed by world seed, round, strategy -- not the arm)
            default_pick = {a: stream(seed, "ctrl-default", rnd, a).shuffled(
                grid[a])[:alloc_default.get(a, 0)] for a in acts}
        else:                                  # the v3 generator's own list
            default_pick = {a: [c for c in grid[a] if c.dflt][
                :alloc_default.get(a, 0)] for a in acts}
        dpool = _dedupe([(a, default_pick[a]) for a in acts])
        info["pool_default"] = [c.label for c in dpool]
        if not (self.use_process and self.has_data()):
            info["alloc"] = dict(alloc_default)
            return dpool, info
        allc = [c for a in acts for c in grid[a]]
        self.estimate(allc, v, ctx, inc_mean, rnd / float(hp["n_rounds"]))
        kappa = hp["gen_kappa"][explore]
        for c in allc:
            pr = stream(seed, "ctrl-gen", rnd, c.action, c.pos)
            c.score = (c.mu - hp["default_pref"] * c.pos
                       + kappa * c.sd * gauss(pr))
        # strategy allocation: diagnosis prior x learned strategy value
        nd = _needs(v)
        mxn = max(nd[a] for a in acts)
        val = {a: max(c.mu for c in grid[a]) for a in acts}
        mxv = max(val.values())
        w = {a: math.exp((nd[a] - mxn) / hp["alloc_temp"]
                         + (val[a] - mxv) / hp["alloc_value_temp"])
             for a in acts}
        n = _alloc_capped(w, {a: len(grid[a]) for a in acts}, size)
        picks = []
        for a in acts:
            ranked = sorted(grid[a], key=lambda c: (-c.score, c.pos))
            picks.append((a, ranked[:n[a]]))
        info["alloc"] = n
        info["learned"] = True
        return _dedupe(picks), info

    # --------------------------------------------------------------- ranking
    def rank(self, pool, v, ctx, inc_mean, rnd, k, explore, prng):
        """Learned ranking: top (k - explore) by estimated value, then
        `explore` exploratory picks in the shared random order. Without
        memory it reduces to the shared random order (= FROZEN)."""
        perm = prng.shuffled(range(len(pool)))
        pos = {c: i for i, c in enumerate(perm)}
        idx = list(range(len(pool)))
        if not self.use_rank:
            return sorted(idx, key=lambda i: pos[i])[:k]
        if any(getattr(c, "mu_r", None) is None for c in pool) or \
                not self.use_process:
            self.estimate(pool, v, ctx, inc_mean,
                          rnd / float(self.hp["n_rounds"]))
        greedy = sorted(idx, key=lambda i: (-pool[i].mu, pos[i]))[
            :max(0, k - explore)]
        rest = [i for i in sorted(idx, key=lambda i: pos[i])
                if i not in greedy]
        return greedy + rest[:explore]

    # ---------------------------------------------------------- observation
    def observe_screen(self, screened, v, ctx, rnd, w_idx):
        """screened: [(cand, [(y, probe_inc)...], cost)]. Immediate
        horizon rows + episodic records + retrieval-weight update."""
        if not self.learn:
            return 0
        n = 0
        eta = self.hp["retrieval_eta"]
        rf = rnd / float(self.hp["n_rounds"])
        for c, ys, cost in screened:
            ymean = sum(y for y, _p in ys) / len(ys)
            if getattr(c, "mu_k", None) is not None and \
                    getattr(c, "mu_r", None) is not None:
                self.log_w[0] -= eta * (ymean - c.mu_r) ** 2
                self.log_w[1] -= eta * (ymean - c.mu_k) ** 2
            for y, pinc in ys:
                self.value.add(vfeat(c, v, self.diag_on, pinc, "screen", rf),
                               y)
                n += 1
            self.episodes.append({
                "world": w_idx, "round": rnd, "label": c.label,
                "action": c.action, "params": list(c.params),
                "ctx": list(ctx), "y": ymean, "n": len(ys),
                "cost": cost, "adopted": False})
        return n

    def observe_track(self, cand, v, rnd, t_prev, t_new, w_idx):
        """Delayed (unbiased) effect of the adopted edit on the tracking
        probes: paired per probe."""
        if not self.learn or t_prev is None or t_new is None:
            return 0
        rf = rnd / float(self.hp["n_rounds"])
        for a, b in zip(t_prev, t_new):
            self.value.add(vfeat(cand, v, self.diag_on, a, "track", rf),
                           b - a, self.hp["track_weight"])
        for e in reversed(self.episodes):
            if e["round"] == rnd and e["label"] == cand.label \
                    and e["world"] == w_idx:
                e["adopted"] = True
                e["track_gain"] = sum(b - a for a, b in
                                      zip(t_prev, t_new)) / len(t_new)
                break
        return len(t_new)

    def refit(self):
        if self.learn:
            self.value.refit()

    def end_world(self, w_idx, rounds, T):
        """rounds: per round {"ctx_pre", "ctx", "plan", "adopted_cand",
        "v", "rnd", "compute"}; T: tracking progress vectors T_0..T_R
        (None where unavailable). Adds return-to-go rows (delayed horizon)
        for adopted edits, knob rows for every round-level decision, and
        the world-level transfer record."""
        R = len(rounds)
        TR = T[R] if len(T) > R else None
        out = {"return_rows": 0, "knob_rows": 0}
        rec_rounds = []
        for i, rd in enumerate(rounds):
            tp = T[i] if i < len(T) else None
            G = None
            if tp is not None and TR is not None:
                G = sum(b - a for a, b in zip(tp, TR)) / len(TR)
            rec_rounds.append({"world": w_idx, "round": rd["rnd"],
                               "ctx": rd["ctx"], "plan": rd["plan"],
                               "return": G, "compute": rd["compute"]})
            if not self.learn or G is None:
                continue
            c = rd.get("adopted_cand")
            if c is not None:
                rf = rd["rnd"] / float(self.hp["n_rounds"])
                for a, b in zip(tp, TR):
                    self.value.add(vfeat(c, rd["v"], self.diag_on, a,
                                         "return", rf),
                                   b - a, self.hp["return_weight"])
                    out["return_rows"] += 1
                for e in reversed(self.episodes):
                    if e["round"] == rd["rnd"] and e["label"] == c.label \
                            and e["world"] == w_idx:
                        e["return_gain"] = G
                        break
            for k in KNOB_ORDER:
                ctxk = rd["ctx_pre"] if k == "attempt" else rd["ctx"]
                self.knob[k].add(kfeat(k, rd["plan"][k], ctxk), G)
                out["knob_rows"] += 1
        T0 = T[0] if T else None
        transfer = (sum(b - a for a, b in zip(T0, TR)) / len(TR)
                    if T0 is not None and TR is not None else None)
        self.rounds.extend(rec_rounds)
        self.worlds.append({"world": w_idx, "transfer": transfer})
        if self.learn:
            self.value.refit()
            for m in self.knob.values():
                if m.n:
                    m.refit()
        out["transfer"] = transfer
        return out


def _needs(v):
    missing, order, explore, overspec, compose, residual = v[:6]
    solved_frac, ent, dead, n_mac = v[6], v[7], v[8], v[9]
    return {"MINE": missing + 0.5 * solved_frac,
            "RESID": residual,
            "COMPOSE": compose + 0.5 * n_mac,
            "PRIOR": order + 0.25,
            "EXPLORE": explore + 2.0 * (1.0 - ent),
            "PRUNE": overspec + dead}


def _alloc_capped(w, cap, size):
    """Proportional allocation with a floor of one per strategy, capped by
    how many variants a strategy has; surplus goes to the others."""
    acts = sorted(w, key=lambda a: ACTIONS.index(a))
    n = {a: min(1, cap[a]) for a in acts}
    left = size - sum(n.values())
    while left > 0:
        open_ = [a for a in acts if n[a] < cap[a]]
        if not open_:
            break
        tot = sum(w[a] for a in open_)
        raw = {a: left * w[a] / tot for a in open_}
        add = {a: min(cap[a] - n[a], int(math.floor(raw[a])))
               for a in open_}
        if sum(add.values()) == 0:
            a = max(open_, key=lambda a: (raw[a], -ACTIONS.index(a)))
            add[a] = 1
        for a in open_:
            n[a] += add[a]
        left = size - sum(n.values())
    return n


def _dedupe(picks):
    """Pool assembly exactly as the v3 generator: strategies in fixed
    order, a variant is skipped if its config duplicates one already in the
    pool or equals the incumbent."""
    pool, shas = [], set()
    for a in ACTIONS:
        items = dict(picks).get(a, [])
        for j, c in enumerate(items):
            sh = c.cfg.sha()
            if sh in shas or c.inc_sha == sh:
                continue
            shas.add(sh)
            pool.append(c)
    return pool
