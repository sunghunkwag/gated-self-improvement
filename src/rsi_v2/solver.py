"""
rsi_v2.solver -- the OBJECT-LEVEL solver.

A token-level evolutionary program search (the omniforge xv_search_task
algorithm, generalised to a growable vocabulary). What the improver may
change is DATA, never this code: a SolverConfig holds

  macros      learned multi-primitive tokens (content-addressed: the token
              name IS its primitive expansion, e.g. "<sort.diff>")
  weights     sampling prior over all tokens
  max_tokens  program length bound, in tokens
  p_random    fraction of fresh random children per generation

Every execution is charged to a Meter; the search stops at its budget.
It sees only a PublicTask (train examples) and returns a program that is
exact on ALL train examples, or None. Held-out checking is the caller's.
"""
import bisect

from . import substrate as S

MAX_EXPANDED = 16
POP = 32
BASE_MAX_TOKENS = 4
BASE_P_RANDOM = 0.3


def macro_token(prims):
    return "<" + ".".join(prims) + ">"


def is_macro(tok):
    return tok.startswith("<")


def expansion(tok):
    if tok.startswith("<"):
        return tuple(tok[1:-1].split("."))
    return (tok,)


class SolverConfig(object):
    __slots__ = ("macros", "weights", "max_tokens", "p_random")

    def __init__(self, macros=(), weights=None, max_tokens=BASE_MAX_TOKENS,
                 p_random=BASE_P_RANDOM):
        self.macros = tuple(macros)          # tuple of token names
        if weights is None:
            weights = {t: 1.0 for t in S.NAMES}
        self.weights = dict(weights)
        self.max_tokens = int(max_tokens)
        self.p_random = float(p_random)

    def tokens(self):
        return tuple(S.NAMES) + self.macros

    def copy(self):
        return SolverConfig(self.macros, self.weights, self.max_tokens,
                            self.p_random)

    def to_json(self):
        return {"macros": list(self.macros),
                "weights": {k: round(v, 9) for k, v in
                            sorted(self.weights.items())},
                "max_tokens": self.max_tokens,
                "p_random": round(self.p_random, 9)}

    @staticmethod
    def from_json(d):
        return SolverConfig(d["macros"], d["weights"], d["max_tokens"],
                            d["p_random"])

    def sha(self):
        return S.sha256_text(S.canon(self.to_json()))[:16]


def base_config():
    return SolverConfig()


class Compiled(object):
    """Search-ready view of a config (cumulative weights, fn tuples)."""

    __slots__ = ("toks", "cum", "total", "fns", "max_tokens", "p_random")

    def __init__(self, cfg):
        self.toks = cfg.tokens()
        acc, cum = 0.0, []
        for t in self.toks:
            acc += max(cfg.weights.get(t, 0.0), 0.0)
            cum.append(acc)
        self.cum, self.total = cum, acc
        self.fns = {t: S.fns_of(expansion(t)) for t in self.toks}
        self.max_tokens = cfg.max_tokens
        self.p_random = cfg.p_random

    def sample(self, prng):
        i = bisect.bisect_left(self.cum, prng.unit() * self.total)
        return self.toks[min(i, len(self.toks) - 1)]

    def random_prog(self, prng):
        n = 1 + prng.below(self.max_tokens)
        return tuple(self.sample(prng) for _ in range(n))

    def mutate(self, p, prng):
        p = list(p)
        for _ in range(1 + prng.below(2)):
            r = prng.unit()
            if r < 0.35 and len(p) < self.max_tokens:
                p.insert(prng.below(len(p) + 1), self.sample(prng))
            elif r < 0.65 and len(p) > 1:
                p.pop(prng.below(len(p)))
            elif p:
                p[prng.below(len(p))] = self.sample(prng)
        return tuple(p) if p else (self.sample(prng),)

    def crossover(self, a, b, prng):
        c = a[:prng.below(len(a) + 1)] + b[prng.below(len(b) + 1):]
        c = c[:self.max_tokens]
        return c if c else (a[:1] or b[:1])

    def prog_fns(self, p):
        out = ()
        for t in p:
            out += self.fns[t]
        return out


class SearchResult(object):
    __slots__ = ("program", "spent", "best_fit", "best_prog", "diversity",
                 "parent_exhausted")

    def __init__(self, program, spent, best_fit, best_prog, diversity,
                 parent_exhausted):
        self.program = program
        self.spent = spent
        self.best_fit = best_fit
        self.best_prog = best_prog
        self.diversity = diversity
        self.parent_exhausted = parent_exhausted


def _partial(out, y):
    if out == y:
        return 1.0
    denom = max(len(y), len(out), 1)
    m = 0
    for a, b in zip(out, y):
        if a == b:
            m += 1
    return 0.85 * (m / float(denom)) + (0.1 if len(out) == len(y) else 0.0)


def search(comp, public, budget, prng, meter):
    """EA over token programs. `public` must be a PublicTask."""
    sub = meter.child(budget, "search")
    train = [(list(x), list(y)) for x, y in public.train]
    order = sorted(range(len(train)), key=lambda i: (len(train[i][0]), i))
    guide = [train[i] for i in order[:2]]
    rest = [train[i] for i in order[2:]]
    memo = {}
    best = [-2.0, None]
    found = [None]
    parent_hit = [False]

    def fit(p):
        f = memo.get(p)
        if f is not None:
            return f
        fns = comp.prog_fns(p)
        if not fns or len(fns) > MAX_EXPANDED:
            memo[p] = -1.0
            return -1.0
        tot = 0.0
        for x, y in guide:
            tot += _partial(sub.run(fns, x), y)
        f = tot / len(guide)
        if f >= 1.0:
            for x, y in rest:
                if sub.run(fns, x) != y:
                    f = 0.999
                    break
            else:
                found[0] = p
        memo[p] = f
        if f > best[0]:
            best[0], best[1] = f, p
        return f

    ps = POP
    pop = []
    proposals = 0
    cap_props = 4 * budget + 64
    try:
        while len(pop) < ps and proposals < cap_props:
            proposals += 1
            p = comp.random_prog(prng)
            f = fit(p)
            if found[0] is not None:
                raise StopIteration
            pop.append((f, p))
        while proposals < cap_props:
            pop.sort(key=lambda t: -t[0])
            elite = pop[:max(2, ps // 4)]
            newpop = list(elite)
            while len(newpop) < ps and proposals < cap_props:
                proposals += 1
                r = prng.unit()
                if r < comp.p_random:
                    c = comp.random_prog(prng)
                elif r < comp.p_random + (1.0 - comp.p_random) / 2.0:
                    c = comp.mutate(_tourn(pop, prng)[1], prng)
                else:
                    c = comp.crossover(_tourn(pop, prng)[1],
                                       _tourn(pop, prng)[1], prng)
                f = fit(c)
                if found[0] is not None:
                    raise StopIteration
                newpop.append((f, c))
            pop = newpop
    except StopIteration:
        pass
    except S.BudgetExceeded as e:
        if e.args and e.args[0] != "search":
            parent_hit[0] = True
    div = len({p for _f, p in pop}) / float(max(1, len(pop)))
    return SearchResult(found[0], sub.spent, best[0], best[1], div,
                        parent_hit[0])


def _tourn(pop, prng, k=3):
    b = pop[prng.below(len(pop))]
    for _ in range(k - 1):
        c = pop[prng.below(len(pop))]
        if c[0] > b[0]:
            b = c
    return b


def check(comp, prog, examples, meter):
    """Exact on every example (charged)."""
    if prog is None:
        return False
    fns = comp.prog_fns(prog)
    for x, y in examples:
        if meter.run(fns, list(x)) != list(y):
            return False
    return True
