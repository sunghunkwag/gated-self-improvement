"""
rsi_v2.substrate -- deterministic PRNG, the 20 list primitives, METERED
execution.

FROZEN FILE (hashed into the preregistration): the primitive semantics are
verbatim copies of omniforge's lf_* substrate (the test suite proves the
behaviour fingerprint is identical), and every program execution anywhere
in rsi_v2 goes through `execute`, which increments a process-global
counter. Arms may only execute programs through a `Meter`, which charges
the arm's budget; the harness asserts `global delta == metered spend` for
every arm, so any uncharged ("hidden") compute is detected.
"""
import hashlib
import json

MAXLEN = 24
_MASK = (1 << 64) - 1


class XorShift64Star(object):
    """SHA-256-seeded xorshift64* (identical to omniforge.lf_XorShift64Star).
    Every random decision in rsi_v2 draws from a stream whose key names
    (seed, purpose, round, index) -- never an arm/condition name."""

    __slots__ = ("state",)

    def __init__(self, seed_text):
        d = hashlib.sha256(str(seed_text).encode("utf-8")).digest()
        self.state = int.from_bytes(d[:8], "big") or 11400714819323198485

    def u64(self):
        x = self.state
        x ^= x >> 12
        x ^= (x << 25) & _MASK
        x ^= x >> 27
        self.state = x
        return (x * 2685821657736338717) & _MASK

    def below(self, n):
        return self.u64() % n if n > 0 else 0

    def unit(self):
        return self.u64() / float(1 << 64)

    def choice(self, seq):
        return seq[self.below(len(seq))]

    def shuffled(self, seq):
        out = list(seq)
        for i in range(len(out) - 1, 0, -1):
            j = self.below(i + 1)
            out[i], out[j] = out[j], out[i]
        return out


def _c(x):
    return [v & 255 for v in x][:MAXLEN]


PRIMS = {
    "inc": lambda x: _c([v + 1 for v in x]),
    "dec": lambda x: _c([v - 1 for v in x]),
    "double": lambda x: _c([v * 2 for v in x]),
    "half": lambda x: _c([v // 2 for v in x]),
    "neg": lambda x: _c([255 - v for v in x]),
    "add5": lambda x: _c([v + 5 for v in x]),
    "mod10": lambda x: _c([v % 10 for v in x]),
    "reverse": lambda x: _c(list(reversed(x))),
    "sort": lambda x: _c(sorted(x)),
    "rotate": lambda x: _c(x[1:] + x[:1] if x else []),
    "dedupe": lambda x: _c([v for i, v in enumerate(x)
                            if i == 0 or v != x[i - 1]]),
    "evens": lambda x: _c([v for v in x if v % 2 == 0]),
    "big": lambda x: _c([v for v in x if v > 128]),
    "tail": lambda x: _c(x[1:]),
    "init": lambda x: _c(x[:-1]),
    "cumsum": lambda x: _c([sum(x[:i + 1]) for i in range(len(x))]),
    "cummax": lambda x: _c([max(x[:i + 1]) for i in range(len(x))]),
    "diff": lambda x: _c([x[i] - (x[i - 1] if i else 0)
                          for i in range(len(x))]),
    "dup": lambda x: _c([v for v in x for _ in range(2)]),
    "pairsum": lambda x: _c([x[i] + x[i + 1] for i in range(len(x) - 1)]),
}
NAMES = tuple(sorted(PRIMS))
PROBE_INPUTS = ([3, 1, 4, 1, 5, 9], [200, 7, 7, 90], [0, 255, 128, 64],
                [10, 20, 30, 40, 50], [9, 9, 2], [2, 2, 2, 2],
                [17, 4, 200, 3, 3, 88])


class _ExecCounter(object):
    __slots__ = ("n",)

    def __init__(self):
        self.n = 0


EXEC = _ExecCounter()


def fns_of(prims):
    return tuple(PRIMS[p] for p in prims)


def execute(fns, xs):
    """Run a primitive-function tuple on one input. The ONLY execution
    path in rsi_v2; counted globally."""
    EXEC.n += 1
    v = list(xs)
    for f in fns:
        v = f(v)
    return v


class BudgetExceeded(Exception):
    pass


class HiddenComputeError(AssertionError):
    pass


class Meter(object):
    """Hierarchical execution budget. A child meter charges its parent;
    exceeding either cap raises BudgetExceeded before executing."""

    __slots__ = ("cap", "spent", "parent", "label")

    def __init__(self, cap, label="", parent=None):
        self.cap = int(cap)
        self.spent = 0
        self.parent = parent
        self.label = label

    def left(self):
        own = self.cap - self.spent
        if self.parent is not None:
            return min(own, self.parent.left())
        return own

    def child(self, cap, label=""):
        return Meter(cap, label, self)

    def charge(self):
        m = self
        while m is not None:
            if m.spent >= m.cap:
                raise BudgetExceeded(m.label)
            m = m.parent
        m = self
        while m is not None:
            m.spent += 1
            m = m.parent

    def run(self, fns, xs):
        self.charge()
        return execute(fns, xs)


def signature(fns, meter):
    """Behaviour signature on the fixed probe inputs (charged)."""
    return tuple(tuple(meter.run(fns, xs)) for xs in PROBE_INPUTS)


def canon(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def sha256_text(s):
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def fingerprint():
    """Behaviour fingerprint of the primitive set (compare: omniforge)."""
    rows = []
    for nm in NAMES:
        f = PRIMS[nm]
        rows.append([nm, [f(list(x)) for x in PROBE_INPUTS]])
    return sha256_text(canon(rows))[:16]
