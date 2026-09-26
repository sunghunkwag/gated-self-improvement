"""
rsi_v2.stats -- paired statistics (deterministic, stdlib only).

Same sign-flip permutation / percentile bootstrap as omniforge lf_perm_p /
lf_boot_ci, plus a one-sided variant and Holm step-down.
"""
from .substrate import XorShift64Star


def perm_p(diffs, tag, n_perm=20000, one_sided=False):
    n = len(diffs)
    obs = sum(diffs) / float(n)
    prng = XorShift64Star("rsi_v2|stats|perm|%s" % tag)
    hits = 0
    for _ in range(n_perm):
        s = 0.0
        for d in diffs:
            s += d if prng.unit() < 0.5 else -d
        m = s / n
        if one_sided:
            if m >= obs - 1e-12:
                hits += 1
        elif abs(m) >= abs(obs) - 1e-12:
            hits += 1
    return (hits + 1) / float(n_perm + 1)


def boot_ci(diffs, tag, n_boot=5000):
    n = len(diffs)
    prng = XorShift64Star("rsi_v2|stats|boot|%s" % tag)
    means = []
    for _ in range(n_boot):
        s = 0.0
        for _i in range(n):
            s += diffs[prng.below(n)]
        means.append(s / n)
    means.sort()
    return means[int(0.025 * n_boot)], means[min(int(0.975 * n_boot),
                                                  n_boot - 1)]


def holm(pvals):
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (name, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[name] = running
    return out


def summary(diffs, tag, one_sided=False):
    n = len(diffs)
    mean = sum(diffs) / float(n)
    sd = (sum((d - mean) ** 2 for d in diffs) / max(1, n - 1)) ** 0.5
    lo, hi = boot_ci(diffs, tag)
    return {"n": n, "mean": mean, "sd": sd, "ci95": [lo, hi],
            "wins": sum(1 for d in diffs if d > 0),
            "ties": sum(1 for d in diffs if d == 0),
            "losses": sum(1 for d in diffs if d < 0),
            "p": perm_p(diffs, tag, one_sided=one_sided),
            "dz": (mean / sd) if sd > 0 else 0.0}
