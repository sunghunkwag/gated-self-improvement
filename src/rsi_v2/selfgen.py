"""
rsi_v2.selfgen -- SELF-GENERATED curriculum score (diagnostic ONLY).

A system that grows its own vocabulary can always make up tasks that its
own vocabulary solves. This module measures exactly that -- tasks sampled
from the arm's OWN final config (2-3 tokens drawn from its own prior over
its own macros) -- so the report can show it NEXT TO the external score
and demonstrate that the two are different quantities. It is never used
for any headline, gate or claim (the report generator reads the primary
metric from evaluate.py results only, and a test enforces it).
"""
from . import substrate as S
from . import solver as SV

N_SELF = 8
SELF_BUDGET = 1500


def self_score(cfg, seed):
    comp = SV.Compiled(cfg)
    m = S.Meter(10 ** 12, "selfgen")
    solved = 0
    made = 0
    for k in range(N_SELF):
        prng = S.XorShift64Star("rsi_v2|selfgen|%s|%d" % (seed, k))
        for _try in range(50):
            n = 2 + prng.below(2)
            prog = tuple(comp.sample(prng) for _ in range(n))
            fns = comp.prog_fns(prog)
            if len(fns) > SV.MAX_EXPANDED:
                continue
            xs = [[prng.below(256) for _ in range(6 + prng.below(4))]
                  for _ in range(4)]
            ys = [m.run(fns, x) for x in xs]
            if sum(1 for y in ys if len(y) >= 2) < 3:
                continue
            tx = [[prng.below(256) for _ in range(10 + prng.below(5))]
                  for _ in range(4)]
            test = [(x, m.run(fns, x)) for x in tx]
            break
        else:
            continue
        made += 1
        from .tasks import PublicTask
        res = SV.search(comp, PublicTask(list(zip(xs, ys))), SELF_BUDGET,
                        S.XorShift64Star("rsi_v2|selfgen-s|%s|%d"
                                         % (seed, k)), m)
        if res.program is not None and SV.check(comp, res.program, test, m):
            solved += 1
    return {"self_solved": solved, "self_made": made}
