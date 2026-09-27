"""
rsi_v3.evaluate -- the IMMUTABLE FINAL-HOLDOUT evaluator (frozen, hashed).

Identical scoring rule to rsi_v2.evaluate: a final-holdout task counts as
solved iff the arm's final solver, given only the task's train examples and
EVAL_BUDGET executions, finds a program exact on every train AND hidden
test example. The search stream is keyed by (seed, split, task index) only
-- never the arm -- and FINAL HOLDOUT rows are readable only here, and not
at all while `sealed()` is active (the whole improvement phase runs sealed).
"""
from rsi_v2 import substrate as S
from rsi_v2 import solver as SV
from . import tasks as T

EVAL_BUDGET = 3000
PRIMARY_SPLIT = "holdout_ext"

_ACCESS = {"sealed": False}


class HoldoutSealedError(AssertionError):
    pass


def final_rows(seed, split):
    if _ACCESS["sealed"]:
        raise HoldoutSealedError("final-holdout access while sealed")
    assert split in T.FINAL_SPLITS
    return T.seed_manifest(seed)[split]


class sealed(object):
    def __enter__(self):
        self._prev = _ACCESS["sealed"]
        _ACCESS["sealed"] = True
        return self

    def __exit__(self, *exc):
        _ACCESS["sealed"] = self._prev
        return False


def eval_stream(seed, split, k, stream=0):
    return S.XorShift64Star("rsi_v3|eval|s%d|%s|%s|%d"
                            % (stream, seed, split, k))


def evaluate(cfg, seed, stream=0, budget=EVAL_BUDGET):
    comp = SV.Compiled(cfg)
    rows_by = {sp: final_rows(seed, sp) for sp in T.FINAL_SPLITS}
    g0 = S.EXEC.n
    search_m = S.Meter(10 ** 12, "eval-search")
    score_m = S.Meter(10 ** 12, "eval-score")
    out = {"by_split": {}, "per_task": {}}
    for split in T.FINAL_SPLITS:
        flags = []
        for k, row in enumerate(rows_by[split]):
            m = search_m.child(budget, "eval-task")
            res = SV.search(comp, T.PublicTask(row["train"]), budget,
                            eval_stream(seed, split, k, stream), m)
            assert res.spent <= budget
            ok = res.program is not None and SV.check(
                comp, res.program, row["train"] + row["test"], score_m)
            flags.append(1 if ok else 0)
        out["by_split"][split] = sum(flags)
        out["per_task"][split] = flags
    out["search_execs"] = search_m.spent
    out["score_execs"] = score_m.spent
    if S.EXEC.n - g0 != search_m.spent + score_m.spent:
        raise S.HiddenComputeError("evaluator executed unmetered programs")
    return out
