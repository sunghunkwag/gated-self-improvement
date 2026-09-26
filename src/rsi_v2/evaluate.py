"""
rsi_v2.evaluate -- the IMMUTABLE held-out evaluator.

FROZEN FILE (hashed into the preregistration). The primary metric is
computed here and only here:

    solved(task) = the final SolverConfig's search, given ONLY the task's
                   train examples and EVAL_BUDGET executions, returns a
                   program that is exact on every train example AND on
                   every hidden test example (longer, unseen inputs).

Common random numbers: the search stream for holdout task k of seed s is
'rsi_v2|eval|s<stream>|<seed>|<split>|<k>' -- it does not contain the arm
name, so every arm is evaluated under identical randomness. stream=0 for
all arms; stream=1 only for the deliberate COLD_REPLICATE noise floor.

Holdout rows are read from the frozen manifest inside this module; the
improvement phase has no code path to them (tasks.improver_view strips
them, and the test suite trips a wire on any access during improvement).
"""
from . import substrate as S
from . import solver as SV
from . import tasks as T

EVAL_BUDGET = 3000
HOLDOUT_SPLITS = ("holdout_in", "holdout_ext")
PRIMARY_SPLIT = "holdout_ext"

_ACCESS = {"sealed": False}


class HoldoutSealedError(AssertionError):
    pass


def holdout_rows(seed, split):
    if _ACCESS["sealed"]:
        raise HoldoutSealedError("holdout access during improvement phase")
    assert split in HOLDOUT_SPLITS
    return T.seed_manifest(seed)[split]


class sealed(object):
    """Context manager: any holdout read inside raises."""

    def __enter__(self):
        self._prev = _ACCESS["sealed"]
        _ACCESS["sealed"] = True
        return self

    def __exit__(self, *exc):
        _ACCESS["sealed"] = self._prev
        return False


def eval_stream(seed, split, k, stream=0):
    return S.XorShift64Star("rsi_v2|eval|s%d|%s|%s|%d"
                            % (stream, seed, split, k))


def evaluate(cfg, seed, stream=0, budget=EVAL_BUDGET):
    """Returns {'by_split': {split: solved}, 'per_task': {split: [0/1]},
    'search_execs', 'score_execs'}. Deterministic."""
    comp = SV.Compiled(cfg)
    rows_by = {sp: holdout_rows(seed, sp) for sp in HOLDOUT_SPLITS}
    g0 = S.EXEC.n  # after task construction: only solver/scorer counted
    search_m = S.Meter(10 ** 12, "eval-search")
    score_m = S.Meter(10 ** 12, "eval-score")
    out = {"by_split": {}, "per_task": {}}
    for split in HOLDOUT_SPLITS:
        rows = rows_by[split]
        flags = []
        for k, row in enumerate(rows):
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
