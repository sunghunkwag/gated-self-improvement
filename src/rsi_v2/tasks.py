"""
rsi_v2.tasks -- FROZEN external task families and per-seed splits.

FROZEN FILE (hashed into the preregistration; the per-seed manifests it
generates are hashed too). Written by the experimenter, never by the
system under test. Nothing here is tuned to any arm.

Generative structure (all randomness from fixed seed strings below):
  * a hidden MOTIF LIBRARY: 8 "atoms" (2-primitive behaviours that no single
    primitive reproduces) and 6 "compounds" (two atoms back to back, whose
    behaviour no program of <= 3 primitives reproduces);
  * 16 TASK FAMILIES, each = 3 motifs + 2 "glue" primitives; a task is a
    chain of 1-3 motifs from its family with occasional glue between them
    (<= 12 primitives). This gives a difficulty LADDER: 2-primitive tasks
    are easy, 5-12 primitive tasks are unreachable for a 4-token searcher
    that has not grown its own vocabulary.
  * per seed, the families are shuffled into 10 TRAIN families and 6
    EXTERNAL families. Train/probe/holdout_in tasks come from train
    families; holdout_ext tasks come only from external families (novel
    motif combinations the system never trained or probed on).
  * every task in a seed has a DISTINCT behaviour signature across all
    four splits (behaviour-level, not just ID-level, isolation).

The solver never sees task ids, families, generating programs, or the
test inputs of any holdout task: it gets `public_view(task)` (train
examples only).
"""
from . import substrate as S

TASKS_VERSION = "rsi_v2-tasks-1"
N_ATOMS = 8
N_COMPOUNDS = 6
N_FAMILIES = 16
N_TRAIN_FAMILIES = 10
SPLITS = (("train", 40), ("probe", 60), ("holdout_in", 16),
          ("holdout_ext", 24))
MAX_TASK_LEN = 12
UNIT_COUNTS = (1, 1, 2, 2, 3)
GLUE_P = 0.25

_BIG = 10 ** 12


def _sig(prims):
    return S.signature(S.fns_of(prims), S.Meter(_BIG, "task-gen"))


def _nonempty(sig):
    return sum(1 for o in sig if len(o) > 0)


_LIB = [None]


def motif_library():
    """(atoms, compounds): lists of primitive tuples. Deterministic."""
    if _LIB[0] is not None:
        return _LIB[0]
    ident = _sig(())
    short = {ident}
    for a in S.NAMES:
        short.add(_sig((a,)))
    prng = S.XorShift64Star("rsi_v2|motif-library|atoms|1")
    atoms, atom_sigs = [], set()
    while len(atoms) < N_ATOMS:
        a, b = prng.choice(S.NAMES), prng.choice(S.NAMES)
        if a == b:
            continue
        sg = _sig((a, b))
        if sg in short or sg in atom_sigs or _nonempty(sg) < 5:
            continue
        atoms.append((a, b))
        atom_sigs.add(sg)
    upto3 = set(short)
    for a in S.NAMES:
        for b in S.NAMES:
            upto3.add(_sig((a, b)))
    for a in S.NAMES:
        for b in S.NAMES:
            for c in S.NAMES:
                upto3.add(_sig((a, b, c)))
    prng = S.XorShift64Star("rsi_v2|motif-library|compounds|1")
    comps, comp_sigs = [], set()
    tries = 0
    while len(comps) < N_COMPOUNDS:
        tries += 1
        assert tries < 100000, "compound generation stuck"
        i, j = prng.below(N_ATOMS), prng.below(N_ATOMS)
        if i == j:
            continue
        prog = atoms[i] + atoms[j]
        sg = _sig(prog)
        if sg in upto3 or sg in comp_sigs or _nonempty(sg) < 5:
            continue
        comps.append(prog)
        comp_sigs.add(sg)
    _LIB[0] = (atoms, comps)
    return _LIB[0]


_FAM = [None]


def families():
    """16 families: {'name', 'units': [prim tuples], 'glue': [prims]}."""
    if _FAM[0] is not None:
        return _FAM[0]
    atoms, comps = motif_library()
    prng = S.XorShift64Star("rsi_v2|families|1")
    out = []
    for k in range(N_FAMILIES):
        n_comp = 1 + prng.below(2)
        cs = prng.shuffled(range(N_COMPOUNDS))[:n_comp]
        as_ = prng.shuffled(range(N_ATOMS))[:3 - n_comp]
        units = [comps[c] for c in cs] + [atoms[a] for a in as_]
        glue = prng.shuffled(S.NAMES)[:2]
        out.append({"name": "F%02d" % k, "units": units, "glue": glue})
    _FAM[0] = out
    return out


def _sample_program(fam, prng):
    while True:
        n = UNIT_COUNTS[prng.below(len(UNIT_COUNTS))]
        prog = []
        for i in range(n):
            if i > 0 and prng.unit() < GLUE_P:
                prog.append(prng.choice(fam["glue"]))
            prog.extend(prng.choice(fam["units"]))
        if len(prog) <= MAX_TASK_LEN:
            return tuple(prog), n


def _inputs(prng, lo, hi, n):
    out = []
    for _ in range(n):
        ln = lo + prng.below(hi - lo + 1)
        out.append([prng.below(256) for _ in range(ln)])
    return out


def _instance(prog, prng):
    m = S.Meter(_BIG, "task-gen")
    fns = S.fns_of(prog)
    tr_in, te_in = _inputs(prng, 6, 9, 4), _inputs(prng, 10, 14, 4)
    tr = [(x, m.run(fns, x)) for x in tr_in]
    te = [(x, m.run(fns, x)) for x in te_in]
    if sum(1 for _x, y in tr if len(y) >= 2) < 3:
        return None  # under-specified: too little output to learn from
    if all(y == x for x, y in tr):
        return None
    if len({tuple(y) for _x, y in tr}) == 1:
        return None
    for nm in S.NAMES:  # reject tasks a single primitive already solves
        f = S.fns_of((nm,))
        if all(m.run(f, x) == y for x, y in tr + te):
            return None
    return tr, te


def _task_id(seed, split, k):
    return S.sha256_text("rsi_v2|task-id|%s|%s|%d" % (seed, split, k))[:12]


_MAN = {}


def seed_manifest(seed):
    """All four splits for one seed. Deterministic; cached per process."""
    if seed in _MAN:
        return _MAN[seed]
    fams = families()
    prng = S.XorShift64Star("rsi_v2|family-split|%s" % seed)
    order = prng.shuffled(range(N_FAMILIES))
    train_f = sorted(order[:N_TRAIN_FAMILIES])
    ext_f = sorted(order[N_TRAIN_FAMILIES:])
    seen = set()
    man = {"seed": seed, "version": TASKS_VERSION,
           "train_families": [fams[i]["name"] for i in train_f],
           "external_families": [fams[i]["name"] for i in ext_f]}
    # interleave the splits (k/n order) so no split monopolises the few
    # short behaviours each family has; behaviours stay globally distinct
    sched = sorted(((k / float(n), si, split, k)
                    for si, (split, n) in enumerate(SPLITS)
                    for k in range(n)))
    rows_by = {split: [] for split, _n in SPLITS}
    for _frac, _si, split, k in sched:
        pool = ext_f if split == "holdout_ext" else train_f
        fam = fams[pool[k % len(pool)]]
        tp = S.XorShift64Star("rsi_v2|task|%s|%s|%d" % (seed, split, k))
        for _try in range(2000):
            prog, n_units = _sample_program(fam, tp)
            sg = _sig(prog)
            if sg in seen:
                continue
            inst = _instance(prog, tp)
            if inst is None:
                continue
            seen.add(sg)
            rows_by[split].append({
                "id": _task_id(seed, split, k), "split": split,
                "family": fam["name"], "program": list(prog),
                "n_units": n_units, "length": len(prog),
                "train": [[x, y] for x, y in inst[0]],
                "test": [[x, y] for x, y in inst[1]]})
            break
        else:
            raise RuntimeError("task generation exhausted")
    for split, _n in SPLITS:
        man[split] = rows_by[split]
    _MAN[seed] = man
    return man


def library_digest():
    atoms, comps = motif_library()
    body = {"version": TASKS_VERSION, "atoms": [list(a) for a in atoms],
            "compounds": [list(c) for c in comps],
            "families": [{"name": f["name"],
                          "units": [list(u) for u in f["units"]],
                          "glue": list(f["glue"])} for f in families()]}
    return S.sha256_text(S.canon(body))


def manifest_digest(seeds):
    h = []
    for s in seeds:
        h.append(S.sha256_text(S.canon(seed_manifest(s))))
    return S.sha256_text(S.canon(h))


# --------------------------------------------------------------------------
# Views. The improver receives train/probe examples only; the solver
# receives a PublicTask (train examples only) for every search it runs.

class PublicTask(object):
    """What a search may see: the train examples. No id, no family, no
    generating program, no test inputs."""

    __slots__ = ("train",)

    def __init__(self, train):
        self.train = tuple((tuple(x), tuple(y)) for x, y in train)


class ImproverTask(object):
    """A train or probe task as the IMPROVER sees it: train + test examples
    (its own training / validation data), still no id/family/program."""

    __slots__ = ("train", "test")

    def __init__(self, row):
        self.train = tuple((tuple(x), tuple(y)) for x, y in row["train"])
        self.test = tuple((tuple(x), tuple(y)) for x, y in row["test"])

    def public(self):
        return PublicTask(self.train)


def improver_view(seed):
    """(train_tasks, probe_batches). Holdout rows are NOT returned."""
    man = seed_manifest(seed)
    train = [ImproverTask(r) for r in man["train"]]
    probes = [ImproverTask(r) for r in man["probe"]]
    n_batches = 5
    per = len(probes) // n_batches
    batches = [probes[i * per:(i + 1) * per] for i in range(n_batches)]
    return train, batches
