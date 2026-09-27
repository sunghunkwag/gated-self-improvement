"""
rsi_v3.tasks -- FROZEN task families with a THREE-WAY family split.

Same hidden motif library and the same generative process as rsi_v2 (the
atoms/compounds are imported unchanged from rsi_v2.tasks), but 24 families
so every seed can be split three ways by FAMILY:

  TRAIN families (12)      -> train (40 tasks)  + holdout_in (16, sealed)
  META-VAL families (6)    -> metaval (96 tasks): the improver's
                              counterfactual tests and the meta-predictor's
                              training data. Cross-family by construction,
                              so the improver's learning signal measures
                              TRANSFER, like the final objective.
  FINAL families (6)       -> holdout_ext (24 tasks): FINAL HOLDOUT, sealed.

Meta-policy learning and gating may use TRAIN and META-VAL only. Every task
of a seed has a distinct behaviour signature across all four splits. The
solver sees PublicTask (train examples only); the improver sees
ImproverTask (train + test examples of TRAIN / META-VAL tasks only).
"""
from rsi_v2 import substrate as S
from rsi_v2 import tasks as T2

TASKS_VERSION = "rsi_v3-tasks-1"
N_FAMILIES = 24
N_TRAIN_FAMILIES = 12
N_METAVAL_FAMILIES = 6
SPLITS = (("train", 40), ("metaval", 96), ("holdout_in", 16),
          ("holdout_ext", 24))
FINAL_SPLITS = ("holdout_in", "holdout_ext")
METAVAL_BATCH = 12

PublicTask = T2.PublicTask
ImproverTask = T2.ImproverTask


def motif_library():
    return T2.motif_library()


_FAM = [None]


def families():
    if _FAM[0] is not None:
        return _FAM[0]
    atoms, comps = motif_library()
    prng = S.XorShift64Star("rsi_v3|families|1")
    out = []
    for k in range(N_FAMILIES):
        n_comp = 1 + prng.below(2)
        cs = prng.shuffled(range(len(comps)))[:n_comp]
        as_ = prng.shuffled(range(len(atoms)))[:3 - n_comp]
        units = [comps[c] for c in cs] + [atoms[a] for a in as_]
        glue = prng.shuffled(S.NAMES)[:2]
        out.append({"name": "G%02d" % k, "units": units, "glue": glue})
    _FAM[0] = out
    return out


def _task_id(seed, split, k):
    return S.sha256_text("rsi_v3|task-id|%s|%s|%d" % (seed, split, k))[:12]


_MAN = {}


def seed_manifest(seed):
    if seed in _MAN:
        return _MAN[seed]
    fams = families()
    prng = S.XorShift64Star("rsi_v3|family-split|%s" % seed)
    order = prng.shuffled(range(N_FAMILIES))
    a, b = N_TRAIN_FAMILIES, N_TRAIN_FAMILIES + N_METAVAL_FAMILIES
    pools = {"train": sorted(order[:a]), "holdout_in": sorted(order[:a]),
             "metaval": sorted(order[a:b]), "holdout_ext": sorted(order[b:])}
    man = {"seed": seed, "version": TASKS_VERSION,
           "train_families": [fams[i]["name"] for i in pools["train"]],
           "metaval_families": [fams[i]["name"] for i in pools["metaval"]],
           "final_families": [fams[i]["name"] for i in pools["holdout_ext"]]}
    seen = set()
    sched = sorted(((k / float(n), si, split, k)
                    for si, (split, n) in enumerate(SPLITS)
                    for k in range(n)))
    rows_by = {split: [] for split, _n in SPLITS}
    for _f, _si, split, k in sched:
        pool = pools[split]
        fam = fams[pool[k % len(pool)]]
        tp = S.XorShift64Star("rsi_v3|task|%s|%s|%d" % (seed, split, k))
        for _try in range(4000):
            prog, n_units = T2._sample_program(fam, tp)
            sg = T2._sig(prog)
            if sg in seen:
                continue
            inst = T2._instance(prog, tp)
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
    body = {"version": TASKS_VERSION, "atoms": [list(x) for x in atoms],
            "compounds": [list(c) for c in comps],
            "families": [{"name": f["name"],
                          "units": [list(u) for u in f["units"]],
                          "glue": list(f["glue"])} for f in families()]}
    return S.sha256_text(S.canon(body))


def manifest_digest(seeds):
    return S.sha256_text(S.canon([S.sha256_text(S.canon(seed_manifest(s)))
                                  for s in seeds]))


def improver_view(seed, n_batches):
    """(train_tasks, metaval_batches). FINAL HOLDOUT rows never leave
    this module through this path."""
    man = seed_manifest(seed)
    train = [ImproverTask(r) for r in man["train"]]
    mv = [ImproverTask(r) for r in man["metaval"]]
    assert n_batches * METAVAL_BATCH <= len(mv)
    batches = [mv[i * METAVAL_BATCH:(i + 1) * METAVAL_BATCH]
               for i in range(n_batches)]
    return train, batches
