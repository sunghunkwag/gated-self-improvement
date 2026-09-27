"""
Executable anti-cheat defenses for rsi_v3 (the meta-learned improver).

Run:  python3 -m unittest discover -s tests -v

Carries every v2 defense over to v3 and adds the v3-specific ones:
FINAL HOLDOUT sealed (meta-val allowed), meta-predictor trained only on
paired META-VAL outcomes, frozen predictor never changes, diagnosis
ablation changes the generated candidates, seed ranges never touch the v2
confirmatory seeds, per-world compute caps, cross-process determinism.
"""
import ast
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from rsi_v2 import substrate as S               # noqa: E402
from rsi_v2 import solver as SV                 # noqa: E402
from rsi_v3.ledger import SafeLedger as Ledger  # noqa: E402
from rsi_v3 import tasks as T                   # noqa: E402
from rsi_v3 import improver as I                # noqa: E402
from rsi_v3 import evaluate as E                # noqa: E402
from rsi_v3 import runner as R                  # noqa: E402
from rsi_v3.metamodel import Ridge              # noqa: E402

TINY = copy.deepcopy(I.HP)
TINY.update({"n_worlds": 2, "n_rounds": 5, "attempt_budget": 100,
             "gate_probe_budget": 60, "gate_round_budget": 3000,
             "pool_size": 10, "single_pool_size": 12, "k_screen": 4})
SEED = 3001


def _src(name):
    with open(os.path.join(SRC, "rsi_v3", name), encoding="utf-8") as f:
        return f.read()


class TestComputeV3(unittest.TestCase):
    def test_equal_caps_and_metered(self):
        caps = set()
        for arm in I.ARMS:
            cfgs, rec = I.run_arm(arm, SEED, TINY)
            caps.add(rec["cap"])
            self.assertEqual(rec["spent"], rec["global_delta"])
            self.assertLessEqual(rec["spent"], rec["cap"])
            for w in rec["worlds"]:
                self.assertLessEqual(w["spent"], rec["world_cap"])
            self.assertEqual(len(cfgs), TINY["n_worlds"])
        self.assertEqual(len(caps), 1)

    def test_hidden_compute_detected(self):
        orig = I.run_recursive

        def cheat(*a, **k):
            out = orig(*a, **k)
            S.execute(S.fns_of(("sort",)), [2, 1])
            return out
        I.run_recursive = cheat
        try:
            with self.assertRaises(S.HiddenComputeError):
                I.run_arm("ADAPTIVE_META", SEED, TINY)
        finally:
            I.run_recursive = orig

    def test_diagnostic_arms_are_not_contenders(self):
        self.assertNotIn("ORACLE_RANK", I.ARMS)
        p, _ = R.load_prereg() if os.path.exists(R.PREREG_PATH) else (
            {"arms": I.ARMS}, None)
        self.assertFalse(set(p["arms"]) & set(I.DIAGNOSTIC_ARMS))


class TestCRNV3(unittest.TestCase):
    def test_streams_never_use_arm(self):
        for name in ("improver.py", "evaluate.py"):
            tree = ast.parse(_src(name))
            for node in ast.walk(tree):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id in ("stream", "eval_stream")):
                    names = {n.id for a in node.args for n in ast.walk(a)
                             if isinstance(n, ast.Name)}
                    self.assertNotIn("arm", names)
                    self.assertNotIn("policy", names)

    def test_frozen_and_adaptive_identical_first_round(self):
        a = I.run_arm("FROZEN_META", SEED, TINY)[1]
        b = I.run_arm("ADAPTIVE_META", SEED, TINY)[1]
        ra, rb = a["worlds"][0]["rounds"][0], b["worlds"][0]["rounds"][0]
        strip = lambda r: json.dumps(
            {k: v for k, v in r.items()
             if k not in ("model_n", "n_rows", "meta_seconds")},
            sort_keys=True)
        self.assertEqual(strip(ra), strip(rb))

    def test_final_eval_identical_for_identical_config(self):
        cfg = SV.base_config()
        x = E.evaluate(cfg, I.world_key(SEED, 1), budget=150)
        y = E.evaluate(cfg.copy(), I.world_key(SEED, 1), budget=150)
        self.assertEqual(x["per_task"], y["per_task"])


class TestSplitsV3(unittest.TestCase):
    def test_three_way_family_split_and_behaviour_disjoint(self):
        for key in (I.world_key(SEED, 1), I.world_key(7001, 3)):
            man = T.seed_manifest(key)
            tr, mv, fi = (set(man["train_families"]),
                          set(man["metaval_families"]),
                          set(man["final_families"]))
            self.assertFalse(tr & mv or tr & fi or mv & fi)
            self.assertEqual({r["family"] for r in man["metaval"]} - mv,
                             set())
            self.assertEqual({r["family"] for r in man["holdout_ext"]} - fi,
                             set())
            sigs = set()
            for split, _n in T.SPLITS:
                for row in man[split]:
                    sg = S.signature(S.fns_of(row["program"]),
                                     S.Meter(10 ** 9))
                    self.assertNotIn(sg, sigs)
                    sigs.add(sg)

    def test_final_holdout_sealed_metaval_allowed(self):
        with E.sealed():
            with self.assertRaises(E.HoldoutSealedError):
                E.final_rows(I.world_key(SEED, 1), "holdout_ext")
            train, batches = T.improver_view(I.world_key(SEED, 1), 2)
            self.assertTrue(batches[0])
            I.run_arm("ADAPTIVE_META", SEED, TINY)

    def test_improver_has_no_final_holdout_path(self):
        for name in ("improver.py", "metamodel.py"):
            tree = ast.parse(_src(name))
            names = {n.attr for n in ast.walk(tree)
                     if isinstance(n, ast.Attribute)}
            names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            for bad in ("final_rows", "seed_manifest", "evaluate", "E",
                        "holdout_ext", "holdout_in"):
                self.assertNotIn(bad, names)

    def test_improver_view_contains_no_final_rows(self):
        key = I.world_key(SEED, 2)
        man = T.seed_manifest(key)
        final = {json.dumps(r["train"]) for s in T.FINAL_SPLITS
                 for r in man[s]}
        train, batches = T.improver_view(key, 8)
        seen = {json.dumps([[list(x), list(y)] for x, y in t.train])
                for t in train + [p for b in batches for p in b]}
        self.assertFalse(final & seen)


class TestMetaPredictor(unittest.TestCase):
    def test_frozen_predictor_never_learns(self):
        m = Ridge(3, 1.0)
        m.frozen = True
        with self.assertRaises(AssertionError):
            m.add([1.0, 0.0, 0.0], 1.0)
        _cfgs, rec = I.run_arm("FROZEN_META", SEED, TINY)
        self.assertEqual(rec["model_n"], 0)
        self.assertTrue(all(w == 0.0 for w in rec["model_w"]))

    def test_adaptive_trains_on_every_candidate_x_probe_row(self):
        _cfgs, rec = I.run_arm("ADAPTIVE_META", SEED, TINY)
        rows = sum(rd["n_rows"] for wd in rec["worlds"]
                   for rd in wd["rounds"])
        screened = sum(1 for wd in rec["worlds"] for rd in wd["rounds"]
                       for s in rd["screened"] if "y_mean" in s)
        half = T.METAVAL_BATCH // 2
        self.assertGreaterEqual(rows, screened * half)
        self.assertEqual(rec["model_n"], rows)

    def test_nocarry_resets_but_carry_persists(self):
        _c, a = I.run_arm("ADAPTIVE_META", SEED, TINY)
        _c, b = I.run_arm("ADAPTIVE_NOCARRY", SEED, TINY)
        rows_last = sum(rd["n_rows"] for rd in b["worlds"][-1]["rounds"])
        self.assertEqual(b["model_n"], rows_last)
        self.assertGreater(a["model_n"], rows_last)

    def test_learning_rows_use_metaval_probes_only(self):
        """The only data a learning arm sees are paired outcomes on META-VAL
        probes: gate_eval is only ever called with meta-val batches."""
        src = _src("improver.py")
        self.assertIn("batches[r - 1]", src)
        tree = ast.parse(src)
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "gate_eval"):
                arg = node.args[1]
                self.assertTrue(isinstance(arg, ast.Name) and arg.id in (
                    "screen", "confirm", "probes"))


class TestDiagnosisAblation(unittest.TestCase):
    def test_diagnosis_changes_generated_candidates(self):
        """Mechanical proof: with the diagnosis switched off, the same
        state produces a different candidate pool (allocation differs),
        on most first rounds across seeds."""
        hp = I.HP        # the real (frozen-candidate) configuration
        differ, n = 0, 0
        for sd in (3001, 3002, 3003, 3004):
            for w in (1, 2):
                key = I.world_key(sd, w)
                train, _b = T.improver_view(key, 1)
                st = I.State()
                m = S.Meter(10 ** 12)
                comp = SV.Compiled(st.cfg)
                traces = [I.attempt(st, comp, t, i, hp["attempt_budget"],
                                    I.stream(key, "attempt", 1, i), m)
                          for i, t in enumerate(train)]
                v, _vd, info = I.diagnose(st, traces, 1, hp)
                p1, _a1, _ = I.generate_pool(st, traces, train, v, info, m,
                                             hp, True, hp["pool_size"])
                p2, _a2, _ = I.generate_pool(st, traces, train, v, info, m,
                                             hp, False, hp["pool_size"])
                n += 1
                differ += [c.label for c in p1] != [c.label for c in p2]
        self.assertGreaterEqual(differ, 6)      # >= 75% of real states

    def test_diagnosis_enters_the_predictor(self):
        """Ranking channel: the matched failure-score features are non-zero
        with diagnosis and exactly zero in the NODIAG ablation."""
        c = I.Cand("MINE", "MINE[0]", SV.base_config(), [0.0] * I.N_DELTA,
                   1.0)
        v = [0.7] * I.N_STATE
        on = I.features(c, v, True, 0.0)
        off = I.features(c, v, False, 0.0)
        self.assertNotEqual(on, off)
        self.assertEqual(len(on), I.feat_dim())

    def test_pruning_requires_evidence(self):
        st = I.State()
        ok = I.availability(st, [], True, {"dead": 0.0, "regress": 0.0},
                            TINY)
        self.assertFalse(ok["PRUNE"])
        st.cfg = SV.SolverConfig(("<sort.diff>",),
                                 dict(st.cfg.weights, **{"<sort.diff>": 1.0}))
        ok = I.availability(st, [], True, {"dead": 0.0, "regress": 0.0},
                            TINY)
        self.assertFalse(ok["PRUNE"])
        ok = I.availability(st, [], True, {"dead": 1.0, "regress": 0.0},
                            TINY)
        self.assertTrue(ok["PRUNE"])


class TestProtocolV3(unittest.TestCase):
    def test_seed_ranges_never_touch_v2_or_each_other(self):
        self.assertFalse(set(R.DEV_ITERATE) & set(R.FORBIDDEN))
        self.assertFalse(set(R.DEV_CHECK) & set(R.FORBIDDEN))
        self.assertFalse(set(R.DEV_ITERATE) & set(R.DEV_CHECK))
        for s in range(1001, 1301):
            self.assertIn(s, R.FORBIDDEN)
        led = Ledger(R.LEDGER_PATH)
        for r in led.records:
            for s in r["body"].get("seeds", []) or []:
                self.assertNotIn(s, R.FORBIDDEN)

    def test_dev_runs_refuse_forbidden_seeds(self):
        with self.assertRaises(SystemExit):
            R._dev_like("DEV", "x", [1001], {}, 1, ["COLD"], R.DEV_ITERATE)

    def test_frozen_protocol_if_frozen(self):
        led = Ledger(R.LEDGER_PATH)
        if not led.find("PREREG_FREEZE"):
            self.skipTest("v3 protocol not frozen yet")
        R.check_frozen(led)
        orig = R.file_hashes

        def tampered():
            h = orig()
            h["rsi_v3/evaluate.py"] = "0" * 64
            return h
        R.file_hashes = tampered
        try:
            with self.assertRaises(AssertionError):
                R.check_frozen(led)
        finally:
            R.file_hashes = orig
        p, _ = R.load_prereg()
        self.assertEqual(p["primary_metric"]["split"], "holdout_ext")
        self.assertEqual(p["statistics"]["alpha"], 0.05)
        seeds = R.parse_seeds(p["seeds"]["confirm"])
        self.assertFalse(set(seeds) & (set(R.FORBIDDEN) | set(R.DEV_ITERATE)
                                       | set(R.DEV_CHECK)))
        fz = led.find("PREREG_FREEZE")[0]["seq"]
        for r in led.records:
            if r["kind"] in ("UNIT_START", "UNIT_END") and \
                    r["body"].get("phase") == "confirm":
                self.assertGreater(r["seq"], fz)


class TestLedgerConcurrency(unittest.TestCase):
    def test_stale_writer_cannot_fork_the_chain(self):
        """Reproduces the dev-11 race: a writer holding a stale copy
        appends after another writer. SafeLedger must re-chain on top of
        the true head, keeping the file verifiable."""
        d = tempfile.mkdtemp()
        try:
            p = os.path.join(d, "l.jsonl")
            a = Ledger(p)
            a.append("X", {"i": 0})
            stale = Ledger(p)
            a.append("X", {"i": 1})
            a.append("X", {"i": 2})
            stale.append("X", {"i": 3})    # stale in-memory head
            fresh = Ledger(p)
            self.assertTrue(fresh.verify())
            self.assertEqual([r["body"]["i"] for r in fresh.records],
                             [0, 1, 2, 3])
        finally:
            shutil.rmtree(d)


class TestDeterminismV3(unittest.TestCase):
    def test_cross_process_determinism(self):
        """Same unit in two fresh processes with different hash seeds."""
        code = ("import sys, json; sys.path.insert(0, %r);"
                "from rsi_v3 import improver as I;"
                "hp = json.loads(%r);"
                "cfgs, rec = I.run_arm('ADAPTIVE_META', %d, hp);"
                "print(json.dumps([c.sha() for c in cfgs] + [rec['spent'],"
                " rec['model_w']]))" % (SRC, json.dumps(TINY), SEED))
        outs = []
        for hs in ("1", "2"):
            env = dict(os.environ, PYTHONHASHSEED=hs)
            outs.append(subprocess.check_output([sys.executable, "-c", code],
                                                env=env).strip())
        self.assertEqual(outs[0], outs[1])


if __name__ == "__main__":
    unittest.main()
