"""
Executable anti-cheat defenses for rsi_v2 (and the v1 audit fix).

Run:  python3 -m unittest discover -s tests -v

Each test targets one way an RSI result could be faked or flattered:
extra hidden compute, condition-specific RNG, holdout leakage, seed /
task-id memorisation, evaluator modification, hard-coded answers,
weakened success criteria, selective deletion of runs, and counting
self-generated-task gains that do not transfer to external tasks.
Tests use tiny budgets so the suite runs in about a minute.
"""
import ast
import copy
import json
import os
import shutil
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

from rsi_v2 import substrate as S          # noqa: E402
from rsi_v2 import tasks as T              # noqa: E402
from rsi_v2 import solver as SV            # noqa: E402
from rsi_v2 import improver as I           # noqa: E402
from rsi_v2 import evaluate as E           # noqa: E402
from rsi_v2 import selfgen as SG           # noqa: E402
from rsi_v2 import runner as R             # noqa: E402
from rsi_v2 import stats as ST             # noqa: E402
from rsi_v2.ledger import Ledger, LedgerError   # noqa: E402

TINY = copy.deepcopy(I.HP)
TINY.update({"attempt_budget": 120, "gate_probe_budget": 80,
             "gate_round_budget": 2400})
SEED = 3


def _src(name):
    with open(os.path.join(SRC, "rsi_v2", name), encoding="utf-8") as f:
        return f.read()


class TestComputeMatching(unittest.TestCase):
    """Extra hidden compute."""

    def test_equal_caps_and_within_cap(self):
        caps, spent = set(), {}
        for arm in I.ARMS:
            cfg, rec = I.run_arm(arm, SEED, TINY)
            caps.add(rec["cap"])
            spent[arm] = rec["spent"]
            self.assertLessEqual(rec["spent"], rec["cap"])
            self.assertEqual(rec["spent"], rec["global_delta"])
        self.assertEqual(len(caps), 1, "arms must share one compute cap")
        self.assertEqual(spent["COLD"], 0)

    def test_phase_budgets_sum_to_same_cap(self):
        n = 40
        R_ = TINY["n_rounds"]
        rec_cap = R_ * (n * TINY["attempt_budget"]
                        + TINY["gate_round_budget"])
        single_cap = (R_ * n * TINY["attempt_budget"]
                      + R_ * TINY["gate_round_budget"])
        self.assertEqual(rec_cap, single_cap)
        self.assertEqual(rec_cap, I.total_cap(TINY, n))

    def test_hidden_compute_in_improver_is_detected(self):
        orig = I.run_recursive

        def cheating(*a, **k):
            st = orig(*a, **k)
            S.execute(S.fns_of(("sort",)), [3, 1, 2])  # unmetered peek
            return st
        I.run_recursive = cheating
        try:
            with self.assertRaises(S.HiddenComputeError):
                I.run_arm("RECURSIVE_FULL", SEED, TINY)
        finally:
            I.run_recursive = orig

    def test_hidden_compute_in_evaluator_is_detected(self):
        orig = SV.search

        def cheating(comp, public, budget, prng, meter):
            S.execute(S.fns_of(("reverse",)), [1, 2])
            return orig(comp, public, budget, prng, meter)
        SV.search = cheating
        try:
            with self.assertRaises(S.HiddenComputeError):
                E.evaluate(SV.base_config(), SEED, budget=50)
        finally:
            SV.search = orig

    def test_meter_is_a_hard_cap(self):
        m = S.Meter(5, "cap")
        fns = S.fns_of(("inc",))
        for _ in range(5):
            m.run(fns, [1])
        with self.assertRaises(S.BudgetExceeded):
            m.run(fns, [1])
        parent = S.Meter(3, "parent")
        child = parent.child(100, "child")
        for _ in range(3):
            child.run(fns, [1])
        with self.assertRaises(S.BudgetExceeded):
            child.run(fns, [1])

    def test_search_respects_budget(self):
        task = T.PublicTask(T.seed_manifest(SEED)["holdout_ext"][-1]["train"])
        for b in (1, 17, 300):
            m = S.Meter(10 ** 9)
            res = SV.search(SV.Compiled(SV.base_config()), task, b,
                            S.XorShift64Star("x"), m)
            self.assertLessEqual(res.spent, b)
            self.assertEqual(m.spent, res.spent)


class TestCommonRandomNumbers(unittest.TestCase):
    """Condition-specific RNG."""

    def test_eval_stream_has_no_arm(self):
        tree = ast.parse(_src("evaluate.py"))
        fn = [n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "evaluate"][0]
        self.assertNotIn("arm", [a.arg for a in fn.args.args])
        self.assertNotIn("arm", _src("evaluate.py").split(
            "def eval_stream")[1].split("def ")[0])

    def test_evaluation_identical_for_identical_config(self):
        cfg = SV.base_config()
        a = E.evaluate(cfg, SEED, budget=200)
        b = E.evaluate(cfg.copy(), SEED, budget=200)
        self.assertEqual(a["per_task"], b["per_task"])
        c = E.evaluate(cfg, SEED, budget=200, stream=1)
        self.assertEqual(c["search_execs"] > 0, True)

    def test_improvement_streams_never_use_arm(self):
        tree = ast.parse(_src("improver.py"))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "stream"):
                names = {n.id for a in node.args for n in ast.walk(a)
                         if isinstance(n, ast.Name)}
                self.assertNotIn("arm", names)

    def test_round1_identical_across_recursive_arms(self):
        """FULL / FROZEN / NODIAG start from the same policy and share every
        stream, so their first round must be bit-identical."""
        recs = {}
        for arm in ("RECURSIVE_FULL", "RECURSIVE_FROZEN",
                    "RECURSIVE_NODIAG"):
            _cfg, rec = I.run_arm(arm, SEED, TINY)
            r1 = dict(rec["rounds"][0])
            r1.pop("theta_after")
            recs[arm] = json.dumps(r1, sort_keys=True)
        self.assertEqual(len(set(recs.values())), 1)

    def test_single_uses_the_recursive_attempt_streams(self):
        """SINGLE_5X restart r of task i == recursive round r of task i."""
        train, _b = T.improver_view(SEED)
        st = I.ImproverState(TINY)
        comp = SV.Compiled(st.cfg)
        m = S.Meter(10 ** 9)
        a = I.attempt_task(st, comp, train[0], 0, 120,
                           I.stream(SEED, "attempt", 1, 0), m)
        st2 = I.ImproverState(TINY)
        b = I.attempt_task(st2, comp, train[0], 0, 120,
                           I.stream(SEED, "attempt", 1, 0), S.Meter(10 ** 9))
        self.assertEqual((a.solved, a.spent, a.best_prog),
                         (b.solved, b.spent, b.best_prog))

    def test_v1_audit_fix_eval_stream_is_arm_independent(self):
        """v1 audit: omniforge's held-out streams were keyed on the arm name
        ('ev|<cond>|...'). Every harness must now use lf_eval_prng."""
        import inspect
        import omniforge as OM
        for fn in (OM.xv_run_unit, OM.xvi_run_unit, OM.gx_run_unit,
                   OM.up_eval_unit):
            src = inspect.getsource(fn)
            self.assertIn("lf_eval_prng(", src)
            self.assertNotIn("'ev|%s|%s|%d|%d' % (cond", src)
        with open(os.path.join(SRC, "rsi_upgrade.py"),
                  encoding="utf-8") as f:
            self.assertIn("OM.lf_eval_prng(cond", f.read())
        a = OM.lf_eval_prng("ROUND5", 101, 3, 0).u64()
        b = OM.lf_eval_prng("R5PLUS", 101, 3, 0).u64()
        c = OM.lf_eval_prng("COLD2", 101, 3, 0).u64()
        self.assertEqual(a, b)          # counterfactual arms: same stream
        self.assertNotEqual(a, c)       # only the noise-floor replicate


class TestHoldoutIsolation(unittest.TestCase):
    """Holdout leakage."""

    def test_behaviour_disjoint_splits(self):
        for seed in (1, 2, SEED, 1001):
            man = T.seed_manifest(seed)
            sigs, ids = {}, set()
            for split, _n in T.SPLITS:
                for row in man[split]:
                    sg = S.signature(S.fns_of(row["program"]),
                                     S.Meter(10 ** 9))
                    self.assertNotIn(sg, sigs, "behaviour shared across "
                                     "%s and %s" % (sigs.get(sg), split))
                    sigs[sg] = split
                    self.assertNotIn(row["id"], ids)
                    ids.add(row["id"])
            fam_ext = {r["family"] for r in man["holdout_ext"]}
            fam_tr = {r["family"] for s in ("train", "probe", "holdout_in")
                      for r in man[s]}
            self.assertFalse(fam_ext & fam_tr,
                             "external families must be unseen")

    def test_improver_view_has_no_holdout(self):
        train, batches = T.improver_view(SEED)
        man = T.seed_manifest(SEED)
        hold = {json.dumps(r["train"]) for s in E.HOLDOUT_SPLITS
                for r in man[s]}
        seen = [json.dumps([[list(x), list(y)] for x, y in t.train])
                for t in train + [p for b in batches for p in b]]
        self.assertFalse(hold & set(seen))
        for t in train:
            for attr in ("id", "family", "program", "split"):
                self.assertFalse(hasattr(t, attr))

    def test_improvement_phase_never_reads_holdout(self):
        with E.sealed():
            with self.assertRaises(E.HoldoutSealedError):
                E.holdout_rows(SEED, "holdout_ext")
            for arm in ("SINGLE_5X", "RECURSIVE_FULL"):
                I.run_arm(arm, SEED, TINY)   # would raise if it peeked

    def test_improver_and_solver_have_no_holdout_code_path(self):
        for name in ("improver.py", "solver.py"):
            tree = ast.parse(_src(name))
            names = {n.attr for n in ast.walk(tree)
                     if isinstance(n, ast.Attribute)}
            names |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            for bad in ("holdout_rows", "seed_manifest", "evaluate", "E"):
                self.assertNotIn(bad, names, "%s references %s"
                                 % (name, bad))


class TestNoMemorisation(unittest.TestCase):
    """Seed / task-id memorisation and hard-coded answers."""

    def test_public_task_exposes_train_only(self):
        self.assertEqual(T.PublicTask.__slots__, ("train",))
        self.assertEqual(T.ImproverTask.__slots__, ("train", "test"))

    def test_task_ids_do_not_affect_results(self):
        cfg = SV.base_config()
        a = E.evaluate(cfg, SEED, budget=150)
        orig = T._task_id
        T._task_id = lambda seed, split, k: "relabelled-%d-%s" % (k, split)
        T._MAN.clear()
        try:
            b = E.evaluate(cfg, SEED, budget=150)
            self.assertTrue(T.seed_manifest(SEED)["train"][0]["id"]
                            .startswith("relabelled"))
        finally:
            T._task_id = orig
            T._MAN.clear()
        self.assertEqual(a["per_task"], b["per_task"])

    def test_solver_api_receives_no_seed_or_id(self):
        tree = ast.parse(_src("solver.py"))
        fn = [n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "search"][0]
        self.assertEqual([a.arg for a in fn.args.args],
                         ["comp", "public", "budget", "prng", "meter"])

    def test_final_config_carries_no_answers(self):
        cfg, _rec = I.run_arm("RECURSIVE_FULL", SEED, TINY)
        blob = json.dumps(cfg.to_json())
        self.assertEqual(set(cfg.to_json()),
                         {"macros", "weights", "max_tokens", "p_random"})
        for tok in cfg.macros:          # only primitive names inside
            for p in SV.expansion(tok):
                self.assertIn(p, S.PRIMS)
        man = T.seed_manifest(SEED)
        for s in E.HOLDOUT_SPLITS:
            for row in man[s]:
                self.assertNotIn(row["id"], blob)

    def test_no_holdout_outputs_hardcoded_in_source(self):
        text = "".join(_src(n) for n in ("solver.py", "improver.py",
                                         "selfgen.py"))
        compact = text.replace(" ", "")
        for seed in (1, 1001, 1300):
            for s in E.HOLDOUT_SPLITS:
                for row in T.seed_manifest(seed)[s]:
                    for _x, y in row["test"]:
                        if len(y) >= 4:
                            self.assertNotIn(json.dumps(y).replace(" ", ""),
                                             compact)

    def test_shuffled_labels_are_not_solved(self):
        """The scorer checks the REAL hidden outputs: a config that solves
        tasks cannot also 'solve' them with another task's test labels."""
        atoms, comps = T.motif_library()
        cfg = SV.base_config()
        for m in atoms + comps:
            tok = SV.macro_token(m)
            cfg.macros += (tok,)
            cfg.weights[tok] = 1.0
        comp = SV.Compiled(cfg)
        rows = T.seed_manifest(SEED)["holdout_ext"]
        solved_true = solved_shuf = 0
        for k, row in enumerate(rows):
            other = rows[(k + 1) % len(rows)]
            res = SV.search(comp, T.PublicTask(row["train"]), 1500,
                            E.eval_stream(SEED, "holdout_ext", k),
                            S.Meter(10 ** 9))
            m = S.Meter(10 ** 9)
            if res.program is not None:
                solved_true += SV.check(comp, res.program, row["test"], m)
                shuf = [(x, y) for (x, _), (_, y) in zip(row["test"],
                                                          other["test"])]
                solved_shuf += SV.check(comp, res.program, shuf, m)
        self.assertGreater(solved_true, 3)
        self.assertEqual(solved_shuf, 0)


class TestFrozenProtocol(unittest.TestCase):
    """Evaluator modification and weakened success criteria."""

    def setUp(self):
        self.led = Ledger(R.LEDGER_PATH)
        if not self.led.find("PREREG_FREEZE"):
            self.skipTest("protocol not frozen yet")

    def test_protocol_hashes_match_freeze(self):
        R.check_frozen(self.led)   # raises on any drift

    def test_evaluator_tampering_is_detected(self):
        orig = R.file_hashes

        def tampered():
            h = orig()
            h["evaluate.py"] = "0" * 64
            return h
        R.file_hashes = tampered
        try:
            with self.assertRaises(AssertionError):
                R.check_frozen(self.led)
        finally:
            R.file_hashes = orig

    def test_weakened_criteria_are_detected(self):
        orig = R.load_prereg

        def weakened():
            p, _sha = orig()
            p = copy.deepcopy(p)
            p["statistics"]["alpha"] = 0.2
            text = json.dumps(p)
            return p, S.sha256_text(text)
        R.load_prereg = weakened
        try:
            with self.assertRaises(AssertionError):
                R.check_frozen(self.led)
        finally:
            R.load_prereg = orig

    def test_prereg_criteria_are_strict(self):
        p, _ = R.load_prereg()
        self.assertEqual(p["statistics"]["alpha"], 0.05)
        self.assertEqual(p["primary_metric"]["field"], "ext")
        self.assertEqual(p["primary_metric"]["split"], "holdout_ext")
        prim = [c for c in p["confirmatory_contrasts"]
                if c["role"] == "primary"][0]
        self.assertEqual((prim["a"], prim["b"]),
                         ("RECURSIVE_FULL", "SINGLE_5X"))
        seeds = R.parse_seeds(p["seeds"]["confirm"])
        self.assertFalse(set(seeds) & set(R.DEV_SEEDS))

    def test_hp_changes_are_detected(self):
        saved = I.HP["attempt_budget"]
        I.HP["attempt_budget"] = saved + 1
        try:
            with self.assertRaises(AssertionError):
                R.check_frozen(self.led)
        finally:
            I.HP["attempt_budget"] = saved

    def test_no_confirm_unit_before_freeze(self):
        fz = self.led.find("PREREG_FREEZE")[0]["seq"]
        for r in self.led.records:
            if r["kind"] in ("UNIT_START", "UNIT_END") and \
                    r["body"].get("phase") == "confirm":
                self.assertGreater(r["seq"], fz)
        for r in self.led.find("DEV_START") + self.led.find("DEV_END"):
            for s in r["body"]["seeds"]:
                self.assertIn(s, R.DEV_SEEDS)


class TestLedger(unittest.TestCase):
    """Selective deletion of failed runs."""

    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.p = os.path.join(self.d, "l.jsonl")
        led = Ledger(self.p)
        for i in range(5):
            led.append("UNIT_START", {"phase": "confirm", "arm": "A",
                                      "seed": i})
            if i != 3:
                led.append("UNIT_END", {"phase": "confirm", "arm": "A",
                                        "seed": i, "ext": i})

    def tearDown(self):
        shutil.rmtree(self.d)

    def _lines(self):
        with open(self.p) as f:
            return f.readlines()

    def _write(self, lines):
        with open(self.p, "w") as f:
            f.writelines(lines)

    def test_intact_ledger_verifies(self):
        self.assertTrue(Ledger(self.p).verify())

    def test_deleted_record_breaks_chain(self):
        lines = self._lines()
        del lines[4]
        self._write(lines)
        with self.assertRaises(LedgerError):
            Ledger(self.p)

    def test_deleted_tail_is_visible_as_abandoned(self):
        led = Ledger(self.p)
        self.assertEqual([(r["body"]["arm"], r["body"]["seed"])
                          for r in led.abandoned("confirm")], [("A", 3)])

    def test_edited_record_breaks_chain(self):
        lines = self._lines()
        lines[1] = lines[1].replace('"ext": 0', '"ext": 9')
        self._write(lines)
        with self.assertRaises(LedgerError):
            Ledger(self.p)

    def test_reordered_records_break_chain(self):
        lines = self._lines()
        lines[1], lines[2] = lines[2], lines[1]
        self._write(lines)
        with self.assertRaises(LedgerError):
            Ledger(self.p)

    def test_missing_units_block_the_verdict(self):
        led = Ledger(self.p)
        tab, missing = R.collect(led, "confirm", ["A"], [0, 1, 2, 3, 4])
        self.assertEqual(missing, [("A", 3)])


class TestSelfGeneratedDoesNotCount(unittest.TestCase):
    """Self-generated-task gains that do not transfer externally."""

    def test_self_score_can_rise_without_external_gain(self):
        """A 'self-gamer' that stuffs its vocabulary with random macros
        makes its OWN generated tasks easy (they are built from those
        macros) without solving more external tasks."""
        prng = S.XorShift64Star("gamer")
        gamer = SV.base_config()
        for _ in range(12):
            m = tuple(prng.choice(S.NAMES) for _ in range(3))
            tok = SV.macro_token(m)
            if tok not in gamer.macros:
                gamer.macros += (tok,)
                gamer.weights[tok] = 20.0
        cold = SV.base_config()
        sg_g = SG.self_score(gamer, SEED)["self_solved"]
        sg_c = SG.self_score(cold, SEED)["self_solved"]
        ext_g = E.evaluate(gamer, SEED, budget=600)["by_split"]["holdout_ext"]
        ext_c = E.evaluate(cold, SEED, budget=600)["by_split"]["holdout_ext"]
        self.assertGreaterEqual(sg_g, 1)
        self.assertLessEqual(ext_g, ext_c + 1)
        # whatever those numbers are, the verdict ignores self scores:
        self.assertIsNotNone(sg_c)

    def test_report_primary_uses_external_metric_only(self):
        src = _src("runner.py")
        body = src.split("def build_report")[1].split("\ndef ")[0]
        self.assertNotIn("self_solved\"]", body.split("rep[\"means\"]")[0])
        self.assertIn('metric = prereg["primary_metric"]["field"]', body)
        self.assertIn('assert metric == "ext"', body)


class TestStats(unittest.TestCase):
    def test_perm_and_holm(self):
        self.assertLess(ST.perm_p([1] * 20, "t", one_sided=True), 0.001)
        self.assertGreater(ST.perm_p([1, -1] * 10, "t2", one_sided=True),
                           0.3)
        h = ST.holm({"a": 0.01, "b": 0.04})
        self.assertAlmostEqual(h["a"], 0.02)
        self.assertAlmostEqual(h["b"], 0.04)


class TestSubstrate(unittest.TestCase):
    def test_primitives_match_omniforge(self):
        """rsi_v2 primitives are the omniforge lf_* primitives."""
        with open(os.path.join(SRC, "omniforge.py"), encoding="utf-8") as f:
            src = f.read()
        a = src.index("def lf__c(x):")
        b = src.index("lf_NAMES = sorted(lf_PRIMS)", a)
        ns = {"lf_MAXLEN": 24}
        exec(src[a:b], ns)
        for nm in S.NAMES:
            for x in S.PROBE_INPUTS + ([],):
                self.assertEqual(S.PRIMS[nm](list(x)),
                                 ns["lf_PRIMS"][nm](list(x)), nm)
        self.assertEqual(sorted(ns["lf_PRIMS"]), list(S.NAMES))

    def test_determinism(self):
        a = R.run_unit(("RECURSIVE_FULL", SEED, TINY))
        b = R.run_unit(("RECURSIVE_FULL", SEED, TINY))
        a.pop("elapsed_s")
        b.pop("elapsed_s")
        self.assertEqual(json.dumps(a, sort_keys=True),
                         json.dumps(b, sort_keys=True))


if __name__ == "__main__":
    unittest.main()


class TestPipelineDryRun(unittest.TestCase):
    """freeze -> confirm -> report on temp paths (tiny budget, 1 seed), so
    the frozen report code is exercised before any real confirmatory run;
    also checks that freezing is one-shot and re-runs are refused."""

    def test_end_to_end(self):
        d = tempfile.mkdtemp()
        saved = (R.LEDGER_PATH, R.PREREG_PATH, R.CONFIRM_LOG, dict(I.HP))
        try:
            R.LEDGER_PATH = os.path.join(d, "ledger.jsonl")
            R.PREREG_PATH = os.path.join(d, "prereg.json")
            R.CONFIRM_LOG = os.path.join(d, "confirm.jsonl")
            I.HP.clear()
            I.HP.update(TINY)
            with open(os.path.join(ROOT, "results",
                                   "PREREGISTRATION_V2.json")) as f:
                p = json.load(f)
            p["seeds"]["confirm"] = "9001"
            p["hp_sha"] = S.sha256_text(S.canon(I.HP))
            with open(R.PREREG_PATH, "w") as f:
                json.dump(p, f)
            R.cmd_freeze()
            with self.assertRaises(SystemExit):
                R.cmd_freeze()                       # one-shot
            R.cmd_confirm(1)
            led = Ledger(R.LEDGER_PATH)
            n_end = len(led.find("UNIT_END"))
            R.cmd_confirm(1)                         # nothing re-run
            self.assertEqual(len(Ledger(R.LEDGER_PATH).find("UNIT_END")),
                             n_end)
            rep = R.build_report(Ledger(R.LEDGER_PATH))
            self.assertEqual(rep["missing_units"], [])
            self.assertIn(rep["verdict"], ("PRIMARY PASSED",
                                           "PRIMARY NOT SUPPORTED (null)"))
            self.assertEqual(set(rep["confirmatory"]),
                             {c["name"] for c in p["confirmatory_contrasts"]})
            self.assertEqual(len({tuple(v) for v in rep["caps"].values()}),
                             1)
        finally:
            R.LEDGER_PATH, R.PREREG_PATH, R.CONFIRM_LOG = saved[:3]
            I.HP.clear()
            I.HP.update(saved[3])
            shutil.rmtree(d)
