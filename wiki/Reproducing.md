# Reproducing

## Requirements
Python 3.8+ standard library only. No pip installs, no GPU, no internet.

## Fast (seconds–minutes)
```bash
python3 -m unittest discover -s tests -v
(cd src && python3 -m rsi_v2 verify && python3 -m rsi_v2 report)
(cd src && python3 -m rsi_v3 verify)             # v3 frozen protocol + ledger
python3 experiments/pc_analysis.py pc-devcheck-01 # process-controller go/no-go tables
python3 experiments/v3_devcheck_analysis.py v3devcheck-01-dev10config
python3 src/tforge.py selftest
python3 src/transferforge.py run 1 11 300 && python3 src/transferforge.py report
python3 src/omniforge.py selftest
```

## Full batteries (hours, any CPU)
Use the Kaggle kernels in `experiments/kaggle/` (they self-slice and checkpoint),
or run the drivers directly, e.g.:
```bash
python3 experiments/audit_v1_rerun.py run 101 200 4      # v1 audit rerun (shared vs arm-keyed streams)
(cd src && python3 -m rsi_v2 confirm --workers 4)          # v2 battery (needs a frozen prereg; see UPGRADE_V2_RESULTS.md)
(cd src && python3 -m rsi_v3 report --out /tmp/rep.json)   # v3 frozen report from the ledger (confirm already run)
python3 experiments/v3_repro_check.py v3devcheck-01-dev10config ADAPTIVE_META:3101   # re-run a v3 unit, compare
python3 src/sdt_layer.py run 101 140 99999 holdout      # gate-integrity, n=40
python3 src/openforge.py run OPEN 1 5000 99999          # open-ended long run
```
Everything is deterministic in the seed and resumable from its state file.
