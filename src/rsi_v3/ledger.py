"""
rsi_v3.ledger -- the append-only hash-chained ledger, made safe for
concurrent writers.

rsi_v2.ledger.Ledger keeps the chain head in memory; a long-running process
that appends after another process has written produces a mis-chained
record (this happened once during v3 development -- see the LEDGER_REPAIR
record and results/ledger/rsi_v3_ledger.orphans.jsonl). SafeLedger takes an
exclusive OS lock, re-reads and re-verifies the file, and only then appends
on top of the true current head. rsi_v2.ledger itself is unchanged (it is
hashed into the frozen v2 protocol).
"""
import fcntl
import json
import os
import time

from rsi_v2 import ledger as L2
from rsi_v2 import substrate as S

LedgerError = L2.LedgerError


class SafeLedger(L2.Ledger):
    def __init__(self, path):
        L2.Ledger.__init__(self, path)       # full load + full verify
        self._offset = os.path.getsize(path) if os.path.exists(path) else 0

    def _reload(self):
        """Incremental: read only bytes appended since our last view and
        verify them against our (already verified) chain head. A file that
        shrank or does not extend our view is reloaded and verified fully."""
        if not os.path.exists(self.path):
            self.records, self._offset = [], 0
            return
        size = os.path.getsize(self.path)
        if size < self._offset:
            L2.Ledger.__init__(self, self.path)
            self._offset = size
            return
        with open(self.path, "r", encoding="utf-8") as f:
            f.seek(self._offset)
            tail = f.read()
        prev = self.head
        for line in tail.splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            if r.get("seq") != len(self.records) or r["prev"] != prev or \
                    L2._hash(prev, r["kind"], r["body"]) != r["hash"]:
                raise LedgerError("record %d: chain broken" % len(
                    self.records))
            self.records.append(r)
            prev = r["hash"]
        self._offset = size

    def append(self, kind, body):
        body = dict(body)
        body.setdefault("utc", time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime()))
        d = os.path.dirname(self.path)
        if d and not os.path.isdir(d):
            os.makedirs(d)
        lock_path = self.path + ".lock"
        with open(lock_path, "a") as lk:
            fcntl.flock(lk.fileno(), fcntl.LOCK_EX)
            try:
                self._reload()           # catch up with other writers
                rec = {"seq": len(self.records), "kind": kind, "body": body,
                       "prev": self.head}
                rec["hash"] = L2._hash(rec["prev"], kind, body)
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, sort_keys=True) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                self.records.append(rec)
                self._offset = os.path.getsize(self.path)
            finally:
                fcntl.flock(lk.fileno(), fcntl.LOCK_UN)
        return rec["hash"]

    def find(self, kind, **match):
        self._reload()
        return L2.Ledger.find(self, kind, **match)


def digest(rec):
    return S.sha256_text(json.dumps(rec, sort_keys=True))
