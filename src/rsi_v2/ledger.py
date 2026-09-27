"""
rsi_v2.ledger -- append-only, hash-chained experiment ledger.

Every dev iteration, preregistration freeze, confirmatory unit START and
END (including failed, crashed and null runs) is appended here. Each
record commits to the previous record's hash, so deleting, reordering or
editing any line breaks `verify()`. A START without a matching END is
reported as an abandoned run, never silently dropped.
"""
import json
import os
import time

from . import substrate as S

GENESIS = "0" * 64


def _hash(prev, kind, body):
    return S.sha256_text(prev + S.canon({"kind": kind, "body": body}))


class LedgerError(AssertionError):
    pass


class Ledger(object):
    def __init__(self, path):
        self.path = path
        self.records = []
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.records.append(json.loads(line))
        self.verify()

    @property
    def head(self):
        return self.records[-1]["hash"] if self.records else GENESIS

    def verify(self):
        prev = GENESIS
        for i, r in enumerate(self.records):
            if r.get("seq") != i:
                raise LedgerError("record %d: sequence gap/reorder" % i)
            if r["prev"] != prev:
                raise LedgerError("record %d: prev-hash mismatch" % i)
            if _hash(prev, r["kind"], r["body"]) != r["hash"]:
                raise LedgerError("record %d: content hash mismatch" % i)
            prev = r["hash"]
        return True

    def append(self, kind, body):
        body = dict(body)
        body.setdefault("utc", time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                             time.gmtime()))
        rec = {"seq": len(self.records), "kind": kind, "body": body,
               "prev": self.head}
        rec["hash"] = _hash(rec["prev"], kind, body)
        d = os.path.dirname(self.path)
        if d and not os.path.isdir(d):
            os.makedirs(d)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self.records.append(rec)
        return rec["hash"]

    def find(self, kind, **match):
        return [r for r in self.records if r["kind"] == kind and
                all(r["body"].get(k) == v for k, v in match.items())]

    def abandoned(self, phase):
        """START records of `phase` with no END (crash, kill, deletion)."""
        ends = {(r["body"]["arm"], r["body"]["seed"])
                for r in self.find("UNIT_END", phase=phase)}
        return [r for r in self.find("UNIT_START", phase=phase)
                if (r["body"]["arm"], r["body"]["seed"]) not in ends]
