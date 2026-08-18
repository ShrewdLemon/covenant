"""Common shape of a check outcome.

Every check produces a rate against a stated threshold, not just a verdict,
and writes a hash-stamped YAML record so the run is citable evidence.
A failing check is a covenant breach and exits non-zero in CI.

Records are replayable: the record contains only what the check computed
from its inputs, so the same inputs produce byte-identical records at the
same path. Run timestamps live in an append-only ``runs.log`` sidecar,
not in the record.
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from covenant.hashing import sha256_canonical
from covenant.store import Store


class CheckRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    check: str
    model_name: str
    passed: bool
    metrics: dict[str, float]
    thresholds: dict[str, float]
    n_evaluated: int
    inputs: dict[str, str]  # sha256 of model file, data snapshot, covenants
    config: dict[str, Any] = Field(default_factory=dict)
    details: dict[str, Any] = Field(default_factory=dict)
    record_sha256: str = ""

    def stamp(self) -> CheckRecord:
        body = self.model_dump(mode="json")
        body.pop("record_sha256")
        self.record_sha256 = sha256_canonical(body)
        return self

    def write(self, store: Store) -> str:
        """Idempotent: the path is derived from the record hash, an existing
        record is never rewritten, and every run is appended to runs.log."""
        store.init()
        path = store.check_path(self.model_name, self.check, self.record_sha256[:12])
        if not path.exists():
            store.write_check(
                self.model_name, self.check, self.record_sha256[:12],
                self.model_dump(mode="json"),
            )
        self._log_run(store, path)
        return str(path)

    def _log_run(self, store: Store, path: Path) -> None:
        log = store.root / "checks" / self.model_name / "runs.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now(dt.UTC).isoformat(timespec="seconds")
        verdict = "PASS" if self.passed else "BREACH"
        with open(log, "a") as f:
            f.write(f"{stamp} {self.check} {self.record_sha256[:12]} {verdict}\n")
