"""Common shape of a check outcome.

Every check produces a rate against a stated threshold, not just a verdict,
and writes a hash-stamped YAML record so the run is citable evidence.
A failing check is a covenant breach and exits non-zero in CI.
"""

from __future__ import annotations

import datetime as dt
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from covenant.hashing import sha256_canonical
from covenant.store import Store


class CheckRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    check: str
    model_name: str
    created_at: dt.datetime
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
        store.init()
        path = store.write_check(
            self.model_name, self.check, self.record_sha256[:12], self.model_dump(mode="json")
        )
        return str(path)
