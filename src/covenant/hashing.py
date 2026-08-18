"""Content addressing for models, data snapshots and records.

Hashes are taken over the artefact bytes as supplied (never re-serialised:
pickle bytes are not stable across library versions), over canonical JSON
for structured records, and over a stable row/column digest for dataframes.
Library versions are recorded alongside hashes so a reader can reproduce
the context in which a hash was taken.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    """Serialise to JSON with sorted keys and fixed separators.

    The same logical record always produces the same bytes, regardless of
    dict insertion order.
    """
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_fallback)


def _json_fallback(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if hasattr(obj, "isoformat"):  # date, datetime
        return obj.isoformat()
    raise TypeError(f"not canonically serialisable: {type(obj)!r}")


def sha256_canonical(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def sha256_dataframe(df: pd.DataFrame) -> str:
    """Deterministic digest of a dataframe's schema and values.

    Row order and column order both matter: a reordered snapshot is a
    different snapshot. Uses pandas' stable per-row hashing rather than
    parquet bytes so the digest does not depend on a parquet writer version.
    """
    h = hashlib.sha256()
    schema = [(str(c), str(t)) for c, t in zip(df.columns, df.dtypes, strict=True)]
    h.update(canonical_json(schema).encode("utf-8"))
    row_hashes = pd.util.hash_pandas_object(df, index=False).to_numpy()
    h.update(np.ascontiguousarray(row_hashes).tobytes())
    return h.hexdigest()


def version_id(model_sha: str, data_sha: str, covenants_sha: str) -> str:
    """A model version is identified by what it is, was trained on, and claims.

    Governance fields (owner, review date, tier) can be amended without
    creating a new version; a re-fit, a new snapshot or a changed covenant
    always creates one.
    """
    full = sha256_canonical(
        {"model": model_sha, "data": data_sha, "covenants": covenants_sha}
    )
    return full[:12]
