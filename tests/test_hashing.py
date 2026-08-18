from __future__ import annotations

import pandas as pd
from hypothesis import given
from hypothesis import strategies as st

from covenant.hashing import (
    canonical_json,
    sha256_canonical,
    sha256_dataframe,
    version_id,
)

scalars = st.one_of(st.none(), st.booleans(), st.integers(), st.text(max_size=10))
json_objects = st.dictionaries(st.text(max_size=8), scalars, max_size=6)


@given(json_objects)
def test_canonical_json_is_insertion_order_invariant(d: dict) -> None:
    reordered = dict(reversed(list(d.items())))
    assert canonical_json(d) == canonical_json(reordered)
    assert sha256_canonical(d) == sha256_canonical(reordered)


def test_canonical_json_fixed_separators() -> None:
    assert canonical_json({"b": 1, "a": [1, 2]}) == '{"a":[1,2],"b":1}'


def test_dataframe_hash_stable_and_sensitive() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": ["a", "b", "c"]})
    assert sha256_dataframe(df) == sha256_dataframe(df.copy())

    changed = df.copy()
    changed.loc[1, "x"] = 2.5
    assert sha256_dataframe(df) != sha256_dataframe(changed)

    # column order is part of the snapshot's identity
    assert sha256_dataframe(df) != sha256_dataframe(df[["y", "x"]])
    # row order too
    assert sha256_dataframe(df) != sha256_dataframe(df.iloc[::-1].reset_index(drop=True))
    # dtype matters even when values compare equal
    as_int = df.assign(x=df["x"].astype("int64"))
    assert sha256_dataframe(df) != sha256_dataframe(as_int)


def test_version_id_is_stable_and_short() -> None:
    a = version_id("m" * 64, "d" * 64, "c" * 64)
    assert a == version_id("m" * 64, "d" * 64, "c" * 64)
    assert len(a) == 12
    assert a != version_id("m" * 64, "d" * 64, "x" * 64)
