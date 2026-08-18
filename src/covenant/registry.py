"""Registration: turn (model, training snapshot, covenants, governance)
into a content-addressed inventory record in the user's git repo."""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from covenant.hashing import sha256_canonical, sha256_dataframe, sha256_file, version_id
from covenant.model import CovenantModel, library_versions, load_model
from covenant.schema import (
    DataInfo,
    GovernanceRecord,
    Hashes,
    ModelCovenants,
    ModelInfo,
    ModelRecord,
)
from covenant.store import Store, read_yaml


class RegistrationError(ValueError):
    pass


def _friendly(err: ValidationError, doc: str) -> RegistrationError:
    problems = []
    for e in err.errors():
        loc = ".".join(str(p) for p in e["loc"]) or "(top level)"
        problems.append(f"  {loc}: {e['msg']}")
    return RegistrationError(f"{doc} is not valid:\n" + "\n".join(problems))


def load_covenants(path: str | Path) -> ModelCovenants:
    raw = read_yaml(path)
    if raw is None:
        raise RegistrationError(f"{path} is empty")
    try:
        return ModelCovenants.model_validate(raw)
    except ValidationError as err:
        raise _friendly(err, f"covenants file {path}") from err


def load_governance(path: str | Path) -> GovernanceRecord:
    raw = read_yaml(path)
    if raw is None:
        raise RegistrationError(f"{path} is empty")
    try:
        return GovernanceRecord.model_validate(raw)
    except ValidationError as err:
        raise _friendly(err, f"governance file {path}") from err


def load_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix in (".parquet", ".pq"):
        return pd.read_parquet(path)
    if path.suffix == ".csv":
        return pd.read_csv(path)
    raise RegistrationError(f"unsupported data format {path.suffix!r}; use .csv or .parquet")


def register(
    model_path: str | Path,
    data_path: str | Path,
    covenants_path: str | Path,
    governance_path: str | Path,
    store: Store,
    now: dt.datetime | None = None,
) -> ModelRecord:
    covenants = load_covenants(covenants_path)
    governance = load_governance(governance_path)
    data = load_data(data_path)

    missing = [f for f in covenants.feature_names() if f not in data.columns]
    if missing:
        raise RegistrationError(
            f"declared features not present in the training snapshot: {missing}"
        )

    estimator = load_model(model_path)
    try:
        CovenantModel(estimator, covenants.feature_names(), positive_class=covenants.positive_class)
    except (TypeError, ValueError) as err:
        raise RegistrationError(str(err)) from err

    hashes = Hashes(
        model_sha256=sha256_file(model_path),
        data_sha256=sha256_dataframe(data),
        covenants_sha256=sha256_canonical(covenants.model_dump(mode="json")),
    )
    vid = version_id(hashes.model_sha256, hashes.data_sha256, hashes.covenants_sha256)

    record = ModelRecord(
        version_id=vid,
        model_name=covenants.model_name,
        created_at=now or dt.datetime.now(dt.UTC),
        hashes=hashes,
        model=ModelInfo(
            model_class=f"{type(estimator).__module__}.{type(estimator).__qualname__}",
            library_versions=library_versions(),
        ),
        data=DataInfo(
            path=str(data_path),
            n_rows=len(data),
            n_cols=data.shape[1],
            columns=[str(c) for c in data.columns],
        ),
        governance=governance,
        covenants=covenants,
    )

    store.init()
    store.write_record(covenants.model_name, vid, record.model_dump(mode="json"))
    return record
