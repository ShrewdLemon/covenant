"""Champion/challenger comparison as a hashed, replayable record.

"Our model beats the simple thing" should be a measured claim with an
interval, not an assertion — and the measurement should be symmetric:
both models scored on the same snapshot, deltas computed with a *paired*
bootstrap (identical resample rows applied to both score vectors), so the
interval reflects the difference, not two overlapping marginal CIs.

This is evidence, not a gate: the record carries no pass/fail and the
command exits 0 whenever the comparison could be computed. Score both
models on a holdout snapshot neither was fitted to if you want the
comparison to mean anything out of sample — the record says which
snapshot it got.
"""

from __future__ import annotations

from pathlib import Path

from covenant import metrics
from covenant.checks.reason_codes import CheckSetupError
from covenant.hashing import sha256_canonical, sha256_dataframe, sha256_file
from covenant.model import CovenantModel, load_model
from covenant.registry import load_covenants, load_data
from covenant.schema import ModelCovenants

_METRICS = (
    ("roc_auc", metrics.roc_auc, "higher"),
    ("ks", metrics.ks_statistic, "higher"),
    ("brier", metrics.brier, "lower"),
    ("ece", metrics.ece, "lower"),
)


def compare_models(
    model_a_path: str | Path,
    model_b_path: str | Path,
    data_path: str | Path,
    covenants_path: str | Path,
    config_overrides: dict | None = None,
) -> dict:
    """Score both models on the covenant's features over one snapshot and
    return a hash-stamped comparison record (A minus B deltas)."""
    covenants: ModelCovenants = load_covenants(covenants_path)
    config = covenants.report
    if config_overrides:
        config = config.model_copy(update=config_overrides)
    if not config.target_column:
        raise CheckSetupError(
            "the covenants declare no report.target_column; covenant compare "
            "needs the 0/1 outcome column (1 = bad) to score both models"
        )

    data = load_data(data_path).reset_index(drop=True)
    features = covenants.feature_names()
    categorical = covenants.categorical_features()
    missing = [f for f in features if f not in data.columns]
    if missing:
        raise CheckSetupError(f"data lacks declared features: {missing}")
    if config.target_column not in data.columns:
        raise CheckSetupError(
            f"report.target_column {config.target_column!r} is not a column "
            "of the snapshot"
        )
    numeric = [f for f in features if f not in categorical]
    data[numeric] = data[numeric].astype(float)
    y = data[config.target_column].astype(float).to_numpy()

    # Each model scores with its own recorded inputs (a challenger need not
    # share the champion's feature set); the covenant's declared list stands
    # in when an estimator records none.
    def bind(path: str | Path, label: str) -> CovenantModel:
        estimator = load_model(path)
        inputs = [str(c) for c in getattr(estimator, "feature_names_in_", [])] or features
        absent = [c for c in inputs if c not in data.columns]
        if absent:
            raise CheckSetupError(
                f"{label} ({Path(path).name}) needs columns the snapshot "
                f"lacks: {absent}"
            )
        return CovenantModel(estimator, inputs, positive_class=covenants.positive_class)

    model_a = bind(model_a_path, "model A")
    model_b = bind(model_b_path, "model B")
    p_a = model_a.p_bad(data)
    p_b = model_b.p_bad(data)

    boot = {"n_boot": config.n_bootstrap, "seed": config.random_state}
    per_model: dict[str, dict[str, list[float]]] = {"model_a": {}, "model_b": {}}
    deltas: dict[str, dict] = {}
    for name, fn, better in _METRICS:
        point_a, lo_a, hi_a = metrics.bootstrap_ci(fn, y, p_a, **boot)
        point_b, lo_b, hi_b = metrics.bootstrap_ci(fn, y, p_b, **boot)
        diff, lo_d, hi_d = metrics.paired_bootstrap_diff(fn, y, p_a, p_b, **boot)
        per_model["model_a"][name] = [round(v, 6) for v in (point_a, lo_a, hi_a)]
        per_model["model_b"][name] = [round(v, 6) for v in (point_b, lo_b, hi_b)]
        deltas[name] = {
            "a_minus_b": [round(v, 6) for v in (diff, lo_d, hi_d)],
            "better": better,
            "significant": bool(lo_d > 0 or hi_d < 0),
        }

    record = {
        "kind": "compare",
        "model_name": covenants.model_name,
        "inputs": {
            "model_a_sha256": sha256_file(model_a_path),
            "model_b_sha256": sha256_file(model_b_path),
            "data_sha256": sha256_dataframe(load_data(data_path)),
            "covenants_sha256": sha256_canonical(covenants.model_dump(mode="json")),
        },
        "model_files": {
            "model_a": Path(model_a_path).name,
            "model_b": Path(model_b_path).name,
        },
        "config": {
            "target_column": config.target_column,
            "n_bootstrap": config.n_bootstrap,
            "random_state": config.random_state,
            "n_rows": int(len(data)),
        },
        "metrics": per_model,
        "deltas": deltas,
        "note": (
            "paired bootstrap: identical resample rows applied to both score "
            "vectors; deltas are model_a minus model_b on the supplied "
            "snapshot. Evidence, not a gate — and only out-of-sample if the "
            "snapshot is one neither model was fitted to."
        ),
        "record_sha256": "",
    }
    body = dict(record)
    body.pop("record_sha256")
    record["record_sha256"] = sha256_canonical(body)
    return record
