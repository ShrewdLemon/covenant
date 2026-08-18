"""Check 2: does the model actually move risk in the directions the
documentation declares?

Feature-highlighting explanations silently assume monotonicity (Barocas,
Selbst & Raghavan, 2019): "improve the flagged feature" is only honest
advice if improving it cannot raise the score. So declared directions are
tested three ways:

* **declared** — the covenant (`direction:` per feature);
* **configured** — monotone constraints read off the estimator itself
  (XGBoost/LightGBM ``monotone_constraints``, sklearn ``monotonic_cst``),
  when present; a configured direction that contradicts the declared one is
  a breach on its own;
* **empirical** — synthetic dominance pairs (hold every other feature
  fixed, move one, ``p_bad`` must not move against the declared direction)
  and ICE paths swept over the feature's empirical quantiles, both counted
  as violation rates against a stated threshold.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from covenant.checks.base import CheckRecord
from covenant.checks.reason_codes import CheckSetupError
from covenant.hashing import sha256_canonical, sha256_dataframe, sha256_file
from covenant.model import CovenantModel, load_model
from covenant.registry import load_covenants, load_data
from covenant.schema import Direction, ModelCovenants, MonotonicityCheckConfig

CHECK_NAME = "monotonicity"

_CONSTRAINT_PARAM_KEYS = ("monotone_constraints", "monotonic_cst")


def configured_directions(
    estimator: Any, feature_names: list[str], bad_class_index: int = 1
) -> dict[str, Direction] | None:
    """Monotone constraints configured on the estimator, as directions on
    ``p_bad``, or None when there are no constraints we can read and align.

    Constraints are interpreted as acting on the positive-class score
    (class index 1); when ``p_bad`` is the other column the sign flips.
    """
    est = estimator
    if hasattr(est, "steps"):  # sklearn Pipeline: constraints sit on the final step
        est = est.steps[-1][1]
    if not hasattr(est, "get_params"):
        return None
    raw = None
    try:
        params = est.get_params(deep=False)
    except Exception:
        return None
    for key in _CONSTRAINT_PARAM_KEYS:
        if params.get(key) is not None:
            raw = params[key]
            break
    if raw is None:
        return None

    if isinstance(raw, str):  # xgboost accepts "(1,-1,0)"
        parts = raw.strip().strip("()").split(",")
        raw = [int(p) for p in parts if p.strip()]
    if isinstance(raw, dict):
        by_feature = {str(k): int(v) for k, v in raw.items()}
    else:
        values = [int(v) for v in np.asarray(raw).ravel()]
        names = list(getattr(est, "feature_names_in_", [])) or feature_names
        if len(values) != len(names):
            return None  # cannot align a bare sequence confidently
        by_feature = dict(zip([str(n) for n in names], values, strict=True))

    flip = -1 if bad_class_index == 0 else 1
    mapping = {1: Direction.INCREASES_RISK, -1: Direction.DECREASES_RISK, 0: Direction.NONE}
    return {
        name: mapping[int(np.sign(flip * value))]
        for name, value in by_feature.items()
        if name in feature_names
    }


def run_monotonicity_check(
    model_path: str | Path,
    data_path: str | Path,
    covenants_path: str | Path,
    config_overrides: dict | None = None,
) -> CheckRecord:
    covenants: ModelCovenants = load_covenants(covenants_path)
    config: MonotonicityCheckConfig = covenants.checks.monotonicity
    if config_overrides:
        config = config.model_copy(update=config_overrides)

    data = load_data(data_path).reset_index(drop=True)
    features = covenants.feature_names()
    categorical = covenants.categorical_features()
    missing = [f for f in features if f not in data.columns]
    if missing:
        raise CheckSetupError(f"data lacks declared features: {missing}")
    numeric = [f for f in features if f not in categorical]
    data[numeric] = data[numeric].astype(float)

    declared = {
        name: direction
        for name, direction in covenants.declared_directions().items()
        if direction is not Direction.NONE
    }
    if not declared:
        raise CheckSetupError(
            "no feature declares a monotone direction; nothing for the "
            "monotonicity check to test. Set direction: increases_risk or "
            "decreases_risk on at least one feature."
        )

    estimator = load_model(model_path)
    model = CovenantModel(estimator, features, positive_class=covenants.positive_class)
    configured = configured_directions(estimator, features, model.bad_class_index)
    configured_mismatch = [
        {
            "feature": name,
            "declared": declared[name].value,
            "configured": configured[name].value,
        }
        for name in declared
        if configured is not None
        and configured.get(name) not in (None, Direction.NONE)
        and configured[name] is not declared[name]
    ]

    rng = np.random.default_rng(config.random_state)
    per_feature: list[dict] = []
    n_evaluated = 0
    for name, direction in declared.items():
        result = _test_feature(model, data, features, name, direction, config, rng)
        result["configured"] = (
            configured[name].value if configured and name in configured else "absent"
        )
        per_feature.append(result)
        n_evaluated += result["n_pairs"] + result["n_ice_steps"]

    worst = max(max(r["pair_violation_rate"], r["ice_violation_rate"]) for r in per_feature)
    mean_rate = float(
        np.mean([max(r["pair_violation_rate"], r["ice_violation_rate"]) for r in per_feature])
    )
    passed = worst <= config.max_violation_rate and not configured_mismatch

    record = CheckRecord(
        check=CHECK_NAME,
        model_name=covenants.model_name,
        passed=passed,
        metrics={
            "worst_violation_rate": round(worst, 4),
            "mean_violation_rate": round(mean_rate, 4),
            "configured_mismatches": float(len(configured_mismatch)),
        },
        thresholds={"max_violation_rate": config.max_violation_rate},
        n_evaluated=n_evaluated,
        inputs={
            "model_sha256": sha256_file(model_path),
            "data_sha256": sha256_dataframe(load_data(data_path)),
            "covenants_sha256": sha256_canonical(covenants.model_dump(mode="json")),
        },
        config=config.model_dump(),
        details={
            "by_feature": per_feature,
            "configured_constraints_readable": configured is not None,
            "configured_mismatch": configured_mismatch,
            "note": (
                "pair/ICE violation rates are empirical, on synthetic points "
                "built from observed feature values; a configured constraint "
                "that contradicts the declared direction fails the check "
                "regardless of the empirical rates"
            ),
        },
    )
    return record.stamp()


def _test_feature(
    model: CovenantModel,
    data: pd.DataFrame,
    features: list[str],
    name: str,
    direction: Direction,
    config: MonotonicityCheckConfig,
    rng: np.random.Generator,
) -> dict:
    sign = 1.0 if direction is Direction.INCREASES_RISK else -1.0
    col = data[name].to_numpy(dtype=float)

    # Dominance pairs: same applicant, two observed values of this feature.
    base_idx = rng.choice(len(data), size=config.n_pairs, replace=True)
    v_a = rng.choice(col, size=config.n_pairs, replace=True)
    v_b = rng.choice(col, size=config.n_pairs, replace=True)
    lo = np.minimum(v_a, v_b)
    hi = np.maximum(v_a, v_b)
    keep = hi > lo
    base = data.iloc[base_idx[keep]][features].reset_index(drop=True)
    X_lo = base.copy()
    X_lo[name] = lo[keep]
    X_hi = base.copy()
    X_hi[name] = hi[keep]
    p_lo = model.p_bad(X_lo)
    p_hi = model.p_bad(X_hi)
    delta = sign * (p_hi - p_lo)
    pair_violations = delta < -config.tolerance
    pair_rate = float(pair_violations.mean()) if len(delta) else 0.0

    example = None
    if pair_violations.any():
        i = int(np.argmin(delta))
        example = {
            "value_low": round(float(lo[keep][i]), 6),
            "value_high": round(float(hi[keep][i]), 6),
            "p_bad_at_low": round(float(p_lo[i]), 6),
            "p_bad_at_high": round(float(p_hi[i]), 6),
        }

    # ICE paths: sweep the feature over its empirical quantiles for a
    # sample of applicants; every step must respect the declared direction.
    grid = np.unique(np.quantile(col, np.linspace(0.02, 0.98, config.ice_grid_points)))
    n_ice_steps = 0
    ice_rate = 0.0
    if len(grid) >= 2:
        row_idx = rng.choice(len(data), size=config.n_ice_rows, replace=True)
        rows = data.iloc[row_idx][features].reset_index(drop=True)
        sweep = rows.loc[rows.index.repeat(len(grid))].reset_index(drop=True)
        sweep[name] = np.tile(grid, len(rows))
        p = model.p_bad(sweep).reshape(len(rows), len(grid))
        steps = sign * np.diff(p, axis=1)
        n_ice_steps = int(steps.size)
        ice_rate = float((steps < -config.tolerance).mean())

    return {
        "feature": name,
        "declared": direction.value,
        "pair_violation_rate": round(pair_rate, 4),
        "ice_violation_rate": round(ice_rate, 4),
        "n_pairs": int(keep.sum()),
        "n_ice_steps": n_ice_steps,
        "worst_pair": example,
    }
