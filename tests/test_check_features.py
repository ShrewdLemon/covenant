from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from covenant.checks.features import run_features_check
from covenant.checks.reason_codes import CheckSetupError

COVENANTS_TMPL = """\
covenant_schema: 1
model_name: features-test
features:
{features}
reason_codes:
  method: univariate
  top_k: 2
checks:
  features:
    sample_size: 40
    background_size: 20
    random_state: 0
"""


def _frame(cols: list[str], n: int = 150, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.normal(size=(n, len(cols))), columns=cols)
    logit = df.to_numpy() @ np.linspace(1.0, 1.6, len(cols)) + rng.normal(0, 0.5, n)
    df["bad"] = (rng.uniform(size=n) < 1.0 / (1.0 + np.exp(-logit))).astype(int)
    return df


def _setup(
    tmp_path: Path,
    trained_on: list[str],
    declared: list[str],
    df: pd.DataFrame | None = None,
    as_array: bool = False,
) -> tuple[Path, Path, Path]:
    """Fit a small logistic model on `trained_on`, write model, snapshot and
    a covenant declaring `declared`, and return the three paths."""
    if df is None:
        df = _frame(trained_on)
    X = df[trained_on].to_numpy() if as_array else df[trained_on]
    model = LogisticRegression(max_iter=1000).fit(X, df["bad"])
    model_path = tmp_path / "model.joblib"
    joblib.dump(model, model_path)
    data_path = tmp_path / "train.csv"
    df.to_csv(data_path, index=False)
    covenants_path = tmp_path / "covenants.yaml"
    lines = "\n".join(f"  - {{name: {name}}}" for name in declared)
    covenants_path.write_text(COVENANTS_TMPL.format(features=lines))
    return model_path, data_path, covenants_path


def test_clean_model_passes(tmp_path: Path) -> None:
    paths = _setup(tmp_path, ["a", "b", "c"], ["a", "b", "c"])
    record = run_features_check(*paths)
    assert record.passed, record.details
    assert record.metrics == {
        "n_undocumented_used": 0.0,
        "n_declared_unused": 0.0,
        "n_dead": 0.0,
    }
    assert record.details["structural_available"] is True
    assert record.details["undocumented_used"] == []
    assert record.details["declared_unused"] == []
    assert record.details["dead_features"] == []
    assert record.n_evaluated == 40


def test_undocumented_input_breaches(tmp_path: Path) -> None:
    """Model fitted on [a, b, c] under a covenant declaring [a, b]: the model
    sees a column the documentation never mentions."""
    paths = _setup(tmp_path, ["a", "b", "c"], ["a", "b"])
    record = run_features_check(*paths)
    assert not record.passed
    assert record.details["undocumented_used"] == ["c"]
    assert record.metrics["n_undocumented_used"] == 1.0
    assert record.details["declared_unused"] == []
    # The screen still ran (bound to the model's real inputs) and only
    # declared features are screened for deadness.
    assert record.n_evaluated == 40
    assert {d["feature"] for d in record.details["dead_features"]} <= {"a", "b"}


def test_declared_unused_breaches(tmp_path: Path) -> None:
    """Covenant declares [a, b, c] but the model was fitted on [a, b]: the
    documentation claims a feature the model cannot see."""
    df = _frame(["a", "b", "c"])
    paths = _setup(tmp_path, ["a", "b"], ["a", "b", "c"], df=df)
    record = run_features_check(*paths)
    assert not record.passed
    assert record.details["declared_unused"] == ["c"]
    assert record.metrics["n_declared_unused"] == 1.0
    assert record.details["undocumented_used"] == []


def test_dead_feature_warns_without_failing(tmp_path: Path) -> None:
    """A declared constant column is measurably inert: flagged as dead under
    a generous epsilon, while the check still passes."""
    df = _frame(["a", "b"])
    df.insert(2, "flat", 1.0)
    paths = _setup(tmp_path, ["a", "b", "flat"], ["a", "b", "flat"], df=df)
    record = run_features_check(*paths, config_overrides={"dead_feature_epsilon": 0.05})
    assert record.passed, record.details
    assert [d["feature"] for d in record.details["dead_features"]] == ["flat"]
    assert record.details["dead_features"][0]["mean_abs_attribution"] < 0.05
    assert record.metrics["n_dead"] == 1.0
    assert record.thresholds["dead_feature_epsilon"] == 0.05
    assert "documentation-quality" in record.details["note"]


def test_structural_unavailable_relies_on_screen(tmp_path: Path) -> None:
    """An estimator fitted on a bare array records no feature_names_in_; the
    record says so honestly and the attribution screen still runs."""
    paths = _setup(tmp_path, ["a", "b"], ["a", "b"], as_array=True)
    record = run_features_check(*paths)
    assert record.passed
    assert record.details["structural_available"] is False
    assert "feature_names_in_" in record.details["note"]
    assert record.metrics["n_undocumented_used"] == 0.0
    assert record.metrics["n_declared_unused"] == 0.0


def test_missing_model_column_is_setup_error(tmp_path: Path) -> None:
    df = _frame(["a", "b", "c"])
    model_path, data_path, covenants_path = _setup(tmp_path, ["a", "b", "c"], ["a", "b"], df=df)
    df.drop(columns=["c"]).to_csv(data_path, index=False)
    with pytest.raises(CheckSetupError, match="'c'"):
        run_features_check(model_path, data_path, covenants_path)


def test_record_determinism(tmp_path: Path) -> None:
    paths = _setup(tmp_path, ["a", "b", "c"], ["a", "b", "c"])
    a = run_features_check(*paths)
    b = run_features_check(*paths)
    assert a.record_sha256 == b.record_sha256
