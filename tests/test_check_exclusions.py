from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from covenant.checks.exclusions import run_exclusions_check
from covenant.checks.reason_codes import CheckSetupError

FEATURES = ["income", "dti", "utilization"]

COVENANTS_YAML = """\
covenant_schema: 1
model_name: test-exclusions
features:
  - {name: income, direction: decreases_risk}
  - {name: dti, direction: increases_risk}
  - {name: utilization, direction: increases_risk}
excluded:
  - {name: gender, reason: protected attribute (ECOA)}
reason_codes:
  method: difference_from_mean
  top_k: 2
  parameters: {coefficients: coefficients.csv}
checks:
  exclusions:
    sample_size: 40
    background_size: 20
"""


def _fixture(
    root: Path,
    *,
    proxy_gender: bool = False,
    fit_on_gender: bool = False,
    gender_in_snapshot: bool = True,
    fit_on_arrays: bool = False,
) -> tuple[Path, Path, Path]:
    """Model + snapshot + covenants on disk. ``proxy_gender`` makes the
    excluded column a noisy monotone copy of dti (Spearman ~0.9);
    ``fit_on_gender`` fits the estimator on the excluded column too;
    ``fit_on_arrays`` fits on numpy so the estimator records no
    feature_names_in_."""
    rng = np.random.default_rng(3)
    n = 800
    df = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    if proxy_gender:
        df["gender"] = df["dti"] + rng.normal(0.0, 0.5, n)
    else:
        df["gender"] = rng.normal(size=n)
    logit = df[FEATURES].to_numpy() @ np.array([-1.0, 1.1, 0.8])
    if fit_on_gender:
        logit = logit + 1.5 * df["gender"].to_numpy()
    df["bad"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)

    fit_columns = FEATURES + ["gender"] if fit_on_gender else FEATURES
    X_fit = df[fit_columns].to_numpy() if fit_on_arrays else df[fit_columns]
    model = LogisticRegression(max_iter=1000).fit(X_fit, df["bad"])
    model_path = root / "model.joblib"
    joblib.dump(model, model_path)

    snapshot = df.drop(columns=["gender"]) if not gender_in_snapshot else df
    data_path = root / "train.csv"
    snapshot.to_csv(data_path, index=False)
    covenants_path = root / "covenants.yaml"
    covenants_path.write_text(COVENANTS_YAML)
    return model_path, data_path, covenants_path


def test_proxy_gender_is_flagged_and_breaches(tmp_path: Path) -> None:
    record = run_exclusions_check(*_fixture(tmp_path, proxy_gender=True))
    assert not record.passed
    flags = record.details["flagged_pairs"]
    assert {(f["excluded"], f["feature"]) for f in flags} == {("gender", "dti")}
    assert flags[0]["method"] == "spearman_abs"
    assert flags[0]["strength"] > 0.8
    assert record.metrics["n_proxy_flags"] == 1.0
    assert record.metrics["max_association_observed"] > record.thresholds["max_association"]
    assert "surfaced, not proven absent" in record.details["note"]


def test_fail_on_proxies_false_still_flags_but_passes(tmp_path: Path) -> None:
    record = run_exclusions_check(
        *_fixture(tmp_path, proxy_gender=True),
        config_overrides={"fail_on_proxies": False},
    )
    assert record.passed
    assert record.metrics["n_proxy_flags"] == 1.0
    assert record.details["flagged_pairs"]


def test_benign_excluded_variable_passes(tmp_path: Path) -> None:
    record = run_exclusions_check(*_fixture(tmp_path))
    assert record.passed
    assert record.details["flagged_pairs"] == []
    assert record.metrics["max_association_observed"] < 0.5
    gender = record.details["by_variable"][0]
    assert gender["in_snapshot"] is True
    assert gender["reaches_model"] == "no"  # feature_names_in_ is authoritative
    assert record.metrics["max_excluded_attribution_observed"] == 0.0


def test_model_fitted_on_excluded_column_breaches(tmp_path: Path) -> None:
    record = run_exclusions_check(*_fixture(tmp_path, fit_on_gender=True))
    assert not record.passed
    gender = record.details["by_variable"][0]
    assert gender["reaches_model"] == "yes"
    assert gender["attribution_breach"] is True
    assert (
        record.metrics["max_excluded_attribution_observed"]
        > record.thresholds["max_excluded_attribution"]
    )
    assert record.details["flagged_pairs"] == []  # breach comes from attribution alone


def test_absent_from_snapshot_passes_with_note(tmp_path: Path) -> None:
    record = run_exclusions_check(*_fixture(tmp_path, gender_in_snapshot=False))
    assert record.passed
    gender = record.details["by_variable"][0]
    assert gender["in_snapshot"] is False
    assert "not present in snapshot" in gender["note"]
    assert record.metrics["max_association_observed"] == 0.0
    assert record.metrics["n_proxy_flags"] == 0.0


def test_unverifiable_usage_is_noted_not_breached(tmp_path: Path) -> None:
    """Fitted on arrays, so no feature_names_in_; gender is not declared —
    whether it reaches the model cannot be verified, and honesty about
    that is not a breach."""
    record = run_exclusions_check(*_fixture(tmp_path, fit_on_arrays=True))
    assert record.passed
    gender = record.details["by_variable"][0]
    assert gender["reaches_model"] == "unknown"
    assert "cannot be verified" in gender["note"]


def test_no_excluded_variables_is_setup_error(tmp_path: Path) -> None:
    model_path, data_path, _ = _fixture(tmp_path)
    text = COVENANTS_YAML.replace(
        "excluded:\n  - {name: gender, reason: protected attribute (ECOA)}\n", ""
    )
    covenants_path = tmp_path / "covenants_none.yaml"
    covenants_path.write_text(text)
    with pytest.raises(CheckSetupError, match="nothing to check"):
        run_exclusions_check(model_path, data_path, covenants_path)


def test_check_is_deterministic(tmp_path: Path) -> None:
    paths = _fixture(tmp_path, proxy_gender=True, fit_on_gender=True)
    a = run_exclusions_check(*paths)
    b = run_exclusions_check(*paths)
    assert a.record_sha256 == b.record_sha256
    assert not a.passed  # proxy flag and attribution breach both present


def test_model_input_missing_from_snapshot_is_setup_error(tmp_path: Path) -> None:
    """An excluded variable the model reads but the snapshot omits must be a
    hard setup error, never a silent 'absent' pass (review finding)."""
    from covenant.checks.reason_codes import CheckSetupError

    paths = _fixture(tmp_path, fit_on_gender=True, gender_in_snapshot=False)
    with pytest.raises(CheckSetupError, match="gender"):
        run_exclusions_check(*paths)


def test_association_sample_size_bounds_the_screen(tmp_path: Path) -> None:
    """A seeded association sample keeps the screen deterministic and cheap."""
    paths = _fixture(tmp_path, proxy_gender=True)
    full = run_exclusions_check(*paths)
    sampled = run_exclusions_check(*paths, config_overrides={"association_sample_size": 300})
    sampled_again = run_exclusions_check(
        *paths, config_overrides={"association_sample_size": 300}
    )
    assert sampled.record_sha256 == sampled_again.record_sha256
    # the strong proxy stays flagged under sampling
    assert not full.passed and not sampled.passed
