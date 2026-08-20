from __future__ import annotations

from pathlib import Path

import pytest

from covenant.checks.reason_codes import jaccard, run_reason_code_check
from covenant.store import Store


def test_jaccard() -> None:
    assert jaccard(frozenset("ab"), frozenset("ab")) == 1.0
    assert jaccard(frozenset("ab"), frozenset("bc")) == pytest.approx(1 / 3)
    assert jaccard(frozenset(), frozenset()) == 1.0


def test_live_table_passes(fitted: dict, store_dir: Path) -> None:
    record = run_reason_code_check(fitted["model"], fitted["data"], fitted["covenants"])
    assert record.passed, record.metrics
    assert record.metrics["top1_agreement"] >= 0.75
    assert record.metrics["topk_jaccard"] >= 0.60
    # the measured side should be stable across backgrounds on a linear model
    assert record.metrics["background_jaccard"] >= 0.8
    assert record.details["by_score_band"]

    path = record.write(Store(store_dir))
    assert Path(path).exists()
    assert record.record_sha256 and len(record.record_sha256) == 64


def test_stale_table_breaches(fitted: dict) -> None:
    record = run_reason_code_check(fitted["model"], fitted["data"], fitted["covenants_broken"])
    assert not record.passed
    # swapped + zeroed coefficients must show up as visible disagreement
    assert record.metrics["topk_jaccard"] < 0.60
    assert record.details["worst_disagreements"]


def test_check_is_deterministic(fitted: dict) -> None:
    a = run_reason_code_check(fitted["model"], fitted["data"], fitted["covenants"])
    b = run_reason_code_check(fitted["model"], fitted["data"], fitted["covenants"])
    assert a.metrics == b.metrics
    assert a.details["by_score_band"] == b.details["by_score_band"]


def test_threshold_override_and_too_few_denied(fitted: dict) -> None:
    from covenant.checks.reason_codes import CheckSetupError

    with pytest.raises(CheckSetupError, match="too few"):
        run_reason_code_check(
            fitted["model"],
            fitted["data"],
            fitted["covenants"],
            {"decision_threshold": 0.999},
        )


def test_record_write_is_idempotent(fitted: dict, store_dir: Path) -> None:
    store = Store(store_dir)
    a = run_reason_code_check(fitted["model"], fitted["data"], fitted["covenants"])
    path_a = Path(a.write(store))
    bytes_a = path_a.read_bytes()
    b = run_reason_code_check(fitted["model"], fitted["data"], fitted["covenants"])
    path_b = Path(b.write(store))

    assert path_a == path_b
    assert path_b.read_bytes() == bytes_a
    records = [p for p in path_a.parent.glob("reason-codes-*.yaml")]
    assert len(records) == 1
    log = (path_a.parent / "runs.log").read_text().splitlines()
    assert len(log) == 2
    assert all(a.record_sha256[:12] in line for line in log)


def test_categorical_pipeline_runs(fitted_categorical: dict) -> None:
    """A ColumnTransformer/OneHotEncoder pipeline with a categorical covenant
    must run through SHAP without crashing (regression: str - str TypeError)."""
    record = run_reason_code_check(
        fitted_categorical["model"],
        fitted_categorical["data"],
        fitted_categorical["covenants"],
    )
    assert record.n_evaluated > 0
    assert 0.0 <= record.metrics["topk_jaccard"] <= 1.0
    # true-driver reasons against the fitted model should agree far more
    # often than chance over three features
    assert record.metrics["topk_jaccard"] > 0.5


def test_custom_reasons_missing_id_fails_loudly(
    fitted_categorical: dict, tmp_path: Path
) -> None:
    from covenant.declared import DeclaredMethodError

    root = fitted_categorical["root"]
    reasons = (root / "reasons.csv").read_text().splitlines()
    (tmp_path / "reasons.csv").write_text("\n".join(reasons[:51]) + "\n")
    covenants = tmp_path / "covenants.yaml"
    covenants.write_text((root / "covenants.yaml").read_text())
    with pytest.raises(DeclaredMethodError, match="no reasons for"):
        run_reason_code_check(
            fitted_categorical["model"], fitted_categorical["data"], covenants
        )


def test_configurable_background_floor(fitted: dict) -> None:
    """The sensitivity floor is covenant policy, not a hardcoded constant."""
    strict = run_reason_code_check(
        fitted["model"], fitted["data"], fitted["covenants"],
        {"background_stability_floor": 1.0},
    )
    assert strict.details["background_sensitive"] is True
    lax = run_reason_code_check(
        fitted["model"], fitted["data"], fitted["covenants"],
        {"background_stability_floor": 0.0},
    )
    assert lax.details["background_sensitive"] is False


def test_shapley_export_provenance_flag(fitted_categorical: dict, tmp_path: Path) -> None:
    """An export generated from the check's own measured attributions must be
    flagged as indistinguishable (freshness, not independence)."""

    root = fitted_categorical["root"]
    base = run_reason_code_check(
        fitted_categorical["model"], fitted_categorical["data"], fitted_categorical["covenants"]
    )
    assert base.details["shapley_export_provenance"] is None  # method is custom

    # Build a shapley covenant whose artefact IS the measured attributions.
    from covenant.attribution import explain, sample_background
    from covenant.model import CovenantModel
    from covenant.registry import load_covenants, load_data

    cov = load_covenants(fitted_categorical["covenants"])
    data = load_data(fitted_categorical["data"]).reset_index(drop=True)
    feats = cov.feature_names()
    numeric = [f for f in feats if f not in cov.categorical_features()]
    data[numeric] = data[numeric].astype(float)
    from covenant.model import load_model

    model = CovenantModel(load_model(fitted_categorical["model"]), feats)
    p = model.p_bad(data)
    denied = data[p >= 0.5]
    bg = sample_background(data, feats, 30, 0)
    attributions, _ = explain(model, denied[feats], bg, cov.categorical_features(), 0)
    export = attributions.copy()
    export.insert(0, "app_id", denied["app_id"].to_numpy())
    export.to_csv(tmp_path / "attr.csv", index=False)
    covenants = tmp_path / "covenants.yaml"
    covenants.write_text(
        (root / "covenants.yaml")
        .read_text()
        .replace("method: custom", "method: shapley")
        .replace("reasons_file: reasons.csv", "attributions_file: attr.csv")
    )
    record = run_reason_code_check(
        fitted_categorical["model"], fitted_categorical["data"], covenants
    )
    prov = record.details["shapley_export_provenance"]
    assert prov["evaluated"] is True
    assert prov["indistinguishable_from_measured"] is True
    assert "fresh" in prov["note"]
