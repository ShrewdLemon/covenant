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
