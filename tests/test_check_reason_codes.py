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
