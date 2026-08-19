from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import brier_score_loss, roc_auc_score

from covenant.metrics import (
    bootstrap_ci,
    brier,
    csi,
    ece,
    gini,
    ks_statistic,
    paired_bootstrap_diff,
    psi,
    roc_auc,
)

# --- strategies -------------------------------------------------------------


@st.composite
def binary_problem(draw, min_size: int = 4, max_size: int = 60):
    """Labels with both classes present, scores on a coarse lattice so ties
    are common and strictly monotone float transforms cannot merge values."""
    n = draw(st.integers(min_size, max_size))
    y = draw(
        st.lists(st.integers(0, 1), min_size=n, max_size=n).filter(
            lambda v: 0 < sum(v) < len(v)
        )
    )
    p_raw = draw(st.lists(st.integers(0, 1000), min_size=n, max_size=n))
    return np.array(y, dtype=float), np.array(p_raw, dtype=float) / 1000.0


# --- roc_auc ----------------------------------------------------------------


def test_auc_perfect_separation() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert roc_auc(y, p) == 1.0
    assert ks_statistic(y, p) == 1.0
    assert gini(y, p) == 1.0


def test_auc_anti_separation() -> None:
    y = np.array([1, 1, 0, 0])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    assert roc_auc(y, p) == 0.0
    assert gini(y, p) == -1.0


def test_auc_single_class_raises() -> None:
    with pytest.raises(ValueError, match="needs both classes"):
        roc_auc(np.zeros(10), np.linspace(0, 1, 10))
    with pytest.raises(ValueError, match="needs both classes"):
        roc_auc(np.ones(10), np.linspace(0, 1, 10))
    with pytest.raises(ValueError, match="needs both classes"):
        ks_statistic(np.ones(10), np.linspace(0, 1, 10))


def test_auc_constant_scores_both_classes_is_half() -> None:
    y = np.array([0, 1, 0, 1, 1])
    p = np.full(5, 0.5)
    assert roc_auc(y, p) == 0.5  # ties count half via average ranks


def test_length_mismatch_and_empty_raise() -> None:
    with pytest.raises(ValueError, match="different lengths"):
        roc_auc([0, 1], [0.5])
    with pytest.raises(ValueError, match="empty"):
        brier([], [])
    with pytest.raises(ValueError, match="0/1"):
        roc_auc([0, 2], [0.1, 0.9])


@settings(max_examples=100)
@given(binary_problem())
def test_auc_invariant_under_monotone_transforms(problem) -> None:
    y, p = problem
    base = roc_auc(y, p)
    for transform in (np.exp, lambda x: x**3, lambda x: 5.0 * x + 2.0):
        assert abs(roc_auc(y, transform(p)) - base) < 1e-12


@settings(max_examples=100)
@given(binary_problem())
def test_metric_bounds_on_random_inputs(problem) -> None:
    y, p = problem
    assert 0.0 <= roc_auc(y, p) <= 1.0
    assert 0.0 <= ks_statistic(y, p) <= 1.0
    assert -1.0 <= gini(y, p) <= 1.0
    assert 0.0 <= brier(y, p) <= 1.0
    assert 0.0 <= ece(y, p) <= 1.0


# --- sklearn as oracle ------------------------------------------------------


def test_auc_and_brier_match_sklearn_on_200_seeded_problems() -> None:
    for seed in range(200):
        rng = np.random.default_rng(seed)
        n = int(rng.integers(10, 200))
        y = rng.integers(0, 2, size=n)
        y[0], y[1] = 0, 1  # force both classes
        # odd seeds get heavy ties: scores drawn from a four-point lattice
        p = rng.choice([0.1, 0.2, 0.5, 0.8], size=n) if seed % 2 else rng.random(n)
        assert abs(roc_auc(y, p) - roc_auc_score(y, p)) < 1e-9
        assert abs(brier(y, p) - brier_score_loss(y, p)) < 1e-9


# --- calibration ------------------------------------------------------------


def test_ece_zero_when_bins_are_calibrated() -> None:
    assert ece(np.array([0, 1]), np.array([0.0, 1.0])) == 0.0


def test_ece_known_value_single_bin() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.full(4, 0.9)
    assert ece(y, p) == pytest.approx(0.4)  # one bin: |0.9 - 0.5|


def test_ece_skips_empty_bins_and_rejects_non_probabilities() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array([0.05, 0.06, 0.95, 0.96])  # 8 of 10 bins empty
    assert np.isfinite(ece(y, p, n_bins=10))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        ece(y, np.array([0.1, 0.2, 0.3, 1.5]))


# --- stability --------------------------------------------------------------


def test_psi_identical_is_exactly_zero() -> None:
    x = np.random.default_rng(0).normal(size=500)
    assert psi(x, x) == 0.0


def test_psi_grows_with_location_shift() -> None:
    base = np.random.default_rng(1).normal(size=2000)
    shifts = [psi(base, base + d) for d in (0.1, 0.5, 1.5)]
    assert 0.0 < shifts[0] < shifts[1] < shifts[2]


def test_psi_finite_when_actual_has_empty_bins() -> None:
    expected = np.linspace(0.0, 1.0, 1000)
    actual = np.linspace(0.9, 1.0, 200)  # all mass in the top decile
    value = psi(expected, actual)
    assert np.isfinite(value)
    assert value > 0.0


def test_psi_finite_when_actual_outside_expected_range() -> None:
    expected = np.linspace(0.0, 1.0, 500)
    actual = np.linspace(5.0, 6.0, 500)  # entirely outside; open-ended top bin
    assert np.isfinite(psi(expected, actual))


def test_psi_handles_low_cardinality_expected() -> None:
    expected = np.array([0.0] * 50 + [1.0] * 50)  # duplicate quantile edges collapse
    actual = np.array([0.0] * 80 + [1.0] * 20)
    assert np.isfinite(psi(expected, actual))


def test_csi_is_psi_per_feature() -> None:
    rng = np.random.default_rng(2)
    expected_df = pd.DataFrame({"a": rng.normal(size=300), "b": rng.normal(size=300)})
    actual_df = pd.DataFrame(
        {"a": rng.normal(size=300), "b": rng.normal(loc=2.0, size=300)}
    )
    result = csi(expected_df, actual_df, ["a", "b"])
    assert set(result) == {"a", "b"}
    assert result["a"] == psi(expected_df["a"].to_numpy(), actual_df["a"].to_numpy())
    assert result["b"] > result["a"]


def test_csi_missing_feature_raises() -> None:
    frame = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError, match="missing"):
        csi(frame, frame, ["a", "nope"])


# --- bootstrap --------------------------------------------------------------


def _problem(seed: int = 0, n: int = 80) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, size=n)
    y[0], y[1] = 0, 1
    p = np.clip(0.35 * y + 0.5 * rng.random(n), 0.0, 1.0)
    return y, p


def test_bootstrap_ci_deterministic_and_seed_sensitive() -> None:
    y, p = _problem()
    first = bootstrap_ci(roc_auc, y, p, n_boot=200, seed=7)
    second = bootstrap_ci(roc_auc, y, p, n_boot=200, seed=7)
    assert first == second
    other = bootstrap_ci(roc_auc, y, p, n_boot=200, seed=8)
    assert first[0] == other[0]  # point estimate does not depend on the seed
    assert (first[1], first[2]) != (other[1], other[2])


def test_bootstrap_ci_brackets_point_on_seeded_cases() -> None:
    for seed in range(10):
        y, p = _problem(seed=seed)
        for metric in (roc_auc, brier):
            point, lo, hi = bootstrap_ci(metric, y, p, n_boot=300, seed=seed)
            assert lo <= point <= hi


def test_bootstrap_ci_skips_single_class_resamples() -> None:
    rng = np.random.default_rng(3)
    n = 40
    y = np.zeros(n, dtype=int)
    y[:2] = 1  # rare positives: many resamples are single-class and skipped
    p = rng.random(n)
    point, lo, hi = bootstrap_ci(roc_auc, y, p, n_boot=100, seed=0)
    assert np.isfinite([point, lo, hi]).all()
    assert bootstrap_ci(roc_auc, y, p, n_boot=100, seed=0) == (point, lo, hi)


def test_bootstrap_ci_gives_up_after_capped_retries() -> None:
    calls = {"n": 0}

    def flaky(y: np.ndarray, p: np.ndarray) -> float:
        calls["n"] += 1
        if calls["n"] == 1:
            return 0.5  # the full-sample point estimate succeeds
        raise ValueError("resample rejected")

    y, p = _problem()
    with pytest.raises(ValueError, match="valid resamples"):
        bootstrap_ci(flaky, y, p, n_boot=50, seed=0)


def test_bootstrap_ci_propagates_point_estimate_failure() -> None:
    with pytest.raises(ValueError, match="needs both classes"):
        bootstrap_ci(roc_auc, np.zeros(20), np.linspace(0, 1, 20), n_boot=50)


def test_paired_bootstrap_diff_identical_models_is_zero() -> None:
    y, p = _problem()
    diff, lo, hi = paired_bootstrap_diff(roc_auc, y, p, p, n_boot=200, seed=0)
    assert (diff, lo, hi) == (0.0, 0.0, 0.0)


def test_paired_bootstrap_diff_detects_better_model() -> None:
    rng = np.random.default_rng(4)
    n = 300
    y = rng.integers(0, 2, size=n)
    y[0], y[1] = 0, 1
    strong = np.clip(0.6 * y + 0.4 * rng.random(n), 0.0, 1.0)
    weak = np.clip(0.1 * y + 0.9 * rng.random(n), 0.0, 1.0)
    diff, lo, hi = paired_bootstrap_diff(roc_auc, y, strong, weak, n_boot=300, seed=0)
    assert diff > 0.0
    assert lo <= diff <= hi
    assert lo > 0.0  # the interval excludes zero for a clearly better model


def test_paired_bootstrap_diff_deterministic() -> None:
    y, p = _problem(seed=5)
    _, q = _problem(seed=6)
    first = paired_bootstrap_diff(brier, y, p, q, n_boot=150, seed=1)
    assert first == paired_bootstrap_diff(brier, y, p, q, n_boot=150, seed=1)
    assert first != paired_bootstrap_diff(brier, y, p, q, n_boot=150, seed=2)


def test_bootstrap_rejects_bad_params() -> None:
    y, p = _problem()
    with pytest.raises(ValueError, match="alpha"):
        bootstrap_ci(roc_auc, y, p, alpha=1.5)
    with pytest.raises(ValueError, match="n_boot"):
        paired_bootstrap_diff(roc_auc, y, p, p, n_boot=0)
    with pytest.raises(ValueError, match="different lengths"):
        paired_bootstrap_diff(roc_auc, y, p, p[:-1])


class TestReviewRegressions:
    """Locks for defects found in the v0.3 adversarial review."""

    def test_psi_detects_binary_flip(self) -> None:
        from covenant.metrics import psi

        value = psi([0] * 55 + [1] * 45, [0] * 5 + [1] * 95)
        assert value == pytest.approx(1.5726, abs=1e-3)
        for n in (100, 200, 1000):
            assert psi([0] * (n // 2 + 5) + [1] * (n // 2 - 5), [0] * 5 + [1] * (n - 5)) > 0.5

    def test_psi_zero_inflated_shift_detected(self) -> None:
        from covenant.metrics import psi

        expected = [0.0] * 95 + [1.0, 2.0, 3.0, 4.0, 5.0]
        actual = [5.0] * 100
        assert psi(expected, actual) > 1.0

    def test_psi_constant_baseline_measures_total_shift(self) -> None:
        from covenant.metrics import psi

        assert psi([1.0] * 100, [1.0] * 100) == 0.0
        assert psi([1.0] * 100, [2.0] * 100) > 10.0  # total shift, loud

    def test_csi_covers_low_cardinality_features(self) -> None:
        import pandas as pd

        from covenant.metrics import csi

        expected = pd.DataFrame({"flag": [0.0] * 30 + [1.0] * 20, "x": list(range(50))})
        actual = pd.DataFrame({"flag": [1.0] * 50, "x": list(range(50))})
        values = csi(expected, actual, ["flag", "x"])
        assert values["flag"] > 0.5
        assert values["x"] == pytest.approx(0.0, abs=1e-9)
