"""Dependency-light validation metrics for the deterministic report.

Discrimination (AUC/KS/Gini), calibration (Brier/ECE) and stability
(PSI/CSI) are commodity numbers — every validation platform computes
them. Covenant implements them in a few hundred auditable lines of
numpy/scipy rather than importing a metrics library, because a validator
reading the report should be able to read the arithmetic that produced
it, and because byte-replayable reports require owning every code path
(metrics are commodity; determinism and mapping are the product).

Every point estimate is designed to ship with a seeded bootstrap
confidence interval. On the small validation samples typical of the
long tail, a point estimate alone overstates what the data can support —
FINRA's Model Validation Toolkit frames this as the credibility of
metrics under small samples, and Covenant borrows the idea: report the
interval, not just the point.

These functions compute evidence; validators decide what the numbers
mean. Edge cases raise ``ValueError`` with actionable messages instead
of returning NaN, so a mis-specified run fails loudly in CI rather than
producing a plausible-looking record.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy.stats import rankdata

MetricFn = Callable[[np.ndarray, np.ndarray], float]

# Proportions are clipped to this floor before the PSI log so an empty
# bin contributes a large finite term instead of infinity.
PSI_PROPORTION_FLOOR = 1e-6

# bootstrap_ci / paired_bootstrap_diff draw at most this multiple of
# n_boot resamples before giving up on collecting n_boot valid ones.
BOOTSTRAP_MAX_DRAW_FACTOR = 10


def _validate_labels_scores(y: ArrayLike, p: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    y_arr = np.asarray(y, dtype=float).ravel()
    p_arr = np.asarray(p, dtype=float).ravel()
    if len(y_arr) != len(p_arr):
        raise ValueError(f"y and p have different lengths ({len(y_arr)} vs {len(p_arr)})")
    if len(y_arr) == 0:
        raise ValueError("y and p are empty; metrics need at least one row")
    if not np.isfinite(p_arr).all():
        raise ValueError("p contains non-finite values; clean the scores upstream")
    labels = np.unique(y_arr)
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError(
            f"y must contain only 0/1 labels, got values {labels[:5].tolist()}; "
            "encode the outcome as 0 = good, 1 = bad before computing metrics"
        )
    return y_arr, p_arr


def _validate_sample(x: ArrayLike, name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if len(arr) == 0:
        raise ValueError(f"{name} is empty; stability metrics need at least one row")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values; clean the column upstream")
    return arr


def roc_auc(y: ArrayLike, p: ArrayLike) -> float:
    """Area under the ROC curve via the Mann-Whitney rank formulation.

    Average ranks handle tied scores exactly (a tie counts as half a
    concordant pair), so heavily binned scorecard outputs are scored the
    same way a trapezoidal ROC integration would score them.
    """
    y_arr, p_arr = _validate_labels_scores(y, p)
    pos = y_arr == 1.0
    n_pos = int(pos.sum())
    n_neg = len(y_arr) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("needs both classes")
    ranks = rankdata(p_arr)  # average ranks on ties
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def ks_statistic(y: ArrayLike, p: ArrayLike) -> float:
    """Kolmogorov-Smirnov separation: max |F1(t) - F0(t)| over score thresholds."""
    y_arr, p_arr = _validate_labels_scores(y, p)
    pos_scores = np.sort(p_arr[y_arr == 1.0])
    neg_scores = np.sort(p_arr[y_arr == 0.0])
    if len(pos_scores) == 0 or len(neg_scores) == 0:
        raise ValueError("needs both classes")
    thresholds = np.unique(p_arr)
    f1 = np.searchsorted(pos_scores, thresholds, side="right") / len(pos_scores)
    f0 = np.searchsorted(neg_scores, thresholds, side="right") / len(neg_scores)
    return float(np.max(np.abs(f1 - f0)))


def gini(y: ArrayLike, p: ArrayLike) -> float:
    """Accuracy ratio: ``2 * AUC - 1``."""
    return 2.0 * roc_auc(y, p) - 1.0


def brier(y: ArrayLike, p: ArrayLike) -> float:
    """Brier score: mean squared error of the predicted probabilities."""
    y_arr, p_arr = _validate_labels_scores(y, p)
    return float(np.mean((p_arr - y_arr) ** 2))


def ece(y: ArrayLike, p: ArrayLike, n_bins: int = 10) -> float:
    """Expected calibration error over equal-width probability bins.

    ``sum_b (n_b / n) * |mean(p in b) - mean(y in b)|`` with empty bins
    skipped; scores must be probabilities in [0, 1].
    """
    y_arr, p_arr = _validate_labels_scores(y, p)
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if p_arr.min() < 0.0 or p_arr.max() > 1.0:
        raise ValueError(
            "ece needs probabilities in [0, 1]; pass calibrated scores, not raw margins"
        )
    idx = np.minimum((p_arr * n_bins).astype(int), n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    sum_p = np.bincount(idx, weights=p_arr, minlength=n_bins)
    sum_y = np.bincount(idx, weights=y_arr, minlength=n_bins)
    nonempty = counts > 0  # empty bins contribute nothing
    return float(np.sum(np.abs(sum_p[nonempty] - sum_y[nonempty])) / len(y_arr))


def psi(expected: ArrayLike, actual: ArrayLike, n_bins: int = 10) -> float:
    """Population stability index of ``actual`` against ``expected``.

    When ``expected`` has more than ``n_bins`` distinct values, bin edges
    come from its interior quantiles (duplicates collapsed, outer bins
    open-ended so actual values outside the expected range still land in a
    bin). When it has ``n_bins`` or fewer distinct values — binary flags,
    zero-inflated counts, constant columns — quantile edges cannot separate
    the mass points reliably, so PSI is computed at value level over the
    union of the values seen in either sample: every distinct value is its
    own bin, and a shift between values (or to an unseen value) is measured
    instead of vanishing into a single bin. Proportions are clipped to
    ``PSI_PROPORTION_FLOOR`` before the log, then the standard
    ``sum((a - e) * ln(a / e))`` is returned.
    """
    e_arr = _validate_sample(expected, "expected")
    a_arr = _validate_sample(actual, "actual")
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")
    if np.unique(e_arr).size <= n_bins:
        values = np.unique(np.concatenate([e_arr, a_arr]))
        e_prop = np.bincount(np.searchsorted(values, e_arr), minlength=len(values))
        a_prop = np.bincount(np.searchsorted(values, a_arr), minlength=len(values))
    else:
        inner = np.unique(np.quantile(e_arr, np.linspace(0.0, 1.0, n_bins + 1))[1:-1])
        n_effective_bins = len(inner) + 1
        e_prop = np.bincount(
            np.searchsorted(inner, e_arr, side="right"), minlength=n_effective_bins
        )
        a_prop = np.bincount(
            np.searchsorted(inner, a_arr, side="right"), minlength=n_effective_bins
        )
    e_clip = np.clip(e_prop / len(e_arr), PSI_PROPORTION_FLOOR, None)
    a_clip = np.clip(a_prop / len(a_arr), PSI_PROPORTION_FLOOR, None)
    return float(np.sum((a_clip - e_clip) * np.log(a_clip / e_clip)))


def csi(
    expected_df: pd.DataFrame,
    actual_df: pd.DataFrame,
    features: list[str],
    n_bins: int = 10,
) -> dict[str, float]:
    """Characteristic stability index: PSI per feature column."""
    missing_e = [f for f in features if f not in expected_df.columns]
    missing_a = [f for f in features if f not in actual_df.columns]
    if missing_e or missing_a:
        raise ValueError(
            f"csi features missing from expected frame: {missing_e}; "
            f"missing from actual frame: {missing_a}"
        )
    return {
        feature: psi(
            np.asarray(expected_df[feature], dtype=float),
            np.asarray(actual_df[feature], dtype=float),
            n_bins=n_bins,
        )
        for feature in features
    }


def _bootstrap_quantiles(stats: np.ndarray, alpha: float) -> tuple[float, float]:
    lo, hi = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def _validate_bootstrap_params(n_boot: int, alpha: float) -> None:
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def _too_few_resamples(got: int, n_boot: int, max_draws: int) -> ValueError:
    return ValueError(
        f"bootstrap collected only {got}/{n_boot} valid resamples after "
        f"{max_draws} draws; the metric rejects most resamples (e.g. a class "
        "so rare that resamples are usually single-class). Use a larger "
        "sample or a metric defined on this data."
    )


def bootstrap_ci(
    metric_fn: MetricFn,
    y: ArrayLike,
    p: ArrayLike,
    n_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Seeded percentile-bootstrap interval: ``(point, lo, hi)``.

    Rows are resampled with replacement; a resample that breaks the
    metric (``ValueError``, e.g. single-class for AUC) is skipped, with
    total draws capped at ``BOOTSTRAP_MAX_DRAW_FACTOR * n_boot``.
    Deterministic for a given seed.
    """
    _validate_bootstrap_params(n_boot, alpha)
    y_arr = np.asarray(y)
    p_arr = np.asarray(p)
    if len(y_arr) != len(p_arr):
        raise ValueError(f"y and p have different lengths ({len(y_arr)} vs {len(p_arr)})")
    point = float(metric_fn(y_arr, p_arr))

    rng = np.random.default_rng(seed)
    n = len(y_arr)
    stats = np.empty(n_boot)
    got = 0
    max_draws = BOOTSTRAP_MAX_DRAW_FACTOR * n_boot
    for _ in range(max_draws):
        if got == n_boot:
            break
        idx = rng.integers(0, n, size=n)
        try:
            stats[got] = float(metric_fn(y_arr[idx], p_arr[idx]))
        except ValueError:
            continue
        got += 1
    if got < n_boot:
        raise _too_few_resamples(got, n_boot, max_draws)
    lo, hi = _bootstrap_quantiles(stats, alpha)
    return point, lo, hi


def paired_bootstrap_diff(
    metric_fn: MetricFn,
    y: ArrayLike,
    p_a: ArrayLike,
    p_b: ArrayLike,
    n_boot: int = 500,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Seeded bootstrap interval on ``metric(y, p_a) - metric(y, p_b)``.

    The same resample indices are applied to both score vectors — the
    two models are always compared on identical rows, which is what
    makes a champion/challenger comparison honest: unpaired resampling
    would fold row-sampling noise into the difference.
    """
    _validate_bootstrap_params(n_boot, alpha)
    y_arr = np.asarray(y)
    a_arr = np.asarray(p_a)
    b_arr = np.asarray(p_b)
    if not len(y_arr) == len(a_arr) == len(b_arr):
        raise ValueError(
            f"y, p_a and p_b have different lengths "
            f"({len(y_arr)}, {len(a_arr)}, {len(b_arr)})"
        )
    diff = float(metric_fn(y_arr, a_arr)) - float(metric_fn(y_arr, b_arr))

    rng = np.random.default_rng(seed)
    n = len(y_arr)
    stats = np.empty(n_boot)
    got = 0
    max_draws = BOOTSTRAP_MAX_DRAW_FACTOR * n_boot
    for _ in range(max_draws):
        if got == n_boot:
            break
        idx = rng.integers(0, n, size=n)
        try:
            stats[got] = float(metric_fn(y_arr[idx], a_arr[idx])) - float(
                metric_fn(y_arr[idx], b_arr[idx])
            )
        except ValueError:
            continue
        got += 1
    if got < n_boot:
        raise _too_few_resamples(got, n_boot, max_draws)
    lo, hi = _bootstrap_quantiles(stats, alpha)
    return diff, lo, hi
