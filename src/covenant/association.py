"""Association measures for Check 4's proxy screen.

Excluding a variable from a model does not exclude its signal: a retained
feature can carry it. The literature screens for this with plain
association tests between features and protected attributes (arXiv
2511.03807 uses eta-squared for exactly this purpose), so Covenant reports
the pairwise association between each covenant-excluded variable and every
declared feature on a common [0, 1] scale:

* numeric-numeric — |Spearman's rho|, rank-based so a monotone recoding of
  a proxy cannot hide it;
* categorical-numeric — the correlation ratio eta (square root of
  eta-squared);
* categorical-categorical — Cramer's V with the Bergsma (2013) bias
  correction, which removes the upward bias of the classical estimator on
  small samples and large tables.

Kind is decided by dtype alone: pandas-numeric dtypes (ints, floats,
bools) are numeric, everything else is categorical. A numeric column with
few distinct values — say a 0/1-coded attribute — is deliberately *not*
re-classified as categorical, so the method used for a pair is predictable
from the snapshot's schema rather than from its values. Degenerate inputs
(constants, single categories, too-small tables) return 0.0 instead of
NaN, so a screen over many pairs never poisons its maximum.

A pairwise screen surfaces obvious proxies; it never proves absence — a
multivariate proxy can evade every pairwise test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import stats


def _paired_floats(x: ArrayLike, y: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Two equal-length float vectors with non-finite pairs dropped."""
    xf = np.asarray(x, dtype=float).ravel()
    yf = np.asarray(y, dtype=float).ravel()
    if len(xf) != len(yf):
        raise ValueError(f"length mismatch: {len(xf)} vs {len(yf)}")
    keep = np.isfinite(xf) & np.isfinite(yf)
    return xf[keep], yf[keep]


def spearman_abs(x: ArrayLike, y: ArrayLike) -> float:
    """|Spearman's rho| for a numeric-numeric pair, in [0, 1].

    Non-finite pairs are dropped; constant or too-short input returns 0.0 rather
    than NaN (no ranking exists, so no monotone association is evidenced).
    """
    xf, yf = _paired_floats(x, y)
    if len(xf) < 2 or np.ptp(xf) == 0.0 or np.ptp(yf) == 0.0:
        return 0.0
    rho = float(stats.spearmanr(xf, yf)[0])
    if np.isnan(rho):
        return 0.0
    return float(min(abs(rho), 1.0))


def correlation_ratio(categories: ArrayLike, values: ArrayLike) -> float:
    """Correlation ratio eta for a categorical-numeric pair, in [0, 1].

    eta is the square root of eta-squared, the share of the numeric
    variance explained by the category means. Non-finite pairs are dropped;
    constant values or a single category return 0.0.
    """
    cats = pd.Series(list(np.asarray(categories, dtype=object).ravel()))
    vals = np.asarray(values, dtype=float).ravel()
    if len(cats) != len(vals):
        raise ValueError(f"length mismatch: {len(cats)} vs {len(vals)}")
    keep = cats.notna().to_numpy() & np.isfinite(vals)
    cats, vals = cats[keep], vals[keep]
    if len(vals) < 2:
        return 0.0
    grand = vals.mean()
    ss_total = float(((vals - grand) ** 2).sum())
    if ss_total == 0.0:
        return 0.0
    codes, uniques = pd.factorize(cats)
    if len(uniques) < 2:
        return 0.0
    counts = np.bincount(codes)
    means = np.bincount(codes, weights=vals) / counts
    ss_between = float((counts * (means - grand) ** 2).sum())
    eta_squared = min(max(ss_between / ss_total, 0.0), 1.0)
    return float(np.sqrt(eta_squared))


def cramers_v(a: ArrayLike, b: ArrayLike) -> float:
    """Bias-corrected Cramer's V (Bergsma, 2013) for a categorical pair,
    in [0, 1].

    Computed from the chi-squared statistic of the contingency table
    (scipy.stats.chi2_contingency, no Yates correction), with Bergsma's
    correction subtracting the value expected under independence. NaN
    pairs are dropped; tables smaller than 2x2 return 0.0.
    """
    sa = pd.Series(list(np.asarray(a, dtype=object).ravel()), name="a")
    sb = pd.Series(list(np.asarray(b, dtype=object).ravel()), name="b")
    if len(sa) != len(sb):
        raise ValueError(f"length mismatch: {len(sa)} vs {len(sb)}")
    keep = ~(sa.isna() | sb.isna())
    table = pd.crosstab(sa[keep], sb[keep]).to_numpy()
    n = int(table.sum())
    r, k = table.shape
    if n < 3 or r < 2 or k < 2:
        return 0.0
    chi2 = float(stats.chi2_contingency(table, correction=False)[0])
    phi2 = chi2 / n
    phi2_corrected = max(0.0, phi2 - (r - 1) * (k - 1) / (n - 1))
    r_corrected = r - (r - 1) ** 2 / (n - 1)
    k_corrected = k - (k - 1) ** 2 / (n - 1)
    denominator = min(r_corrected, k_corrected) - 1.0
    if denominator <= 0.0:
        return 0.0
    return float(min(np.sqrt(phi2_corrected / denominator), 1.0))


def association(x: pd.Series, y: pd.Series) -> tuple[float, str]:
    """Association strength between two columns, with the method used.

    Dispatch is by dtype alone (see the module docstring): numeric-numeric
    goes to ``spearman_abs``, mixed pairs to ``correlation_ratio``
    (categorical side as the grouping), categorical-categorical to
    ``cramers_v``. Returns ``(strength in [0, 1], method_name)``.
    """
    x_numeric = pd.api.types.is_numeric_dtype(x)
    y_numeric = pd.api.types.is_numeric_dtype(y)
    if x_numeric and y_numeric:
        strength = spearman_abs(
            x.to_numpy(dtype=float, na_value=np.nan),
            y.to_numpy(dtype=float, na_value=np.nan),
        )
        return strength, "spearman_abs"
    if x_numeric != y_numeric:
        cats, vals = (y, x) if x_numeric else (x, y)
        strength = correlation_ratio(
            cats.to_numpy(), vals.to_numpy(dtype=float, na_value=np.nan)
        )
        return strength, "correlation_ratio"
    return cramers_v(x.to_numpy(), y.to_numpy()), "cramers_v"
