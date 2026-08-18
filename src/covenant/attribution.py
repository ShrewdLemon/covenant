"""Attributions: the measured side and the declared side of Check 1.

Measured: SHAP values of ``p_bad`` via the model-agnostic permutation
explainer. Post-hoc attributions are an approximation, not ground truth
(Sudjianto & Zhang, 2021), and they are sensitive to the background
distribution (Pace Analytics, 2024) — so the background is a first-class,
seeded parameter and Check 1 reports sensitivity to it rather than hiding it.

Declared: the reason codes the user's production pipeline would send a
denied applicant, derived by the covenant's declared method
(Krivorotov & Richey, 2022 taxonomy) or supplied as a file.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shap

from covenant.model import CovenantModel
from covenant.schema import ReasonCodeMethod, ReasonCodePolicy


def sample_background(
    train: pd.DataFrame, feature_names: list[str], size: int, random_state: int
) -> pd.DataFrame:
    """A seeded sample of the training snapshot, in declared feature order."""
    frame = train[feature_names]
    if len(frame) <= size:
        return frame.reset_index(drop=True)
    rng = np.random.default_rng(random_state)
    idx = rng.choice(len(frame), size=size, replace=False)
    return frame.iloc[np.sort(idx)].reset_index(drop=True)


def measured_attributions(
    model: CovenantModel,
    X: pd.DataFrame,
    background: pd.DataFrame,
    random_state: int = 0,
    npermutations: int = 4,
) -> pd.DataFrame:
    """Per-row SHAP attributions of ``p_bad``, positive = pushes toward denial.

    Uses the permutation explainer uniformly so pipelines, scorecards and
    boosted trees all go through the same, model-agnostic path. Exact
    fast paths (TreeExplainer, EBM shape functions) are roadmap items.
    """
    features = list(X.columns)
    masker = shap.maskers.Independent(background.to_numpy(), max_samples=len(background))
    n_features = len(features)
    max_evals = (2 * n_features + 1) * npermutations
    try:
        explainer = shap.PermutationExplainer(
            model.p_bad_from_array, masker, seed=random_state
        )
    except TypeError:  # older shap without seed=
        np.random.seed(random_state)
        explainer = shap.PermutationExplainer(model.p_bad_from_array, masker)
    explanation = explainer(X.to_numpy(), max_evals=max_evals, silent=True)
    return pd.DataFrame(explanation.values, columns=features, index=X.index)


def top_k_sets(attributions: pd.DataFrame, k: int) -> list[frozenset[str]]:
    """Top-k denial reasons per row: the k features pushing hardest toward
    denial (largest positive attribution)."""
    cols = np.asarray(attributions.columns)
    values = attributions.to_numpy()
    order = np.argsort(-values, axis=1, kind="stable")[:, :k]
    return [frozenset(cols[row]) for row in order]


def top_1(attributions: pd.DataFrame) -> list[str]:
    cols = np.asarray(attributions.columns)
    return list(cols[np.argmax(attributions.to_numpy(), axis=1)])


# --------------------------------------------------------------------------
# Declared side
# --------------------------------------------------------------------------


class DeclaredMethodError(ValueError):
    pass


def declared_attributions(
    policy: ReasonCodePolicy,
    X: pd.DataFrame,
    covenants_dir: Path,
) -> pd.DataFrame:
    """Attribution matrix implied by the declared reason-code method.

    Higher value = stronger reason for denial, matching the measured side.
    """
    if policy.method is ReasonCodeMethod.DIFFERENCE_FROM_MEAN:
        return _difference_from_mean(policy, X, covenants_dir)
    raise DeclaredMethodError(
        f"declared method {policy.method.value!r} is not implemented in this "
        "version; implemented: difference_from_mean (attributions), "
        "custom (reason-code file). most_points_lost, univariate and "
        "shapley are on the roadmap."
    )


def declared_reason_sets(
    policy: ReasonCodePolicy,
    X: pd.DataFrame,
    covenants_dir: Path,
) -> tuple[list[frozenset[str]], list[str]]:
    """(top-k reason sets, top-1 reason) per row, by the declared method."""
    if policy.method is ReasonCodeMethod.CUSTOM:
        return _custom_reasons(policy, X, covenants_dir)
    attributions = declared_attributions(policy, X, covenants_dir)
    return top_k_sets(attributions, policy.top_k), top_1(attributions)


def _difference_from_mean(
    policy: ReasonCodePolicy, X: pd.DataFrame, covenants_dir: Path
) -> pd.DataFrame:
    """Linear contribution vs the mean applicant: coef * (x - mean) / scale.

    The coefficient table is the artefact a production system actually uses
    to phrase adverse-action letters — which is exactly why it can go stale
    against the deployed model. Columns: feature, coef, optional mean
    (default 0), optional scale (default 1).
    """
    table_ref = policy.parameters.get("coefficients")
    if table_ref is None:
        raise DeclaredMethodError(
            "reason_codes.parameters.coefficients must point to a CSV of "
            "feature,coef[,mean][,scale] for method difference_from_mean"
        )
    table_path = Path(table_ref)
    if not table_path.is_absolute():
        table_path = covenants_dir / table_path
    table = pd.read_csv(table_path)
    if "feature" not in table.columns or "coef" not in table.columns:
        raise DeclaredMethodError(
            f"{table_path} needs columns feature,coef[,mean][,scale]"
        )
    table = table.set_index("feature")
    missing = [c for c in X.columns if c not in table.index]
    if missing:
        raise DeclaredMethodError(
            f"coefficient table {table_path} lacks features: {missing}"
        )
    table = table.reindex(X.columns)
    coef = table["coef"].to_numpy(dtype=float)
    mean = table.get("mean", pd.Series(0.0, index=table.index)).to_numpy(dtype=float)
    scale = table.get("scale", pd.Series(1.0, index=table.index)).to_numpy(dtype=float)
    values = coef * (X.to_numpy(dtype=float) - mean) / scale
    return pd.DataFrame(values, columns=X.columns, index=X.index)


def _custom_reasons(
    policy: ReasonCodePolicy, X: pd.DataFrame, covenants_dir: Path
) -> tuple[list[frozenset[str]], list[str]]:
    """Reason codes exported from the production pipeline: a CSV with columns
    reason_1..reason_k, positionally aligned to the scored data."""
    reasons_ref = policy.parameters.get("reasons_file")
    if reasons_ref is None:
        raise DeclaredMethodError(
            "reason_codes.parameters.reasons_file must point to a CSV of "
            "reason_1..reason_k for method custom"
        )
    reasons_path = Path(reasons_ref)
    if not reasons_path.is_absolute():
        reasons_path = covenants_dir / reasons_path
    table = pd.read_csv(reasons_path)
    reason_cols = [c for c in table.columns if c.startswith("reason_")]
    if not reason_cols:
        raise DeclaredMethodError(f"{reasons_path} has no reason_1..reason_k columns")
    rows = table.iloc[X.index] if X.index.max() < len(table) else None
    if rows is None or len(table) < len(X):
        raise DeclaredMethodError(
            f"{reasons_path} has {len(table)} rows; custom reasons must cover "
            "every row of the scored data, aligned positionally"
        )
    sets = [
        frozenset(str(v) for v in row if pd.notna(v))
        for row in rows[reason_cols].to_numpy()
    ]
    top1 = [str(v) for v in rows[reason_cols[0]]]
    return sets, top1
