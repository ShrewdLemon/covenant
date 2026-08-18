"""Attributions: the measured side and the declared side of Check 1.

Measured: SHAP values of ``p_bad`` via the model-agnostic permutation
explainer. Post-hoc attributions are an approximation, not ground truth
(Sudjianto & Zhang, 2021), and they are sensitive to the background
distribution (Pace Analytics, 2024) — so the background is a first-class,
seeded parameter and Check 1 reports sensitivity to it rather than hiding it.

Declared: the reason codes the user's production pipeline would send a
denied applicant, derived by the covenant's declared method
(Krivorotov & Richey, 2022 taxonomy) or supplied as a file.

Categorical features are encoded to integer codes for SHAP's masker and
decoded back to their labels before the model scores, so pipelines with
one-hot or target encoders see the data they were fitted on. Each
categorical feature receives a single whole-feature attribution.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import shap

from covenant.model import CovenantModel
from covenant.schema import ReasonCodeMethod, ReasonCodePolicy


class CategoryCodec:
    """Bidirectional map between dataframes with categorical columns and the
    all-numeric matrices SHAP's masker requires."""

    def __init__(self, frame: pd.DataFrame, categorical: list[str]) -> None:
        self.columns = list(frame.columns)
        self.categories: dict[str, pd.Index] = {}
        for col in categorical:
            if col not in frame.columns:
                continue
            if frame[col].isna().any():
                raise ValueError(
                    f"categorical feature {col!r} has missing values; impute "
                    "them upstream — SHAP masking over missing categories is "
                    "not supported yet"
                )
            self.categories[col] = pd.Index(pd.unique(frame[col]))

    def encode(self, X: pd.DataFrame) -> np.ndarray:
        out = np.empty((len(X), len(self.columns)), dtype=float)
        for j, col in enumerate(self.columns):
            if col in self.categories:
                codes = pd.Categorical(X[col], categories=self.categories[col]).codes
                if (codes < 0).any():
                    unseen = sorted(set(X[col]) - set(self.categories[col]))
                    raise ValueError(
                        f"categorical feature {col!r} has values outside the "
                        f"training snapshot: {unseen[:5]}"
                    )
                out[:, j] = codes
            else:
                out[:, j] = pd.to_numeric(X[col]).to_numpy(dtype=float)
        return out

    def decode(self, arr: np.ndarray) -> pd.DataFrame:
        frame = pd.DataFrame(arr, columns=self.columns)
        for col, cats in self.categories.items():
            codes = frame[col].round().astype(int).clip(0, len(cats) - 1)
            frame[col] = cats.take(codes)
        return frame


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
    categorical: list[str] | None = None,
    random_state: int = 0,
    npermutations: int = 4,
) -> pd.DataFrame:
    """Per-row SHAP attributions of ``p_bad``, positive = pushes toward denial.

    Uses the permutation explainer uniformly so pipelines, scorecards and
    boosted trees all go through the same, model-agnostic path. Exact
    fast paths (TreeExplainer, EBM shape functions) are roadmap items.
    """
    features = list(X.columns)
    codec = CategoryCodec(pd.concat([X, background], ignore_index=True), categorical or [])

    def score(arr: np.ndarray) -> np.ndarray:
        return model.p_bad(codec.decode(np.atleast_2d(arr)))

    masker = shap.maskers.Independent(
        codec.encode(background), max_samples=len(background)
    )
    n_features = len(features)
    max_evals = (2 * n_features + 1) * npermutations
    try:
        explainer = shap.PermutationExplainer(score, masker, seed=random_state)
    except TypeError:  # older shap without seed=
        np.random.seed(random_state)
        explainer = shap.PermutationExplainer(score, masker)
    explanation = explainer(codec.encode(X), max_evals=max_evals, silent=True)
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
    ids: pd.Series | None = None,
    id_column: str | None = None,
) -> tuple[list[frozenset[str]], list[str]]:
    """(top-k reason sets, top-1 reason) per row, by the declared method."""
    if policy.method is ReasonCodeMethod.CUSTOM:
        if ids is None or id_column is None:
            raise DeclaredMethodError(
                "custom reason codes require checks.reason_codes.id_column"
            )
        return _custom_reasons(policy, ids, id_column, covenants_dir)
    attributions = declared_attributions(policy, X, covenants_dir)
    return top_k_sets(attributions, policy.top_k), top_1(attributions)


def _resolve(ref: str, covenants_dir: Path) -> Path:
    path = Path(ref)
    return path if path.is_absolute() else covenants_dir / path


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
    table_path = _resolve(table_ref, covenants_dir)
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
    try:
        values_matrix = X.to_numpy(dtype=float)
    except (TypeError, ValueError) as err:
        raise DeclaredMethodError(
            "difference_from_mean needs all-numeric features; for models "
            "with categorical features use reason_codes.method: custom and "
            "export the production reasons"
        ) from err
    values = coef * (values_matrix - mean) / scale
    return pd.DataFrame(values, columns=X.columns, index=X.index)


def _custom_reasons(
    policy: ReasonCodePolicy,
    ids: pd.Series,
    id_column: str,
    covenants_dir: Path,
) -> tuple[list[frozenset[str]], list[str]]:
    """Reason codes exported from the production pipeline: a CSV with the
    id column plus reason_1..reason_k, joined to the data on the id."""
    reasons_ref = policy.parameters.get("reasons_file")
    if reasons_ref is None:
        raise DeclaredMethodError(
            "reason_codes.parameters.reasons_file must point to a CSV of "
            f"{id_column},reason_1..reason_k for method custom"
        )
    reasons_path = _resolve(reasons_ref, covenants_dir)
    table = pd.read_csv(reasons_path)
    if id_column not in table.columns:
        raise DeclaredMethodError(
            f"{reasons_path} lacks the id column {id_column!r} declared in "
            "checks.reason_codes.id_column"
        )
    reason_cols = [c for c in table.columns if c.startswith("reason_")]
    if not reason_cols:
        raise DeclaredMethodError(f"{reasons_path} has no reason_1..reason_k columns")
    if table[id_column].duplicated().any():
        dupes = table.loc[table[id_column].duplicated(), id_column].head(5).tolist()
        raise DeclaredMethodError(
            f"{reasons_path} has duplicate ids in {id_column!r}: {dupes}"
        )
    table = table.set_index(id_column)
    missing_mask = ~ids.isin(table.index)
    if missing_mask.any():
        missing_ids = ids[missing_mask].head(5).tolist()
        raise DeclaredMethodError(
            f"{reasons_path} has no reasons for {int(missing_mask.sum())} "
            f"scored rows; first missing {id_column!r} values: {missing_ids}"
        )
    rows = table.loc[ids]
    sets = [
        frozenset(str(v) for v in row if pd.notna(v))
        for row in rows[reason_cols].to_numpy()
    ]
    top1 = [str(v) for v in rows[reason_cols[0]]]
    return sets, top1
