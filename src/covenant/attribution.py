"""The measured side of Covenant's checks: what actually drove the score.

SHAP values of ``p_bad``, positive = pushes toward denial. Post-hoc
attributions are an approximation, not ground truth (Sudjianto & Zhang,
2021), and they are sensitive to the background distribution (Pace
Analytics, 2024) — so the background is a first-class, seeded parameter and
Check 1 reports sensitivity to it rather than hiding it.

Categorical features are encoded to integer codes for SHAP's masker and
decoded back to their labels before the model scores, so pipelines with
one-hot or target encoders see the data they were fitted on. Each
categorical feature receives a single whole-feature attribution.

The declared side (production reason-code methods) lives in
``covenant.declared``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from covenant.model import CovenantModel


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
