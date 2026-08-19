"""The measured side of Covenant's checks: what actually drove the score.

SHAP values of ``p_bad``, positive = pushes toward denial. Post-hoc
attributions are an approximation, not ground truth (Sudjianto & Zhang,
2021), and they are sensitive to the background distribution (Pace
Analytics, 2024) — so the background is a first-class, seeded parameter and
Check 1 reports sensitivity to it rather than hiding it.

When the model admits an exact answer, Covenant computes it instead of
sampling one: linear models get their closed-form Shapley contributions
(Nair, Sudjianto et al., 2022 derive adverse-action attributions for
additive models from first principles), tree ensembles get TreeExplainer,
and everything else falls back to the model-agnostic permutation explainer.
``explain`` returns which path produced the numbers, and check records can
carry that name so a validator knows whether the measured side is exact or
sampled.

Categorical features are encoded to integer codes for SHAP's masker and
decoded back to their labels before the model scores, so pipelines with
one-hot or target encoders see the data they were fitted on. Each
categorical feature receives a single whole-feature attribution.

The declared side (production reason-code methods) lives in
``covenant.declared``.
"""

from __future__ import annotations

import importlib
import sys

import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from covenant.model import CovenantModel

_TREE_CLASSIFIERS: tuple[tuple[str, str], ...] = (
    ("xgboost", "XGBClassifier"),
    ("lightgbm", "LGBMClassifier"),
)


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


def explain(
    model: CovenantModel,
    X: pd.DataFrame,
    background: pd.DataFrame,
    categorical: list[str] | None = None,
    random_state: int = 0,
    npermutations: int = 4,
) -> tuple[pd.DataFrame, str]:
    """Per-row attributions of ``p_bad``, positive = pushes toward denial,
    plus the name of the path that produced them.

    Returns ``(attributions, path)`` with ``path`` one of ``"ebm-exact"``,
    ``"linear-exact"``, ``"tree-shap"`` or ``"permutation-shap"``.
    Selection, most exact first (``"ebm-exact"`` reads an InterpretML EBM's
    own shape functions — see ``_ebm_exact``):

    * ``"linear-exact"`` — no categorical features and the estimator is a
      binary sklearn ``LogisticRegression``, or a ``Pipeline`` whose final
      step is one and whose earlier steps are all ``StandardScaler`` or
      ``"passthrough"`` (an affine per-feature map that composes exactly).
      Contribution_j = beta_eff_j * (x_j - mean(background_j)) where
      beta_eff composes the scalers (coef / scale). This is the exact
      interventional Shapley value of a linear model in **logit space**;
      the permutation path explains **probability space**. Per-row rankings
      are what the checks compare, and for a monotone link those rankings
      are consistent for a linear model. Deterministic, instant, exact.
    * ``"tree-shap"`` — no categorical features and the estimator is a bare
      sklearn ``RandomForestClassifier`` / ``GradientBoostingClassifier`` /
      ``HistGradientBoostingClassifier``, or an xgboost / lightgbm sklearn
      classifier when those libraries are installed. ``shap.TreeExplainer``
      with ``data=background`` (interventional); ``model_output=
      "probability"`` is tried first, falling back to the margin output if
      shap refuses it, and on any TreeExplainer failure the call falls
      through to the permutation path silently — the returned path name is
      then ``"permutation-shap"``, so the record never overstates exactness.
    * ``"permutation-shap"`` — everything else: the model-agnostic
      codec-based permutation explainer over a seeded background, unchanged.

    Post-hoc attributions remain an approximation of the model, not ground
    truth (Sudjianto & Zhang, 2021); the exact paths remove the sampling
    noise, not the interpretive caveats.
    """
    if _is_ebm_classifier(model.estimator):
        frame = _ebm_exact(model, X, background)
        if frame is not None:
            return frame, "ebm-exact"
    active_categories = [c for c in (categorical or []) if c in X.columns]
    if not active_categories:
        beta_eff = _linear_effective_coefficients(model)
        if beta_eff is not None:
            return _linear_exact(model, X, background, beta_eff), "linear-exact"
        if _is_supported_tree_classifier(model.estimator):
            frame = _tree_shap(model, X, background)
            if frame is not None:
                return frame, "tree-shap"
    frame = _permutation_shap(model, X, background, categorical, random_state, npermutations)
    return frame, "permutation-shap"


def _is_ebm_classifier(estimator: object) -> bool:
    try:
        from interpret.glassbox import ExplainableBoostingClassifier
    except ImportError:
        return False
    return isinstance(estimator, ExplainableBoostingClassifier)


def _ebm_exact(
    model: CovenantModel, X: pd.DataFrame, background: pd.DataFrame
) -> pd.DataFrame | None:
    """Exact shape-function contributions for an InterpretML EBM.

    An EBM is inherently interpretable (Sudjianto & Zhang, 2021): its
    per-term contributions ``eval_terms`` are the model, not an
    approximation of it, so no sampling explainer is needed. Contributions
    are in logit space (like linear-exact), centered on the background
    sample's mean per term so the reference point matches the other paths.
    Pairwise interaction terms are split equally between their two features
    — the standard convention, stated here because it is a convention, not
    a theorem. Categorical features need no codec: the EBM consumes its
    raw values. Returns None (falling back to permutation SHAP) whenever
    the estimator's own feature record cannot be aligned with the data.
    """
    est = model.estimator
    names = [str(n) for n in getattr(est, "feature_names_in_", [])]
    if not names or any(n not in X.columns for n in names):
        return None
    term_features = getattr(est, "term_features_", None)
    if term_features is None:
        return None
    try:
        raw = np.asarray(est.eval_terms(X[names]))
        baseline = np.asarray(est.eval_terms(background[names])).mean(axis=0)
    except Exception:
        return None
    centered = raw - baseline
    if not np.isfinite(centered).all():
        return None
    sign = -1.0 if model.bad_class_index == 0 else 1.0
    out = np.zeros((len(X), len(X.columns)))
    column_index = {c: i for i, c in enumerate(X.columns)}
    for term, feature_ids in enumerate(term_features):
        weight = sign / len(feature_ids)
        for feature_id in feature_ids:
            j = column_index.get(names[feature_id])
            if j is not None:
                out[:, j] += weight * centered[:, term]
    return pd.DataFrame(out, columns=list(X.columns), index=X.index)


def measured_attributions(
    model: CovenantModel,
    X: pd.DataFrame,
    background: pd.DataFrame,
    categorical: list[str] | None = None,
    random_state: int = 0,
    npermutations: int = 4,
) -> pd.DataFrame:
    """Per-row SHAP attributions of ``p_bad``, positive = pushes toward denial.

    Thin wrapper over :func:`explain` that keeps the original
    frame-returning signature; use ``explain`` when the caller should record
    which attribution path produced the numbers.
    """
    return explain(model, X, background, categorical, random_state, npermutations)[0]


def _linear_effective_coefficients(model: CovenantModel) -> np.ndarray | None:
    """Effective per-feature slopes of the logit toward ``p_bad``, or None
    when the estimator is not an exactly-decomposable linear model.

    Eligible: a fitted binary ``LogisticRegression``, bare or as the final
    step of a ``Pipeline`` whose earlier steps are all ``StandardScaler`` or
    ``"passthrough"``. A StandardScaler is affine per feature, so the
    composed slope is ``coef / prod(scale)``; the intercept and centring
    terms cancel out of ``x_j - mean(background_j)``.
    """
    estimator = model.estimator
    scaler_steps: list[StandardScaler] = []
    if isinstance(estimator, Pipeline):
        for _, step in estimator.steps[:-1]:
            if isinstance(step, StandardScaler):
                scaler_steps.append(step)
            elif not (isinstance(step, str) and step == "passthrough"):
                return None
        final = estimator.steps[-1][1]
    else:
        final = estimator
    if not isinstance(final, LogisticRegression):
        return None
    classes = getattr(final, "classes_", None)
    coef = getattr(final, "coef_", None)
    if classes is None or coef is None or len(classes) != 2:
        return None
    names_in = getattr(estimator, "feature_names_in_", None)
    if names_in is not None and list(names_in) != model.feature_names:
        return None

    scale = np.ones(len(model.feature_names))
    for step in scaler_steps:
        if step.scale_ is not None:
            scale = scale * np.asarray(step.scale_, dtype=float)
    beta = np.asarray(coef, dtype=float)[0]
    if beta.shape != scale.shape:
        return None
    beta_eff = beta / scale
    return beta_eff if model.bad_class_index == 1 else -beta_eff


def _linear_exact(
    model: CovenantModel, X: pd.DataFrame, background: pd.DataFrame, beta_eff: np.ndarray
) -> pd.DataFrame:
    features = model.feature_names
    mu = background[features].to_numpy(dtype=float).mean(axis=0)
    values = (X[features].to_numpy(dtype=float) - mu) * beta_eff
    frame = pd.DataFrame(values, columns=features, index=X.index)
    if list(X.columns) != features:  # undeclared columns have exactly zero attribution
        frame = frame.reindex(columns=X.columns, fill_value=0.0)
    return frame


def _is_supported_tree_classifier(estimator: object) -> bool:
    if isinstance(
        estimator,
        RandomForestClassifier | GradientBoostingClassifier | HistGradientBoostingClassifier,
    ):
        return True
    for module_name, class_name in _TREE_CLASSIFIERS:
        module = sys.modules.get(module_name)
        if module is None:
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue
        cls = getattr(module, class_name, None)
        if cls is not None and isinstance(estimator, cls):
            return True
    return False


def _tree_shap(
    model: CovenantModel, X: pd.DataFrame, background: pd.DataFrame
) -> pd.DataFrame | None:
    """Interventional TreeExplainer attributions, or None when shap cannot
    produce them (the caller then falls through to permutation)."""
    features = model.feature_names
    try:
        bg = background[features].to_numpy(dtype=float)
        Xv = X[features].to_numpy(dtype=float)
    except Exception:
        return None
    for kwargs in ({"model_output": "probability"}, {}):
        try:
            explainer = shap.TreeExplainer(model.estimator, data=bg, **kwargs)
            raw = explainer.shap_values(Xv, check_additivity=False)
        except Exception:
            continue
        values = _positive_class_slice(raw, model.bad_class_index)
        if (
            values is None
            or values.shape != (len(X), len(features))
            or not np.isfinite(values).all()
        ):
            continue
        frame = pd.DataFrame(values, columns=features, index=X.index)
        if list(X.columns) != features:
            frame = frame.reindex(columns=X.columns, fill_value=0.0)
        return frame
    return None


def _positive_class_slice(raw: object, bad_class_index: int) -> np.ndarray | None:
    """The ``p_bad`` slice of TreeExplainer output, which may arrive as a
    per-class list, an ``(n, p, 2)`` stack, or a single ``(n, p)`` array
    explaining the class-1 score (negated when ``p_bad`` is class 0)."""
    if isinstance(raw, list):
        if len(raw) == 1:
            raw = raw[0]
        elif bad_class_index < len(raw):
            return np.asarray(raw[bad_class_index], dtype=float)
        else:
            return None
    arr = np.asarray(raw, dtype=float)
    if arr.ndim == 3 and bad_class_index < arr.shape[2]:
        return arr[:, :, bad_class_index]
    if arr.ndim == 2:
        return arr if bad_class_index == 1 else -arr
    return None


def _permutation_shap(
    model: CovenantModel,
    X: pd.DataFrame,
    background: pd.DataFrame,
    categorical: list[str] | None,
    random_state: int,
    npermutations: int,
) -> pd.DataFrame:
    """The model-agnostic path: permutation SHAP of ``p_bad`` over a seeded
    background, with categorical columns run through the codec."""
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
