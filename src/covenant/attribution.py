"""The measured side of Covenant's checks: what actually drove the score.

SHAP values of ``p_bad``, positive = pushes toward denial. Post-hoc
attributions are an approximation, not ground truth (Sudjianto & Zhang,
2021), and they are sensitive to the background distribution (Pace
Analytics, 2024) — so the background is a first-class, seeded parameter and
Check 1 reports sensitivity to it rather than hiding it.

When the model admits an exact answer, Covenant computes it instead of
sampling one: linear models — bare or wrapped in standard preprocessing
pipelines, one-hot encoders included — get their closed-form Shapley
contributions (Nair, Sudjianto et al., 2022 derive adverse-action
attributions for additive models from first principles), tree ensembles —
bare or behind simple all-numeric pipeline transforms — get TreeExplainer,
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
from scipy import sparse
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

from covenant.model import CovenantModel

_TREE_CLASSIFIERS: tuple[tuple[str, str], ...] = (
    ("xgboost", "XGBClassifier"),
    ("lightgbm", "LGBMClassifier"),
)

# ColumnTransformer blocks each path can decompose exactly per raw feature.
# "passthrough" and "drop" are always accepted alongside these.
_LINEAR_CT_TRANSFORMS: tuple[type, ...] = (StandardScaler, OneHotEncoder)
_TREE_CT_TRANSFORMS: tuple[type, ...] = (StandardScaler, OneHotEncoder, OrdinalEncoder)


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

    * ``"linear-exact"`` — the estimator is a fitted binary sklearn
      ``LogisticRegression``, in one of two wrapper shapes. Bare, or behind
      ``StandardScaler`` / ``"passthrough"`` pipeline steps (no categorical
      features): contribution_j = beta_eff_j * (x_j - mean(background_j))
      with beta_eff composing the scalers (coef / scale). Or behind a
      ``Pipeline`` carrying a single ``ColumnTransformer`` whose blocks are
      all ``StandardScaler``, ``"passthrough"`` or ``OneHotEncoder`` (any
      ``handle_unknown``), plus optional ``StandardScaler`` /
      ``"passthrough"`` steps around it — the realistic scorecard shape,
      categorical features included, since the encoder consumes them:
      contribution of raw feature f = sum over f's transformed columns j of
      beta_j * (z_j - mean(background z_j)), computed from one transform of
      X and the background through the pipeline's own fitted pre-steps (see
      ``_linear_exact_pipeline``). Both shapes are the exact interventional
      Shapley value of the model in **logit space**; the permutation path
      explains **probability space**. Per-row rankings are what the checks
      compare, and for a monotone link those rankings are consistent for an
      additive model. Deterministic, instant, exact.
    * ``"tree-shap"`` — a supported tree classifier: sklearn
      ``RandomForestClassifier`` / ``GradientBoostingClassifier`` /
      ``HistGradientBoostingClassifier``, or an xgboost / lightgbm sklearn
      classifier when those libraries are installed. Bare (no categorical
      features), or the final step of a ``Pipeline`` whose pre-steps are
      ``StandardScaler`` / ``"passthrough"`` or a single
      ``ColumnTransformer`` of ``StandardScaler`` / ``"passthrough"`` /
      ``OrdinalEncoder`` / ``OneHotEncoder`` blocks; X and the background
      are then transformed once through the fitted pre-steps and
      encoder-expanded columns are summed back to their raw feature (see
      ``_tree_shap_pipeline``). ``shap.TreeExplainer`` with
      ``data=background`` (interventional); ``model_output="probability"``
      is tried first, falling back to the margin output if shap refuses it.
      On any doubt — unknown transformer, column-mapping failure,
      TreeExplainer failure — the call falls through to the permutation
      path silently and the returned path name is ``"permutation-shap"``,
      so the record never overstates exactness.
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
    # The categorical gate is per path: the raw-space paths below need
    # all-numeric columns, while the pipeline paths encode categoricals
    # themselves and decide their own eligibility.
    active_categories = [c for c in (categorical or []) if c in X.columns]
    if not active_categories:
        beta_eff = _linear_effective_coefficients(model)
        if beta_eff is not None:
            return _linear_exact(model, X, background, beta_eff), "linear-exact"
    frame = _linear_exact_pipeline(model, X, background)
    if frame is not None:
        return frame, "linear-exact"
    if not active_categories and _is_supported_tree_classifier(model.estimator):
        frame = _tree_shap(model, X, background)
        if frame is not None:
            return frame, "tree-shap"
    frame = _tree_shap_pipeline(model, X, background)
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


def _attribution_frame(values: np.ndarray, X: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Per-feature values as a frame aligned to ``X``; undeclared columns
    have exactly zero attribution."""
    frame = pd.DataFrame(values, columns=features, index=X.index)
    if list(X.columns) != features:
        frame = frame.reindex(columns=X.columns, fill_value=0.0)
    return frame


def _linear_exact(
    model: CovenantModel, X: pd.DataFrame, background: pd.DataFrame, beta_eff: np.ndarray
) -> pd.DataFrame:
    features = model.feature_names
    mu = background[features].to_numpy(dtype=float).mean(axis=0)
    values = (X[features].to_numpy(dtype=float) - mu) * beta_eff
    return _attribution_frame(values, X, features)


def _pipeline_transform_parts(
    model: CovenantModel, allowed_ct_types: tuple[type, ...], require_ct: bool
) -> tuple[Pipeline, object, np.ndarray] | None:
    """Split an eligible fitted ``Pipeline`` into ``(pre_steps, final
    estimator, transformed-column -> raw-feature index map)``, or None when
    the wrapper is not a shape Covenant can decompose exactly.

    Eligible pre-steps: ``StandardScaler`` / ``"passthrough"`` (which map
    columns 1:1, so they preserve any column-to-feature map) plus at most
    one ``ColumnTransformer`` whose blocks are drawn from
    ``allowed_ct_types`` (or ``"passthrough"`` / ``"drop"``). The map is
    what lets a transformed-space attribution be reported per raw feature
    without guessing; when it cannot be built exactly the caller must fall
    back to a sampled path rather than report an exact one.
    """
    estimator = model.estimator
    if not isinstance(estimator, Pipeline) or len(estimator.steps) < 2:
        return None
    names_in = getattr(estimator, "feature_names_in_", None)
    if names_in is not None and set(map(str, names_in)) != set(model.feature_names):
        return None
    # Same *set* of names suffices: the ColumnTransformer selects its input
    # columns by name and the column-to-feature map below resolves by name,
    # so a covenant that lists the features in a different order than the
    # model was fitted with still gets the exact path.
    ct: ColumnTransformer | None = None
    for _, step in estimator.steps[:-1]:
        if isinstance(step, ColumnTransformer):
            if ct is not None:
                return None
            ct = step
        elif isinstance(step, StandardScaler) or (isinstance(step, str) and step == "passthrough"):
            continue
        else:
            return None
    if ct is None:
        if require_ct:
            return None
        col_to_raw = np.arange(len(model.feature_names))
    else:
        mapped = _transformed_column_map(ct, model.feature_names, allowed_ct_types)
        if mapped is None:
            return None
        col_to_raw = mapped
    return estimator[:-1], estimator.steps[-1][1], col_to_raw


def _transformed_column_map(
    ct: ColumnTransformer, feature_names: list[str], allowed_types: tuple[type, ...]
) -> np.ndarray | None:
    """The raw-feature index of every output column of a fitted
    ``ColumnTransformer``, or None when any output column cannot be traced
    to exactly one raw feature.

    Walks ``transformers_`` in fitted order — the order ``transform``
    concatenates output blocks. Per-column blocks (``StandardScaler``,
    ``OrdinalEncoder``, ``"passthrough"``) map 1:1; a ``OneHotEncoder``
    expands each input into one column per emitted category, with ``drop=``
    and infrequent-category settings read from the fitted encoder's own
    public attributes. Any block type outside ``allowed_types`` (or any
    column spec that cannot be resolved) returns None: a transformer that
    mixes features, such as PCA, admits no exact per-raw-feature
    decomposition, so claiming one would overstate the record.
    """
    transformers = getattr(ct, "transformers_", None)
    if transformers is None:
        return None
    index = {name: j for j, name in enumerate(feature_names)}
    col_to_raw: list[int] = []
    for _, transformer, cols in transformers:
        if isinstance(transformer, str) and transformer == "drop":
            continue
        raw_idx = _resolve_ct_columns(cols, len(feature_names), index)
        if raw_idx is None:
            return None
        if _is_passthrough_block(transformer):
            col_to_raw.extend(raw_idx)
            continue
        if not isinstance(transformer, allowed_types):
            return None
        if isinstance(transformer, OneHotEncoder):
            counts = _one_hot_output_counts(transformer, len(raw_idx))
            if counts is None:
                return None
            for raw_j, count in zip(raw_idx, counts, strict=True):
                col_to_raw.extend([raw_j] * count)
        else:  # per-column transformer: one output column per input column
            col_to_raw.extend(raw_idx)
    return np.asarray(col_to_raw, dtype=int)


def _is_passthrough_block(transformer: object) -> bool:
    """True for a ColumnTransformer block that passes columns through 1:1.

    Fitted ``transformers_`` store ``"passthrough"`` either as the literal
    string (older sklearn) or as the identity ``FunctionTransformer`` newer
    sklearn substitutes for it; a FunctionTransformer with a user-supplied
    func is not accepted — its column semantics cannot be verified."""
    if isinstance(transformer, str) and transformer == "passthrough":
        return True
    return (
        isinstance(transformer, FunctionTransformer)
        and transformer.func is None
        and transformer.inverse_func is None
    )


def _resolve_ct_columns(
    cols: object, n_features: int, index: dict[str, int]
) -> list[int] | None:
    """A ColumnTransformer column spec as raw-feature indices, or None for
    any spec (callable, unknown name, out-of-range position) that cannot be
    resolved deterministically against the declared feature order."""
    if isinstance(cols, slice):
        return list(range(*cols.indices(n_features)))
    if isinstance(cols, str):
        cols = [cols]
    elif isinstance(cols, int | np.integer) and not isinstance(cols, bool):
        cols = [int(cols)]
    try:
        arr = np.asarray(cols)
    except Exception:
        return None
    if arr.ndim != 1:
        return None
    if arr.dtype == np.bool_:
        if arr.size != n_features:
            return None
        return [int(j) for j in np.flatnonzero(arr)]
    out: list[int] = []
    for c in arr.tolist():
        if isinstance(c, str):
            j = index.get(c)
            if j is None:
                return None
            out.append(j)
        elif isinstance(c, int | np.integer) and not isinstance(c, bool):
            j = int(c)
            if not 0 <= j < n_features:
                return None
            out.append(j)
        else:
            return None
    return out


def _one_hot_output_counts(encoder: OneHotEncoder, n_inputs: int) -> list[int] | None:
    """Output columns per input column of a fitted ``OneHotEncoder``, read
    from its public fitted attributes: ``len(categories_[k])``, minus the
    infrequent categories collapsed into one bucket, minus a dropped
    category where ``drop_idx_`` marks one. None on any shape mismatch —
    the caller then falls back rather than guess."""
    categories = getattr(encoder, "categories_", None)
    if categories is None or len(categories) != n_inputs:
        return None
    counts = [len(cats) for cats in categories]
    infrequent = getattr(encoder, "infrequent_categories_", None)
    if infrequent is not None:
        if len(infrequent) != n_inputs:
            return None
        for k, infreq in enumerate(infrequent):
            if infreq is not None and len(infreq) > 0:
                counts[k] -= len(infreq) - 1
    drop_idx = getattr(encoder, "drop_idx_", None)
    if drop_idx is not None:
        if len(drop_idx) != n_inputs:
            return None
        counts = [c - (0 if d is None else 1) for c, d in zip(counts, drop_idx, strict=True)]
    if any(c < 0 for c in counts):
        return None
    return counts


def _linear_exact_pipeline(
    model: CovenantModel, X: pd.DataFrame, background: pd.DataFrame
) -> pd.DataFrame | None:
    """Exact logit-space contributions through a one-hot logistic pipeline,
    or None when the wrapper is not exactly decomposable.

    In logit space the model is exactly additive over transformed columns:
    logit(x) = b0 + sum_j beta_j * z_j(x), and every accepted block maps
    each transformed column z_j to exactly one raw feature (StandardScaler
    is per-column affine; OneHotEncoder expands one raw categorical into
    indicator columns). So the logit is additive over raw features, and the
    interventional Shapley value of raw feature f for row i against the
    background is exactly

        sum over f's transformed columns j of beta_j * (z_ij - mean_bg_j),

    where z is the transformed matrix and mean_bg the background's
    transformed mean. X and the background are transformed once through the
    pipeline's own fitted pre-steps, so ``drop=``, ``handle_unknown=``,
    infrequent-category handling, a trailing StandardScaler between the
    ColumnTransformer and the regression, and transformer weights are all
    honoured by construction (a trailing scaler just means the betas apply
    to the scaled z the pre-steps already produce). This is the same value
    the permutation path estimates, computed exactly and in **logit space**
    — the permutation path samples **probability space**; the docstring of
    :func:`explain` states why rankings agree. Sparse encoder output stays
    sparse: the per-feature sums are a sparse-aware matrix product.
    """
    parts = _pipeline_transform_parts(model, _LINEAR_CT_TRANSFORMS, require_ct=True)
    if parts is None:
        return None
    pre, final, col_to_raw = parts
    if not isinstance(final, LogisticRegression):
        return None
    classes = getattr(final, "classes_", None)
    coef = getattr(final, "coef_", None)
    if classes is None or coef is None or len(classes) != 2:
        return None
    beta = np.asarray(coef, dtype=float)[0]
    if model.bad_class_index == 0:
        beta = -beta
    features = model.feature_names
    try:
        Z = pre.transform(X[features])
        Z_bg = pre.transform(background[features])
    except Exception:
        return None
    n_transformed = len(col_to_raw)
    if beta.shape != (n_transformed,):
        return None
    if Z.shape != (len(X), n_transformed) or Z_bg.shape[1] != n_transformed:
        return None
    mu = np.asarray(Z_bg.mean(axis=0), dtype=float).ravel()
    # weights[j, f] = beta_j iff transformed column j belongs to raw feature
    # f, so Z @ weights sums beta_j * z_ij per raw feature in one product.
    weights = np.zeros((n_transformed, len(features)))
    weights[np.arange(n_transformed), col_to_raw] = beta
    values = np.asarray(Z @ weights) - mu @ weights
    if values.shape != (len(X), len(features)) or not np.isfinite(values).all():
        return None
    return _attribution_frame(values, X, features)


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
    """Interventional TreeExplainer attributions for a bare tree ensemble,
    or None when shap cannot produce them (the caller then falls through to
    permutation)."""
    features = model.feature_names
    try:
        bg = background[features].to_numpy(dtype=float)
        Xv = X[features].to_numpy(dtype=float)
    except Exception:
        return None
    values = _tree_shap_matrix(model.estimator, Xv, bg, model.bad_class_index)
    if values is None:
        return None
    return _attribution_frame(values, X, features)


def _tree_shap_pipeline(
    model: CovenantModel, X: pd.DataFrame, background: pd.DataFrame
) -> pd.DataFrame | None:
    """TreeExplainer attributions through a simple pipeline wrapper, or
    None on any doubt (the caller then falls through to permutation).

    Accepted pre-steps are the all-numeric simple transforms of
    ``_pipeline_transform_parts`` with ``OrdinalEncoder`` also allowed
    inside the ColumnTransformer. X and the background are transformed once
    through the fitted pre-steps, the existing TreeExplainer machinery runs
    on the transformed matrices with the final estimator, and
    transformed-column attributions are summed back to their raw feature
    via the shared column map. Summing encoder-expanded columns is the
    standard grouping convention for interventional SHAP — stated here
    because it is a convention, not the Shapley value of the raw feature as
    a single player; the path name stays ``"tree-shap"``, which is what the
    values are. Unknown transformer types, mapping failures and
    TreeExplainer failures all return None so the record never overstates
    exactness."""
    parts = _pipeline_transform_parts(model, _TREE_CT_TRANSFORMS, require_ct=False)
    if parts is None:
        return None
    pre, final, col_to_raw = parts
    if not _is_supported_tree_classifier(final):
        return None
    features = model.feature_names
    try:
        Z = _dense_matrix(pre.transform(X[features]))
        Z_bg = _dense_matrix(pre.transform(background[features]))
    except Exception:
        return None
    n_transformed = len(col_to_raw)
    if Z.shape != (len(X), n_transformed) or Z_bg.shape[1] != n_transformed:
        return None
    values_transformed = _tree_shap_matrix(final, Z, Z_bg, model.bad_class_index)
    if values_transformed is None:
        return None
    ownership = np.zeros((n_transformed, len(features)))
    ownership[np.arange(n_transformed), col_to_raw] = 1.0
    return _attribution_frame(values_transformed @ ownership, X, features)


def _dense_matrix(matrix: object) -> np.ndarray:
    """A dense float matrix for TreeExplainer, whatever transform emitted."""
    if sparse.issparse(matrix):
        matrix = matrix.toarray()  # type: ignore[attr-defined]
    return np.asarray(matrix, dtype=float)


def _tree_shap_matrix(
    estimator: object, Xv: np.ndarray, bg: np.ndarray, bad_class_index: int
) -> np.ndarray | None:
    """Interventional TreeExplainer values for ``Xv`` (same column space as
    ``bg``), or None when shap cannot produce a finite, well-shaped
    answer. ``model_output="probability"`` first, margin output second."""
    for kwargs in ({"model_output": "probability"}, {}):
        try:
            explainer = shap.TreeExplainer(estimator, data=bg, **kwargs)
            raw = explainer.shap_values(Xv, check_additivity=False)
        except Exception:
            continue
        values = _positive_class_slice(raw, bad_class_index)
        if values is None or values.shape != Xv.shape or not np.isfinite(values).all():
            continue
        return values
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
