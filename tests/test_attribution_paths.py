"""Path selection in covenant.attribution.explain: linear-exact and
tree-shap fast paths against the model-agnostic permutation fallback."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, OrdinalEncoder, StandardScaler

from covenant.attribution import _permutation_shap, explain, measured_attributions, top_1
from covenant.model import CovenantModel

FEATURES = ["income", "dti", "utilization", "delinquencies", "inquiries"]
COEFS = np.array([-1.0, 1.2, 0.9, 0.7, 0.4])


def _numeric_problem(seed: int, n: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    logit = -0.3 + X.to_numpy() @ COEFS + rng.normal(0, 0.6, n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


@pytest.fixture(scope="module")
def numeric_data() -> dict:
    X, y = _numeric_problem(seed=7, n=400)
    return {
        "X": X,
        "y": y,
        "explain": X.head(50),
        "background": X.iloc[300:340].reset_index(drop=True),
    }


@pytest.fixture(scope="module")
def lr_model(numeric_data: dict) -> CovenantModel:
    est = LogisticRegression(max_iter=1000).fit(numeric_data["X"], numeric_data["y"])
    return CovenantModel(est, FEATURES)


@pytest.fixture(scope="module")
def pipeline_model(numeric_data: dict) -> CovenantModel:
    est = Pipeline(
        [("scale", StandardScaler()), ("logreg", LogisticRegression(max_iter=1000))]
    ).fit(numeric_data["X"], numeric_data["y"])
    return CovenantModel(est, FEATURES)


@pytest.fixture(scope="module")
def forest_model(numeric_data: dict) -> CovenantModel:
    est = RandomForestClassifier(n_estimators=40, max_depth=3, random_state=0).fit(
        numeric_data["X"], numeric_data["y"]
    )
    return CovenantModel(est, FEATURES)


CAT_FEATURES = ["income", "dti", "home_ownership"]
OWNERSHIP_EFFECT = {"RENT": 0.7, "MORTGAGE": 0.0, "OWN": -0.5}


@pytest.fixture(scope="module")
def categorical_setup() -> dict:
    rng = np.random.default_rng(11)
    n = 300
    df = pd.DataFrame(
        {
            "income": rng.normal(0, 1, n),
            "dti": rng.normal(0, 1, n),
            "home_ownership": rng.choice(list(OWNERSHIP_EFFECT), size=n),
        }
    )
    logit = (
        -1.1 * df["income"]
        + 1.3 * df["dti"]
        + df["home_ownership"].map(OWNERSHIP_EFFECT)
        + rng.normal(0, 0.6, n)
    )
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    est = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", "passthrough", ["income", "dti"]),
                        ("cat", OneHotEncoder(), ["home_ownership"]),
                    ]
                ),
            ),
            ("logreg", LogisticRegression(max_iter=1000)),
        ]
    ).fit(df[CAT_FEATURES], y)
    # Same data behind a MinMaxScaler block: affine, but deliberately
    # outside the accepted transformer set, so this pipeline exercises the
    # model-agnostic permutation fallback.
    agnostic = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", MinMaxScaler(), ["income", "dti"]),
                        ("cat", OneHotEncoder(), ["home_ownership"]),
                    ]
                ),
            ),
            ("logreg", LogisticRegression(max_iter=1000)),
        ]
    ).fit(df[CAT_FEATURES], y)
    return {
        "model": CovenantModel(est, CAT_FEATURES),
        "agnostic_model": CovenantModel(agnostic, CAT_FEATURES),
        "explain": df.head(15)[CAT_FEATURES],
        "background": df.iloc[200:225][CAT_FEATURES].reset_index(drop=True),
    }


def _top1_agreement(a: pd.DataFrame, b: pd.DataFrame) -> float:
    return float(np.mean([x == y for x, y in zip(top_1(a), top_1(b), strict=True)]))


# ---------------------------------------------------------------- linear-exact


def test_linear_exact_bare_lr(numeric_data: dict, lr_model: CovenantModel) -> None:
    Xe, bg = numeric_data["explain"], numeric_data["background"]
    frame, path = explain(lr_model, Xe, bg)
    assert path == "linear-exact"
    mu = bg.to_numpy(dtype=float).mean(axis=0)
    expected = (Xe.to_numpy(dtype=float) - mu) * lr_model.estimator.coef_[0]
    np.testing.assert_array_equal(frame.to_numpy(), expected)
    assert list(frame.columns) == FEATURES
    assert list(frame.index) == list(Xe.index)


def test_linear_exact_standard_scaler_pipeline(
    numeric_data: dict, pipeline_model: CovenantModel
) -> None:
    Xe, bg = numeric_data["explain"], numeric_data["background"]
    frame, path = explain(pipeline_model, Xe, bg)
    assert path == "linear-exact"
    scaler = pipeline_model.estimator.named_steps["scale"]
    logreg = pipeline_model.estimator.named_steps["logreg"]
    beta_eff = logreg.coef_[0] / scaler.scale_
    mu = bg.to_numpy(dtype=float).mean(axis=0)
    expected = (Xe.to_numpy(dtype=float) - mu) * beta_eff
    np.testing.assert_array_equal(frame.to_numpy(), expected)


def test_linear_exact_top1_agrees_with_permutation(
    numeric_data: dict, lr_model: CovenantModel
) -> None:
    Xe, bg = numeric_data["explain"], numeric_data["background"]
    linear, path = explain(lr_model, Xe, bg)
    assert path == "linear-exact"
    permutation = _permutation_shap(lr_model, Xe, bg, None, 0, 4)
    assert _top1_agreement(linear, permutation) >= 0.80


def test_measured_attributions_wrapper_matches_explain(
    numeric_data: dict, lr_model: CovenantModel
) -> None:
    Xe, bg = numeric_data["explain"], numeric_data["background"]
    wrapped = measured_attributions(lr_model, Xe, bg)
    frame, _ = explain(lr_model, Xe, bg)
    pd.testing.assert_frame_equal(wrapped, frame)


# ------------------------------------------------------------------- tree-shap


def test_tree_path_selected_and_close_to_permutation(
    numeric_data: dict, forest_model: CovenantModel
) -> None:
    Xe = numeric_data["X"].head(40)
    bg = numeric_data["background"]
    frame, path = explain(forest_model, Xe, bg)
    assert path == "tree-shap"
    assert frame.shape == (40, len(FEATURES))
    assert np.isfinite(frame.to_numpy()).all()
    permutation = _permutation_shap(forest_model, Xe, bg, None, 0, 4)
    assert _top1_agreement(frame, permutation) >= 0.70


# ---------------------------------------------------------- permutation-shap


def test_unsupported_transform_routes_to_permutation(categorical_setup: dict) -> None:
    """A MinMaxScaler block is affine but deliberately outside the accepted
    transformer set, so the one-hot pipeline around it takes the
    model-agnostic path. (The eligible passthrough-plus-one-hot pipeline in
    this fixture now takes linear-exact — tested further down.)"""
    frame, path = explain(
        categorical_setup["agnostic_model"],
        categorical_setup["explain"],
        categorical_setup["background"],
        categorical=["home_ownership"],
    )
    assert path == "permutation-shap"
    assert list(frame.columns) == CAT_FEATURES
    assert np.isfinite(frame.to_numpy()).all()


# ---------------------------------------------------------------- determinism


def test_linear_path_deterministic(numeric_data: dict, lr_model: CovenantModel) -> None:
    Xe, bg = numeric_data["explain"], numeric_data["background"]
    frame_a, path_a = explain(lr_model, Xe, bg)
    frame_b, path_b = explain(lr_model, Xe, bg)
    assert path_a == path_b == "linear-exact"
    pd.testing.assert_frame_equal(frame_a, frame_b)


def test_tree_path_deterministic(numeric_data: dict, forest_model: CovenantModel) -> None:
    Xe = numeric_data["X"].head(20)
    bg = numeric_data["background"]
    frame_a, path_a = explain(forest_model, Xe, bg)
    frame_b, path_b = explain(forest_model, Xe, bg)
    assert path_a == path_b == "tree-shap"
    pd.testing.assert_frame_equal(frame_a, frame_b)


def test_permutation_path_deterministic(categorical_setup: dict) -> None:
    args = (
        categorical_setup["agnostic_model"],
        categorical_setup["explain"],
        categorical_setup["background"],
        ["home_ownership"],
    )
    frame_a, path_a = explain(*args)
    frame_b, path_b = explain(*args)
    assert path_a == path_b == "permutation-shap"
    pd.testing.assert_frame_equal(frame_a, frame_b)


# ---------------------------------------------------------------- performance


def _wide_problem(p: int, n: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    cols = [f"f{i:02d}" for i in range(p)]
    X = pd.DataFrame(rng.normal(size=(n, p)), columns=cols)
    beta = rng.normal(size=p)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-(X.to_numpy() @ beta)))).astype(int)
    return X, y


@pytest.mark.slow
def test_linear_exact_forty_features_five_hundred_rows_is_fast() -> None:
    X, y = _wide_problem(p=40, n=2000, seed=3)
    est = Pipeline(
        [("scale", StandardScaler()), ("logreg", LogisticRegression(max_iter=1000))]
    ).fit(X, y)
    model = CovenantModel(est, list(X.columns))
    Xe = X.head(500)
    bg = X.iloc[1000:1100].reset_index(drop=True)
    start = time.perf_counter()
    frame, path = explain(model, Xe, bg)
    elapsed = time.perf_counter() - start
    assert path == "linear-exact"
    assert frame.shape == (500, 40)
    assert elapsed < 10.0


@pytest.mark.slow
def test_permutation_forty_features_completes_in_budget() -> None:
    X, y = _wide_problem(p=40, n=600, seed=5)
    # MinMaxScaler is not an eligible affine step, so this pipeline takes the
    # model-agnostic permutation path.
    est = Pipeline(
        [("scale", MinMaxScaler()), ("logreg", LogisticRegression(max_iter=1000))]
    ).fit(X, y)
    model = CovenantModel(est, list(X.columns))
    Xe = X.head(40)
    bg = X.iloc[500:525].reset_index(drop=True)
    start = time.perf_counter()
    frame, path = explain(model, Xe, bg)
    elapsed = time.perf_counter() - start
    assert path == "permutation-shap"
    assert frame.shape == (40, 40)
    assert np.isfinite(frame.to_numpy()).all()
    assert elapsed < 90.0


# ----------------------------------- exact paths through realistic pipelines

ONE_HOT_FEATURES = ["income", "dti", "utilization", "home_ownership"]
NUMERIC_COLS = ["income", "dti", "utilization"]


def _one_hot_problem(seed: int, n: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "income": rng.normal(0, 1, n),
            "dti": rng.normal(0, 1, n),
            "utilization": rng.normal(0, 1, n),
            "home_ownership": rng.choice(list(OWNERSHIP_EFFECT), size=n),
        }
    )
    logit = (
        -1.4 * df["income"]
        + 1.6 * df["dti"]
        + 1.1 * df["utilization"]
        + df["home_ownership"].map(OWNERSHIP_EFFECT)
        + rng.normal(0, 0.4, n)
    )
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return df, y


@pytest.fixture(scope="module")
def one_hot_setup() -> dict:
    """The realistic scorecard shape: ColumnTransformer of scaled numerics
    plus one-hot categoricals into a LogisticRegression."""
    df, y = _one_hot_problem(seed=23, n=600)
    est = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), NUMERIC_COLS),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), ["home_ownership"]),
                    ]
                ),
            ),
            ("logreg", LogisticRegression(max_iter=2000)),
        ]
    ).fit(df[ONE_HOT_FEATURES], y)
    return {
        "frame": df,
        "y": y,
        "model": CovenantModel(est, ONE_HOT_FEATURES),
        "explain": df.head(40)[ONE_HOT_FEATURES],
        "background": df.iloc[500:540][ONE_HOT_FEATURES].reset_index(drop=True),
    }


@pytest.fixture(scope="module")
def ordinal_tree_setup() -> dict:
    """An all-numeric-after-encoding pipeline into a boosted tree."""
    df, y = _one_hot_problem(seed=31, n=600)
    est = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", "passthrough", NUMERIC_COLS),
                        ("cat", OrdinalEncoder(), ["home_ownership"]),
                    ]
                ),
            ),
            ("hgb", HistGradientBoostingClassifier(random_state=0, max_iter=60)),
        ]
    ).fit(df[ONE_HOT_FEATURES], y)
    return {
        "model": CovenantModel(est, ONE_HOT_FEATURES),
        "explain": df.head(30)[ONE_HOT_FEATURES],
        "background": df.iloc[500:525][ONE_HOT_FEATURES].reset_index(drop=True),
    }


def test_one_hot_logistic_pipeline_is_linear_exact(one_hot_setup: dict) -> None:
    """The realistic scorecard is exactly additive in logit space, so its
    attributions must be the closed form, not a sampled estimate: the
    contribution of raw feature f is the sum over f's transformed columns j
    of beta_j * (z_j - mean(background z_j)). Verified against a transformed
    matrix built by hand from the fitted scaler and encoder."""
    model = one_hot_setup["model"]
    Xe, bg = one_hot_setup["explain"], one_hot_setup["background"]
    frame, path = explain(model, Xe, bg, categorical=["home_ownership"])
    assert path == "linear-exact"
    assert list(frame.columns) == ONE_HOT_FEATURES
    assert list(frame.index) == list(Xe.index)

    prep = model.estimator.named_steps["prep"]
    scaler = prep.named_transformers_["num"]
    encoder = prep.named_transformers_["cat"]

    def transform(d: pd.DataFrame) -> np.ndarray:
        return np.hstack(
            [
                scaler.transform(d[NUMERIC_COLS]),
                encoder.transform(d[["home_ownership"]]).toarray(),
            ]
        )

    z, z_bg = transform(Xe), transform(bg)
    beta = model.estimator.named_steps["logreg"].coef_[0]
    per_column = beta * (z - z_bg.mean(axis=0))
    expected = np.column_stack(
        [per_column[:, 0], per_column[:, 1], per_column[:, 2], per_column[:, 3:].sum(axis=1)]
    )
    np.testing.assert_allclose(frame.to_numpy(), expected, rtol=0, atol=1e-10)


def test_one_hot_linear_exact_top1_matches_permutation(one_hot_setup: dict) -> None:
    """The exact path computes the same value the permutation path
    estimates, so per-row top-1 reasons must agree on nearly every row."""
    model = one_hot_setup["model"]
    Xe, bg = one_hot_setup["explain"], one_hot_setup["background"]
    frame, path = explain(model, Xe, bg, categorical=["home_ownership"])
    assert path == "linear-exact"
    permutation = _permutation_shap(model, Xe, bg, ["home_ownership"], 0, 4)
    assert _top1_agreement(frame, permutation) >= 0.90


def test_conftest_style_categorical_pipeline_now_linear_exact(categorical_setup: dict) -> None:
    """The passthrough-plus-OneHotEncoder logistic pipeline (the shape of
    tests/conftest.py's fitted_categorical fixture, built inline in this
    module's fixture) previously fell back to permutation SHAP; it is
    exactly decomposable and now takes linear-exact."""
    frame, path = explain(
        categorical_setup["model"],
        categorical_setup["explain"],
        categorical_setup["background"],
        categorical=["home_ownership"],
    )
    assert path == "linear-exact"
    assert list(frame.columns) == CAT_FEATURES
    assert np.isfinite(frame.to_numpy()).all()


def test_ordinal_hgb_pipeline_gets_tree_shap(ordinal_tree_setup: dict) -> None:
    model = ordinal_tree_setup["model"]
    Xe, bg = ordinal_tree_setup["explain"], ordinal_tree_setup["background"]
    frame, path = explain(model, Xe, bg, categorical=["home_ownership"])
    assert path == "tree-shap"
    assert frame.shape == (len(Xe), len(ONE_HOT_FEATURES))
    assert np.isfinite(frame.to_numpy()).all()
    permutation = _permutation_shap(model, Xe, bg, ["home_ownership"], 0, 4)
    assert _top1_agreement(frame, permutation) >= 0.70


def test_pca_pipeline_falls_back_to_permutation(numeric_data: dict) -> None:
    """PCA mixes raw features, so no exact per-raw-feature decomposition
    exists; claiming one would overstate the record."""
    est = Pipeline(
        [
            ("pca", PCA(n_components=3, random_state=0)),
            ("logreg", LogisticRegression(max_iter=1000)),
        ]
    ).fit(numeric_data["X"], numeric_data["y"])
    model = CovenantModel(est, FEATURES)
    frame, path = explain(model, numeric_data["explain"].head(10), numeric_data["background"])
    assert path == "permutation-shap"
    assert frame.shape == (10, len(FEATURES))
    assert np.isfinite(frame.to_numpy()).all()


def test_one_hot_booster_pipeline_finite(one_hot_setup: dict) -> None:
    """One-hot columns into a booster: whichever path serves it (tree-shap
    with expanded columns summed back to raw features, or the permutation
    fallback), the frame must be well-shaped and finite — the record may
    never carry an overstated or broken answer."""
    df, y = one_hot_setup["frame"], one_hot_setup["y"]
    est = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), NUMERIC_COLS),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), ["home_ownership"]),
                    ]
                ),
            ),
            ("hgb", HistGradientBoostingClassifier(random_state=0, max_iter=40)),
        ]
    ).fit(df[ONE_HOT_FEATURES], y)
    model = CovenantModel(est, ONE_HOT_FEATURES)
    Xe = one_hot_setup["explain"].head(20)
    frame, path = explain(model, Xe, one_hot_setup["background"], categorical=["home_ownership"])
    assert path in {"tree-shap", "permutation-shap"}
    assert frame.shape == (20, len(ONE_HOT_FEATURES))
    assert list(frame.columns) == ONE_HOT_FEATURES
    assert np.isfinite(frame.to_numpy()).all()


def test_one_hot_linear_path_deterministic(one_hot_setup: dict) -> None:
    args = (
        one_hot_setup["model"],
        one_hot_setup["explain"],
        one_hot_setup["background"],
        ["home_ownership"],
    )
    frame_a, path_a = explain(*args)
    frame_b, path_b = explain(*args)
    assert path_a == path_b == "linear-exact"
    pd.testing.assert_frame_equal(frame_a, frame_b)


def test_ordinal_tree_path_deterministic(ordinal_tree_setup: dict) -> None:
    args = (
        ordinal_tree_setup["model"],
        ordinal_tree_setup["explain"],
        ordinal_tree_setup["background"],
        ["home_ownership"],
    )
    frame_a, path_a = explain(*args)
    frame_b, path_b = explain(*args)
    assert path_a == path_b == "tree-shap"
    pd.testing.assert_frame_equal(frame_a, frame_b)
