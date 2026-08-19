"""Path selection in covenant.attribution.explain: linear-exact and
tree-shap fast paths against the model-agnostic permutation fallback."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, StandardScaler

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
    return {
        "model": CovenantModel(est, CAT_FEATURES),
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


def test_categorical_pipeline_routes_to_permutation(categorical_setup: dict) -> None:
    frame, path = explain(
        categorical_setup["model"],
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
        categorical_setup["model"],
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
