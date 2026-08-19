"""Integration tests for the optional-dependency adapters and paths.

Each block skips cleanly when its library is absent, so the core suite
never depends on the heavy extras; CI runs one job with
``covenants[integrations]`` installed to exercise them all.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from covenant.attribution import explain
from covenant.model import CovenantModel, load_model

# ---------------------------------------------------------------------------
# OptBinning scorecard adapter
# ---------------------------------------------------------------------------


def _fit_optbinning_scorecard():
    optbinning = pytest.importorskip("optbinning")
    rng = np.random.default_rng(0)
    n = 1200
    df = pd.DataFrame(
        {
            "income": rng.normal(50, 15, n),
            "dti": rng.uniform(0, 1, n),
            "home": rng.choice(["RENT", "OWN", "MORTGAGE"], n),
        }
    )
    logit = -0.05 * (df["income"] - 50) + 2.5 * (df["dti"] - 0.5) + (df["home"] == "RENT") * 0.6
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).to_numpy().astype(int)
    process = optbinning.BinningProcess(
        variable_names=["income", "dti", "home"], categorical_variables=["home"]
    )
    scorecard = optbinning.Scorecard(
        binning_process=process,
        estimator=LogisticRegression(),
        scaling_method="min_max",
        scaling_method_params={"min": 300, "max": 850},
    )
    scorecard.fit(df, y)
    df["bad"] = y
    return scorecard, df


class TestOptBinningAdapter:
    def test_export_uses_exact_splits(self, tmp_path: Path) -> None:
        scorecard, _ = _fit_optbinning_scorecard()
        from covenant.adapters import export_scorecard_points

        out = export_scorecard_points(scorecard, tmp_path / "points.csv")
        table = pd.read_csv(out)
        assert set(table.columns) == {"feature", "bin_lower", "bin_upper", "value", "points"}

        splits = scorecard.binning_process_.get_binned_variable("income").splits
        income = table[table["feature"] == "income"]
        # full-precision bounds, not the summary table's 2-decimal display
        # strings (pandas' default CSV parser can be 1 ulp off repr's exact
        # round-trip, so compare tightly rather than bit-for-bit)
        exported_uppers = np.sort(income["bin_upper"].dropna().astype(float).to_numpy())
        np.testing.assert_allclose(
            exported_uppers, np.sort([float(s) for s in splits]), rtol=1e-12
        )
        # unbounded outer bins have empty bounds
        assert income["bin_lower"].isna().sum() == 1
        assert income["bin_upper"].isna().sum() == 1

        home = table[table["feature"] == "home"]
        assert set(home["value"]) == {"RENT", "OWN", "MORTGAGE"}
        assert home[["bin_lower", "bin_upper"]].isna().all().all()
        # Special/Missing never exported
        assert "Special" not in set(table.get("value", pd.Series(dtype=str)).dropna())

    def test_reversed_scorecard_negates_points(self, tmp_path: Path) -> None:
        scorecard, _ = _fit_optbinning_scorecard()
        from covenant.adapters import export_scorecard_points

        normal = pd.read_csv(export_scorecard_points(scorecard, tmp_path / "a.csv"))
        reversed_ = pd.read_csv(
            export_scorecard_points(
                scorecard, tmp_path / "b.csv", higher_points_lower_risk=False
            )
        )
        assert np.allclose(normal["points"], -reversed_["points"])

    def test_scorecard_end_to_end_check_passes(self, tmp_path: Path) -> None:
        """The exported table against the very scorecard that produced it:
        declared most_points_lost reasons must agree with measured behaviour."""
        from covenant.adapters import export_scorecard_points
        from covenant.checks.reason_codes import run_reason_code_check

        scorecard, df = _fit_optbinning_scorecard()
        export_scorecard_points(scorecard, tmp_path / "points.csv")
        joblib.dump(scorecard, tmp_path / "model.joblib")
        df.to_csv(tmp_path / "train.csv", index=False)
        (tmp_path / "covenants.yaml").write_text(
            """\
covenant_schema: 1
model_name: optbinning-scorecard
features:
  - {name: income, direction: decreases_risk}
  - {name: dti, direction: increases_risk}
  - {name: home, dtype: categorical}
reason_codes:
  method: most_points_lost
  top_k: 2
  parameters:
    points_table: points.csv
checks:
  reason_codes:
    decision_threshold: 0.5
    max_denied_sample: 40
    background_size: 30
    # a 3-feature binned scorecard ties often near the boundary; the top-1
    # policy is set accordingly while jaccard stays comfortably strict
    min_top1_agreement: 0.60
    min_topk_jaccard: 0.70
"""
        )
        record = run_reason_code_check(
            tmp_path / "model.joblib", tmp_path / "train.csv", tmp_path / "covenants.yaml"
        )
        assert record.passed, record.metrics
        assert record.metrics["topk_jaccard"] > 0.75
        assert record.details["attribution_path"] == "permutation-shap"


# ---------------------------------------------------------------------------
# InterpretML EBM exact path
# ---------------------------------------------------------------------------


class TestEbmExactPath:
    @pytest.fixture(scope="class")
    def ebm_setup(self):
        interpret = pytest.importorskip("interpret.glassbox")
        rng = np.random.default_rng(3)
        n = 900
        df = pd.DataFrame(
            {
                "income": rng.normal(0, 1, n),
                "dti": rng.normal(0, 1, n),
                "home": rng.choice(["RENT", "OWN"], n),
            }
        )
        logit = -1.2 * df["income"] + 1.5 * df["dti"] + (df["home"] == "RENT") * 0.8
        y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).to_numpy().astype(int)
        ebm = interpret.ExplainableBoostingClassifier(random_state=0, interactions=2)
        ebm.fit(df, y)
        model = CovenantModel(ebm, ["income", "dti", "home"])
        return model, df

    def test_path_and_determinism(self, ebm_setup) -> None:
        model, df = ebm_setup
        X, background = df.iloc[:60], df.iloc[100:180]
        a1, p1 = explain(model, X, background, categorical=["home"])
        a2, p2 = explain(model, X, background, categorical=["home"])
        assert p1 == p2 == "ebm-exact"
        assert a1.equals(a2)
        assert np.isfinite(a1.to_numpy()).all()

    def test_ranks_agree_with_permutation(self, ebm_setup) -> None:
        from covenant.attribution import _permutation_shap, top_1

        model, df = ebm_setup
        X, background = df.iloc[:40], df.iloc[100:140]
        exact, path = explain(model, X, background, categorical=["home"])
        assert path == "ebm-exact"
        sampled = _permutation_shap(model, X, background, ["home"], 0, 4)
        agreement = np.mean(
            [a == b for a, b in zip(top_1(exact), top_1(sampled), strict=True)]
        )
        assert agreement >= 0.7


# ---------------------------------------------------------------------------
# xgboost / lightgbm: configured constraints and tree-shap
# ---------------------------------------------------------------------------


def _booster_data():
    rng = np.random.default_rng(5)
    n = 800
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "c"])
    logit = 1.2 * X["a"] - 1.0 * X["b"] + rng.normal(0, 0.5, n)
    y = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X, y


class TestXgboost:
    def test_configured_directions_and_monotonicity(self, tmp_path: Path) -> None:
        xgboost = pytest.importorskip("xgboost")
        from covenant.checks.monotonicity import configured_directions, run_monotonicity_check
        from covenant.schema import Direction

        X, y = _booster_data()
        est = xgboost.XGBClassifier(
            monotone_constraints="(1,-1,0)", n_estimators=40, random_state=0
        )
        est.fit(X, y)
        directions = configured_directions(est, ["a", "b", "c"])
        assert directions == {
            "a": Direction.INCREASES_RISK,
            "b": Direction.DECREASES_RISK,
            "c": Direction.NONE,
        }

        joblib.dump(est, tmp_path / "model.joblib")
        frame = X.assign(bad=y)
        frame.to_csv(tmp_path / "train.csv", index=False)
        (tmp_path / "covenants.yaml").write_text(
            """\
covenant_schema: 1
model_name: xgb-test
features:
  - {name: a, direction: increases_risk}
  - {name: b, direction: decreases_risk}
  - {name: c}
reason_codes: {method: difference_from_mean}
"""
        )
        record = run_monotonicity_check(
            tmp_path / "model.joblib", tmp_path / "train.csv", tmp_path / "covenants.yaml"
        )
        assert record.passed, record.details
        assert record.details["configured_constraints_readable"] is True
        by_feature = {r["feature"]: r for r in record.details["by_feature"]}
        assert by_feature["a"]["configured"] == "increases_risk"

    def test_tree_shap_path(self) -> None:
        xgboost = pytest.importorskip("xgboost")
        X, y = _booster_data()
        est = xgboost.XGBClassifier(n_estimators=40, random_state=0)
        est.fit(X, y)
        model = CovenantModel(est, ["a", "b", "c"])
        frame, path = explain(model, X.iloc[:30], X.iloc[100:160])
        assert path in ("tree-shap", "permutation-shap")
        assert np.isfinite(frame.to_numpy()).all()


class TestLightgbm:
    def test_configured_directions_list_form(self) -> None:
        lightgbm = pytest.importorskip("lightgbm")
        from covenant.checks.monotonicity import configured_directions
        from covenant.schema import Direction

        X, y = _booster_data()
        est = lightgbm.LGBMClassifier(
            monotone_constraints=[1, -1, 0], n_estimators=40, random_state=0, verbose=-1
        )
        est.fit(X, y)
        directions = configured_directions(est, ["a", "b", "c"])
        assert directions == {
            "a": Direction.INCREASES_RISK,
            "b": Direction.DECREASES_RISK,
            "c": Direction.NONE,
        }


# ---------------------------------------------------------------------------
# skops loading
# ---------------------------------------------------------------------------


class TestSkops:
    def test_load_and_score_skops_artefact(self, tmp_path: Path) -> None:
        skops_io = pytest.importorskip("skops.io")
        X, y = _booster_data()
        est = LogisticRegression(max_iter=500).fit(X, y)
        path = tmp_path / "model.skops"
        skops_io.dump(est, path)

        loaded = load_model(path)
        model = CovenantModel(loaded, ["a", "b", "c"])
        original = est.predict_proba(X.iloc[:5])[:, 1]
        assert np.allclose(model.p_bad(X.iloc[:5]), original)

    def test_garbage_skops_file_fails_loudly(self, tmp_path: Path) -> None:
        pytest.importorskip("skops.io")
        path = tmp_path / "model.skops"
        path.write_bytes(b"not a skops file")
        with pytest.raises(ValueError, match="skops artefact"):
            load_model(path)
