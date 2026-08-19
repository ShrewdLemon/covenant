"""Pure table logic for the declared reason-code methods — no model, no SHAP.

Each method reads the production artefact that actually phrases the
adverse-action letters (a points table, a bins table, an attributions
export). The tests hand-build tiny artefacts with known answers and assert
exact attributions and top-k sets, plus the error paths that mean the user
set the artefact up wrong."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from covenant.declared import (
    DeclaredMethodError,
    declared_attributions,
    declared_reason_sets,
)
from covenant.schema import ReasonCodeMethod, ReasonCodePolicy

POINTS_CSV = """\
feature,bin_lower,bin_upper,value,points
income,,40,,10
income,40,inf,,30
dti,-inf,20,,25
dti,20,,,5
home,,,RENT,5
home,,,MORTGAGE,15
home,,,OWN,20
"""


def _points_policy(tmp_path: Path, csv: str = POINTS_CSV, top_k: int = 2) -> ReasonCodePolicy:
    (tmp_path / "points.csv").write_text(csv)
    return ReasonCodePolicy(
        method=ReasonCodeMethod.MOST_POINTS_LOST,
        top_k=top_k,
        parameters={"points_table": "points.csv"},
    )


def _mixed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "income": [55.0, 10.0],
            "dti": [25.0, 5.0],
            "home": ["RENT", "MORTGAGE"],
        }
    )


class TestMostPointsLost:
    def test_exact_attributions_incl_categorical(self, tmp_path: Path) -> None:
        policy = _points_policy(tmp_path)
        atts = declared_attributions(policy, _mixed_frame(), tmp_path)
        expected = pd.DataFrame(
            {
                # max points: income 30, dti 25, home 20; lost = max - bin
                "income": [0.0, 20.0],
                "dti": [20.0, 0.0],
                "home": [15.0, 5.0],
            }
        )
        pd.testing.assert_frame_equal(atts, expected)

    def test_top_k_and_top_1(self, tmp_path: Path) -> None:
        policy = _points_policy(tmp_path, top_k=2)
        sets, top1 = declared_reason_sets(policy, _mixed_frame(), tmp_path)
        assert top1 == ["dti", "income"]
        assert sets == [frozenset({"dti", "home"}), frozenset({"income", "home"})]

    def test_lower_bound_is_inclusive_upper_exclusive(self, tmp_path: Path) -> None:
        policy = _points_policy(tmp_path)
        X = pd.DataFrame({"income": [40.0, 39.999]})
        atts = declared_attributions(policy, X, tmp_path)
        # 40 falls in [40, inf) -> 30 points -> 0 lost; just below -> 20 lost
        assert atts["income"].tolist() == [0.0, 20.0]

    def test_unbounded_bins_empty_cells(self, tmp_path: Path) -> None:
        csv = "feature,bin_lower,bin_upper,points\nscore,,40,10\nscore,40,,30\n"
        policy = _points_policy(tmp_path, csv=csv)
        X = pd.DataFrame({"score": [-1e12, 40.0, 1e12]})
        atts = declared_attributions(policy, X, tmp_path)
        assert atts["score"].tolist() == [20.0, 0.0, 0.0]

    def test_unbounded_bins_explicit_inf(self, tmp_path: Path) -> None:
        csv = "feature,bin_lower,bin_upper,points\nscore,-inf,40,10\nscore,40,inf,30\n"
        policy = _points_policy(tmp_path, csv=csv)
        X = pd.DataFrame({"score": [-1e12, 1e12]})
        atts = declared_attributions(policy, X, tmp_path)
        assert atts["score"].tolist() == [20.0, 0.0]

    def test_value_in_no_bin_names_feature_and_value(self, tmp_path: Path) -> None:
        csv = "feature,bin_lower,bin_upper,points\nincome,0,50,10\nincome,50,70,30\n"
        policy = _points_policy(tmp_path, csv=csv)
        X = pd.DataFrame({"income": [60.0, 75.0]})
        with pytest.raises(DeclaredMethodError, match="falls in no declared bin") as exc:
            declared_attributions(policy, X, tmp_path)
        assert "'income'" in str(exc.value)
        assert "75.0" in str(exc.value)

    def test_categorical_value_in_no_bin(self, tmp_path: Path) -> None:
        policy = _points_policy(tmp_path)
        X = _mixed_frame()
        X.loc[0, "home"] = "SUBLET"
        with pytest.raises(DeclaredMethodError, match="falls in no declared bin") as exc:
            declared_attributions(policy, X, tmp_path)
        assert "'home'" in str(exc.value)
        assert "'SUBLET'" in str(exc.value)

    def test_feature_absent_from_table(self, tmp_path: Path) -> None:
        policy = _points_policy(tmp_path)
        X = _mixed_frame()
        X["new_feature"] = 1.0
        with pytest.raises(DeclaredMethodError, match="no bins for feature 'new_feature'"):
            declared_attributions(policy, X, tmp_path)

    def test_malformed_table_no_points_column(self, tmp_path: Path) -> None:
        csv = "feature,bin_lower,bin_upper\nincome,,40\nincome,40,\n"
        policy = _points_policy(tmp_path, csv=csv)
        with pytest.raises(DeclaredMethodError, match="needs columns"):
            declared_attributions(policy, pd.DataFrame({"income": [1.0]}), tmp_path)

    def test_malformed_table_no_bin_columns(self, tmp_path: Path) -> None:
        csv = "feature,points\nincome,10\n"
        policy = _points_policy(tmp_path, csv=csv)
        with pytest.raises(DeclaredMethodError, match="no bin columns"):
            declared_attributions(policy, pd.DataFrame({"income": [1.0]}), tmp_path)

    def test_missing_parameter(self, tmp_path: Path) -> None:
        policy = ReasonCodePolicy(method=ReasonCodeMethod.MOST_POINTS_LOST)
        with pytest.raises(DeclaredMethodError, match="parameters.points_table"):
            declared_attributions(policy, pd.DataFrame({"income": [1.0]}), tmp_path)


UNIVARIATE_CSV = """\
feature,bin_lower,bin_upper,value
util,,0.3,0.02
util,0.3,0.6,0.05
util,0.6,,0.20
dti,,20,0.04
dti,20,,0.10
"""


def _univariate_policy(tmp_path: Path, csv: str = UNIVARIATE_CSV) -> ReasonCodePolicy:
    (tmp_path / "bins.csv").write_text(csv)
    return ReasonCodePolicy(
        method=ReasonCodeMethod.UNIVARIATE,
        top_k=1,
        parameters={"bins_table": "bins.csv"},
    )


class TestUnivariate:
    def test_exact_attributions_vs_table_mean(self, tmp_path: Path) -> None:
        policy = _univariate_policy(tmp_path)
        X = pd.DataFrame({"util": [0.7, 0.1], "dti": [25.0, 10.0]})
        atts = declared_attributions(policy, X, tmp_path)
        # util bin mean = (0.02+0.05+0.20)/3 = 0.09; dti bin mean = 0.07
        expected = pd.DataFrame(
            {"util": [0.20 - 0.09, 0.02 - 0.09], "dti": [0.10 - 0.07, 0.04 - 0.07]}
        )
        pd.testing.assert_frame_equal(atts, expected)

    def test_top_1(self, tmp_path: Path) -> None:
        policy = _univariate_policy(tmp_path)
        X = pd.DataFrame({"util": [0.7, 0.1], "dti": [25.0, 10.0]})
        sets, top1 = declared_reason_sets(policy, X, tmp_path)
        assert top1 == ["util", "dti"]
        assert sets == [frozenset({"util"}), frozenset({"dti"})]

    def test_categorical_bins_use_category_column(self, tmp_path: Path) -> None:
        csv = "feature,category,value\nhome,RENT,0.12\nhome,OWN,0.03\n"
        policy = _univariate_policy(tmp_path, csv=csv)
        X = pd.DataFrame({"home": ["RENT", "OWN"]})
        atts = declared_attributions(policy, X, tmp_path)
        assert atts["home"].tolist() == pytest.approx([0.12 - 0.075, 0.03 - 0.075])

    def test_malformed_table_no_value_column(self, tmp_path: Path) -> None:
        csv = "feature,bin_lower,bin_upper\nutil,,0.5\nutil,0.5,\n"
        policy = _univariate_policy(tmp_path, csv=csv)
        with pytest.raises(DeclaredMethodError, match="needs columns"):
            declared_attributions(policy, pd.DataFrame({"util": [0.1]}), tmp_path)


SHAPLEY_CSV = """\
app_id,income,dti
A1,0.5,0.1
A2,-0.2,0.4
A3,0.0,0.0
"""


def _shapley_policy(tmp_path: Path, csv: str = SHAPLEY_CSV, top_k: int = 1) -> ReasonCodePolicy:
    (tmp_path / "attributions.csv").write_text(csv)
    return ReasonCodePolicy(
        method=ReasonCodeMethod.SHAPLEY,
        top_k=top_k,
        parameters={"attributions_file": "attributions.csv"},
    )


class TestShapley:
    def test_joins_on_id_not_position(self, tmp_path: Path) -> None:
        policy = _shapley_policy(tmp_path)
        X = pd.DataFrame({"income": [1.0, 2.0], "dti": [3.0, 4.0]})
        ids = pd.Series(["A2", "A1"])  # reversed relative to file order
        sets, top1 = declared_reason_sets(policy, X, tmp_path, ids=ids, id_column="app_id")
        assert top1 == ["dti", "income"]
        assert sets == [frozenset({"dti"}), frozenset({"income"})]

    def test_requires_id_column(self, tmp_path: Path) -> None:
        policy = _shapley_policy(tmp_path)
        X = pd.DataFrame({"income": [1.0], "dti": [2.0]})
        with pytest.raises(DeclaredMethodError, match="checks.reason_codes.id_column"):
            declared_reason_sets(policy, X, tmp_path)

    def test_not_reachable_via_declared_attributions(self, tmp_path: Path) -> None:
        policy = _shapley_policy(tmp_path)
        with pytest.raises(DeclaredMethodError, match="declared_reason_sets"):
            declared_attributions(policy, pd.DataFrame({"income": [1.0]}), tmp_path)

    def test_duplicate_ids(self, tmp_path: Path) -> None:
        csv = "app_id,income,dti\nA1,0.5,0.1\nA1,0.2,0.3\n"
        policy = _shapley_policy(tmp_path, csv=csv)
        X = pd.DataFrame({"income": [1.0], "dti": [2.0]})
        ids = pd.Series(["A1"])
        with pytest.raises(DeclaredMethodError, match="duplicate ids"):
            declared_reason_sets(policy, X, tmp_path, ids=ids, id_column="app_id")

    def test_missing_ids(self, tmp_path: Path) -> None:
        policy = _shapley_policy(tmp_path)
        X = pd.DataFrame({"income": [1.0, 2.0], "dti": [3.0, 4.0]})
        ids = pd.Series(["A1", "A9"])
        with pytest.raises(DeclaredMethodError, match="no attributions for 1") as exc:
            declared_reason_sets(policy, X, tmp_path, ids=ids, id_column="app_id")
        assert "A9" in str(exc.value)

    def test_missing_feature_column(self, tmp_path: Path) -> None:
        csv = "app_id,income\nA1,0.5\n"
        policy = _shapley_policy(tmp_path, csv=csv)
        X = pd.DataFrame({"income": [1.0], "dti": [2.0]})
        ids = pd.Series(["A1"])
        with pytest.raises(DeclaredMethodError, match="lacks attribution columns") as exc:
            declared_reason_sets(policy, X, tmp_path, ids=ids, id_column="app_id")
        assert "dti" in str(exc.value)

    def test_missing_id_column_in_file(self, tmp_path: Path) -> None:
        csv = "applicant,income,dti\nA1,0.5,0.1\n"
        policy = _shapley_policy(tmp_path, csv=csv)
        X = pd.DataFrame({"income": [1.0], "dti": [2.0]})
        ids = pd.Series(["A1"])
        with pytest.raises(DeclaredMethodError, match="lacks the id column 'app_id'"):
            declared_reason_sets(policy, X, tmp_path, ids=ids, id_column="app_id")

    def test_missing_parameter(self, tmp_path: Path) -> None:
        policy = ReasonCodePolicy(method=ReasonCodeMethod.SHAPLEY)
        X = pd.DataFrame({"income": [1.0]})
        ids = pd.Series(["A1"])
        with pytest.raises(DeclaredMethodError, match="parameters.attributions_file"):
            declared_reason_sets(policy, X, tmp_path, ids=ids, id_column="app_id")
