from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from covenant.checks.monotonicity import configured_directions, run_monotonicity_check
from covenant.schema import Direction
from covenant.store import Store


def test_correct_directions_pass(fitted: dict, store_dir: Path) -> None:
    """A logistic model is monotone by construction; correctly declared
    directions must show a zero violation rate."""
    record = run_monotonicity_check(fitted["model"], fitted["data"], fitted["covenants"])
    assert record.passed, record.details
    assert record.metrics["worst_violation_rate"] == 0.0
    assert record.details["configured_constraints_readable"] is False
    assert len(record.details["by_feature"]) == 5

    path = Path(record.write(Store(store_dir)))
    assert path.exists()


def test_flipped_direction_breaches(fitted: dict, tmp_path: Path) -> None:
    flipped = tmp_path / "covenants_flipped.yaml"
    flipped.write_text(
        fitted["covenants"].read_text().replace(
            "{name: dti, direction: increases_risk}",
            "{name: dti, direction: decreases_risk}",
        )
    )
    record = run_monotonicity_check(fitted["model"], fitted["data"], flipped)
    assert not record.passed
    dti = next(r for r in record.details["by_feature"] if r["feature"] == "dti")
    assert max(dti["pair_violation_rate"], dti["ice_violation_rate"]) > 0.9
    assert dti["worst_pair"] is not None


def test_no_declared_directions_is_setup_error(fitted: dict, tmp_path: Path) -> None:
    from covenant.checks.reason_codes import CheckSetupError

    text = fitted["covenants"].read_text()
    for direction in ("increases_risk", "decreases_risk"):
        text = text.replace(f"direction: {direction}", "direction: none")
    undirected = tmp_path / "covenants_none.yaml"
    undirected.write_text(text)
    with pytest.raises(CheckSetupError, match="no feature declares"):
        run_monotonicity_check(fitted["model"], fitted["data"], undirected)


def test_configured_mismatch_breaches(fitted: dict, tmp_path: Path) -> None:
    """A booster constrained opposite to the covenant fails on the
    declared-vs-configured comparison, whatever the empirical rates say."""
    from conftest import FEATURES

    frame = fitted["frame"]
    constraints = {f: 0 for f in FEATURES}
    constraints["income"] = 1  # covenant says income decreases risk
    gbm = HistGradientBoostingClassifier(random_state=0, monotonic_cst=constraints)
    gbm.fit(frame[FEATURES], frame["bad"])
    model_path = tmp_path / "gbm.joblib"
    joblib.dump(gbm, model_path)

    record = run_monotonicity_check(model_path, fitted["data"], fitted["covenants"])
    assert not record.passed
    assert record.details["configured_constraints_readable"] is True
    mismatches = record.details["configured_mismatch"]
    assert {m["feature"] for m in mismatches} == {"income"}
    assert mismatches[0]["configured"] == "increases_risk"


def test_check_is_deterministic(fitted: dict) -> None:
    a = run_monotonicity_check(fitted["model"], fitted["data"], fitted["covenants"])
    b = run_monotonicity_check(fitted["model"], fitted["data"], fitted["covenants"])
    assert a.record_sha256 == b.record_sha256


class _FakeBooster:
    def __init__(self, constraints) -> None:
        self._constraints = constraints
        self.feature_names_in_ = np.array(["a", "b", "c"])

    def get_params(self, deep: bool = False) -> dict:
        return {"monotone_constraints": self._constraints}


@pytest.mark.parametrize(
    "raw",
    ["(1,-1,0)", [1, -1, 0], {"a": 1, "b": -1, "c": 0}],
    ids=["xgboost-string", "sequence", "dict"],
)
def test_configured_directions_forms(raw) -> None:
    result = configured_directions(_FakeBooster(raw), ["a", "b", "c"])
    assert result == {
        "a": Direction.INCREASES_RISK,
        "b": Direction.DECREASES_RISK,
        "c": Direction.NONE,
    }


def test_configured_directions_flip_for_bad_class_zero() -> None:
    result = configured_directions(_FakeBooster("(1,-1,0)"), ["a", "b", "c"], bad_class_index=0)
    assert result["a"] is Direction.DECREASES_RISK
    assert result["b"] is Direction.INCREASES_RISK


def test_configured_directions_absent() -> None:
    assert configured_directions(_FakeBooster(None), ["a", "b", "c"]) is None


def test_configured_directions_refuse_pipeline_positional_guess() -> None:
    """A bare constraint sequence on a pipeline's final step must not be
    aligned to the covenant's feature order (confirmed misalignment bug)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder

    rng = np.random.default_rng(0)
    import pandas as pd

    frame = pd.DataFrame(
        {
            "cat": rng.choice(["a", "b"], 200),
            "num1": rng.normal(size=200),
            "num2": rng.normal(size=200),
        }
    )
    y = (frame["num1"] + rng.normal(0, 0.5, 200) > 0).astype(int)
    pipe = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", "passthrough", ["num1", "num2"]),
                        ("cat", OrdinalEncoder(), ["cat"]),
                    ]
                ),
            ),
            (
                "gbm",
                HistGradientBoostingClassifier(
                    monotonic_cst=[1, -1, 0], random_state=0
                ),
            ),
        ]
    )
    pipe.fit(frame, y)
    # Covenant declares features as [cat, num1, num2]; the transformed order
    # is [num1, num2, cat]. Positional alignment would label 'cat' as
    # increases_risk. The only safe answer is None (unreadable).
    assert configured_directions(pipe, ["cat", "num1", "num2"]) is None
