from __future__ import annotations

from pathlib import Path

import joblib
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier
from typer.testing import CliRunner

from covenant.cli import app
from covenant.compare import compare_models

runner = CliRunner()


@pytest.fixture(scope="module")
def challenger(tmp_path_factory: pytest.TempPathFactory, fitted: dict) -> Path:
    from conftest import FEATURES

    frame = fitted["frame"]
    gbm = HistGradientBoostingClassifier(random_state=0)
    gbm.fit(frame[FEATURES], frame["bad"])
    path = tmp_path_factory.mktemp("challenger") / "gbm.joblib"
    joblib.dump(gbm, path)
    return path


def test_compare_record_shape_and_determinism(fitted: dict, challenger: Path) -> None:
    overrides = {"target_column": "bad", "n_bootstrap": 60}
    a = compare_models(fitted["model"], challenger, fitted["data"], fitted["covenants"], overrides)
    b = compare_models(fitted["model"], challenger, fitted["data"], fitted["covenants"], overrides)
    assert a["record_sha256"] == b["record_sha256"]
    for name in ("roc_auc", "ks", "brier", "ece"):
        diff, lo, hi = a["deltas"][name]["a_minus_b"]
        assert lo <= diff <= hi
        point, plo, phi = a["metrics"]["model_a"][name]
        assert plo <= point <= phi
    assert "paired bootstrap" in a["note"]


def test_compare_missing_target_is_setup_error(fitted: dict, challenger: Path) -> None:
    from covenant.checks.reason_codes import CheckSetupError

    with pytest.raises(CheckSetupError, match="target_column"):
        compare_models(fitted["model"], challenger, fitted["data"], fitted["covenants"])


def test_compare_cli_writes_record(fitted: dict, challenger: Path, tmp_path: Path) -> None:
    store = tmp_path / ".covenant"
    result = runner.invoke(
        app,
        [
            "compare",
            str(fitted["model"]),
            str(challenger),
            str(fitted["data"]),
            "--covenants",
            str(fitted["covenants"]),
            "--store",
            str(store),
            "--target",
            "bad",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "A-B" in result.output and "roc_auc" in result.output
    records = list((store / "compare").glob("compare-*.yaml"))
    assert len(records) == 1
