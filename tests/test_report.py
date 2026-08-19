from __future__ import annotations

import hashlib
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from covenant.checks.reason_codes import CheckSetupError
from covenant.report import build_report

REPORT_FEATURES = ["income", "dti", "utilization", "delinquencies"]

REPORT_COVENANTS_YAML = """\
covenant_schema: 1
model_name: report-test
features:
  - {name: income, direction: decreases_risk}
  - {name: dti}
  - {name: utilization}
  - {name: delinquencies}
reason_codes:
  method: univariate
  top_k: 3
report:
  target_column: bad
  time_column: month
  n_bootstrap: 60
"""

NO_TARGET_COVENANTS_YAML = REPORT_COVENANTS_YAML.replace("  target_column: bad\n", "")


@pytest.fixture(scope="module")
def report_assets(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """A fitted logistic scorecard, a snapshot with a monthly time column,
    a shifted holdout, and covenants configured for the report."""
    root = tmp_path_factory.mktemp("report_fixture")
    rng = np.random.default_rng(3)
    n = 500
    df = pd.DataFrame(
        rng.normal(size=(n, len(REPORT_FEATURES))), columns=REPORT_FEATURES
    )
    logit = (
        -0.5
        - 1.0 * df["income"]
        + 1.1 * df["dti"]
        + 0.8 * df["utilization"]
        + 0.5 * df["delinquencies"]
        + rng.normal(0, 0.7, n)
    )
    df["bad"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    df["month"] = rng.integers(1, 13, size=n)

    model = LogisticRegression(max_iter=1000).fit(df[REPORT_FEATURES], df["bad"])
    model_path = root / "model.joblib"
    joblib.dump(model, model_path)
    data_path = root / "train.csv"
    df.to_csv(data_path, index=False)

    shifted = df.copy()
    shifted["utilization"] = shifted["utilization"] + 0.8
    holdout_path = root / "holdout.csv"
    shifted.to_csv(holdout_path, index=False)

    covenants_path = root / "covenants.yaml"
    covenants_path.write_text(REPORT_COVENANTS_YAML)
    no_target_path = root / "covenants_no_target.yaml"
    no_target_path.write_text(NO_TARGET_COVENANTS_YAML)

    return {
        "model": model_path,
        "data": data_path,
        "holdout": holdout_path,
        "covenants": covenants_path,
        "covenants_no_target": no_target_path,
    }


@pytest.fixture(scope="module")
def built(report_assets: dict, tmp_path_factory: pytest.TempPathFactory) -> Path:
    out = tmp_path_factory.mktemp("report_out") / "run"
    return build_report(
        report_assets["model"], report_assets["data"], report_assets["covenants"], out
    )


def test_report_renders_all_sections(built: Path) -> None:
    assert built.name == "report.md"
    assert built.exists()
    text = built.read_text()
    assert "AUC" in text
    assert "report_sha256:" in text
    assert "Maps to:" in text
    assert "## 7. Challenger" in text
    assert "PSI vs slice 1" in text  # the drift table rendered
    assert "Verdict:" in text  # the monotonicity section rendered
    figures = built.parent / "figures"
    for name in ("roc.png", "calibration.png", "challenger.png"):
        assert (figures / name).exists(), name


def test_byte_identical_rerender_and_hash_replay(
    built: Path, report_assets: dict, tmp_path: Path
) -> None:
    """THE property: same inputs, byte-identical report.md and PNGs, and the
    embedded report_sha256 replays from the body with the hash blanked."""
    again = build_report(
        report_assets["model"],
        report_assets["data"],
        report_assets["covenants"],
        tmp_path / "again",
    )
    first_bytes = built.read_bytes()
    assert first_bytes == again.read_bytes()

    first_figures = sorted((built.parent / "figures").glob("*.png"))
    again_figures = sorted((again.parent / "figures").glob("*.png"))
    assert [f.name for f in first_figures] == [f.name for f in again_figures]
    for a, b in zip(first_figures, again_figures, strict=True):
        assert a.read_bytes() == b.read_bytes(), a.name

    text = first_bytes.decode("utf-8")
    match = re.search(r"report_sha256: ([0-9a-f]{64})", text)
    assert match is not None
    embedded = match.group(1)
    blanked = text.replace(
        f"report_sha256: {embedded}", "report_sha256: " + "0" * 64, 1
    )
    assert hashlib.sha256(blanked.encode("utf-8")).hexdigest() == embedded


def test_holdout_renders_positive_psi(report_assets: dict, tmp_path: Path) -> None:
    path = build_report(
        report_assets["model"],
        report_assets["data"],
        report_assets["covenants"],
        tmp_path / "out",
        holdout_path=report_assets["holdout"],
    )
    text = path.read_text()
    match = re.search(r"Score PSI \(train -> holdout\): ([0-9.]+)", text)
    assert match is not None
    assert float(match.group(1)) > 0.0
    assert "| utilization |" in text  # the per-feature CSI table rendered


def test_missing_target_column_is_setup_error(report_assets: dict, tmp_path: Path) -> None:
    with pytest.raises(CheckSetupError, match="report.target_column"):
        build_report(
            report_assets["model"],
            report_assets["data"],
            report_assets["covenants_no_target"],
            tmp_path / "out",
        )


def test_report_bytes_independent_of_path_spelling(
    report_assets: dict, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The rendered bytes must not depend on how the input paths were spelled
    (review finding: the footer used to embed the paths verbatim)."""
    absolute = build_report(
        report_assets["model"],
        report_assets["data"],
        report_assets["covenants"],
        tmp_path / "abs",
    )
    monkeypatch.chdir(report_assets["model"].parent)
    relative = build_report(
        report_assets["model"].name,
        report_assets["data"].name,
        report_assets["covenants"].name,
        tmp_path / "rel",
    )
    assert absolute.read_bytes() == relative.read_bytes()
