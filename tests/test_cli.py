from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from covenant.cli import app

runner = CliRunner()


def combined_output(result) -> str:
    """stdout plus stderr, across click versions that do or don't mix them."""
    try:
        return result.output + result.stderr
    except ValueError:  # stderr not separately captured: already in output
        return result.output


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "distribution: covenants" in result.output


def test_init_writes_templates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0
    assert (tmp_path / "covenants.yaml").exists()
    assert (tmp_path / "governance.yaml").exists()

    again = runner.invoke(app, ["init"])
    assert again.exit_code == 0
    assert "left untouched" in again.output


def test_report_without_target_column_exits_2(fitted: dict, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        [
            "report",
            str(fitted["model"]),
            str(fitted["data"]),
            "--covenants",
            str(fitted["covenants"]),
            "--out",
            str(tmp_path / "report"),
        ],
    )
    assert result.exit_code == 2
    assert "target_column" in combined_output(result)


def test_report_cli_renders(fitted: dict, tmp_path: Path) -> None:
    out = tmp_path / "report"
    result = runner.invoke(
        app,
        [
            "report",
            str(fitted["model"]),
            str(fitted["data"]),
            "--covenants",
            str(fitted["covenants"]),
            "--out",
            str(out),
            "--target",
            "bad",
        ],
    )
    assert result.exit_code == 0, combined_output(result)
    assert (out / "report.md").exists()
    body = (out / "report.md").read_text()
    assert "report_sha256:" in body
    assert "AUC" in body


def test_register_and_diff_with_prefixes(fitted: dict, tmp_path: Path) -> None:
    store = str(tmp_path / ".covenant")
    base = [
        str(fitted["model"]),
        str(fitted["data"]),
        "--governance",
        str(fitted["governance"]),
        "--store",
        store,
    ]
    a = runner.invoke(app, ["register", *base, "--covenants", str(fitted["covenants"])])
    assert a.exit_code == 0, a.output
    b = runner.invoke(app, ["register", *base, "--covenants", str(fitted["covenants_broken"])])
    assert b.exit_code == 0, b.output

    version_a = a.output.split("version ")[1].split()[0]
    version_b = b.output.split("version ")[1].split()[0]
    result = runner.invoke(
        app, ["diff", "test-scorecard", version_a[:4], version_b[:4], "--store", store]
    )
    assert result.exit_code == 0, result.output
    assert "coefficients_stale.csv" in result.output
    assert "created_at" not in result.output

    with_all = runner.invoke(
        app,
        ["diff", "test-scorecard", version_a[:4], version_b[:4], "--store", store, "--all"],
    )
    assert "created_at" in with_all.output

    shown = runner.invoke(app, ["show", "test-scorecard", version_a[:4], "--store", store])
    assert shown.exit_code == 0
    assert f"version_id: {version_a}" in shown.output


def test_register_invalid_governance_exits_2(fitted: dict, tmp_path: Path) -> None:
    bad = tmp_path / "governance_bad.yaml"
    bad.write_text(fitted["governance"].read_text().replace("materiality:", "materialityz:"))
    result = runner.invoke(
        app,
        [
            "register",
            str(fitted["model"]),
            str(fitted["data"]),
            "--covenants",
            str(fitted["covenants"]),
            "--governance",
            str(bad),
            "--store",
            str(tmp_path / ".covenant"),
        ],
    )
    assert result.exit_code == 2
    assert "error:" in combined_output(result)


def test_unexpected_error_is_exit_2_not_traceback(tmp_path: Path) -> None:
    garbage = tmp_path / "model.joblib"
    garbage.write_bytes(b"not a model")
    data = tmp_path / "data.csv"
    data.write_text("dti\n0.1\n")
    covenants = tmp_path / "covenants.yaml"
    covenants.write_text(
        "covenant_schema: 1\nmodel_name: x\n"
        "features: [{name: dti, direction: increases_risk}]\n"
        "reason_codes: {method: difference_from_mean}\n"
    )
    result = runner.invoke(
        app,
        ["check", "monotonicity", str(garbage), str(data), "--covenants", str(covenants)],
    )
    assert result.exit_code == 2
    assert "error:" in combined_output(result)
    assert "Traceback" not in combined_output(result)


def test_check_monotonicity_cli_exit_codes(fitted: dict, tmp_path: Path) -> None:
    store = str(tmp_path / ".covenant")
    passing = runner.invoke(
        app,
        [
            "check",
            "monotonicity",
            str(fitted["model"]),
            str(fitted["data"]),
            "--covenants",
            str(fitted["covenants"]),
            "--store",
            store,
        ],
    )
    assert passing.exit_code == 0, passing.output
    assert "PASS" in passing.output

    flipped = tmp_path / "covenants_flipped.yaml"
    flipped.write_text(
        fitted["covenants"].read_text().replace(
            "{name: dti, direction: increases_risk}",
            "{name: dti, direction: decreases_risk}",
        )
    )
    breach = runner.invoke(
        app,
        [
            "check",
            "monotonicity",
            str(fitted["model"]),
            str(fitted["data"]),
            "--covenants",
            str(flipped),
            "--store",
            store,
        ],
    )
    assert breach.exit_code == 1, breach.output
    assert "BREACH" in breach.output

    listed = runner.invoke(app, ["checks", "test-scorecard", "--store", store])
    assert listed.exit_code == 0
    assert "monotonicity" in listed.output


def test_check_all_combined_exit_code(fitted: dict, tmp_path: Path) -> None:
    store = str(tmp_path / ".covenant")
    result = runner.invoke(
        app,
        [
            "check",
            "all",
            str(fitted["model"]),
            str(fitted["data"]),
            "--covenants",
            str(fitted["covenants_broken"]),
            "--store",
            store,
        ],
    )
    # reason codes breach on the stale table; monotonicity passes
    assert result.exit_code == 1, result.output
    assert "reason-codes" in result.output
    assert "BREACH" in result.output
    assert "monotonicity" in result.output


def test_check_all_fails_when_configured_check_cannot_run(
    fitted: dict, tmp_path: Path
) -> None:
    """A missing reason-code artefact must not abort checks 2-4, but an
    incomplete gate must not pass either (stranger-test blocker): the run
    reports the skip AND exits 2."""
    covenants = tmp_path / "covenants.yaml"
    covenants.write_text(
        fitted["covenants"].read_text().replace("coefficients_live.csv", "missing.csv")
    )
    result = runner.invoke(
        app,
        [
            "check",
            "all",
            str(fitted["model"]),
            str(fitted["data"]),
            "--covenants",
            str(covenants),
            "--store",
            str(tmp_path / ".covenant"),
        ],
    )
    out = combined_output(result)
    assert "reason-codes   SKIPPED" in out
    assert "monotonicity" in out and "PASS" in out  # the rest still ran
    assert "incomplete gate must not pass" in out
    assert result.exit_code == 2, out


def test_check_all_exits_2_when_nothing_ran(fitted: dict, tmp_path: Path) -> None:
    """A gate that checks nothing must not pass (review finding)."""
    covenants = tmp_path / "covenants.yaml"
    text = fitted["covenants"].read_text()
    text = text.replace("coefficients_live.csv", "missing.csv")
    for direction in ("increases_risk", "decreases_risk"):
        text = text.replace(f"direction: {direction}", "direction: none")
    text = text.replace("excluded:\n  - {name: gender, reason: protected attribute}\n", "")
    covenants.write_text(text)
    result = runner.invoke(
        app,
        [
            "check",
            "all",
            str(fitted["model"]),
            str(fitted["data"]),
            "--covenants",
            str(covenants),
            "--store",
            str(tmp_path / ".covenant"),
        ],
    )
    out = combined_output(result)
    # reason-codes (missing artefact), monotonicity (no directions) and
    # exclusions (no excluded vars) are all skipped: incomplete gate, exit 2.
    assert "SKIPPED" in out
    assert result.exit_code == 2
