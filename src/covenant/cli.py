"""The covenant CLI: register, check, diff, report.

Exit codes are the contract: 0 = pass, 1 = covenant breach (a check failed),
2 = usage or setup error. That is what lets a check sit in CI and block a
deployment.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from covenant import __version__
from covenant.store import STORE_DIR, Store, diff_records

app = typer.Typer(
    name="covenant",
    help="Covenant tests for credit models: governance as code.",
    no_args_is_help=True,
    pretty_exceptions_show_locals=False,
)
check_app = typer.Typer(help="Run covenant checks; a breach exits non-zero.", no_args_is_help=True)
app.add_typer(check_app, name="check")

StoreOption = typer.Option(STORE_DIR, "--store", help="Path of the .covenant store.")


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Print version and exit."),
) -> None:
    if version:
        typer.echo(f"covenant {__version__} (distribution: covenants)")
        raise typer.Exit(0)


@app.command()
def init(store: str = StoreOption) -> None:
    """Create the .covenant store and template covenants/governance files."""
    Store(store).init()
    wrote = []
    for name, template in _TEMPLATES.items():
        path = Path(name)
        if not path.exists():
            path.write_text(template)
            wrote.append(name)
    typer.echo(f"initialised {store}/")
    for name in wrote:
        typer.echo(f"wrote template {name} — edit it before registering")
    if not wrote:
        typer.echo("covenants.yaml and governance.yaml already exist; left untouched")


@app.command()
def register(
    model: Path = typer.Argument(help="Persisted estimator (joblib/pickle) with predict_proba."),
    data: Path = typer.Argument(help="Training snapshot (.csv or .parquet)."),
    covenants: Path = typer.Option("covenants.yaml", "--covenants", help="The model's covenants."),
    governance: Path = typer.Option("governance.yaml", "--governance", help="Governance record."),
    store: str = StoreOption,
) -> None:
    """Register a model version: content-addressed record in the store."""
    from covenant.registry import RegistrationError
    from covenant.registry import register as do_register

    try:
        record = do_register(model, data, covenants, governance, Store(store))
    except (RegistrationError, FileNotFoundError) as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(2) from err
    typer.echo(f"registered {record.model_name} version {record.version_id}")
    typer.echo(f"  model    sha256:{record.hashes.model_sha256[:12]}")
    typer.echo(f"  data     sha256:{record.hashes.data_sha256[:12]} ({record.data.n_rows} rows)")
    typer.echo(f"  covenants sha256:{record.hashes.covenants_sha256[:12]}")
    typer.echo(f"  record   {Store(store).record_path(record.model_name, record.version_id)}")


@check_app.command("reason-codes")
def check_reason_codes(
    model: Path = typer.Argument(help="Persisted estimator with predict_proba."),
    data: Path = typer.Argument(help="Data snapshot to score (.csv or .parquet)."),
    covenants: Path = typer.Option("covenants.yaml", "--covenants", help="The model's covenants."),
    store: str = StoreOption,
    threshold: float | None = typer.Option(
        None, "--threshold", help="Override checks.reason_codes.decision_threshold."
    ),
    as_json: bool = typer.Option(False, "--json", help="Print the check record as JSON."),
) -> None:
    """Check 1: declared adverse-action reasons vs measured attributions."""
    from covenant.attribution import DeclaredMethodError
    from covenant.checks.reason_codes import CheckSetupError, run_reason_code_check
    from covenant.registry import RegistrationError

    overrides = {"decision_threshold": threshold} if threshold is not None else None
    try:
        record = run_reason_code_check(model, data, covenants, overrides)
    except (CheckSetupError, DeclaredMethodError, RegistrationError, FileNotFoundError) as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(2) from err

    path = record.write(Store(store))
    if as_json:
        typer.echo(json.dumps(record.model_dump(mode="json"), indent=2, default=str))
    else:
        _print_reason_code_report(record, path)
    raise typer.Exit(0 if record.passed else 1)


def _print_reason_code_report(record, path: str) -> None:
    verdict = "PASS" if record.passed else "BREACH (fail)"
    m, t = record.metrics, record.thresholds
    typer.echo(f"check reason-codes — {record.model_name}: {verdict}")
    typer.echo(
        f"  top-1 agreement  {m['top1_agreement']:.3f}"
        f"  (threshold {t['min_top1_agreement']:.2f})"
    )
    typer.echo(
        f"  top-k jaccard    {m['topk_jaccard']:.3f}"
        f"  (threshold {t['min_topk_jaccard']:.2f})"
    )
    typer.echo(
        f"  background stability of measured side: {m['background_jaccard']:.3f}"
        + ("  [sensitive — treat measured side with caution]"
           if record.details.get("background_sensitive") else "")
    )
    typer.echo(f"  n denied evaluated: {record.n_evaluated}")
    bands = record.details.get("by_score_band", [])
    if bands:
        typer.echo("  by score band (denied applicants, near boundary first):")
        for band in bands:
            lo, hi = band["p_bad_range"]
            typer.echo(
                f"    p_bad {lo:.3f}-{hi:.3f}  n={band['n']:<5d}"
                f" top-1 {band['top1_agreement']:.3f}  jaccard {band['topk_jaccard']:.3f}"
            )
    worst = record.details.get("worst_disagreements", [])
    if worst:
        typer.echo("  worst disagreements (declared vs measured):")
        for w in worst[:5]:
            typer.echo(
                f"    row {w['row']:<6d} p_bad {w['p_bad']:.3f}"
                f"  declared {', '.join(w['declared'])}"
                f"  | measured {', '.join(w['measured'])}"
            )
    typer.echo(f"  record: {path}")


@app.command()
def diff(
    model_name: str = typer.Argument(help="Registered model name."),
    version_a: str = typer.Argument(help="First version id."),
    version_b: str = typer.Argument(help="Second version id."),
    store: str = StoreOption,
) -> None:
    """Diff two registered versions of a model."""
    s = Store(store)
    try:
        a = s.read_record(model_name, version_a)
        b = s.read_record(model_name, version_b)
    except FileNotFoundError as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(2) from err
    lines = diff_records(a, b)
    if not lines:
        typer.echo("records are identical")
        return
    for line in lines:
        typer.echo(line)


@app.command("list")
def list_models(store: str = StoreOption) -> None:
    """List registered models and their versions."""
    s = Store(store)
    models = s.list_models()
    if not models:
        typer.echo(f"no models registered under {store}/")
        return
    for name in models:
        versions = s.list_versions(name)
        typer.echo(f"{name}: {', '.join(versions)}")


@app.command()
def report() -> None:
    """Deterministic validation report — not yet implemented (planned v0.2)."""
    typer.echo(
        "covenant report is planned for v0.2: discrimination, calibration, "
        "stability, drift and challenger lift with bootstrap CIs, rendered "
        "deterministically and mapped to SR 26-2 / FREE-AI.",
        err=True,
    )
    raise typer.Exit(2)


_TEMPLATES = {
    "covenants.yaml": """\
# The model's covenants: testable claims about its behaviour.
covenant_schema: 1
model_name: my-scorecard
features:
  - name: income
    dtype: numeric
    direction: decreases_risk
  - name: dti
    dtype: numeric
    direction: increases_risk
excluded:
  - name: gender
    reason: protected attribute (ECOA)
reason_codes:
  # how production derives adverse-action reasons:
  # difference_from_mean | custom | shapley | most_points_lost | univariate
  method: difference_from_mean
  top_k: 4
  parameters:
    coefficients: coefficients.csv   # feature,coef[,mean][,scale]
checks:
  reason_codes:
    min_top1_agreement: 0.75
    min_topk_jaccard: 0.60
    decision_threshold: 0.5
""",
    "governance.yaml": """\
# Governance record for the registered model version.
owner:
  name: Your Name
  email: you@example.com
intended_use: >
  Describe the decision this model supports and its population.
limitations:
  - Describe a known limitation.
materiality:
  tier: 2            # 1 = highest materiality
  justification: >
    Why this tier: exposure, purpose, portfolio share. This field is
    mandatory and must be substantive.
review_date: 2027-01-01
vendor: null          # or {name: ..., product: ..., version: ...}
""",
}


if __name__ == "__main__":
    app()
