"""The covenant CLI: register, check, diff, report.

Exit codes are the contract: 0 = pass, 1 = covenant breach (a check failed),
2 = usage or setup error. That is what lets a check sit in CI and block a
deployment. Unexpected exceptions are caught and reported as exit 2;
``--debug`` re-raises them with the full traceback.
"""

from __future__ import annotations

import functools
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

_DEBUG = False


def _guard(fn):
    """Last-resort handler: anything unexpected becomes `error: …`, exit 2."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except (typer.Exit, typer.Abort, SystemExit, KeyboardInterrupt):
            raise
        except Exception as err:
            if _DEBUG:
                raise
            typer.echo(f"error: {err}", err=True)
            raise typer.Exit(2) from err

    return wrapper


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Print version and exit."),
    debug: bool = typer.Option(False, "--debug", help="Re-raise unexpected errors."),
) -> None:
    global _DEBUG
    _DEBUG = debug
    if version:
        typer.echo(f"covenant {__version__} (distribution: covenants)")
        raise typer.Exit(0)


@app.command()
@_guard
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
@_guard
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
    except (RegistrationError, FileNotFoundError, ValueError) as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(2) from err
    typer.echo(f"registered {record.model_name} version {record.version_id}")
    typer.echo(f"  model    sha256:{record.hashes.model_sha256[:12]}")
    typer.echo(f"  data     sha256:{record.hashes.data_sha256[:12]} ({record.data.n_rows} rows)")
    typer.echo(f"  covenants sha256:{record.hashes.covenants_sha256[:12]}")
    typer.echo(f"  record   {Store(store).record_path(record.model_name, record.version_id)}")


def _run_check(runner, model: Path, data: Path, covenants: Path, overrides: dict | None):
    """Run a check callable, mapping known failures to exit 2."""
    from covenant.checks.reason_codes import CheckSetupError
    from covenant.declared import DeclaredMethodError
    from covenant.registry import RegistrationError

    try:
        return runner(model, data, covenants, overrides)
    except (CheckSetupError, DeclaredMethodError, RegistrationError, FileNotFoundError) as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(2) from err


@check_app.command("reason-codes")
@_guard
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
    from covenant.checks.reason_codes import run_reason_code_check

    overrides = {"decision_threshold": threshold} if threshold is not None else None
    record = _run_check(run_reason_code_check, model, data, covenants, overrides)
    path = record.write(Store(store))
    if as_json:
        typer.echo(json.dumps(record.model_dump(mode="json"), indent=2, default=str))
    else:
        _print_reason_code_report(record, path)
    raise typer.Exit(0 if record.passed else 1)


@check_app.command("monotonicity")
@_guard
def check_monotonicity(
    model: Path = typer.Argument(help="Persisted estimator with predict_proba."),
    data: Path = typer.Argument(help="Data snapshot (.csv or .parquet)."),
    covenants: Path = typer.Option("covenants.yaml", "--covenants", help="The model's covenants."),
    store: str = StoreOption,
    as_json: bool = typer.Option(False, "--json", help="Print the check record as JSON."),
) -> None:
    """Check 2: declared vs configured vs empirical monotone directions."""
    from covenant.checks.monotonicity import run_monotonicity_check

    record = _run_check(run_monotonicity_check, model, data, covenants, None)
    path = record.write(Store(store))
    if as_json:
        typer.echo(json.dumps(record.model_dump(mode="json"), indent=2, default=str))
    else:
        _print_monotonicity_report(record, path)
    raise typer.Exit(0 if record.passed else 1)


@check_app.command("features")
@_guard
def check_features(
    model: Path = typer.Argument(help="Persisted estimator with predict_proba."),
    data: Path = typer.Argument(help="Data snapshot (.csv or .parquet)."),
    covenants: Path = typer.Option("covenants.yaml", "--covenants", help="The model's covenants."),
    store: str = StoreOption,
    as_json: bool = typer.Option(False, "--json", help="Print the check record as JSON."),
) -> None:
    """Check 3: declared features vs the features the model actually uses."""
    from covenant.checks.features import run_features_check

    record = _run_check(run_features_check, model, data, covenants, None)
    path = record.write(Store(store))
    if as_json:
        typer.echo(json.dumps(record.model_dump(mode="json"), indent=2, default=str))
    else:
        _print_features_report(record, path)
    raise typer.Exit(0 if record.passed else 1)


@check_app.command("exclusions")
@_guard
def check_exclusions(
    model: Path = typer.Argument(help="Persisted estimator with predict_proba."),
    data: Path = typer.Argument(help="Data snapshot (.csv or .parquet)."),
    covenants: Path = typer.Option("covenants.yaml", "--covenants", help="The model's covenants."),
    store: str = StoreOption,
    as_json: bool = typer.Option(False, "--json", help="Print the check record as JSON."),
) -> None:
    """Check 4: excluded variables stay out, and obvious proxies are surfaced."""
    from covenant.checks.exclusions import run_exclusions_check

    record = _run_check(run_exclusions_check, model, data, covenants, None)
    path = record.write(Store(store))
    if as_json:
        typer.echo(json.dumps(record.model_dump(mode="json"), indent=2, default=str))
    else:
        _print_exclusions_report(record, path)
    raise typer.Exit(0 if record.passed else 1)


@check_app.command("all")
@_guard
def check_all(
    model: Path = typer.Argument(help="Persisted estimator with predict_proba."),
    data: Path = typer.Argument(help="Data snapshot (.csv or .parquet)."),
    covenants: Path = typer.Option("covenants.yaml", "--covenants", help="The model's covenants."),
    store: str = StoreOption,
) -> None:
    """Run every configured check; one summary, one combined exit code."""
    from covenant.checks.exclusions import run_exclusions_check
    from covenant.checks.features import run_features_check
    from covenant.checks.monotonicity import run_monotonicity_check
    from covenant.checks.reason_codes import CheckSetupError, run_reason_code_check

    runners = [
        ("reason-codes", run_reason_code_check, _summary_reason_codes),
        ("monotonicity", run_monotonicity_check, _summary_monotonicity),
        ("features", run_features_check, _summary_features),
        ("exclusions", run_exclusions_check, _summary_exclusions),
    ]
    worst_exit = 0
    lines = []
    for name, runner, summarise in runners:
        try:
            record = runner(model, data, covenants, None)
        except CheckSetupError as err:
            lines.append(f"  {name:<14s} SKIPPED  {err}")
            continue
        record.write(Store(store))
        verdict = "PASS  " if record.passed else "BREACH"
        lines.append(f"  {name:<14s} {verdict}  {summarise(record)}")
        if not record.passed:
            worst_exit = max(worst_exit, 1)
    typer.echo("covenant check all")
    for line in lines:
        typer.echo(line)
    raise typer.Exit(worst_exit)


def _summary_reason_codes(record) -> str:
    m = record.metrics
    return f"top-1 {m['top1_agreement']:.3f}  jaccard {m['topk_jaccard']:.3f}"


def _summary_monotonicity(record) -> str:
    m = record.metrics
    extra = ""
    if m.get("configured_mismatches"):
        extra = f"  configured mismatches {int(m['configured_mismatches'])}"
    return f"worst violation {m['worst_violation_rate']:.3f}{extra}"


def _summary_features(record) -> str:
    m = record.metrics
    return (
        f"undocumented {int(m['n_undocumented_used'])}"
        f"  unused {int(m['n_declared_unused'])}"
        f"  dead {int(m['n_dead'])}"
    )


def _summary_exclusions(record) -> str:
    m = record.metrics
    return (
        f"max association {m['max_association_observed']:.2f}"
        f"  proxy flags {int(m['n_proxy_flags'])}"
    )


def _print_features_report(record, path: str) -> None:
    verdict = "PASS" if record.passed else "BREACH (fail)"
    details = record.details
    typer.echo(f"check features — {record.model_name}: {verdict}")
    if not details.get("structural_available", True):
        typer.echo("  model exposes no input names; structural comparison unavailable")
    for label, key in (
        ("used by the model but undeclared", "undocumented_used"),
        ("declared but not a model input", "declared_unused"),
    ):
        values = details.get(key, [])
        if values:
            typer.echo(f"  {label}: {', '.join(values)}")
    dead = details.get("dead_features", [])
    if dead:
        typer.echo("  documented but measurably inert (warning, not a breach):")
        for row in dead:
            typer.echo(
                f"    {row['feature']:<26s} mean |attribution| "
                f"{row['mean_abs_attribution']:.5f}"
            )
    typer.echo(f"  record: {path}")


def _print_exclusions_report(record, path: str) -> None:
    verdict = "PASS" if record.passed else "BREACH (fail)"
    m = record.metrics
    typer.echo(f"check exclusions — {record.model_name}: {verdict}")
    typer.echo(
        f"  max association observed {m['max_association_observed']:.3f}"
        f"  (threshold {record.thresholds['max_association']:.2f})"
    )
    flagged = record.details.get("flagged_pairs", [])
    if flagged:
        typer.echo("  potential proxies (surfaced, not proven absent):")
        for pair in flagged:
            typer.echo(
                f"    {pair['excluded']} ~ {pair['feature']}"
                f"  {pair['strength']:.3f} ({pair['method']})"
            )
    for row in record.details.get("by_variable", []):
        if isinstance(row, dict) and row.get("note"):
            typer.echo(f"  {row.get('name', '?')}: {row['note']}")
    typer.echo(f"  record: {path}")


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
    path_used = record.details.get("attribution_path")
    if path_used:
        typer.echo(f"  measured via: {path_used}")
    placebo = record.details.get("placebo")
    if placebo and not placebo.get("skipped"):
        flag = "  [noisy]" if placebo.get("noisy") else ""
        typer.echo(
            f"  placebo ({placebo['feature']}): measured top-k shift "
            f"{placebo['measured_topk_shift']:.3f}{flag}"
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


def _print_monotonicity_report(record, path: str) -> None:
    verdict = "PASS" if record.passed else "BREACH (fail)"
    m, t = record.metrics, record.thresholds
    typer.echo(f"check monotonicity — {record.model_name}: {verdict}")
    typer.echo(
        f"  worst violation rate {m['worst_violation_rate']:.3f}"
        f"  (threshold {t['max_violation_rate']:.2f})"
    )
    readable = record.details.get("configured_constraints_readable")
    typer.echo(
        "  configured constraints: "
        + ("read from estimator" if readable else "none found on estimator")
    )
    typer.echo("  by feature (declared direction, empirical violation rates):")
    for row in record.details.get("by_feature", []):
        typer.echo(
            f"    {row['feature']:<26s} {row['declared']:<15s}"
            f" configured {row['configured']:<15s}"
            f" pairs {row['pair_violation_rate']:.3f}  ice {row['ice_violation_rate']:.3f}"
        )
    mismatches = record.details.get("configured_mismatch", [])
    if mismatches:
        typer.echo("  configured constraints contradicting the covenant:")
        for mm in mismatches:
            typer.echo(
                f"    {mm['feature']}: declared {mm['declared']}, "
                f"configured {mm['configured']}"
            )
    typer.echo(f"  record: {path}")


@app.command()
@_guard
def diff(
    model_name: str = typer.Argument(help="Registered model name."),
    version_a: str = typer.Argument(help="First version id (prefix ok)."),
    version_b: str = typer.Argument(help="Second version id (prefix ok)."),
    store: str = StoreOption,
    show_all: bool = typer.Option(
        False, "--all", help="Include timestamp fields normally suppressed."
    ),
) -> None:
    """Diff two registered versions of a model."""
    s = Store(store)
    try:
        a = s.read_record(model_name, s.resolve_version(model_name, version_a))
        b = s.read_record(model_name, s.resolve_version(model_name, version_b))
    except FileNotFoundError as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(2) from err
    lines = diff_records(a, b, ignore=() if show_all else ("created_at",))
    if not lines:
        typer.echo("records are identical")
        return
    for line in lines:
        typer.echo(line)


@app.command()
@_guard
def show(
    model_name: str = typer.Argument(help="Registered model name."),
    version: str = typer.Argument(help="Version id (prefix ok)."),
    store: str = StoreOption,
) -> None:
    """Print a registered record."""
    s = Store(store)
    try:
        path = s.record_path(model_name, s.resolve_version(model_name, version))
    except FileNotFoundError as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(2) from err
    typer.echo(path.read_text(), nl=False)


@app.command("checks")
@_guard
def list_check_records(
    model_name: str = typer.Argument(help="Registered model name."),
    store: str = StoreOption,
) -> None:
    """List check records for a model, with verdicts."""
    from covenant.store import read_yaml

    s = Store(store)
    paths = s.list_checks(model_name)
    if not paths:
        typer.echo(f"no check records for {model_name!r} under {store}/")
        return
    for path in paths:
        record = read_yaml(path)
        verdict = "PASS  " if record.get("passed") else "BREACH"
        typer.echo(f"{verdict} {record.get('check', '?'):<14s} {path.name}")


@app.command("list")
@_guard
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
@_guard
def report(
    model: Path = typer.Argument(help="Persisted estimator with predict_proba."),
    data: Path = typer.Argument(help="Data snapshot (.csv or .parquet) with the target column."),
    covenants: Path = typer.Option("covenants.yaml", "--covenants", help="The model's covenants."),
    out: Path = typer.Option("covenant-report", "--out", help="Output directory."),
    holdout: Path | None = typer.Option(
        None, "--holdout", help="Optional holdout snapshot for stability (PSI/CSI)."
    ),
    target: str | None = typer.Option(
        None, "--target", help="Override report.target_column from the covenants."
    ),
) -> None:
    """Deterministic validation report: same inputs, same bytes.

    Discrimination, calibration, stability, drift, monotonicity and a
    logistic challenger's lift, with bootstrap confidence intervals, every
    section mapped to the regulatory ask (docs/MAPPING.md)."""
    from covenant.checks.reason_codes import CheckSetupError
    from covenant.declared import DeclaredMethodError
    from covenant.registry import RegistrationError
    from covenant.report import build_report

    overrides = {"target_column": target} if target else None
    try:
        path = build_report(
            model, data, covenants, out, holdout_path=holdout, config_overrides=overrides
        )
    except (CheckSetupError, DeclaredMethodError, RegistrationError, FileNotFoundError) as err:
        typer.echo(f"error: {err}", err=True)
        raise typer.Exit(2) from err
    typer.echo(f"report written: {path}")
    typer.echo("re-render with the same inputs reproduces these bytes exactly")


_TEMPLATES = {
    "covenants.yaml": """\
# The model's covenants: testable claims about its behaviour.
covenant_schema: 1
model_name: my-scorecard
# positive_class: 1        # label of the bad class in classes_; default index 1
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
    # id_column: application_id      # required when method is custom/shapley
  monotonicity:
    max_violation_rate: 0.05
  exclusions:
    max_association: 0.5             # proxy screen; tune to your book
report:
  target_column: bad                 # 0/1 outcome column, needed by `covenant report`
  # time_column: application_month   # enables drift-by-slice
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
