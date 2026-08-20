"""The deterministic validation report: same inputs, byte-identical bytes.

Metrics are commodity — ValidMind, PiML/MoDeVa, FINRA's Model Validation
Toolkit and every validation platform compute AUC, KS, Brier, PSI. What
none of them ship, and what this module treats as the product, is a report
that is *replayable evidence*: identical model, snapshot and covenants
produce a byte-identical ``report.md`` and byte-identical figures, with
the report's own SHA-256 embedded (over its body with the hash blanked)
and a manifest of figure digests so one hash transitively covers the
bundle. The RBI FREE-AI survey (Aug 2025) found only ~21% of AI-using
regulated entities monitored drift and ~18% kept audit logs — the report
is the low-cost version of both habits.

Every point estimate ships with a seeded bootstrap interval; FINRA's
Model Validation Toolkit frames this as the credibility of metrics under
small samples, and on the small validation sets of the long tail the
interval, not the point, is the honest number. Each section ends with a
plain-words "Maps to:" note pointing at ``docs/MAPPING.md`` — never at a
regulatory section number this project has not pinned.

A stranger test of the published package reshaped what the report says
about itself. Every metrics table is labelled in-sample or out-of-sample —
in-sample numbers are computed on the rows the model was fitted to and
flatter it, and when a labelled holdout is supplied the out-of-sample
table renders first as the headline. All four covenant checks
(reason-codes, monotonicity, features, exclusions) embed their verdicts
and record hashes: a breached covenant renders as BREACH, because the
report is evidence, not a gate. And when a governance record is supplied
(``governance_path``), the report embeds the owner, intended use,
limitations, materiality tier with its justification verbatim, review
date and vendor block.

Nothing here claims compliance: Covenant produces evidence; your
validators and auditors decide.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

from covenant import metrics
from covenant.challenger import challenger_scores
from covenant.checks import (
    CheckRecord,
    run_exclusions_check,
    run_features_check,
    run_monotonicity_check,
    run_reason_code_check,
)
from covenant.checks.reason_codes import CheckSetupError
from covenant.declared import DeclaredMethodError
from covenant.hashing import sha256_bytes, sha256_canonical, sha256_dataframe, sha256_file
from covenant.model import CovenantModel, library_versions, load_model
from covenant.registry import load_covenants, load_data, load_governance
from covenant.schema import GovernanceRecord, ModelCovenants, ReportConfig

matplotlib.use("Agg")  # fixed headless backend; pyplot is imported after this, lazily

_HASH_PLACEHOLDER = "0" * 64
_EVIDENCE_LINE = "Covenant produces evidence; your validators and auditors decide."
_MAPPING = "docs/MAPPING.md"

# Deterministic rendering: fixed dpi and font so identical inputs give
# identical PNG bytes; savefig() below also strips the Software metadata
# matplotlib would otherwise stamp with its version string.
_RC: dict[str, Any] = {
    "figure.dpi": 100,
    "savefig.dpi": 100,
    "font.family": "DejaVu Sans",
    "font.size": 10,
}

_PRIMARY_COLOR = "#1f77b4"
_CHALLENGER_COLOR = "#d62728"
_REFERENCE_COLOR = "#888888"


def build_report(
    model_path: str | Path,
    data_path: str | Path,
    covenants_path: str | Path,
    out_dir: str | Path,
    holdout_path: str | Path | None = None,
    config_overrides: dict | None = None,
    *,
    governance_path: str | Path | None = None,
) -> Path:
    """Render ``out_dir/report.md`` plus ``out_dir/figures/*.png``.

    Returns the path to ``report.md``. Deterministic by construction: no
    timestamps, all randomness through ``numpy.random.default_rng`` seeded
    from ``report.random_state``, fixed matplotlib rcParams, stripped PNG
    metadata. ``out_dir`` is created if missing and existing files are
    overwritten; the destination never affects the rendered bytes.

    When ``holdout_path`` carries the outcome column, discrimination and
    calibration render out-of-sample numbers as the headline, in-sample
    second, both labelled. When ``governance_path`` is supplied, the
    governance record (owner, intended use, limitations, materiality with
    its justification, review date, vendor) is embedded in section 2.
    """
    covenants: ModelCovenants = load_covenants(covenants_path)
    governance: GovernanceRecord | None = None
    if governance_path is not None:
        governance = load_governance(governance_path)
    config: ReportConfig = covenants.report
    if config_overrides:
        config = config.model_copy(update=config_overrides)
    if not config.target_column:
        raise CheckSetupError(
            "the covenants declare no report.target_column; set report.target_column "
            "to the 0/1 outcome column of the snapshot (1 = bad) so the report can "
            "compare scores with outcomes"
        )

    raw = load_data(data_path)
    data = raw.reset_index(drop=True)
    features = covenants.feature_names()
    categorical = covenants.categorical_features()
    missing = [f for f in features if f not in data.columns]
    if missing:
        raise CheckSetupError(f"data lacks declared features: {missing}")
    numeric = [f for f in features if f not in categorical]
    data[numeric] = data[numeric].astype(float)
    y = _validated_target(data, config.target_column)

    estimator = load_model(model_path)
    model = CovenantModel(estimator, features, positive_class=covenants.positive_class)
    p = model.p_bad(data)

    holdout_raw = None
    holdout = None
    y_hold: np.ndarray | None = None
    p_hold: np.ndarray | None = None
    if holdout_path is not None:
        holdout_raw = load_data(holdout_path)
        holdout = holdout_raw.reset_index(drop=True)
        missing_h = [f for f in features if f not in holdout.columns]
        if missing_h:
            raise CheckSetupError(f"holdout lacks declared features: {missing_h}")
        holdout[numeric] = holdout[numeric].astype(float)
        p_hold = model.p_bad(holdout)
        if config.target_column in holdout.columns:
            y_hold = _validated_target(holdout, config.target_column, where="holdout snapshot")

    out = Path(out_dir)
    figures_dir = out / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_paths: list[Path] = []

    holdout_eval = None if y_hold is None or p_hold is None else (y_hold, p_hold)
    holdout_supplied = holdout is not None
    sections = [
        _governance_section(governance),
        _discrimination_section(
            y, p, holdout_eval, holdout_supplied, config, figures_dir, fig_paths
        ),
        _calibration_section(
            y, p, holdout_eval, holdout_supplied, config, figures_dir, fig_paths
        ),
        _stability_section(p, data, holdout, p_hold, numeric, config),
        _drift_section(p, data, config),
        _checks_section(model_path, data_path, covenants_path),
        _challenger_section(y, p, data, features, categorical, config, figures_dir, fig_paths),
    ]
    header = _header_section(
        covenants, config, model_path, raw, holdout_raw, governance, fig_paths
    )
    footer = _footer_section(
        model_path, data_path, covenants_path, holdout_path, governance_path, config_overrides
    )

    body = "\n\n".join([header, *sections, footer]) + "\n"
    digest = sha256_bytes(body.encode("utf-8"))
    final = body.replace(
        f"report_sha256: {_HASH_PLACEHOLDER}", f"report_sha256: {digest}", 1
    )
    path = out / "report.md"
    path.write_bytes(final.encode("utf-8"))
    return path


def _validated_target(data: pd.DataFrame, target: str, where: str = "snapshot") -> np.ndarray:
    if target not in data.columns:
        raise CheckSetupError(
            f"report.target_column {target!r} is not a column of the {where}; "
            "set report.target_column to the 0/1 outcome column present in the "
            "data (1 = bad)"
        )
    y = pd.to_numeric(data[target]).to_numpy(dtype=float)
    labels = np.unique(y)
    if not np.isin(labels, (0.0, 1.0)).all():
        raise CheckSetupError(
            f"report.target_column {target!r} in the {where} must contain only "
            f"0/1 labels (1 = bad), got values {labels[:5].tolist()}"
        )
    if len(labels) < 2:
        raise CheckSetupError(
            f"report.target_column {target!r} has a single class; discrimination "
            f"metrics need both goods and bads in the {where}"
        )
    return y


def _f(value: float) -> str:
    return f"{value:.4f}"


def _header_section(
    covenants: ModelCovenants,
    config: ReportConfig,
    model_path: str | Path,
    raw: pd.DataFrame,
    holdout_raw: pd.DataFrame | None,
    governance: GovernanceRecord | None,
    fig_paths: list[Path],
) -> str:
    versions = ", ".join(f"{k} {v}" for k, v in sorted(library_versions().items()))
    rows = [
        ("Model name", f"`{covenants.model_name}`"),
        ("Model file sha256", f"`{sha256_file(model_path)}`"),
        ("Data snapshot sha256", f"`{sha256_dataframe(raw)}`"),
        ("Covenants sha256", f"`{sha256_canonical(covenants.model_dump(mode='json'))}`"),
    ]
    if governance is not None:
        rows.append(
            ("Governance sha256", f"`{sha256_canonical(governance.model_dump(mode='json'))}`")
        )
    if holdout_raw is not None:
        rows.append(("Holdout sha256", f"`{sha256_dataframe(holdout_raw)}`"))
    rows += [
        ("Snapshot shape", f"{len(raw)} rows x {raw.shape[1]} columns"),
        (
            "Report config",
            f"n_bootstrap={config.n_bootstrap}, n_bins={config.n_bins}, "
            f"random_state={config.random_state}, "
            f"challenger={'on' if config.challenger else 'off'}",
        ),
        ("Library versions", versions),
    ]
    manifest = [
        f"| figures/{path.name} | `{sha256_file(path)}` |"
        for path in sorted(fig_paths, key=lambda pth: pth.name)
    ]
    lines = [
        f"# Validation report: {covenants.model_name}",
        "",
        "Covenant renders this report deterministically: the same model, snapshot "
        "and covenants produce byte-identical markdown and figures. Nothing in it "
        f"is a claim of compliance — {_EVIDENCE_LINE}",
        "",
        "## 1. Identity and audit trail",
        "",
        "| Field | Value |",
        "|---|---|",
        *[f"| {field} | {value} |" for field, value in rows],
        "",
        f"report_sha256: {_HASH_PLACEHOLDER}",
        "",
        "Verification convention: `report_sha256` is the SHA-256 of this file's "
        "bytes after replacing the 64 hex digits on the `report_sha256:` line with "
        "64 ASCII `0` characters. To verify, blank the digest exactly that way, "
        "hash the file, and compare. The manifest below embeds each figure's "
        "digest in the hashed body, so the one hash transitively covers every PNG.",
        "",
        "| Figure | sha256 |",
        "|---|---|",
        *manifest,
        "",
        "Maps to: the model-inventory and audit-trail asks — the content hashes tie "
        f"this evidence to one exact model, snapshot and covenants ({_MAPPING}).",
    ]
    return "\n".join(lines)


def _governance_section(governance: GovernanceRecord | None) -> str:
    lines = ["## 2. Governance", ""]
    if governance is None:
        lines += [
            "Governance record not supplied — pass --governance to embed owner, "
            "materiality and review date.",
        ]
    else:
        if governance.vendor is None:
            vendor = "none (in-house)"
        else:
            vendor = f"{governance.vendor.name}, {governance.vendor.product}"
            if governance.vendor.version:
                vendor += f", version {governance.vendor.version}"
            if governance.vendor.contact:
                vendor += f" ({governance.vendor.contact})"
        lines += [
            "| Field | Value |",
            "|---|---|",
            f"| Owner | {governance.owner.name} ({governance.owner.email}) |",
            f"| Materiality tier | {governance.materiality.tier} |",
            f"| Review date | {governance.review_date.isoformat()} |",
            f"| Vendor | {vendor} |",
            "",
            f"Intended use: {governance.intended_use.strip()}",
            "",
        ]
        if governance.limitations:
            lines += [
                "Limitations, as recorded by the owner:",
                "",
                *[f"- {limitation.strip()}" for limitation in governance.limitations],
                "",
            ]
        else:
            lines += ["Limitations: none recorded in the governance file.", ""]
        lines += [
            f"Materiality tier {governance.materiality.tier} justification, "
            "verbatim from the governance record:",
            "",
            *[f"> {line}" for line in governance.materiality.justification.strip().splitlines()],
        ]
    lines += [
        "",
        "Maps to: the accountability and materiality asks — a named owner, a "
        f"tier justified rather than asserted, and a review date ({_MAPPING}).",
    ]
    return "\n".join(lines)


# Why in-sample numbers are labelled: the model was fitted to those rows, so
# it is graded on answers it has already seen — the honest reading needs the
# label next to the number, not in a footnote.
_IN_SAMPLE_CAVEAT = (
    "the same rows the model was fitted to. In-sample numbers flatter the "
    "model — it is graded on answers it has already seen — so they overstate "
    "the performance a new applicant book would see."
)


def _metric_table(rows: list[tuple[str, float, float, float]]) -> list[str]:
    return [
        "| Metric | Point | CI lower | CI upper |",
        "|---|---|---|---|",
        *[f"| {name} | {_f(v)} | {_f(lo)} | {_f(hi)} |" for name, v, lo, hi in rows],
    ]


def _out_of_sample_missing_note(holdout_supplied: bool, config: ReportConfig) -> str:
    if holdout_supplied:
        return (
            f"The supplied holdout lacks the outcome column `{config.target_column}`, "
            "so out-of-sample numbers cannot be computed here; add outcomes to the "
            "holdout to put an out-of-sample table above this one as the headline."
        )
    return (
        "No holdout was supplied, so only in-sample numbers can be shown; pass a "
        "labelled holdout snapshot (CLI: `--holdout`) to put out-of-sample numbers "
        "above these as the headline."
    )


def _discrimination_section(
    y: np.ndarray,
    p: np.ndarray,
    holdout_eval: tuple[np.ndarray, np.ndarray] | None,
    holdout_supplied: bool,
    config: ReportConfig,
    figures_dir: Path,
    fig_paths: list[Path],
) -> str:
    boot = {"n_boot": config.n_bootstrap, "seed": config.random_state}

    def rows_for(y_: np.ndarray, p_: np.ndarray) -> list[tuple[str, float, float, float]]:
        return [
            ("AUC", *metrics.bootstrap_ci(metrics.roc_auc, y_, p_, **boot)),
            ("Gini", *metrics.bootstrap_ci(metrics.gini, y_, p_, **boot)),
            ("KS", *metrics.bootstrap_ci(metrics.ks_statistic, y_, p_, **boot)),
        ]

    _render_roc(figures_dir / "roc.png", y, p, fig_paths)
    lines = [
        "## 3. Discrimination",
        "",
        f"Point estimates with 95% bootstrap confidence intervals "
        f"({config.n_bootstrap} seeded resamples, seed {config.random_state}). "
        "FINRA's Model Validation Toolkit frames small samples as a credibility "
        "problem for point estimates; the interval is the honest number.",
        "",
    ]
    if holdout_eval is not None:
        y_hold, p_hold = holdout_eval
        lines += [
            f"Out-of-sample — computed on the holdout snapshot ({len(y_hold)} rows), "
            "which the model was not fitted to. These are the headline numbers:",
            "",
            *_metric_table(rows_for(y_hold, p_hold)),
            "",
            f"In-sample — computed on the training snapshot ({len(y)} rows), "
            f"{_IN_SAMPLE_CAVEAT} Read them after the out-of-sample table above:",
            "",
            *_metric_table(rows_for(y, p)),
        ]
    else:
        lines += [
            f"In-sample — computed on the training snapshot ({len(y)} rows), "
            f"{_IN_SAMPLE_CAVEAT}",
            "",
            *_metric_table(rows_for(y, p)),
            "",
            _out_of_sample_missing_note(holdout_supplied, config),
        ]
    lines += [
        "",
        "![ROC curve (in-sample)](figures/roc.png)",
        "",
        "Maps to: the outcome-analysis and ongoing-monitoring asks — discrimination "
        f"measured with intervals, not bare points ({_MAPPING}).",
    ]
    return "\n".join(lines)


def _calibration_section(
    y: np.ndarray,
    p: np.ndarray,
    holdout_eval: tuple[np.ndarray, np.ndarray] | None,
    holdout_supplied: bool,
    config: ReportConfig,
    figures_dir: Path,
    fig_paths: list[Path],
) -> str:
    def ece_fn(y_: np.ndarray, p_: np.ndarray) -> float:
        return metrics.ece(y_, p_, n_bins=config.n_bins)

    boot = {"n_boot": config.n_bootstrap, "seed": config.random_state}

    def rows_for(y_: np.ndarray, p_: np.ndarray) -> list[tuple[str, float, float, float]]:
        return [
            ("Brier", *metrics.bootstrap_ci(metrics.brier, y_, p_, **boot)),
            (f"ECE ({config.n_bins} bins)", *metrics.bootstrap_ci(ece_fn, y_, p_, **boot)),
        ]

    _render_calibration(figures_dir / "calibration.png", y, p, config.n_bins, fig_paths)
    lines = [
        "## 4. Calibration",
        "",
        f"Brier score and expected calibration error over {config.n_bins} "
        "equal-width probability bins, each with a 95% bootstrap confidence "
        f"interval ({config.n_bootstrap} seeded resamples, seed "
        f"{config.random_state}).",
        "",
    ]
    if holdout_eval is not None:
        y_hold, p_hold = holdout_eval
        lines += [
            f"Out-of-sample — computed on the holdout snapshot ({len(y_hold)} rows), "
            "which the model was not fitted to. These are the headline numbers:",
            "",
            *_metric_table(rows_for(y_hold, p_hold)),
            "",
            f"In-sample — computed on the training snapshot ({len(y)} rows), "
            f"{_IN_SAMPLE_CAVEAT} Read them after the out-of-sample table above:",
            "",
            *_metric_table(rows_for(y, p)),
        ]
    else:
        lines += [
            f"In-sample — computed on the training snapshot ({len(y)} rows), "
            f"{_IN_SAMPLE_CAVEAT}",
            "",
            *_metric_table(rows_for(y, p)),
            "",
            _out_of_sample_missing_note(holdout_supplied, config),
        ]
    lines += [
        "",
        "![Reliability diagram (in-sample)](figures/calibration.png)",
        "",
        "Maps to: the outcome-analysis ask — predicted probabilities compared with "
        f"observed outcomes ({_MAPPING}).",
    ]
    return "\n".join(lines)


def _stability_section(
    p: np.ndarray,
    data: pd.DataFrame,
    holdout: pd.DataFrame | None,
    p_hold: np.ndarray | None,
    numeric: list[str],
    config: ReportConfig,
) -> str:
    lines = ["## 5. Stability", ""]
    if holdout is None:
        lines += [
            "A single snapshot was supplied, so train-to-holdout stability cannot "
            "be measured — one sample has nothing to be stable against. Pass a "
            "holdout snapshot (CLI: `--holdout`) to render the score PSI and the "
            "per-feature CSI table.",
        ]
    else:
        assert p_hold is not None  # computed in build_report whenever holdout is
        score_psi = metrics.psi(p, p_hold, n_bins=config.n_bins)
        lines += [
            f"Score PSI (train -> holdout): {_f(score_psi)} "
            f"(bin edges from {config.n_bins} quantiles of the training scores).",
            "",
        ]
        if numeric:
            csi_values = metrics.csi(data, holdout, numeric, n_bins=config.n_bins)
            lines += [
                "Per-feature CSI — numeric features only; the quantile-bin screen "
                "does not bin categorical features:",
                "",
                "| Feature | CSI |",
                "|---|---|",
                *[f"| {name} | {_f(csi_values[name])} |" for name in numeric],
                "",
            ]
        else:
            lines += [
                "All declared features are categorical; the per-feature CSI screen "
                "covers numeric features only, so no CSI table is rendered.",
                "",
            ]
        lines += [
            "No stability threshold is asserted here: Covenant reports the index; "
            "your validators decide what binds.",
        ]
    lines += [
        "",
        "Maps to: the ongoing-monitoring and drift asks — population stability "
        f"between the snapshots supplied ({_MAPPING}).",
    ]
    return "\n".join(lines)


def _drift_section(p: np.ndarray, data: pd.DataFrame, config: ReportConfig) -> str:
    lines = ["## 6. Drift", ""]
    time_col = config.time_column
    if time_col is None:
        lines += [
            "No report.time_column is configured, so the snapshot cannot be "
            "sliced in time and score drift is not assessed here. Set "
            "report.time_column to render score PSI across time slices.",
        ]
    elif len(p) < 8:
        lines += [
            f"The snapshot has only {len(p)} rows — too few to split into four "
            "time slices; score drift is not assessed here.",
        ]
    else:
        if time_col not in data.columns:
            raise CheckSetupError(
                f"report.time_column {time_col!r} is not a column of the snapshot; "
                "fix report.time_column or remove it to skip the drift section"
            )
        t = data[time_col].to_numpy()
        order = np.argsort(t, kind="stable")
        p_sorted = p[order]
        t_sorted = t[order]
        slices = np.array_split(np.arange(len(p_sorted)), 4)
        base = p_sorted[slices[0]]
        rows = [
            f"| 1 (baseline) | {len(slices[0])} | "
            f"{_fmt_time(t_sorted[slices[0][0]])} - {_fmt_time(t_sorted[slices[0][-1]])} | - |"
        ]
        for k, idx in enumerate(slices[1:], start=2):
            value = metrics.psi(base, p_sorted[idx], n_bins=config.n_bins)
            rows.append(
                f"| {k} | {len(idx)} | "
                f"{_fmt_time(t_sorted[idx[0]])} - {_fmt_time(t_sorted[idx[-1]])} | {_f(value)} |"
            )
        lines += [
            f"The snapshot is sorted by `{time_col}` (stable sort) and split into "
            "four equal row-count slices; each later slice's scores are compared "
            "with the first slice's by PSI. The RBI FREE-AI survey found only "
            "about a fifth of AI-using regulated entities monitored drift (as "
            "documented in docs/research-guide.md); this table is the low-cost "
            "version of that habit.",
            "",
            f"| Slice | Rows | {time_col} range | PSI vs slice 1 |",
            "|---|---|---|---|",
            *rows,
        ]
    lines += [
        "",
        f"Maps to: the drift-monitoring ask ({_MAPPING}).",
    ]
    return "\n".join(lines)


def _checks_section(
    model_path: str | Path, data_path: str | Path, covenants_path: str | Path
) -> str:
    """All four covenant checks, run exactly as ``covenant check`` runs them.

    A check that cannot run — a missing artefact, nothing declared to test —
    renders an honest "Not run" line instead of failing the render; a breach
    renders as BREACH because the report is evidence, not a gate.
    """
    checks: list[tuple[str, str, Any, Any]] = [
        ("Reason codes", "reason-codes", run_reason_code_check, _reason_codes_body),
        ("Monotonicity", "monotonicity", run_monotonicity_check, _monotonicity_body),
        ("Features", "features", run_features_check, _features_body),
        ("Exclusions", "exclusions", run_exclusions_check, _exclusions_body),
    ]
    maps_to = {
        "reason-codes": (
            "Maps to: the adverse-action accuracy ask — the reasons production "
            f"would send, tested against the drivers actually measured ({_MAPPING})."
        ),
        "monotonicity": (
            "Maps to: the conceptual-soundness ask — declared behaviour tested "
            f"against measured behaviour ({_MAPPING})."
        ),
        "features": (
            "Maps to: the inventory-accuracy ask — the declared dependency list "
            f"held against the features the artefact actually consumes ({_MAPPING})."
        ),
        "exclusions": (
            "Maps to: the exclusions ask — excluded variables measured, obvious "
            f"proxies surfaced, not proven absent ({_MAPPING})."
        ),
    }

    outcomes: list[tuple[str, str, CheckRecord | None, str | None, Any]] = []
    for title, name, runner, body_fn in checks:
        try:
            record = runner(model_path, data_path, covenants_path)
            outcomes.append((title, name, record, None, body_fn))
        except (CheckSetupError, DeclaredMethodError) as err:
            outcomes.append((title, name, None, str(err), body_fn))

    summary_rows = []
    for _, name, record, _, _ in outcomes:
        if record is None:
            summary_rows.append(f"| `{name}` | not run | - |")
        else:
            verdict = "PASS" if record.passed else "BREACH"
            summary_rows.append(f"| `{name}` | {verdict} | `{record.record_sha256}` |")

    lines = [
        "## 7. Covenant checks",
        "",
        "The four covenant checks run here exactly as `covenant check <name>` "
        "runs them, on the same model, snapshot and covenants hashed in "
        "section 1. The report is evidence, not a gate: a breached covenant "
        "renders as **BREACH** instead of stopping the render, because "
        "refusing to produce the document would hide the very finding a "
        "validator needs to see. Each record sha256 is the hash of the same "
        "record `covenant check` writes to the store, so a verdict here is "
        f"citable against a stored record. {_EVIDENCE_LINE}",
        "",
        "| Check | Verdict | Record sha256 |",
        "|---|---|---|",
        *summary_rows,
    ]
    for i, (title, name, record, error, body_fn) in enumerate(outcomes, start=1):
        lines += ["", f"### 7.{i} {title} (`{name}`)", ""]
        if record is None:
            lines += [f"Not run: {error}"]
        else:
            lines += body_fn(record)
        lines += ["", maps_to[name]]
    return "\n".join(lines)


def _reason_codes_body(record: CheckRecord) -> list[str]:
    verdict = "PASS" if record.passed else "BREACH"
    m, t = record.metrics, record.thresholds
    return [
        f"Verdict: **{verdict}** — the adverse-action reasons the production "
        "artefact would send, tested against measured SHAP attributions over "
        f"{record.n_evaluated} denied applicants. A verdict is agreement rates "
        "against thresholds the covenants declare, not a compliance claim. "
        f"Check record sha256: `{record.record_sha256}`.",
        "",
        "| Metric | Observed | Declared threshold |",
        "|---|---|---|",
        f"| Top-1 agreement | {_f(m['top1_agreement'])} | "
        f">= {_f(t['min_top1_agreement'])} (min_top1_agreement) |",
        f"| Top-k Jaccard | {_f(m['topk_jaccard'])} | "
        f">= {_f(t['min_topk_jaccard'])} (min_topk_jaccard) |",
        f"| Background Jaccard | {_f(m['background_jaccard'])} | "
        "reported, not thresholded |",
    ]


def _monotonicity_body(record: CheckRecord) -> list[str]:
    verdict = "PASS" if record.passed else "BREACH"
    return [
        f"Verdict: **{verdict}** — worst violation rate "
        f"{_f(record.metrics['worst_violation_rate'])} against the declared "
        f"threshold max_violation_rate = {_f(record.thresholds['max_violation_rate'])}, "
        f"with {int(record.metrics['configured_mismatches'])} configured-constraint "
        "mismatch(es). A verdict is a rate against a threshold the covenants "
        "declare, not a compliance claim. Check record sha256: "
        f"`{record.record_sha256}`.",
        "",
        "| Feature | Declared | Configured | Pair violation rate | ICE violation rate |",
        "|---|---|---|---|---|",
        *[
            f"| {r['feature']} | {r['declared']} | {r['configured']} | "
            f"{_f(r['pair_violation_rate'])} | {_f(r['ice_violation_rate'])} |"
            for r in record.details["by_feature"]
        ],
    ]


def _features_body(record: CheckRecord) -> list[str]:
    verdict = "PASS" if record.passed else "BREACH"
    m = record.metrics
    epsilon = record.thresholds["dead_feature_epsilon"]
    lines = [
        f"Verdict: **{verdict}** — the declared feature list compared with the "
        "features the artefact actually consumes; an undocumented input or a "
        "declared-but-unused feature is a breach. A dead feature (documented "
        "but measurably inert) is a documentation-quality warning, never a "
        f"breach. Check record sha256: `{record.record_sha256}`.",
        "",
        "| Metric | Observed | Declared threshold |",
        "|---|---|---|",
        f"| Undocumented inputs used | {int(m['n_undocumented_used'])} | 0 (any is a breach) |",
        f"| Declared features unused | {int(m['n_declared_unused'])} | 0 (any is a breach) |",
        f"| Dead features (warning) | {int(m['n_dead'])} | "
        f"attribution share < {_f(epsilon)} counts as dead |",
    ]
    if not record.details["structural_available"]:
        lines += [
            "",
            "The estimator does not record `feature_names_in_`, so the "
            "structural comparison is unavailable and only the attribution "
            "screen ran; it can surface inert documented features but cannot "
            "see undocumented inputs.",
        ]
    dead = record.details["dead_features"]
    if dead:
        lines += [
            "",
            "Dead features: " + ", ".join(f"`{d['feature']}`" for d in dead) + ".",
        ]
    return lines


def _exclusions_body(record: CheckRecord) -> list[str]:
    verdict = "PASS" if record.passed else "BREACH"
    m, t = record.metrics, record.thresholds
    lines = [
        f"Verdict: **{verdict}** — each declared exclusion screened two ways: "
        "measured attribution where the variable reaches the model, and "
        "pairwise association against every declared feature (the proxy "
        f"screen). Check record sha256: `{record.record_sha256}`.",
        "",
        "| Metric | Observed | Declared threshold |",
        "|---|---|---|",
        f"| Max excluded-variable attribution share | "
        f"{m['max_excluded_attribution_observed']:.6f} | "
        f"< {_f(t['max_excluded_attribution'])} (max_excluded_attribution) |",
        f"| Max excluded-vs-feature association | {_f(m['max_association_observed'])} | "
        f"flagged above {_f(t['max_association'])} (max_association) |",
        f"| Proxy pairs flagged | {int(m['n_proxy_flags'])} | - |",
        "",
    ]
    pair = record.details["max_association_pair"]
    if pair is None:
        lines += [
            "No excluded-vs-feature association pair was screened. Proxies are "
            "surfaced, not proven absent: a weak pairwise association cannot "
            "rule out a multivariate proxy.",
        ]
    else:
        lines += [
            f"Strongest association surfaced: excluded `{pair['excluded']}` vs "
            f"declared feature `{pair['feature']}` — {_f(pair['strength'])} "
            f"({pair['method']}). Proxies are surfaced, not proven absent: a "
            "weak pairwise association cannot rule out a multivariate proxy.",
        ]
    return lines


def _challenger_section(
    y: np.ndarray,
    p: np.ndarray,
    data: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    config: ReportConfig,
    figures_dir: Path,
    fig_paths: list[Path],
) -> str:
    lines = ["## 8. Challenger", ""]
    if not config.challenger:
        lines += [
            "Disabled in the report config (report.challenger: false); no "
            "challenger comparison is rendered.",
        ]
    else:
        assert config.target_column is not None  # validated in build_report
        p_ch = challenger_scores(
            data, features, categorical, config.target_column, random_state=config.random_state
        )
        boot = {"n_boot": config.n_bootstrap, "seed": config.random_state}
        rows = [
            (
                "AUC",
                metrics.roc_auc(y, p),
                metrics.roc_auc(y, p_ch),
                metrics.paired_bootstrap_diff(metrics.roc_auc, y, p, p_ch, **boot),
            ),
            (
                "KS",
                metrics.ks_statistic(y, p),
                metrics.ks_statistic(y, p_ch),
                metrics.paired_bootstrap_diff(metrics.ks_statistic, y, p, p_ch, **boot),
            ),
        ]
        _render_challenger(figures_dir / "challenger.png", y, p, p_ch, fig_paths)
        lines += [
            "The challenger is a plain logistic pipeline (one-hot encoding, "
            "scaling, logistic regression) predicted out of fold via 5-fold "
            f"stratified cross-validation, seed {config.random_state}. Deltas are "
            "primary minus challenger with paired bootstrap intervals "
            f"({config.n_bootstrap} resamples on identical rows for both models).",
            "",
            "| Metric | Primary | Challenger | Delta (primary - challenger) | 95% CI |",
            "|---|---|---|---|---|",
            *[
                f"| {name} | {_f(prim)} | {_f(chal)} | {_f(d)} | [{_f(lo)}, {_f(hi)}] |"
                for name, prim, chal, (d, lo, hi) in rows
            ],
            "",
            "Honesty asymmetry, stated so the table cannot oversell: the "
            "challenger is scored out of fold while the primary model is scored "
            "in-sample on the same snapshot, which flatters the primary. Read the "
            "comparison as a floor for the challenger, not a horse race — a "
            'positive delta turns "our model beats the simple thing" into a '
            "measured claim with an interval.",
            "",
            "![Primary vs challenger ROC](figures/challenger.png)",
        ]
    lines += [
        "",
        "Maps to: the outcome-analysis ask — benchmarking against a simpler "
        f"alternative, with an interval ({_MAPPING}).",
    ]
    return "\n".join(lines)


def _footer_section(
    model_path: str | Path,
    data_path: str | Path,
    covenants_path: str | Path,
    holdout_path: str | Path | None,
    governance_path: str | Path | None,
    config_overrides: dict | None,
) -> str:
    command = (
        f"covenant report {Path(model_path).name} {Path(data_path).name} "
        f"--covenants {Path(covenants_path).name}"
    )
    if holdout_path is not None:
        command += f" --holdout {Path(holdout_path).name}"
    if governance_path is not None:
        command += f" --governance {Path(governance_path).name}"
    if config_overrides and config_overrides.get("target_column"):
        command += f" --target {config_overrides['target_column']}"
    lines = [
        "## 9. Reproduce this report",
        "",
        "```",
        command,
        "```",
        "",
        "The command is written with file basenames so these bytes never depend "
        "on how the input paths were spelled — run it from the directory holding "
        "the inputs; the content hashes in section 1, not paths, identify them. "
        "`--out` chooses the destination directory and does not change the "
        "rendered bytes: re-running with the same model, snapshot and covenants "
        "reproduces this file and every figure byte for byte.",
        "",
        _EVIDENCE_LINE,
        "",
        "Maps to: the audit-trail ask — a replayable invocation over "
        f"hash-identified inputs ({_MAPPING}).",
    ]
    return "\n".join(lines)


def _fmt_time(value: object) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _roc_points(y: np.ndarray, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(-p, kind="stable")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1.0)
    fp = np.cumsum(y_sorted == 0.0)
    tpr = np.concatenate(([0.0], tp / tp[-1]))
    fpr = np.concatenate(([0.0], fp / fp[-1]))
    return fpr, tpr


def _reliability_points(
    y: np.ndarray, p: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    counts = np.bincount(idx, minlength=n_bins)
    sum_p = np.bincount(idx, weights=p, minlength=n_bins)
    sum_y = np.bincount(idx, weights=y, minlength=n_bins)
    nonempty = counts > 0
    return sum_p[nonempty] / counts[nonempty], sum_y[nonempty] / counts[nonempty]


def _save_figure(fig: Any, path: Path, fig_paths: list[Path]) -> None:
    import matplotlib.pyplot as plt

    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)
    fig_paths.append(path)


def _render_roc(path: Path, y: np.ndarray, p: np.ndarray, fig_paths: list[Path]) -> None:
    import matplotlib.pyplot as plt

    fpr, tpr = _roc_points(y, p)
    auc = metrics.roc_auc(y, p)
    with plt.rc_context(_RC):  # type: ignore[arg-type]  # keys are valid rcParams
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        ax.plot(fpr, tpr, color=_PRIMARY_COLOR, label=f"primary, in-sample (AUC {auc:.4f})")
        ax.plot([0, 1], [0, 1], "--", color=_REFERENCE_COLOR, linewidth=1.0, label="chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curve (in-sample, training snapshot)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        _save_figure(fig, path, fig_paths)


def _render_calibration(
    path: Path, y: np.ndarray, p: np.ndarray, n_bins: int, fig_paths: list[Path]
) -> None:
    import matplotlib.pyplot as plt

    mean_p, mean_y = _reliability_points(y, p, n_bins)
    with plt.rc_context(_RC):  # type: ignore[arg-type]  # keys are valid rcParams
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        ax.plot([0, 1], [0, 1], "--", color=_REFERENCE_COLOR, linewidth=1.0, label="perfect")
        ax.plot(mean_p, mean_y, marker="o", color=_PRIMARY_COLOR, label="model (in-sample)")
        ax.set_xlabel("Mean predicted p_bad")
        ax.set_ylabel("Observed bad rate")
        ax.set_title(f"Reliability diagram, in-sample ({n_bins} equal-width bins)")
        ax.legend(loc="upper left")
        fig.tight_layout()
        _save_figure(fig, path, fig_paths)


def _render_challenger(
    path: Path, y: np.ndarray, p: np.ndarray, p_ch: np.ndarray, fig_paths: list[Path]
) -> None:
    import matplotlib.pyplot as plt

    fpr_p, tpr_p = _roc_points(y, p)
    fpr_c, tpr_c = _roc_points(y, p_ch)
    auc_p = metrics.roc_auc(y, p)
    auc_c = metrics.roc_auc(y, p_ch)
    with plt.rc_context(_RC):  # type: ignore[arg-type]  # keys are valid rcParams
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        ax.plot(fpr_p, tpr_p, color=_PRIMARY_COLOR, label=f"primary (AUC {auc_p:.4f})")
        ax.plot(
            fpr_c, tpr_c, color=_CHALLENGER_COLOR,
            label=f"challenger, out-of-fold (AUC {auc_c:.4f})",
        )
        ax.plot([0, 1], [0, 1], "--", color=_REFERENCE_COLOR, linewidth=1.0, label="chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("Primary (in-sample) vs challenger (out-of-fold)")
        ax.legend(loc="lower right")
        fig.tight_layout()
        _save_figure(fig, path, fig_paths)
