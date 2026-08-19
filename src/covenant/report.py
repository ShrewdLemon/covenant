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

Nothing here claims compliance: Covenant produces evidence; your
validators and auditors decide.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

from covenant import metrics
from covenant.challenger import challenger_scores
from covenant.checks.monotonicity import run_monotonicity_check
from covenant.checks.reason_codes import CheckSetupError
from covenant.hashing import sha256_bytes, sha256_canonical, sha256_dataframe, sha256_file
from covenant.model import CovenantModel, library_versions, load_model
from covenant.registry import load_covenants, load_data
from covenant.schema import ModelCovenants, ReportConfig

matplotlib.use("Agg")  # fixed headless backend; pyplot is imported after this, lazily

_HASH_PLACEHOLDER = "0" * 64
_EVIDENCE_LINE = "Covenant produces evidence; your validators and auditors decide."
_MAPPING = "docs/MAPPING.md"

# Deterministic rendering: fixed dpi and font so identical inputs give
# identical PNG bytes; savefig() below also strips the Software metadata
# matplotlib would otherwise stamp with its version string.
_RC = {
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
) -> Path:
    """Render ``out_dir/report.md`` plus ``out_dir/figures/*.png``.

    Returns the path to ``report.md``. Deterministic by construction: no
    timestamps, all randomness through ``numpy.random.default_rng`` seeded
    from ``report.random_state``, fixed matplotlib rcParams, stripped PNG
    metadata. ``out_dir`` is created if missing and existing files are
    overwritten; the destination never affects the rendered bytes.
    """
    covenants: ModelCovenants = load_covenants(covenants_path)
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
    if holdout_path is not None:
        holdout_raw = load_data(holdout_path)
        holdout = holdout_raw.reset_index(drop=True)
        missing_h = [f for f in features if f not in holdout.columns]
        if missing_h:
            raise CheckSetupError(f"holdout lacks declared features: {missing_h}")
        holdout[numeric] = holdout[numeric].astype(float)

    out = Path(out_dir)
    figures_dir = out / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig_paths: list[Path] = []

    sections = [
        _discrimination_section(y, p, config, figures_dir, fig_paths),
        _calibration_section(y, p, config, figures_dir, fig_paths),
        _stability_section(p, data, holdout, model, numeric, config),
        _drift_section(p, data, config),
        _monotonicity_section(model_path, data_path, covenants_path),
        _challenger_section(y, p, data, features, categorical, config, figures_dir, fig_paths),
    ]
    header = _header_section(
        covenants, config, model_path, raw, holdout_raw, fig_paths
    )
    footer = _footer_section(model_path, data_path, covenants_path, holdout_path, config_overrides)

    body = "\n\n".join([header, *sections, footer]) + "\n"
    digest = sha256_bytes(body.encode("utf-8"))
    final = body.replace(
        f"report_sha256: {_HASH_PLACEHOLDER}", f"report_sha256: {digest}", 1
    )
    path = out / "report.md"
    path.write_bytes(final.encode("utf-8"))
    return path


def _validated_target(data: pd.DataFrame, target: str) -> np.ndarray:
    if target not in data.columns:
        raise CheckSetupError(
            f"report.target_column {target!r} is not a column of the snapshot; "
            "set report.target_column to the 0/1 outcome column present in the "
            "data (1 = bad)"
        )
    y = pd.to_numeric(data[target]).to_numpy(dtype=float)
    labels = np.unique(y)
    if not np.isin(labels, (0.0, 1.0)).all():
        raise CheckSetupError(
            f"report.target_column {target!r} must contain only 0/1 labels "
            f"(1 = bad), got values {labels[:5].tolist()}"
        )
    if len(labels) < 2:
        raise CheckSetupError(
            f"report.target_column {target!r} has a single class; discrimination "
            "metrics need both goods and bads in the snapshot"
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
    fig_paths: list[Path],
) -> str:
    versions = ", ".join(f"{k} {v}" for k, v in sorted(library_versions().items()))
    rows = [
        ("Model name", f"`{covenants.model_name}`"),
        ("Model file sha256", f"`{sha256_file(model_path)}`"),
        ("Data snapshot sha256", f"`{sha256_dataframe(raw)}`"),
        ("Covenants sha256", f"`{sha256_canonical(covenants.model_dump(mode='json'))}`"),
    ]
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


def _discrimination_section(
    y: np.ndarray,
    p: np.ndarray,
    config: ReportConfig,
    figures_dir: Path,
    fig_paths: list[Path],
) -> str:
    boot = {"n_boot": config.n_bootstrap, "seed": config.random_state}
    rows = [
        ("AUC", *metrics.bootstrap_ci(metrics.roc_auc, y, p, **boot)),
        ("Gini", *metrics.bootstrap_ci(metrics.gini, y, p, **boot)),
        ("KS", *metrics.bootstrap_ci(metrics.ks_statistic, y, p, **boot)),
    ]
    _render_roc(figures_dir / "roc.png", y, p, fig_paths)
    lines = [
        "## 2. Discrimination",
        "",
        f"Point estimates with 95% bootstrap confidence intervals "
        f"({config.n_bootstrap} seeded resamples, seed {config.random_state}). "
        "FINRA's Model Validation Toolkit frames small samples as a credibility "
        "problem for point estimates; the interval is the honest number.",
        "",
        "| Metric | Point | CI lower | CI upper |",
        "|---|---|---|---|",
        *[f"| {name} | {_f(v)} | {_f(lo)} | {_f(hi)} |" for name, v, lo, hi in rows],
        "",
        "![ROC curve](figures/roc.png)",
        "",
        "Maps to: the outcome-analysis and ongoing-monitoring asks — discrimination "
        f"measured with intervals, not bare points ({_MAPPING}).",
    ]
    return "\n".join(lines)


def _calibration_section(
    y: np.ndarray,
    p: np.ndarray,
    config: ReportConfig,
    figures_dir: Path,
    fig_paths: list[Path],
) -> str:
    def ece_fn(y_: np.ndarray, p_: np.ndarray) -> float:
        return metrics.ece(y_, p_, n_bins=config.n_bins)

    boot = {"n_boot": config.n_bootstrap, "seed": config.random_state}
    rows = [
        ("Brier", *metrics.bootstrap_ci(metrics.brier, y, p, **boot)),
        (f"ECE ({config.n_bins} bins)", *metrics.bootstrap_ci(ece_fn, y, p, **boot)),
    ]
    _render_calibration(figures_dir / "calibration.png", y, p, config.n_bins, fig_paths)
    lines = [
        "## 3. Calibration",
        "",
        f"Brier score and expected calibration error over {config.n_bins} "
        "equal-width probability bins, each with a 95% bootstrap confidence "
        f"interval ({config.n_bootstrap} seeded resamples, seed "
        f"{config.random_state}).",
        "",
        "| Metric | Point | CI lower | CI upper |",
        "|---|---|---|---|",
        *[f"| {name} | {_f(v)} | {_f(lo)} | {_f(hi)} |" for name, v, lo, hi in rows],
        "",
        "![Reliability diagram](figures/calibration.png)",
        "",
        "Maps to: the outcome-analysis ask — predicted probabilities compared with "
        f"observed outcomes ({_MAPPING}).",
    ]
    return "\n".join(lines)


def _stability_section(
    p: np.ndarray,
    data: pd.DataFrame,
    holdout: pd.DataFrame | None,
    model: CovenantModel,
    numeric: list[str],
    config: ReportConfig,
) -> str:
    lines = ["## 4. Stability", ""]
    if holdout is None:
        lines += [
            "A single snapshot was supplied, so train-to-holdout stability cannot "
            "be measured — one sample has nothing to be stable against. Pass a "
            "holdout snapshot (CLI: `--holdout`) to render the score PSI and the "
            "per-feature CSI table.",
        ]
    else:
        p_hold = model.p_bad(holdout)
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
    lines = ["## 5. Drift", ""]
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


def _monotonicity_section(
    model_path: str | Path, data_path: str | Path, covenants_path: str | Path
) -> str:
    lines = ["## 6. Monotonicity", ""]
    try:
        record = run_monotonicity_check(model_path, data_path, covenants_path)
    except CheckSetupError as err:
        lines += [
            f"Not run: {err}",
            "",
            "Maps to: the conceptual-soundness ask — declared behaviour tested "
            f"against measured behaviour ({_MAPPING}).",
        ]
        return "\n".join(lines)

    verdict = "PASS" if record.passed else "BREACH"
    lines += [
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
        "",
        "Maps to: the conceptual-soundness ask — declared behaviour tested "
        f"against measured behaviour ({_MAPPING}).",
    ]
    return "\n".join(lines)


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
    lines = ["## 7. Challenger", ""]
    if not config.challenger:
        lines += [
            "Disabled in the report config (report.challenger: false); no "
            "challenger comparison is rendered.",
        ]
    else:
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
    config_overrides: dict | None,
) -> str:
    command = (
        f"covenant report {Path(model_path).name} {Path(data_path).name} "
        f"--covenants {Path(covenants_path).name}"
    )
    if holdout_path is not None:
        command += f" --holdout {Path(holdout_path).name}"
    if config_overrides and config_overrides.get("target_column"):
        command += f" --target {config_overrides['target_column']}"
    lines = [
        "## 8. Reproduce this report",
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


def _save_figure(fig: object, path: Path, fig_paths: list[Path]) -> None:
    import matplotlib.pyplot as plt

    fig.savefig(path, metadata={"Software": None})
    plt.close(fig)
    fig_paths.append(path)


def _render_roc(path: Path, y: np.ndarray, p: np.ndarray, fig_paths: list[Path]) -> None:
    import matplotlib.pyplot as plt

    fpr, tpr = _roc_points(y, p)
    auc = metrics.roc_auc(y, p)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        ax.plot(fpr, tpr, color=_PRIMARY_COLOR, label=f"primary (AUC {auc:.4f})")
        ax.plot([0, 1], [0, 1], "--", color=_REFERENCE_COLOR, linewidth=1.0, label="chance")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.set_title("ROC curve")
        ax.legend(loc="lower right")
        fig.tight_layout()
        _save_figure(fig, path, fig_paths)


def _render_calibration(
    path: Path, y: np.ndarray, p: np.ndarray, n_bins: int, fig_paths: list[Path]
) -> None:
    import matplotlib.pyplot as plt

    mean_p, mean_y = _reliability_points(y, p, n_bins)
    with plt.rc_context(_RC):
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        ax.plot([0, 1], [0, 1], "--", color=_REFERENCE_COLOR, linewidth=1.0, label="perfect")
        ax.plot(mean_p, mean_y, marker="o", color=_PRIMARY_COLOR, label="model")
        ax.set_xlabel("Mean predicted p_bad")
        ax.set_ylabel("Observed bad rate")
        ax.set_title(f"Reliability diagram ({n_bins} equal-width bins)")
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
    with plt.rc_context(_RC):
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
