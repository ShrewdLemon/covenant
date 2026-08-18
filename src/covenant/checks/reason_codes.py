"""Check 1: do the adverse-action reason codes the model would send a denied
applicant agree with the features that measurably drove the decision?

Declared side: the covenant's reason-code method (the production artefact —
a coefficient table, a reasons export). Measured side: SHAP attributions of
``p_bad``. Consistency is FinRegLab's sense of the word: the degree to which
the two identify the same drivers. Disagreement concentrated near the
decision boundary is expected (Krivorotov & Richey, 2022), so the record
stratifies by score band rather than reporting one number and hiding where
it comes from.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from covenant.attribution import (
    declared_reason_sets,
    measured_attributions,
    sample_background,
    top_1,
    top_k_sets,
)
from covenant.checks.base import CheckRecord
from covenant.hashing import sha256_canonical, sha256_dataframe, sha256_file
from covenant.model import CovenantModel, load_model
from covenant.registry import load_covenants, load_data
from covenant.schema import ModelCovenants, ReasonCodeCheckConfig

CHECK_NAME = "reason-codes"

BACKGROUND_SENSITIVITY_FLOOR = 0.8


class CheckSetupError(ValueError):
    pass


def jaccard(a: frozenset, b: frozenset) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def run_reason_code_check(
    model_path: str | Path,
    data_path: str | Path,
    covenants_path: str | Path,
    config_overrides: dict | None = None,
) -> CheckRecord:
    covenants: ModelCovenants = load_covenants(covenants_path)
    config: ReasonCodeCheckConfig = covenants.checks.reason_codes
    if config_overrides:
        config = config.model_copy(update=config_overrides)

    data = load_data(data_path).reset_index(drop=True)
    features = covenants.feature_names()
    categorical = covenants.categorical_features()
    missing = [f for f in features if f not in data.columns]
    if missing:
        raise CheckSetupError(f"data lacks declared features: {missing}")
    if config.id_column and config.id_column not in data.columns:
        raise CheckSetupError(
            f"checks.reason_codes.id_column {config.id_column!r} is not a "
            "column of the data snapshot"
        )
    numeric = [f for f in features if f not in categorical]
    data[numeric] = data[numeric].astype(float)

    estimator = load_model(model_path)
    model = CovenantModel(estimator, features, positive_class=covenants.positive_class)

    p_bad = model.p_bad(data)
    denied_idx = np.flatnonzero(p_bad >= config.decision_threshold)
    if len(denied_idx) < 10:
        raise CheckSetupError(
            f"only {len(denied_idx)} applicants score above the decision "
            f"threshold {config.decision_threshold}; too few to evaluate "
            "reason codes. Lower checks.reason_codes.decision_threshold."
        )
    rng = np.random.default_rng(config.random_state)
    if len(denied_idx) > config.max_denied_sample:
        denied_idx = np.sort(
            rng.choice(denied_idx, size=config.max_denied_sample, replace=False)
        )
    X_denied = data.loc[denied_idx, features]
    p_denied = p_bad[denied_idx]

    # Declared side: what the production pipeline would tell the applicant.
    ids = data.loc[denied_idx, config.id_column] if config.id_column else None
    declared_sets, declared_top1 = declared_reason_sets(
        covenants.reason_codes,
        X_denied,
        Path(covenants_path).resolve().parent,
        ids=ids,
        id_column=config.id_column,
    )

    # Measured side: SHAP under a seeded background, plus a second background
    # so the record reports how much the measured side itself moves.
    k = covenants.reason_codes.top_k
    background_a = sample_background(data, features, config.background_size, config.random_state)
    attributions = measured_attributions(
        model, X_denied, background_a, categorical, config.random_state
    )
    measured_sets = top_k_sets(attributions, k)
    measured_top1 = top_1(attributions)

    background_b = sample_background(
        data, features, config.background_size, config.random_state + 1
    )
    attributions_b = measured_attributions(
        model, X_denied, background_b, categorical, config.random_state + 1
    )
    measured_sets_b = top_k_sets(attributions_b, k)
    background_jaccard = float(
        np.mean([jaccard(a, b) for a, b in zip(measured_sets, measured_sets_b, strict=True)])
    )

    row_jaccard = np.array(
        [jaccard(d, m) for d, m in zip(declared_sets, measured_sets, strict=True)]
    )
    top1_hits = np.array([d == m for d, m in zip(declared_top1, measured_top1, strict=True)])
    top1_agreement = float(top1_hits.mean())
    topk_jaccard = float(row_jaccard.mean())

    strata = _stratify(p_denied, top1_hits, row_jaccard)
    worst = _worst_disagreements(
        denied_idx, p_denied, declared_sets, measured_sets, row_jaccard
    )

    passed = (
        top1_agreement >= config.min_top1_agreement
        and topk_jaccard >= config.min_topk_jaccard
    )

    record = CheckRecord(
        check=CHECK_NAME,
        model_name=covenants.model_name,
        passed=passed,
        metrics={
            "top1_agreement": round(top1_agreement, 4),
            "topk_jaccard": round(topk_jaccard, 4),
            "background_jaccard": round(background_jaccard, 4),
        },
        thresholds={
            "min_top1_agreement": config.min_top1_agreement,
            "min_topk_jaccard": config.min_topk_jaccard,
        },
        n_evaluated=len(denied_idx),
        inputs={
            "model_sha256": sha256_file(model_path),
            "data_sha256": sha256_dataframe(load_data(data_path)),
            "covenants_sha256": sha256_canonical(covenants.model_dump(mode="json")),
        },
        config={
            **config.model_dump(),
            "declared_method": covenants.reason_codes.method.value,
            "top_k": k,
        },
        details={
            "by_score_band": strata,
            "worst_disagreements": worst,
            "background_sensitive": background_jaccard < BACKGROUND_SENSITIVITY_FLOOR,
            "note": (
                "measured side is a SHAP approximation of the model, not ground "
                "truth; background_jaccard reports how stable it is across two "
                "seeded backgrounds"
            ),
        },
    )
    return record.stamp()


def _stratify(
    p_denied: np.ndarray, top1_hits: np.ndarray, row_jaccard: np.ndarray, bands: int = 5
) -> list[dict]:
    """Agreement by score band among denied applicants, lowest scores first.

    The lowest band sits nearest the decision boundary, where disagreement
    is expected to concentrate."""
    edges = np.quantile(p_denied, np.linspace(0, 1, bands + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        edges = np.array([p_denied.min(), p_denied.max()])
    out = []
    labels = np.clip(np.searchsorted(edges, p_denied, side="right") - 1, 0, len(edges) - 2)
    for band in range(len(edges) - 1):
        mask = labels == band
        if not mask.any():
            continue
        out.append(
            {
                "p_bad_range": [round(float(edges[band]), 4), round(float(edges[band + 1]), 4)],
                "n": int(mask.sum()),
                "top1_agreement": round(float(top1_hits[mask].mean()), 4),
                "topk_jaccard": round(float(row_jaccard[mask].mean()), 4),
            }
        )
    return out


def _worst_disagreements(
    denied_idx: np.ndarray,
    p_denied: np.ndarray,
    declared_sets: list[frozenset],
    measured_sets: list[frozenset],
    row_jaccard: np.ndarray,
    limit: int = 10,
) -> list[dict]:
    order = np.argsort(row_jaccard, kind="stable")[:limit]
    rows = []
    for i in order:
        if row_jaccard[i] == 1.0:
            break
        rows.append(
            {
                "row": int(denied_idx[i]),
                "p_bad": round(float(p_denied[i]), 4),
                "declared": sorted(declared_sets[i]),
                "measured": sorted(measured_sets[i]),
                "jaccard": round(float(row_jaccard[i]), 4),
            }
        )
    return rows
