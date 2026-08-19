"""Check 3: does the model consume the features the covenant declares —
no more and no fewer?

The covenant's feature list is the inventory's statement of what the model
depends on, and supervisors expect that statement to be inspectable and
true of the artefact (SR 26-2 expects the inventory to give
enterprise-level visibility of dependencies; RBI FREE-AI, 2025, expects
inventories to stand up to supervisory inspection). Two failure modes
drift the record away from the model: an input the documentation never
mentions, and a documented feature the model cannot see — a claim about
behaviour the artefact cannot have.

Two comparisons, with different weight:

* **structural** (breach) — the estimator's own ``feature_names_in_``
  against the declared list, in both directions. Exact where the estimator
  records its inputs; recorded as unavailable in the details, never
  silently skipped, where it does not.
* **attribution screen** (warning) — mean |SHAP attribution| of each
  declared feature over a seeded sample of the snapshot. A feature below
  ``dead_feature_epsilon`` is *dead*: documented but measurably inert.
  Post-hoc attributions are an approximation, not ground truth (Sudjianto
  & Zhang, 2021), and sensitive to the background choice (Pace Analytics,
  2024), so a dead feature is surfaced as a documentation-quality warning,
  never a breach — the covenant's claim is unsupported by measured
  behaviour, not contradicted by it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from covenant.attribution import measured_attributions, sample_background
from covenant.checks.base import CheckRecord
from covenant.checks.reason_codes import CheckSetupError
from covenant.hashing import sha256_canonical, sha256_dataframe, sha256_file
from covenant.model import CovenantModel, load_model
from covenant.registry import load_covenants, load_data
from covenant.schema import FeaturesCheckConfig, ModelCovenants

CHECK_NAME = "features"


def run_features_check(
    model_path: str | Path,
    data_path: str | Path,
    covenants_path: str | Path,
    config_overrides: dict | None = None,
) -> CheckRecord:
    covenants: ModelCovenants = load_covenants(covenants_path)
    config: FeaturesCheckConfig = covenants.checks.features
    if config_overrides:
        config = config.model_copy(update=config_overrides)

    data = load_data(data_path).reset_index(drop=True)
    declared = covenants.feature_names()
    categorical = covenants.categorical_features()

    estimator = load_model(model_path)

    # Structural comparison: the estimator's own record of its inputs (on a
    # pipeline, feature_names_in_ is the raw columns its first step was fed).
    raw_inputs = getattr(estimator, "feature_names_in_", None)
    structural_available = raw_inputs is not None
    if structural_available:
        model_inputs = [str(c) for c in raw_inputs]
        undocumented_used = [c for c in model_inputs if c not in declared]
        declared_unused = [f for f in declared if f not in model_inputs]
        score_features = model_inputs
    else:
        undocumented_used: list[str] = []
        declared_unused: list[str] = []
        score_features = declared

    missing = [c for c in score_features if c not in data.columns]
    if missing:
        raise CheckSetupError(
            f"the data snapshot lacks columns the model needs to score: "
            f"{missing}; add them to the snapshot so the attribution screen "
            "can run"
        )
    numeric = [f for f in declared if f not in categorical and f in data.columns]
    data[numeric] = data[numeric].astype(float)

    # Attribution screen. The model is bound to its real inputs when they are
    # known — a model using undocumented columns could not score over the
    # declared list alone — and the declared features are read back off the
    # attribution result.
    model = CovenantModel(estimator, score_features, positive_class=covenants.positive_class)
    rng = np.random.default_rng(config.random_state)
    if len(data) > config.sample_size:
        idx = np.sort(rng.choice(len(data), size=config.sample_size, replace=False))
    else:
        idx = np.arange(len(data))
    X = data.iloc[idx][score_features].reset_index(drop=True)
    background = sample_background(
        data, score_features, config.background_size, config.random_state
    )
    attribution_categorical = [c for c in score_features if c in categorical] + [
        c for c in undocumented_used if not pd.api.types.is_numeric_dtype(data[c])
    ]
    attributions = measured_attributions(
        model, X, background, attribution_categorical, config.random_state
    )
    mean_abs = attributions.abs().mean()
    dead_features = [
        {"feature": f, "mean_abs_attribution": round(float(mean_abs[f]), 6)}
        for f in declared
        if f in attributions.columns and float(mean_abs[f]) < config.dead_feature_epsilon
    ]

    passed = not undocumented_used and not declared_unused

    if structural_available:
        structural_note = (
            "declared vs used was compared structurally against the "
            "estimator's feature_names_in_"
        )
    else:
        structural_note = (
            "the estimator does not record feature_names_in_, so the "
            "structural comparison is unavailable and only the attribution "
            "screen ran; it can surface inert documented features but "
            "cannot see undocumented inputs"
        )
    note = (
        structural_note
        + "; dead features are documented but measurably inert — a "
        "documentation-quality warning, not a behavioural contradiction — "
        "so they never fail the check"
    )

    record = CheckRecord(
        check=CHECK_NAME,
        model_name=covenants.model_name,
        passed=passed,
        metrics={
            "n_undocumented_used": float(len(undocumented_used)),
            "n_declared_unused": float(len(declared_unused)),
            "n_dead": float(len(dead_features)),
        },
        thresholds={"dead_feature_epsilon": config.dead_feature_epsilon},
        n_evaluated=len(X),
        inputs={
            "model_sha256": sha256_file(model_path),
            "data_sha256": sha256_dataframe(load_data(data_path)),
            "covenants_sha256": sha256_canonical(covenants.model_dump(mode="json")),
        },
        config=config.model_dump(),
        details={
            "undocumented_used": undocumented_used,
            "declared_unused": declared_unused,
            "dead_features": dead_features,
            "structural_available": structural_available,
            "note": note,
        },
    )
    return record.stamp()
