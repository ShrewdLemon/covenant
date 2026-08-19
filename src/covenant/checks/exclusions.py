"""Check 4: does the model keep its hands off what the covenant excludes?

An exclusion is a testable claim — "this variable plays no role in
scoring" — and it can fail two ways: the variable still reaches the model,
or a retained feature carries its signal. Both are screened:

* **attribution screen** — when an excluded variable is among the model's
  actual inputs (``feature_names_in_`` when the estimator records them,
  the declared features otherwise), its mean |SHAP attribution| of
  ``p_bad`` must stay below a stated threshold. Attributions are an
  approximation, not ground truth (Sudjianto & Zhang, 2021), and are
  background-sensitive (Pace Analytics, 2024), so the sample and
  background are seeded parameters recorded in the config.
* **proxy screen** — pairwise association between the excluded variable
  and every declared feature (|Spearman|, correlation ratio, or
  bias-corrected Cramer's V by dtype; cf. the eta-squared screens of
  arXiv 2511.03807), flagged above a stated threshold. Proxies are
  surfaced, not proven absent: a weak pairwise association cannot rule
  out a multivariate proxy.

When the estimator records no input names and the excluded variable is not
a declared feature, whether it reaches the model cannot be verified from
the artefact; the record says so instead of guessing either way.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from covenant.association import association
from covenant.attribution import explain, sample_background
from covenant.checks.base import CheckRecord
from covenant.checks.reason_codes import CheckSetupError
from covenant.hashing import sha256_canonical, sha256_dataframe, sha256_file
from covenant.model import CovenantModel, load_model
from covenant.registry import load_covenants, load_data
from covenant.schema import ExclusionsCheckConfig, ModelCovenants

CHECK_NAME = "exclusions"


def run_exclusions_check(
    model_path: str | Path,
    data_path: str | Path,
    covenants_path: str | Path,
    config_overrides: dict | None = None,
) -> CheckRecord:
    covenants: ModelCovenants = load_covenants(covenants_path)
    config: ExclusionsCheckConfig = covenants.checks.exclusions
    if config_overrides:
        config = config.model_copy(update=config_overrides)

    if not covenants.excluded:
        raise CheckSetupError(
            "the covenant lists no excluded variables; nothing to check. "
            "Declare them under excluded: (name + reason), or skip the "
            "exclusions check for this model."
        )

    data = load_data(data_path).reset_index(drop=True)
    features = covenants.feature_names()
    categorical = covenants.categorical_features()
    missing = [f for f in features if f not in data.columns]
    if missing:
        raise CheckSetupError(f"data lacks declared features: {missing}")
    numeric = [f for f in features if f not in categorical]
    data[numeric] = data[numeric].astype(float)

    estimator = load_model(model_path)
    raw_inputs = getattr(estimator, "feature_names_in_", None)
    model_inputs = None if raw_inputs is None else [str(c) for c in raw_inputs]

    # Where each excluded variable stands relative to the model's inputs.
    # feature_names_in_ is authoritative when present — and consulted before
    # snapshot presence, so an excluded variable the model reads but the
    # snapshot omits is a hard setup error (the attribution screen needs the
    # column), never a silent "absent" pass. The declared feature list
    # stands in when the estimator records no input names; with neither,
    # usage is unverifiable.
    reach: dict[str, str] = {}
    for excluded in covenants.excluded:
        if model_inputs is not None and excluded.name in model_inputs:
            reach[excluded.name] = "yes"
        elif excluded.name not in data.columns:
            reach[excluded.name] = "absent"
        elif model_inputs is not None:
            reach[excluded.name] = "no"
        elif excluded.name in features:
            reach[excluded.name] = "yes"
        else:
            reach[excluded.name] = "unknown"

    # One seeded attribution pass covers every excluded variable that
    # reaches the model, with CovenantModel bound to the model's real input
    # list so scoring sees exactly the columns the estimator was fitted on.
    attributions: pd.DataFrame | None = None
    attribution_path: str | None = None
    n_attributed = 0
    if any(status == "yes" for status in reach.values()):
        input_features = model_inputs if model_inputs is not None else features
        missing_inputs = [c for c in input_features if c not in data.columns]
        if missing_inputs:
            raise CheckSetupError(
                f"the model's inputs {missing_inputs} are not columns of the "
                "data snapshot; the attribution screen needs every model "
                "input present to score"
            )
        model = CovenantModel(
            estimator, input_features, positive_class=covenants.positive_class
        )
        cat_columns = [
            c
            for c in input_features
            if c in categorical or not pd.api.types.is_numeric_dtype(data[c])
        ]
        X = sample_background(data, input_features, config.sample_size, config.random_state)
        background = sample_background(
            data, input_features, config.background_size, config.random_state + 1
        )
        attributions, attribution_path = explain(
            model, X, background, cat_columns, config.random_state
        )
        n_attributed = len(X)

    by_variable: list[dict] = []
    flagged_pairs: list[dict] = []
    attribution_breach = False
    max_attribution = 0.0
    max_association = 0.0
    n_pairs = 0

    for excluded in covenants.excluded:
        status = reach[excluded.name]
        entry: dict = {"name": excluded.name, "reason": excluded.reason}
        if status == "absent":
            entry["in_snapshot"] = False
            entry["reaches_model"] = "no"
            entry["note"] = "not present in snapshot; nothing to screen"
            by_variable.append(entry)
            continue

        entry["in_snapshot"] = True

        # Proxy screen: association against every declared feature, flagged
        # above the threshold — whether or not the variable reaches the model.
        variable_flags: list[dict] = []
        for feature in features:
            if feature == excluded.name:
                continue
            strength, method = association(data[excluded.name], data[feature])
            n_pairs += 1
            max_association = max(max_association, strength)
            if strength > config.max_association:
                pair = {
                    "excluded": excluded.name,
                    "feature": feature,
                    "strength": round(strength, 4),
                    "method": method,
                }
                variable_flags.append(pair)
                flagged_pairs.append(pair)
        entry["n_proxy_flags"] = len(variable_flags)

        if status == "yes":
            entry["reaches_model"] = "yes"
            assert attributions is not None  # computed above for this status
            # Threshold on the excluded variable's *share* of total mean
            # |attribution| mass: scale-invariant across attribution paths
            # (logit-space linear-exact vs probability-space permutation).
            mean_abs = float(attributions[excluded.name].abs().mean())
            total_mass = float(attributions.abs().mean().sum())
            share = mean_abs / total_mass if total_mass > 0 else 0.0
            max_attribution = max(max_attribution, share)
            breach = share >= config.max_excluded_attribution
            entry["mean_abs_attribution"] = round(mean_abs, 6)
            entry["attribution_share"] = round(share, 6)
            entry["attribution_breach"] = breach
            if breach:
                attribution_breach = True
                entry["note"] = "the covenant excludes it; the model measurably uses it"
        elif status == "no":
            entry["reaches_model"] = "no"
        else:
            entry["reaches_model"] = "unknown"
            entry["note"] = (
                f"the estimator records no feature_names_in_ and "
                f"{excluded.name!r} is not a declared feature; whether it "
                "reaches the model cannot be verified from the artefact"
            )
        by_variable.append(entry)

    passed = not attribution_breach and (not flagged_pairs or not config.fail_on_proxies)

    record = CheckRecord(
        check=CHECK_NAME,
        model_name=covenants.model_name,
        passed=passed,
        metrics={
            "max_excluded_attribution_observed": round(max_attribution, 6),
            "max_association_observed": round(max_association, 4),
            "n_proxy_flags": float(len(flagged_pairs)),
        },
        thresholds={
            "max_excluded_attribution": config.max_excluded_attribution,
            "max_association": config.max_association,
        },
        n_evaluated=n_pairs + n_attributed,
        inputs={
            "model_sha256": sha256_file(model_path),
            "data_sha256": sha256_dataframe(load_data(data_path)),
            "covenants_sha256": sha256_canonical(covenants.model_dump(mode="json")),
        },
        config=config.model_dump(),
        details={
            "by_variable": by_variable,
            "flagged_pairs": flagged_pairs,
            "attribution_path": attribution_path,
            "note": (
                "the proxy screen reports pairwise association between "
                "excluded variables and declared features: proxies are "
                "surfaced, not proven absent, and a weak pairwise "
                "association does not rule out a multivariate proxy. "
                "Covenant produces evidence; validators decide."
            ),
        },
    )
    return record.stamp()
