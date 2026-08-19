"""Build the German Credit demo: real data, a real governance failure.

The Statlog German Credit dataset (Hofmann, 1994; UCI / OpenML "credit-g",
1,000 applications) ships with two columns no modern lender may score:
``personal_status`` — which encodes **sex and marital status** — and
``foreign_worker``. ``data/german_credit.csv`` is a verbatim copy of the
OpenML frame (plus an ``application_id`` and the target recoded to
``bad`` 0/1) committed to the repo so the demo is deterministic and
offline; see the README for provenance.

Two models, one covenant:

* ``model_leaky.joblib`` — fitted on **all** columns, the two protected
  ones included. Its scores look fine; nothing about its accuracy betrays
  what it consumes. ``covenant check features`` catches the undocumented
  inputs and ``covenant check exclusions`` measures the excluded columns
  actually driving scores — both breach.
* ``model_clean.joblib`` — fitted on the 18 documented features only.
  Every check passes, and the proxy screen still *surfaces* the real
  associations it finds (personal_status ~ num_dependents at 0.28, ~ age
  at 0.25) below the declared threshold — surfaced, not proven absent.

Also written: a train/holdout split (800/200, seeded) so the report's
stability section computes real PSI/CSI, and a production-style SHAP
export (``clean_attributions.csv``) so Check 1's ``shapley`` method runs
against a real artefact.

Deterministic: fixed seed, committed data, no network access.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).parent
SEED = 19940601  # Statlog's year, arbitrarily

EXCLUDED = ["personal_status", "foreign_worker"]


def make_pipeline(features: list[str], numeric: list[str]) -> Pipeline:
    numeric_used = [c for c in features if c in numeric]
    categorical_used = [c for c in features if c not in numeric]
    return Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", StandardScaler(), numeric_used),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_used),
                    ]
                ),
            ),
            ("logreg", LogisticRegression(max_iter=2000)),
        ]
    )


def main() -> None:
    df = pd.read_csv(HERE / "data" / "german_credit.csv")
    feature_columns = [c for c in df.columns if c not in ("application_id", "bad")]
    clean_features = [c for c in feature_columns if c not in EXCLUDED]
    numeric = [c for c in feature_columns if pd.api.types.is_numeric_dtype(df[c])]

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(df))
    train = df.iloc[np.sort(order[:800])].reset_index(drop=True)
    holdout = df.iloc[np.sort(order[800:])].reset_index(drop=True)
    train.to_csv(HERE / "train.csv", index=False)
    holdout.to_csv(HERE / "holdout.csv", index=False)

    leaky = make_pipeline(feature_columns, numeric)
    leaky.fit(train[feature_columns], train["bad"])
    joblib.dump(leaky, HERE / "model_leaky.joblib")

    clean = make_pipeline(clean_features, numeric)
    clean.fit(train[clean_features], train["bad"])
    joblib.dump(clean, HERE / "model_clean.joblib")

    # Production-style attribution export for Check 1's shapley method:
    # the artefact a real letter-generation system would consume, keyed on
    # application_id, produced from the clean model with a seeded background.
    from covenant.attribution import explain, sample_background
    from covenant.model import CovenantModel

    model = CovenantModel(clean, clean_features)
    categorical = [c for c in clean_features if c not in numeric]
    background = sample_background(train, clean_features, 40, 0)
    denied = train[model.p_bad(train) >= 0.5].reset_index(drop=True)
    attributions, path = explain(
        model, denied[clean_features], background, categorical, 0, npermutations=2
    )
    export = attributions.round(6)
    export.insert(0, "application_id", denied["application_id"].to_numpy())
    export.to_csv(HERE / "clean_attributions.csv", index=False)

    bad_rate = leaky.predict_proba(train[feature_columns])[:, 1].mean()
    print(f"wrote both models, train/holdout split and the attribution export to {HERE}")
    print(f"attribution export path: {path}; mean p_bad (leaky, train): {bad_rate:.3f}")


if __name__ == "__main__":
    main()
