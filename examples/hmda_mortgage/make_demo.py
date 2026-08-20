"""Build the HMDA mortgage demo: real fair-lending data, a real governance failure.

`data/hmda_ri_2023.csv.gz` is CFPB public HMDA data (2023, Rhode Island,
originated + denied applications; US government work, freely
redistributable): 25,273 applications, 25.7% denied, with the reported
`derived_race`, `derived_sex` and `derived_ethnicity` columns alongside the
underwriting variables.

FRAMING — read before running. The target here is ``denied``: the lender's
**historical decision**, not creditworthiness. A model fitted to it
reproduces whatever bias those historical decisions contain. That is
exactly why the governance checks matter: the covenant *excludes*
`derived_race`, `derived_sex` and `derived_ethnicity`, and Covenant
verifies that the deployed artefact honours that exclusion and surfaces
proxies for it. This demo demonstrates the verification of documented
claims; it does not endorse modelling denial decisions.

Two models, one covenant:

* ``model_leaky.joblib`` — fitted on **all** columns, the three protected
  ones included. Its scores look fine; ``covenant check features`` catches
  the undocumented inputs and ``covenant check exclusions`` measures the
  excluded columns actually driving scores — both breach.
* ``model_clean.joblib`` — fitted on the 10 documented underwriting
  features only. Every check passes, and the proxy screen still *surfaces*
  the real associations it finds (strongest: derived_sex ~ income) below
  the declared threshold — surfaced, not proven absent.

Also written: a seeded 80/20 train/holdout split of a seeded 10,000-row
working sample (plain CSV, so the report's stability section computes real
PSI/CSI in a few minutes end to end) and a production-style SHAP export
(``clean_attributions.csv``) keyed on ``application_id`` for denied
applicants, so Check 1's ``shapley`` method runs against a real artefact.

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
SEED = 2023  # the HMDA collection year

DOCUMENTED = [
    "loan_amount",
    "income",
    "property_value",
    "loan_to_value_ratio",
    "debt_to_income_ratio",
    "applicant_age",
    "loan_purpose",
    "loan_type",
    "occupancy_type",
    "derived_dwelling_category",
]
NUMERIC = ["loan_amount", "income", "property_value", "loan_to_value_ratio"]
EXCLUDED = ["derived_race", "derived_sex", "derived_ethnicity"]

# A seeded working sample keeps the permutation-SHAP export and the checks
# inside a few minutes; the full 25,273 applications stay committed.
WORKING_SAMPLE = 10_000
TRAIN_FRACTION = 0.8


def make_pipeline(features: list[str]) -> Pipeline:
    numeric_used = [c for c in features if c in NUMERIC]
    categorical_used = [c for c in features if c not in NUMERIC]
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
    df = pd.read_csv(HERE / "data" / "hmda_ri_2023.csv.gz")
    all_features = DOCUMENTED + EXCLUDED

    rng = np.random.default_rng(SEED)
    working = df.iloc[np.sort(rng.choice(len(df), size=WORKING_SAMPLE, replace=False))]
    working = working.reset_index(drop=True)
    order = rng.permutation(len(working))
    n_train = int(len(working) * TRAIN_FRACTION)
    train = working.iloc[np.sort(order[:n_train])].reset_index(drop=True)
    holdout = working.iloc[np.sort(order[n_train:])].reset_index(drop=True)
    # covenant's load_data reads .csv/.parquet, so the working split is
    # written as plain CSV; the committed source stays gzipped.
    train.to_csv(HERE / "train.csv", index=False)
    holdout.to_csv(HERE / "holdout.csv", index=False)

    leaky = make_pipeline(all_features)
    leaky.fit(train[all_features], train["denied"])
    joblib.dump(leaky, HERE / "model_leaky.joblib")

    clean = make_pipeline(DOCUMENTED)
    clean.fit(train[DOCUMENTED], train["denied"])
    joblib.dump(clean, HERE / "model_clean.joblib")

    # The covenant declares monotone directions only where the fitted model
    # supports them; print the evidence so the declaration is checkable.
    coefs = clean.named_steps["logreg"].coef_[0][: len(NUMERIC)]
    for name, coef in zip(NUMERIC, coefs, strict=False):
        print(f"clean-model coefficient (scaled space) {name}: {coef:+.4f}")

    # Production-style attribution export for Check 1's shapley method: the
    # artefact an adverse-action letter system would consume, keyed on
    # application_id, produced from the clean model with a seeded background.
    from covenant.attribution import explain, sample_background
    from covenant.model import CovenantModel

    model = CovenantModel(clean, DOCUMENTED)
    categorical = [c for c in DOCUMENTED if c not in NUMERIC]
    background = sample_background(train, DOCUMENTED, 40, 0)
    denied = train[model.p_bad(train) >= 0.5].reset_index(drop=True)
    attributions, path = explain(
        model, denied[DOCUMENTED], background, categorical, 0, npermutations=2
    )
    export = attributions.round(6)
    export.insert(0, "application_id", denied["application_id"].to_numpy())
    export.to_csv(HERE / "clean_attributions.csv", index=False)

    denial_rate = float(train["denied"].mean())
    print(f"wrote both models, train/holdout split and the attribution export to {HERE}")
    print(
        f"attribution export: {len(denied)} denied applicants via {path}; "
        f"train denial rate {denial_rate:.3f}"
    )


if __name__ == "__main__":
    main()
