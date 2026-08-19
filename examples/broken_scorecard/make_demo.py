"""Build the broken-scorecard demo.

Part 1 — reason codes (Check 1). Fits a logistic scorecard and writes two
adverse-action coefficient tables:

* ``coefficients_live.csv`` — derived from the deployed model. Reason codes
  from this table agree with the model's measured behaviour.
* ``coefficients_stale.csv`` — a stale artefact from "a previous model
  version": two strong coefficients swapped, one zeroed. This is the
  failure mode Sei AI describes — the model that issues the decision is
  not the model that produces the reasons — and it is invisible until you
  test for it. ``covenant check reason-codes`` fails it.

Part 2 — monotonicity (Check 2). The true effect of ``loan_amount`` is
U-shaped, but ``covenants_gbm.yaml`` declares it monotone increasing —
documentation written for the old linear scorecard. Two gradient-boosted
models, one covenant:

* ``model_gbm_unconstrained.joblib`` — learns the U-shape; measured
  behaviour contradicts the declared direction → breach.
* ``model_gbm_constrained.joblib`` — fitted with ``monotonic_cst`` matching
  the covenant; behaves as documented → pass.

Deterministic: fixed seeds throughout, so CI can assert on exit codes.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
SEED = 20260818
N = 4000

FEATURES = [
    "income",
    "dti",
    "utilization",
    "delinquencies_24m",
    "inquiries_6m",
    "age_of_oldest_line_months",
    "loan_amount",
    "employment_years",
]


def make_data(rng: np.random.Generator) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "income": rng.lognormal(mean=10.8, sigma=0.5, size=N).round(0),
            "dti": np.clip(rng.normal(0.32, 0.13, N), 0.01, 0.95).round(3),
            "utilization": np.clip(rng.beta(2.0, 3.5, N), 0, 1).round(3),
            "delinquencies_24m": rng.poisson(0.6, N),
            "inquiries_6m": rng.poisson(1.2, N),
            "age_of_oldest_line_months": rng.gamma(6.0, 20.0, N).round(0),
            "loan_amount": rng.lognormal(mean=9.6, sigma=0.7, size=N).round(0),
            "employment_years": np.clip(rng.gamma(2.2, 3.0, N), 0, 45).round(1),
        }
    )

    z = (df - df.mean()) / df.std()
    # loan_amount's true effect is U-shaped: very small and very large loans
    # are risky. A linear scorecard averages it away; a GBM learns it — and
    # then contradicts documentation that declares the feature monotone.
    logit = (
        -1.1
        - 0.9 * z["income"]
        + 1.1 * z["dti"]
        + 0.9 * z["utilization"]
        + 0.7 * z["delinquencies_24m"]
        + 0.45 * z["inquiries_6m"]
        - 0.5 * z["age_of_oldest_line_months"]
        + 0.25 * z["loan_amount"]
        + 0.6 * (z["loan_amount"] ** 2 - 1)
        - 0.35 * z["employment_years"]
        + rng.normal(0, 0.9, N)
    )
    df["bad"] = (rng.uniform(size=N) < 1 / (1 + np.exp(-logit))).astype(int)
    # Excluded variable (never fed to the model) for the Check 4 proxy
    # screen, and an application month for the report's drift slices —
    # both independent of the features by construction.
    df["gender"] = rng.choice(["F", "M"], size=N)
    df["application_month"] = rng.integers(1, 13, size=N)
    df.insert(0, "application_id", [f"A{i:06d}" for i in range(N)])
    return df


def main() -> None:
    rng = np.random.default_rng(SEED)
    df = make_data(rng)

    X, y = df[FEATURES], df["bad"]
    model = Pipeline(
        [("scaler", StandardScaler()), ("logreg", LogisticRegression(max_iter=1000))]
    )
    model.fit(X, y)

    joblib.dump(model, HERE / "model.joblib")
    df.to_csv(HERE / "train.csv", index=False)

    # Live table: exactly the deployed model's linear contribution in raw
    # units — coef is the standardized coefficient, mean/scale come from the
    # fitted scaler, so coef * (x - mean) / scale reproduces the model.
    scaler: StandardScaler = model.named_steps["scaler"]
    logreg: LogisticRegression = model.named_steps["logreg"]
    live = pd.DataFrame(
        {
            "feature": FEATURES,
            "coef": logreg.coef_[0],
            "mean": scaler.mean_,
            "scale": scaler.scale_,
        }
    )
    live.to_csv(HERE / "coefficients_live.csv", index=False)

    # Stale table: reason codes from "the previous model". Swap the two
    # strongest drivers (income <-> dti, opposite signs) and zero out
    # utilization. Scores are untouched — only the explanations are wrong.
    stale = live.copy().set_index("feature")
    stale.loc["income", "coef"], stale.loc["dti", "coef"] = (
        stale.loc["dti", "coef"],
        stale.loc["income", "coef"],
    )
    stale.loc["utilization", "coef"] = 0.0
    stale.reset_index().to_csv(HERE / "coefficients_stale.csv", index=False)

    # Part 2: two GBMs against one covenant that declares every direction.
    directions = {
        "income": -1,
        "dti": 1,
        "utilization": 1,
        "delinquencies_24m": 1,
        "inquiries_6m": 1,
        "age_of_oldest_line_months": -1,
        "loan_amount": 1,  # the covenant's claim; the true effect is U-shaped
        "employment_years": -1,
    }
    unconstrained = HistGradientBoostingClassifier(random_state=SEED)
    unconstrained.fit(X, y)
    joblib.dump(unconstrained, HERE / "model_gbm_unconstrained.joblib")

    constrained = HistGradientBoostingClassifier(random_state=SEED, monotonic_cst=directions)
    constrained.fit(X, y)
    joblib.dump(constrained, HERE / "model_gbm_constrained.joblib")

    denied = (model.predict_proba(X)[:, 1] >= 0.5).mean()
    print(f"wrote model.joblib, both GBMs, train.csv and coefficient tables to {HERE}")
    print(f"denial rate at threshold 0.5: {denied:.1%}")


if __name__ == "__main__":
    main()
