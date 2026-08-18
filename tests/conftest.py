from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

FEATURES = ["income", "dti", "utilization", "delinquencies", "inquiries"]
COEFS = np.array([-1.0, 1.2, 0.9, 0.7, 0.4])

COVENANTS_YAML = """\
covenant_schema: 1
model_name: test-scorecard
features:
  - {{name: income, direction: decreases_risk}}
  - {{name: dti, direction: increases_risk}}
  - {{name: utilization, direction: increases_risk}}
  - {{name: delinquencies, direction: increases_risk}}
  - {{name: inquiries, direction: increases_risk}}
excluded:
  - {{name: gender, reason: protected attribute}}
reason_codes:
  method: difference_from_mean
  top_k: 3
  parameters:
    coefficients: {coefficients}
checks:
  reason_codes:
    min_top1_agreement: 0.75
    min_topk_jaccard: 0.60
    decision_threshold: 0.5
    max_denied_sample: 50
    background_size: 40
"""

GOVERNANCE_YAML = """\
owner:
  name: Test Owner
  email: test@example.com
intended_use: Unit-test scorecard for the covenant test suite.
limitations: [synthetic data only]
materiality:
  tier: 3
  justification: Test fixture with no exposure whatsoever; lowest tier.
review_date: 2027-01-01
vendor: null
"""


def _make_frame(rng: np.random.Generator, n: int) -> pd.DataFrame:
    X = pd.DataFrame(rng.normal(size=(n, len(FEATURES))), columns=FEATURES)
    logit = -0.4 + X.to_numpy() @ COEFS + rng.normal(0, 0.7, n)
    X["bad"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-logit))).astype(int)
    return X


@pytest.fixture(scope="session")
def fitted(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """A fitted logistic scorecard with data, live and stale coefficient
    tables, covenants (good and broken) and governance, all on disk."""
    root = tmp_path_factory.mktemp("fixture")
    rng = np.random.default_rng(7)
    df = _make_frame(rng, 800)

    model = LogisticRegression(max_iter=1000).fit(df[FEATURES], df["bad"])
    model_path = root / "model.joblib"
    joblib.dump(model, model_path)
    data_path = root / "train.csv"
    df.to_csv(data_path, index=False)

    live = pd.DataFrame(
        {"feature": FEATURES, "coef": model.coef_[0], "mean": df[FEATURES].mean().to_numpy()}
    )
    live.to_csv(root / "coefficients_live.csv", index=False)

    stale = live.copy().set_index("feature")
    stale.loc["income", "coef"], stale.loc["dti", "coef"] = (
        stale.loc["dti", "coef"],
        stale.loc["income", "coef"],
    )
    stale.loc["utilization", "coef"] = 0.0
    stale.reset_index().to_csv(root / "coefficients_stale.csv", index=False)

    covenants_good = root / "covenants.yaml"
    covenants_good.write_text(COVENANTS_YAML.format(coefficients="coefficients_live.csv"))
    covenants_broken = root / "covenants_broken.yaml"
    covenants_broken.write_text(COVENANTS_YAML.format(coefficients="coefficients_stale.csv"))
    governance = root / "governance.yaml"
    governance.write_text(GOVERNANCE_YAML)

    return {
        "root": root,
        "model": model_path,
        "data": data_path,
        "covenants": covenants_good,
        "covenants_broken": covenants_broken,
        "governance": governance,
        "frame": df,
    }


@pytest.fixture()
def store_dir(tmp_path: Path) -> Path:
    return tmp_path / ".covenant"
