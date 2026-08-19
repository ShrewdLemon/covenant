"""A plain logistic challenger for the validation report.

Benchmarking against a simpler alternative is standard outcome analysis:
if the primary model cannot beat a five-fold logistic pipeline on its own
training snapshot, its complexity is unearned and a validator should ask
why it was bought. Covenant fits the cheapest credible challenger — one
hot encoding, scaling, logistic regression — and scores it honestly, so
the report can put an interval on the lift instead of asserting it
(the paired resampling lives in ``covenant.metrics.paired_bootstrap_diff``,
which compares both models on identical rows).

The honesty asymmetry, stated here and surfaced wherever these scores are
compared: the challenger is scored **out of fold** while the primary model
is scored **in-sample** on the same snapshot, which flatters the primary.
The comparison is therefore a floor for the challenger, not a horse race —
"our GBM beats the simple thing" becomes a measured claim with an
interval, not an assumption.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from covenant.checks.reason_codes import CheckSetupError

N_FOLDS = 5


def challenger_scores(
    data: pd.DataFrame,
    features: list[str],
    categorical: list[str],
    target: str,
    random_state: int = 0,
) -> np.ndarray:
    """Out-of-fold ``p_bad`` from a plain logistic challenger.

    ``target`` names the 0/1 outcome column of ``data`` (1 = bad). The
    challenger is ``Pipeline(ColumnTransformer(numeric passthrough +
    OneHotEncoder(handle_unknown="ignore")), StandardScaler(with_mean=
    False), LogisticRegression(max_iter=1000))`` predicted out of fold via
    5-fold ``StratifiedKFold(shuffle=True)`` with a fixed seed, so the
    scores are deterministic for a given ``random_state``.

    The honesty asymmetry: these scores are out-of-fold while the primary
    model is scored in-sample on the same snapshot, which flatters the
    primary. Read any comparison as a floor for the challenger, not a
    horse race.
    """
    if target not in data.columns:
        raise CheckSetupError(
            f"challenger target column {target!r} is not a column of the data; "
            "set report.target_column to the 0/1 outcome column (1 = bad)"
        )
    missing = [f for f in features if f not in data.columns]
    if missing:
        raise CheckSetupError(f"data lacks declared features: {missing}")
    y = pd.to_numeric(data[target]).to_numpy(dtype=float)
    labels = np.unique(y)
    if not np.isin(labels, (0.0, 1.0)).all() or len(labels) < 2:
        raise CheckSetupError(
            f"challenger target {target!r} must contain both 0 and 1 labels "
            f"(1 = bad), got values {labels[:5].tolist()}"
        )
    y_int = y.astype(int)
    rarest = int(np.bincount(y_int, minlength=2).min())
    if rarest < N_FOLDS:
        raise CheckSetupError(
            f"the rarer class has only {rarest} rows; {N_FOLDS}-fold stratified "
            "cross-validation needs at least one row of each class per fold"
        )

    numeric = [f for f in features if f not in categorical]
    cats = [f for f in features if f in categorical]
    pipeline = Pipeline(
        [
            (
                "prep",
                ColumnTransformer(
                    [
                        ("num", "passthrough", numeric),
                        ("cat", OneHotEncoder(handle_unknown="ignore"), cats),
                    ]
                ),
            ),
            ("scale", StandardScaler(with_mean=False)),
            ("logreg", LogisticRegression(max_iter=1000)),
        ]
    )
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=random_state)
    proba = cross_val_predict(pipeline, data[features], y_int, cv=cv, method="predict_proba")
    return np.asarray(proba)[:, 1]
