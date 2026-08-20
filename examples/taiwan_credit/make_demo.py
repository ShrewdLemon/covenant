"""Build the Taiwan Credit demo: 30,000 real clients, a real governance failure.

The "Default of Credit Card Clients" dataset (Yeh & Lien, 2009; UCI #350,
via OpenML) records 30,000 Taiwanese credit-card clients from 2005 with
their credit limit, six months of repayment status / bill / payment
history — and three demographic columns: ``sex``, ``education`` and
``marriage``. ``data/taiwan_credit.csv.gz`` is a committed copy (plus a
``client_id`` and the target recoded to ``bad`` 0/1, 22.1% bad rate) so
the demo is deterministic and offline; see the README for provenance.
Covenant's ``load_data`` reads .csv/.parquet, so this script decompresses
the committed .csv.gz into plain ``train.csv`` / ``holdout.csv`` working
files (pandas reads the gz natively).

The covenant excludes ``sex`` and ``marriage``. ``education`` stays a
documented feature: it is a standard predictor in the credit-scoring
literature for this dataset (Yeh & Lien themselves model it), and keeping
it documented is precisely the kind of policy decision a covenant records.

Two models, one covenant:

* ``model_leaky.joblib`` — fitted on **all 23** columns, ``sex`` and
  ``marriage`` included. Its AUC is indistinguishable from the clean
  model's; nothing about its scores betrays what it consumes.
  ``covenant check features`` catches the undocumented inputs and
  ``covenant check exclusions`` measures the excluded columns actually
  driving scores — both breach.
* ``model_clean.joblib`` — fitted on the 21 documented features only.
  Every check passes, and the proxy screen still *surfaces* the real
  associations it finds (marriage ~ age at 0.47 — married clients are
  older) below the declared threshold — surfaced, not proven absent.

Also written: a 24,000/6,000 train/holdout split (seeded) so the report's
stability section computes real PSI/CSI, and a production-style SHAP
export (``clean_attributions.csv``) for every denied applicant, keyed on
``client_id``, so Check 1's ``shapley`` method runs against a real
artefact. At this scale permutation SHAP is the cost centre: the export
covers ~1,800 denied applicants and is the slowest single step of the
build (seconds, not minutes — see the README for measured timings).

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
SEED = 20050901  # September 2005: the last billing month in the data

EXCLUDED = ["sex", "marriage"]


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
    df = pd.read_csv(HERE / "data" / "taiwan_credit.csv.gz")
    feature_columns = [c for c in df.columns if c not in ("client_id", "bad")]
    clean_features = [c for c in feature_columns if c not in EXCLUDED]
    numeric = [c for c in feature_columns if pd.api.types.is_numeric_dtype(df[c])]

    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(df))
    train = df.iloc[np.sort(order[:24000])].reset_index(drop=True)
    holdout = df.iloc[np.sort(order[24000:])].reset_index(drop=True)
    train.to_csv(HERE / "train.csv", index=False)
    holdout.to_csv(HERE / "holdout.csv", index=False)

    leaky = make_pipeline(feature_columns, numeric)
    leaky.fit(train[feature_columns], train["bad"])
    joblib.dump(leaky, HERE / "model_leaky.joblib")

    clean = make_pipeline(clean_features, numeric)
    clean.fit(train[clean_features], train["bad"])
    joblib.dump(clean, HERE / "model_clean.joblib")

    # Production-style attribution export for Check 1's shapley method: the
    # artefact a real letter-generation system would consume, keyed on
    # client_id, produced from the clean model with a seeded background for
    # every denied applicant in the training snapshot (~1,800 of 24,000).
    from covenant.attribution import explain, sample_background
    from covenant.model import CovenantModel

    model = CovenantModel(clean, clean_features)
    categorical = [c for c in clean_features if c not in numeric]
    # size and seed match checks.reason_codes.background_size / random_state
    # in covenants.yaml, where the choice is justified from measurement
    background = sample_background(train, clean_features, 120, 0)
    denied = train[model.p_bad(train) >= 0.5].reset_index(drop=True)
    attributions, path = explain(
        model, denied[clean_features], background, categorical, 0, npermutations=2
    )
    export = attributions.round(6)
    export.insert(0, "client_id", denied["client_id"].to_numpy())
    export.to_csv(HERE / "clean_attributions.csv", index=False)

    auc_gap = None
    try:
        from sklearn.metrics import roc_auc_score

        p_leaky = leaky.predict_proba(train[feature_columns])[:, 1]
        p_clean = clean.predict_proba(train[clean_features])[:, 1]
        auc_gap = roc_auc_score(train["bad"], p_leaky) - roc_auc_score(train["bad"], p_clean)
    except ImportError:  # pragma: no cover - sklearn is a hard dependency anyway
        pass
    print(f"wrote both models, train/holdout split and the attribution export to {HERE}")
    print(f"attribution export path: {path}; denied applicants exported: {len(denied)}")
    if auc_gap is not None:
        print(f"AUC(leaky) - AUC(clean) on train: {auc_gap:+.4f} — the scores don't tell you")


if __name__ == "__main__":
    main()
