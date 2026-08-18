# The broken scorecard

One synthetic credit dataset, two governance failures nothing else catches —
because in both, the *scores* are fine and only the *claims about the model*
are wrong.

Build the artifacts (deterministic, seeded):

```bash
python make_demo.py
```

## 1. Stale reason codes (Check 1)

`model.joblib` is a logistic scorecard. Adverse-action reasons are derived
from a coefficient table — and `coefficients_stale.csv` is the table from
"the previous model version": two strong coefficients swapped, one zeroed.

```bash
covenant check reason-codes model.joblib train.csv --covenants covenants_broken.yaml   # BREACH, exit 1
covenant check reason-codes model.joblib train.csv --covenants covenants_fixed.yaml    # PASS,   exit 0
```

## 2. Documentation says monotone, model says otherwise (Check 2)

The true effect of `loan_amount` is U-shaped — small and large loans are
risky. `covenants_gbm.yaml` declares it monotone increasing, wording carried
over from the linear scorecard's docs. Same covenant, two models:

```bash
covenant check monotonicity model_gbm_unconstrained.joblib train.csv --covenants covenants_gbm.yaml  # BREACH, exit 1
covenant check monotonicity model_gbm_constrained.joblib   train.csv --covenants covenants_gbm.yaml  # PASS,   exit 0
```

The unconstrained `HistGradientBoostingClassifier` learns the U-shape and
contradicts the covenant. The constrained one was fitted with
`monotonic_cst` matching the covenant, so declared, configured and empirical
directions all agree — the check output shows the three-way comparison.

Registration works on any of them:

```bash
covenant register model.joblib train.csv --covenants covenants_fixed.yaml --governance governance.yaml
```

CI (`.github/workflows/ci.yml`) rebuilds these artifacts and asserts every
exit code above on each commit.
