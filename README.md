# Covenant

**pytest for credit-model governance.** A model's documentation is a set of
testable claims about its behaviour — so Covenant tests them.

```bash
pip install covenants     # distribution name is plural
import covenant           # import name is singular
covenant --help           # CLI
```

In lending, a covenant is a promise a borrower makes and a lender verifies.
Here, the promises are the model's: which features it uses, which direction
each pushes risk, which variables are excluded, and how its adverse-action
reason codes are derived. Covenant records those promises as flat YAML in
your own git repo, and ships checks that **fail CI when the model's measured
behaviour contradicts them** — a covenant breach.

## Why

Every lender using a credit model is expected to answer four questions at any
moment: what is the model and where did it come from, does it still work, is
it fair and explainable, and can you prove all of that to an auditor. In the
US that expectation is SR 26-2 (which supersedes SR 11-7, ties governance
intensity to materiality, and expects a live inventory); in India, the RBI's
FREE-AI framework pulls every regulated entity the same way — its 2025 survey
found only ~15% of AI-using entities used interpretation tools and ~18% kept
audit logs.

Even well-governed models have a gap no existing tool addresses: the
documentation and the model drift apart silently. The reason-code table says
"top reason: high DTI" while the deployed model's top driver is something
else, because the table came from a previous version. Regulators call this an
accuracy problem (CFPB Circular 2022-03: adverse-action reasons must reflect
the factors *actually* scored); researchers have measured it
(Krivorotov & Richey 2022; FinRegLab 2023); nobody ships a tool that tests
for it. That test is Covenant's Check 1.

## What it does (v0.1)

**`covenant register`** — an inventory entry per model version,
content-addressed by SHA-256 of the model artefact, the training snapshot and
the covenants file. Stored as YAML under `.covenant/` in your repo, so
history is `git log` and `covenant diff` shows exactly what changed between
two versions. Governance fields — owner, intended use, materiality tier with
a **mandatory justification**, review date, vendor block for third-party
models — are validated strictly: a typo'd key or an empty justification is
rejected with a readable error, not silently accepted.

**`covenant check reason-codes`** — Check 1. For the applicants the model
would deny, compare the top-k adverse-action reasons your production method
produces (a coefficient table, or a reasons file exported from your pipeline)
against the features that measurably drove the decision (SHAP attributions).
Reports top-1 agreement and top-k Jaccard against thresholds you set in the
covenants file, stratified by score band — disagreement concentrates near the
decision boundary, and the record shows where. Exits non-zero on breach, so
it can sit in CI and block a deployment.

The measured side is honest about itself: post-hoc attribution is an
approximation, and it is sensitive to the SHAP background sample. The
background is a seeded, first-class parameter, and every check record reports
the stability of the measured side across two backgrounds.

## Quickstart

```bash
covenant init                 # creates .covenant/ + template YAMLs
$EDITOR covenants.yaml governance.yaml

covenant register model.joblib train.csv
# registered my-scorecard version 3f9c1a2b8d4e

covenant check reason-codes model.joblib train.csv
# check reason-codes — my-scorecard: PASS
#   top-1 agreement  0.94  (threshold 0.75)
#   top-k jaccard    0.87  (threshold 0.60)
#   ...
#   record: .covenant/checks/my-scorecard/reason-codes-a1b2c3d4e5f6.yaml

covenant diff my-scorecard 3f9c1a2b8d4e 7e2d9c0b1a3f
```

The model contract is `predict_proba` over a dataframe — scikit-learn
estimators and pipelines work as-is.

## The demo: a broken scorecard

`examples/broken_scorecard/` fits a logistic scorecard, then derives
adverse-action reasons from a **stale coefficient table** — two strong
coefficients swapped, one zeroed, exactly the "reasons come from the previous
model" failure mode. Scores are untouched; only the explanations are wrong,
which is why nothing else catches it.

```bash
python examples/broken_scorecard/make_demo.py
cd examples/broken_scorecard

covenant check reason-codes model.joblib train.csv --covenants covenants_broken.yaml
# BREACH (fail) — exit code 1, with the disagreement table

covenant check reason-codes model.joblib train.csv --covenants covenants_fixed.yaml
# PASS — exit code 0
```

CI runs both and asserts the exit codes, so the README's central claim is
itself tested on every commit.

## What Covenant is not

- **Not a metrics library.** ValidMind's platform and library, Wells Fargo's
  PiML/MoDeVa, FINRA's Model Validation Toolkit and several credit-risk
  libraries compute the metrics an SR 26-2 report needs; Evidently and
  NannyML monitor drift. Covenant does not compete on metrics. It adds the
  layer none of them ship: a git-native, diffable record of what a model
  *claims*, and CI checks that fail when measured behaviour contradicts the
  claims.
- **Not a compliance guarantee.** Covenant produces evidence; your
  validators and auditors decide. Passing a check never means "compliant".
- **Not an adjudicator.** When declared reasons and measured attributions
  disagree, Covenant reports the disagreement and where it concentrates; it
  does not rule on which side is right.

## Roadmap

- `covenant check monotonicity` — declared vs configured vs empirical
  monotone directions (dominance pairs + ICE slopes; reads
  `monotone_constraints` from tree boosters).
- `covenant check features` / `check exclusions` — declared vs used
  features; excluded variables and their obvious proxies (association
  screen — surfacing proxies, not proving absence).
- `covenant report` — discrimination, calibration, stability, drift and a
  logistic challenger's lift, bootstrap CIs, rendered deterministically
  (same inputs → same bytes) and mapped line-by-line to SR 26-2 / FREE-AI.
- Declared-method coverage: `most_points_lost` (scorecards), `univariate`,
  B-Shap; placebo perturbation sub-check.
- Adapters: OptBinning `Scorecard`, InterpretML EBM (exact shape-function
  contributions instead of SHAP), XGBoost/LightGBM constraint reading.

## License

Apache-2.0.
