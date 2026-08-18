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

## Status

| Piece | State |
|---|---|
| `covenant register` / `diff` / `show` / `list` — content-addressed inventory | ✅ shipped |
| Check 1 `reason-codes` — declared adverse-action reasons vs measured attributions | ✅ shipped |
| Check 2 `monotonicity` — declared vs configured vs empirical directions | ✅ shipped |
| `covenant check all` — combined gate | ✅ shipped |
| Check 3 `features` / Check 4 `exclusions` & proxy screen | ⏳ next |
| Declared methods `most_points_lost`, `univariate`, B-Shap; placebo sub-check | ⏳ planned |
| `covenant report` — deterministic, SR 26-2 / FREE-AI-mapped validation report | ⏳ planned (v0.3) |

## What it does

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

**`covenant check monotonicity`** — Check 2. Feature-highlighting
explanations silently assume monotonicity (Barocas, Selbst & Raghavan):
"improve the flagged feature" is only honest advice if improving it cannot
raise the score. Each declared direction is tested three ways — **declared**
(the covenant), **configured** (monotone constraints read off XGBoost /
LightGBM / sklearn boosters, when present), and **empirical** (dominance
pairs that move one feature while holding the rest fixed, plus ICE paths
swept over the feature's quantiles). Violation rates per feature against a
threshold you set; a configured constraint that contradicts the covenant is
a breach on its own.

**`covenant check all`** — every configured check, one summary, one combined
exit code for CI.

Check records are replayable: identical inputs produce byte-identical,
hash-addressed records at the same path, and run timestamps live in an
append-only `runs.log` beside them.

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
```
```text
check reason-codes — demo-scorecard: BREACH (fail)
  top-1 agreement  0.500  (threshold 0.75)
  top-k jaccard    0.461  (threshold 0.60)
  background stability of measured side: 0.880
  n denied evaluated: 120
  by score band (denied applicants, near boundary first):
    p_bad 0.501-0.596  n=24    top-1 0.542  jaccard 0.467
    ...
  worst disagreements (declared vs measured):
    row 15     p_bad 0.943  declared age_of_oldest_line_months, delinquencies_24m, ...
  record: .covenant/checks/demo-scorecard/reason-codes-….yaml
# exit code 1
```
```bash
covenant check reason-codes model.joblib train.csv --covenants covenants_fixed.yaml
# PASS — top-1 0.983, jaccard 0.943, exit code 0
```

The same demo carries the monotonicity story: the true effect of
`loan_amount` is U-shaped, but the covenant — written for the old linear
scorecard — declares it monotone increasing. Same covenant, two models:

```bash
covenant check monotonicity model_gbm_unconstrained.joblib train.csv --covenants covenants_gbm.yaml
# BREACH — the GBM learned the U-shape; loan_amount ICE violation rate 0.45, exit 1

covenant check monotonicity model_gbm_constrained.joblib train.csv --covenants covenants_gbm.yaml
# PASS — fitted with monotonic_cst matching the covenant; declared = configured = empirical, exit 0
```

CI runs all four and asserts the exit codes, so the README's central claims
are themselves tested on every commit.

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

- `covenant check features` / `check exclusions` — declared vs used
  features; excluded variables and their obvious proxies (association
  screen — surfacing proxies, not proving absence).
- `covenant report` — discrimination, calibration, stability, drift and a
  logistic challenger's lift, bootstrap CIs, rendered deterministically
  (same inputs → same bytes) and mapped line-by-line to SR 26-2 / FREE-AI,
  plus `docs/MAPPING.md`.
- Declared-method coverage: `most_points_lost` (scorecards), `univariate`,
  B-Shap; placebo perturbation sub-check.
- Exact attribution fast paths: LinearExplainer for linear pipelines,
  TreeExplainer for tree ensembles, EBM shape functions — permutation SHAP
  stays as the model-agnostic fallback, and the record names the path used.
- Adapters: OptBinning `Scorecard`, InterpretML EBM.

## License

Apache-2.0.
