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
| Check 1 `reason-codes` — declared adverse-action reasons vs measured attributions, with placebo sub-check | ✅ shipped |
| Check 2 `monotonicity` — declared vs configured vs empirical directions | ✅ shipped |
| Check 3 `features` — declared vs actually-used features | ✅ shipped |
| Check 4 `exclusions` — exclusions honoured, obvious proxies surfaced | ✅ shipped |
| `covenant check all` — combined gate | ✅ shipped |
| All four K&R declared methods (`difference_from_mean`, `most_points_lost`, `univariate`, `shapley` export) plus a production reasons file (`custom`) | ✅ shipped |
| Exact attribution fast paths (ebm-exact, linear-exact, tree-shap) with the path named in every record | ✅ shipped |
| `covenant report` — deterministic validation report, mapped to SR 26-2 / FREE-AI ([docs/MAPPING.md](docs/MAPPING.md)) | ✅ shipped |
| OptBinning `Scorecard` adapter, InterpretML EBM exact path, `.skops` safe loading | ✅ shipped |
| Docs site — [shrewdlemon.github.io/covenant](https://shrewdlemon.github.io/covenant/) | ✅ shipped |

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

**`covenant check features`** — Check 3. Are the declared features the used
features? A feature the model reads but the covenant never mentions is a
breach; so is a documented feature the model cannot see. Documented features
with measurably zero attribution are surfaced as warnings — dead
documentation, not a behavioural contradiction.

**`covenant check exclusions`** — Check 4. Excluded variables are testable
claims too: if one still reaches the model, its measured attribution must be
≈ 0; and every excluded variable is screened for association (|Spearman|,
correlation ratio, bias-corrected Cramér's V) against the used features.
Proxies are **surfaced, not proven absent** — the check says exactly that.

**`covenant check all`** — every configured check, one summary, one combined
exit code for CI.

**`covenant report`** — the deterministic validation report: discrimination
(AUC/Gini/KS), calibration (Brier/ECE), stability (PSI/CSI vs a holdout),
drift by time slice, the monotonicity check's verdict, and a plain logistic
challenger's lift — discrimination, calibration and challenger estimates
carry seeded bootstrap confidence intervals, every section is mapped to the
regulatory ask, and the whole thing renders byte-identically from the same
inputs. The report embeds its own hash and each figure's hash, so it is
citable evidence years later.

The measured side names its method: EBMs are explained by their own shape
functions, linear models get closed-form exact contributions, tree ensembles
get TreeExplainer, everything else falls back to permutation SHAP — and every attribution-based check record states which
path produced its numbers. Check records are replayable: identical inputs produce
byte-identical, hash-addressed records at the same path, and run timestamps
live in an append-only `runs.log` beside them.

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

## The demos

### Real data: German Credit and the protected attributes

[`examples/german_credit/`](examples/german_credit/) runs Covenant on the
Statlog German Credit dataset (UCI / OpenML `credit-g`, 1,000 real
applications, committed to the repo for offline determinism). The dataset
famously contains `personal_status` — which encodes **sex and marital
status** — and `foreign_worker`. Two models, one covenant that excludes
both columns:

```bash
python examples/german_credit/make_demo.py
cd examples/german_credit

covenant check features model_leaky.joblib train.csv --covenants covenants.yaml
# BREACH — used by the model but undeclared: personal_status, foreign_worker; exit 1
covenant check exclusions model_leaky.joblib train.csv --covenants covenants.yaml
# BREACH — the covenant excludes them; the model measurably uses them; exit 1

covenant check all model_clean.joblib train.csv --covenants covenants.yaml
# PASS on all four checks — with the real associations the proxy screen
# found (personal_status ~ num_dependents at 0.28) surfaced below threshold
covenant report model_clean.joblib train.csv --covenants covenants.yaml --governance governance.yaml --holdout holdout.csv --out report/
# headline out-of-sample AUC 0.772 [0.692, 0.840]; all four check verdicts and the
# governance record embedded; byte-identical re-render
```

The leaky model's accuracy gives nothing away — only testing the artefact
against its documented claims catches it. CI asserts all of these exit
codes on every commit.

### More real data: Taiwan and US mortgages

[`examples/taiwan_credit/`](examples/taiwan_credit/) — 30,000 real
credit-card clients (Yeh & Lien 2009, UCI). `sex` and `marriage` are
excluded by covenant; the leaky model breaches, the clean one passes with
the real associations surfaced (marriage ~ age at 0.47, honestly below a
measured threshold), and the whole 30k-row sequence runs in about a minute.

[`examples/hmda_mortgage/`](examples/hmda_mortgage/) — 25,273 real 2023
Rhode Island mortgage applications (HMDA, CFPB public data) with
`derived_race`, `derived_sex` and `derived_ethnicity` excluded by covenant.
The target is the lender's historical decision, and the demo says exactly
what that means: verifying documented exclusions against the artefact, not
endorsing the model. CI asserts every exit code for both.

### Synthetic: the broken scorecard

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
  top-1 agreement  0.492  (threshold 0.75)
  top-k jaccard    0.461  (threshold 0.60)
  background stability of measured side: 0.877
  measured via: linear-exact
  n denied evaluated: 120
  by score band (denied applicants, near boundary first):
    p_bad 0.500-0.577  n=24    top-1 0.333  jaccard 0.478
    ...
  worst disagreements (declared vs measured):
    row 280    p_bad 0.952  declared age_of_oldest_line_months, employment_years, ...
  record: .covenant/checks/demo-scorecard/reason-codes-….yaml
# exit code 1
```
```bash
covenant check reason-codes model.joblib train.csv --covenants covenants_fixed.yaml
# PASS — top-1 0.975, jaccard 0.940, exit code 0
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

## Integrations

Optional extras keep the core light:

```bash
pip install "covenants[scorecard]"   # OptBinning: export_scorecard_points -> most_points_lost table
pip install "covenants[interpret]"   # InterpretML EBM: exact shape-function attributions (path "ebm-exact")
pip install "covenants[skops]"       # .skops model loading: refuses untrusted types instead of executing them
pip install "covenants[boosters]"    # xgboost/lightgbm: constraint reading + TreeExplainer coverage
pip install "covenants[integrations]"  # all of the above
```

## Roadmap

- Recourse validity (does following the reason code actually flip the
  decision?), robustness/resilience and fairness — deferred to the tools
  that already do them well (DiCE/CARLA, PiML, Fairlearn/SolasAI).

## License

Apache-2.0.
