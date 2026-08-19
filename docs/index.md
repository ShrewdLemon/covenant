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

Covenant produces evidence; your validators and auditors decide. Nothing it
outputs is a claim of compliance.

## The four checks and the report

| Command | Tests |
|---|---|
| `covenant check reason-codes` | Declared adverse-action reasons vs measured attributions, stratified by score band, with background-sensitivity and a placebo sub-check |
| `covenant check monotonicity` | Declared vs configured (constraints read off the estimator) vs empirical monotone directions |
| `covenant check features` | Declared features vs the features the model actually uses; inert documented features surfaced as warnings |
| `covenant check exclusions` | Excluded variables measurably excluded, plus an association screen — proxies are *surfaced, not proven absent* |
| `covenant check all` | Everything above, one summary, one combined exit code |
| `covenant report` | Deterministic validation report — discrimination, calibration, stability, drift, monotonicity and a challenger's lift with bootstrap CIs; byte-identical on re-render, self-hashed |

Every check writes a hash-stamped, replayable YAML record and exits
non-zero on breach, so it can sit in CI and block a deployment. The
inventory (`covenant register` / `diff` / `show` / `list`) is
content-addressed flat YAML in your own repository: history is `git log`,
comparison is a diff.

## Quickstart

```bash
covenant init                 # creates .covenant/ + template YAMLs
$EDITOR covenants.yaml governance.yaml

covenant register model.joblib train.csv
covenant check all model.joblib train.csv
covenant report model.joblib train.csv --out report/
```

The model contract is `predict_proba` over a dataframe — scikit-learn
estimators and pipelines work as-is, and adapters cover OptBinning
scorecards (`covenant.adapters.export_scorecard_points`) and InterpretML
EBMs (exact shape-function attributions, no SHAP sampling). Persist models
as `.skops` files for safe loading (`pip install "covenants[skops]"`).

## Where to go next

- [Regulatory mapping](MAPPING.md) — each SR 26-2 / RBI FREE-AI ask, the
  Covenant artefact that answers it, and its status; citations marked TODO
  until pinned against the primary texts.
- [Research guide](research-guide.md) — the landscape, the papers, and
  where the gap this project fills actually is.
- [GitHub](https://github.com/ShrewdLemon/covenant) — source, issues, the
  broken-scorecard demo asserted in CI, and the changelog.
