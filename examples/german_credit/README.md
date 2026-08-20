# German Credit: a real dataset, a real governance failure

The Statlog German Credit dataset (Hofmann, 1994 — UCI / OpenML
`credit-g`, 1,000 applications) ships with two columns no modern lender
may score: **`personal_status`**, which encodes *sex and marital status*
("male single", "female div/dep/mar", …), and **`foreign_worker`**, a
nationality marker. That makes it the perfect real-world demonstration of
what Covenant checks: a model that quietly consumes protected attributes
looks exactly like one that doesn't — its accuracy tells you nothing —
and only a test against the model's *documented claims* catches it.

`data/german_credit.csv` is a verbatim copy of the OpenML frame (plus an
`application_id` and the target recoded to `bad` ∈ {0,1}), committed here
so the demo is deterministic and runs offline. Source: UCI Machine
Learning Repository / OpenML dataset 31 (`credit-g`), public and freely
redistributed; see the module docstring of `make_demo.py`.

## Build

```bash
python make_demo.py     # ~7s: two models, train/holdout split, SHAP export
```

Two models are fitted on the same 800-row training split:

- `model_leaky.joblib` — one-hot + logistic pipeline on **all 20 columns**,
  the two protected ones included.
- `model_clean.joblib` — the same pipeline on the **18 documented features**.

One covenant (`covenants.yaml`) declares the 18 features (four with
monotone directions), excludes the two protected columns with reasons,
and points Check 1 at `clean_attributions.csv` — a production-style
per-applicant SHAP export for denied applications, joined on
`application_id`.

## The leaky model breaches

```text
$ covenant check features model_leaky.joblib train.csv --covenants covenants.yaml
check features — german-credit: BREACH (fail)
  used by the model but undeclared: personal_status, foreign_worker
# exit 1

$ covenant check exclusions model_leaky.joblib train.csv --covenants covenants.yaml
check exclusions — german-credit: BREACH (fail)
  max association observed 0.283  (threshold 0.50)
  personal_status: the covenant excludes it; the model measurably uses it
  foreign_worker: the covenant excludes it; the model measurably uses it
# exit 1
```

## The clean model passes — with the proxies still surfaced

```text
$ covenant check all model_clean.joblib train.csv --covenants covenants.yaml
covenant check all
  reason-codes   PASS    top-1 1.000  jaccard 1.000
  monotonicity   PASS    worst violation 0.000
  features       PASS    undocumented 0  unused 0  dead 0
  exclusions     PASS    max association 0.28  proxy flags 0
# exit 0
```

The exclusions check reports the strongest real association it found —
`personal_status` ~ `num_dependents` at 0.28, `~ age` at 0.25 — below the
declared 0.5 threshold. Proxies are **surfaced, not proven absent**: a
pairwise screen cannot rule out a multivariate proxy, and the record says
so.

## The report, on real outcomes

```bash
covenant report model_clean.joblib train.csv --covenants covenants.yaml \
    --governance governance.yaml --holdout holdout.csv --out report/
```

Real numbers with real intervals, and since v0.5 the headline is
**out-of-sample**: holdout AUC 0.772 [0.692, 0.840] leads, the in-sample
0.827 follows explicitly labeled as flattering the model. Score PSI
train → holdout 0.059 with a per-feature CSI table, all four covenant
check verdicts embedded with their record hashes, the governance record
(owner, materiality tier and its justification verbatim), and the
logistic challenger comparison. Rendered byte-identically on every
re-run; CI proves it with a double render and `diff -r`.

## Why this demo matters

The stale-coefficient demo (`../broken_scorecard/`) shows the failure
nobody else tests for. This one shows it on data reviewers recognise,
with a failure mode regulators actually cite: the documentation says the
protected attribute is excluded, and the deployed artefact disagrees.
Scores identical in quality; claims measurably false; caught in CI.
