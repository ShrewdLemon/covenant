# HMDA mortgage denials: verifying an exclusion on real fair-lending data

**Read this framing first — it is the point of the demo.** The target
here is `denied`: the lender's **historical decision**, not
creditworthiness. A model fitted to it reproduces whatever bias those
historical decisions contain. That is exactly why the governance checks
matter: the covenant *excludes* `derived_race`, `derived_sex` and
`derived_ethnicity`, and Covenant verifies that the deployed artefact
honours that exclusion and surfaces proxies for it. This demo
demonstrates the verification of documented claims; it does not endorse
modelling denial decisions.

`data/hmda_ri_2023.csv.gz` is public CFPB HMDA data — 2023, Rhode
Island, originated + denied applications (US government work, freely
redistributable): 25,273 applications, 25.7% denied. Ten underwriting
columns (numeric `loan_amount`, `income`, `property_value`,
`loan_to_value_ratio`; HMDA range strings `debt_to_income_ratio` and
`applicant_age`; categorical `loan_purpose`, `loan_type`,
`occupancy_type`, `derived_dwelling_category`) sit alongside the three
protected `derived_*` columns. Committed here so the demo is
deterministic and runs offline.

## Build

```bash
python make_demo.py     # ~18s: two models, train/holdout split, SHAP export
```

`make_demo.py` draws a seeded 10,000-application working sample (the
full 25,273 stay committed), splits it 80/20 into `train.csv` /
`holdout.csv` (written as plain CSV — Covenant's data loader reads
.csv/.parquet), and fits two one-hot + logistic pipelines on the same
8,000-row training split:

- `model_leaky.joblib` — on **all 13 columns**, the three protected ones
  included.
- `model_clean.joblib` — on the **10 documented features** only.

One covenant (`covenants.yaml`) declares the 10 features, excludes the
three `derived_*` columns with reasons, and points Check 1 at
`clean_attributions.csv` — a production-style per-applicant SHAP export
for the 1,246 denied applicants, joined on `application_id` (`top_k: 4`,
mirroring Regulation B's at-most-four adverse-action reasons).

Directions are declared only where the fitted model supports them.
`make_demo.py` prints the evidence: `loan_amount` −0.397 (decreases
denial risk — small loans deny more in this book) and
`loan_to_value_ratio` +0.221 (increases it). The intuitive "higher
income lowers denial risk" is **not** supported — once the
`debt_to_income_ratio` buckets are in, `income` sits at +0.016 — so no
direction is declared for it, or for `property_value` (+0.040).

## The leaky model breaches

```text
$ covenant check features model_leaky.joblib train.csv --covenants covenants.yaml
check features — hmda-ri-2023: BREACH (fail)
  used by the model but undeclared: derived_race, derived_sex, derived_ethnicity
  documented but measurably inert (warning, not a breach):
    property_value             mean |attribution| 0.00006
# exit 1

$ covenant check exclusions model_leaky.joblib train.csv --covenants covenants.yaml
check exclusions — hmda-ri-2023: BREACH (fail)
  max association observed 0.184  (threshold 0.50)
  derived_race: the covenant excludes it; the model measurably uses it
  derived_sex: the covenant excludes it; the model measurably uses it
  derived_ethnicity: the covenant excludes it; the model measurably uses it
# exit 1
```

The check record puts numbers on "measurably uses it": `derived_race`
carries 7.0% of the leaky model's total attribution mass, `derived_sex`
5.2%, `derived_ethnicity` 5.2% — against a declared ceiling of 0.1%.
Nothing about the leaky model's accuracy betrays any of this.

## The clean model passes — with the real associations surfaced

```text
$ covenant check all model_clean.joblib train.csv --covenants covenants.yaml
covenant check all
  reason-codes   PASS    top-1 1.000  jaccard 1.000
  monotonicity   PASS    worst violation 0.000
  features       PASS    undocumented 0  unused 0  dead 1
  exclusions     PASS    max association 0.18  proxy flags 0
# exit 0
```

The proxy screen measures real associations in this data and reports
them below the declared 0.5 threshold: `derived_sex` ~ `income` at 0.18,
`derived_race` ~ `loan_to_value_ratio` at 0.15, `derived_ethnicity` ~
`loan_to_value_ratio` at 0.13. Proxies are **surfaced, not proven
absent**: a pairwise screen cannot rule out a multivariate proxy, and
every check record says so.

One more honest flag the passing run still raises: `property_value` is
documented but measurably inert (a dead-feature warning, not a breach).
An earlier build also flagged the measured side `background_sensitive`
(0.78 across two seeded 40-row backgrounds, below the 0.8 floor); raising
`background_size` to 150 — nearly free now that the one-hot logistic
pipeline takes the exact attribution path — measures 0.827 and clears the
flag. Run, observe, then set.

## The report, on real outcomes

```bash
covenant report model_clean.joblib train.csv --covenants covenants.yaml \
    --governance governance.yaml --holdout holdout.csv --out report/
```

Real numbers with real intervals: AUC 0.8352 [0.8251, 0.8455] against
the historical decisions (not against creditworthiness — see the
framing), score PSI train → holdout 0.0034 with a per-feature CSI table,
and the challenger comparison ΔAUC +0.0073 [0.0063, 0.0085] — near zero
by construction, since the primary *is* a logistic pipeline. Rendered
byte-identically on every re-run; CI proves it with a double render and
`diff -r`.

## Why this demo matters

This is the failure mode fair-lending supervision actually cites: the
documentation says race, sex and ethnicity are excluded, and the
deployed artefact disagrees. On HMDA data the stakes are legible — the
excluded columns are the ECOA prohibited basis itself, the covenant is
the lender's documented claim, and Covenant is the test that the claim
is true of the artefact, in CI, on every commit.
