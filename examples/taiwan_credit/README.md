# Taiwan Credit: 30,000 real clients, and the scale probe

The "Default of Credit Card Clients" dataset (Yeh & Lien, 2009 — UCI
#350, via OpenML) records 30,000 Taiwanese credit-card clients from 2005:
credit limit, six months of repayment status (`pay_status_1..6`), bills
and payments — and three demographic columns. This covenant excludes
**`sex`** and **`marriage`** as protected attributes. **`education`
stays a documented feature**: it is a standard predictor in the
credit-scoring literature for this dataset (Yeh & Lien themselves model
it), and recording that decision — with the exclusions it sits next to —
is exactly what a covenant is for.

It is also this repo's scale probe: 30× the rows of the German Credit
demo and 21 declared features, which makes permutation SHAP the cost
centre. Every sample size in `covenants.yaml` was set by measuring first;
the full demo sequence runs in about a minute (timings below).

`data/taiwan_credit.csv.gz` is a committed copy of the frame (plus a
`client_id` and the target recoded to `bad` ∈ {0,1}; overall bad rate
0.221), so the demo is deterministic and offline. Source: UCI Machine
Learning Repository dataset 350, via OpenML; public and freely
redistributed. `make_demo.py` decompresses it into plain `train.csv` /
`holdout.csv` working files, which is what the checks read.

## Build

```bash
python make_demo.py     # 15.6s measured: two models, 24,000/6,000 split,
                        # SHAP export for all 1,775 denied train applicants
```

Two one-hot + scaler + logistic pipelines are fitted on the same
24,000-row training split:

- `model_leaky.joblib` — on **all 23 columns**, `sex` and `marriage`
  included.
- `model_clean.joblib` — on the **21 documented features** only.

Measured on the training split, AUC(leaky) − AUC(clean) = **+0.0015**.
The scores are indistinguishable; nothing about accuracy tells you one
model consumes protected attributes.

One covenant (`covenants.yaml`) declares the 21 features, excludes the
two protected columns with reasons, and points Check 1 at
`clean_attributions.csv` — a production-style per-applicant SHAP export
for every denied applicant, joined on `client_id`.

Monotone directions were declared only after reading the fitted clean
model's coefficient signs: `pay_status_1` +0.643, `pay_status_2` +0.101,
`pay_amt_1` −0.277, `limit_bal` −0.096 (scaled space) are robust and
defensible, so those four are declared. `age` measured **+0.094** — the
sign *contradicts* the folk claim that older is safer, so no direction is
declared for it — and the six correlated `bill_amt` columns flip signs
(−0.398 to +0.143), so none of them carry a claim either.

## The leaky model breaches

```text
$ covenant check features model_leaky.joblib train.csv --covenants covenants.yaml
check features — taiwan-credit: BREACH (fail)
  used by the model but undeclared: sex, marriage
# exit 1

$ covenant check exclusions model_leaky.joblib train.csv --covenants covenants.yaml
check exclusions — taiwan-credit: BREACH (fail)
  max association observed 0.469  (threshold 0.55)
  sex: the covenant excludes it; the model measurably uses it
  marriage: the covenant excludes it; the model measurably uses it
# exit 1
```

The check record puts numbers on "measurably uses it": `sex` carries
0.038 of the model's total attribution mass and `marriage` 0.056, against
a declared ceiling of 0.001 — 38× and 56× over.

## The clean model passes — with the proxies still surfaced

```text
$ covenant check all model_clean.joblib train.csv --covenants covenants.yaml
covenant check all
  reason-codes   PASS    top-1 1.000  jaccard 0.965
  monotonicity   PASS    worst violation 0.000
  features       PASS    undocumented 0  unused 0  dead 0
  exclusions     PASS    max association 0.47  proxy flags 0
# exit 0
```

The association threshold was set by measuring first: the strongest real
association between an excluded variable and a documented feature on the
train split is `marriage` ~ `age` at **0.469** (married clients are
older), then `marriage` ~ `education` at 0.131, `marriage` ~ `limit_bal`
at 0.118 and `sex` ~ `age` at 0.092. Nothing observed exceeds 0.5, and
the threshold sits at 0.55 — above the benign maximum, which the record
still reports. Proxies are **surfaced, not proven absent**: a pairwise
screen cannot rule out a multivariate proxy, and the record says so.

## Reason codes at 30k scale

Check 1 joins the production export to 300 sampled denied applicants (of
1,775) on `client_id` and recomputes attributions against the live model:

```text
$ covenant check reason-codes model_clean.joblib train.csv --covenants covenants.yaml
check reason-codes — taiwan-credit: PASS
  top-1 agreement  1.000  (threshold 0.75)
  top-k jaccard    0.965  (threshold 0.60)
  background stability of measured side: 0.916
  measured via: permutation-shap
  n denied evaluated: 300
# exit 0
```

The background size is the measured knob that matters here: with a
40-row background the measured side's own top-4 sets held only a 0.744
jaccard across two seeded backgrounds — below Covenant's 0.8 floor, so
the record would carry a `[sensitive]` flag. Measured again at 80 rows
(0.819) and 120 rows (**0.916**), the covenant declares 120 and the flag
is gone. That is the intended workflow: run, observe, then set.

## The report, on real outcomes

```bash
covenant report model_clean.joblib train.csv --covenants covenants.yaml \
    --holdout holdout.csv --out report/
```

Real numbers with real intervals on 24,000 rows: AUC 0.7264
[0.7180, 0.7359], KS 0.3821, Brier 0.1440, ECE 0.0469, score PSI
train → holdout 0.0018 with a per-feature CSI table, and the logistic
challenger comparison (ΔAUC +0.0025 [0.0019, 0.0031] — near zero by
construction here, since the primary *is* a logistic pipeline). Rendered
byte-identically on every re-run; CI proves it with a double render and
`diff -r`.

## Measured timings (the scale probe's answer)

Wall-clock on this repo's development machine, whole sequence 62s:

| Step | Wall-clock |
|---|---|
| `python make_demo.py` (SHAP export for 1,775 denied is the bulk) | 15.6s |
| `check features` (leaky) | 7.8s |
| `check exclusions` (leaky) | 7.4s |
| `check all` (clean) | 12.7s |
| `report` (n_bootstrap 500) | 9.2s / 9.7s per render |

Permutation SHAP costs a few milliseconds per explained row here
(21 features, 120-row background), so the sizes in `covenants.yaml` are
generous rather than squeezed — the comments in that file name the
measurement behind each one.

## Why this demo matters

The German Credit demo shows the failure on data reviewers recognise;
this one shows the same failure holding up at production-ish scale, plus
the judgment calls scale forces: which monotone directions survive
contact with a fitted model (age's does not), where an association
threshold belongs once you have measured the real associations under it,
and how large an attribution background must be before the measured side
of Check 1 stops moving under its own noise. Scores identical in
quality; claims measurably false; caught in CI.
