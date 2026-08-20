# Reference

The complete schema for the two YAML documents you write (`covenants.yaml`,
`governance.yaml`), the CSV artefact files a reason-code method reads, and
the records and exit codes Covenant produces. Everything on this page is a
statement about what the code does, not advice about what thresholds are
acceptable — the defaults are starting points, and the policy is yours.

Validation is strict throughout: every document rejects unknown keys, so a
typo in a YAML key fails loudly at registration instead of silently
weakening a check. Validation errors are reported per field with a dotted
path (e.g. `checks.reason_codes.decision_threshold: …`).

---

## covenants.yaml

The model's covenants: testable claims about its behaviour. Validated when
you run `covenant register` and again by every `covenant check` command.

### Top-level keys

| Key | Type | Required / default | Meaning |
|---|---|---|---|
| `covenant_schema` | int | default `1` (only `1` is valid) | Schema version of this document. |
| `model_name` | string | **required**; pattern `^[A-Za-z0-9._-]+$`, min length 1 | Inventory name; becomes a directory under `.covenant/models/` and `.covenant/checks/`, so only letters, digits, `.`, `_`, `-`. |
| `positive_class` | int, string or null | default `null` | Label of the bad/default class as it appears in the estimator's `classes_`. When null, the class at index 1 is used. |
| `features` | list | **required**, at least 1 entry | The features the model depends on. Names must be unique (duplicates are rejected). See [features](#features). |
| `excluded` | list | default `[]` | Variables the model must not use. See [excluded](#excluded). |
| `reason_codes` | mapping | **required** | How production derives adverse-action reason codes. See [reason_codes](#reason_codes). |
| `checks` | mapping | default: all check defaults | Per-check thresholds and sampling parameters. See the four `checks.*` sections below. |
| `report` | mapping | default: all report defaults | Settings for `covenant report`. See [report](#report). |

### features[]

Each entry:

| Key | Type | Required / default | Allowed values | Meaning |
|---|---|---|---|---|
| `name` | string | **required** | — | Column name in the data snapshot (and, when the estimator records them, in `feature_names_in_`). |
| `dtype` | string | default `numeric` | `numeric`, `categorical` | How the feature is treated in attribution and association screens. |
| `direction` | string | default `none` | `increases_risk`, `decreases_risk`, `none` | Declared monotone direction with respect to risk (`p_bad`). `none` claims no monotonicity. |
| `description` | string or null | default `null` | — | Free-text documentation; not used by any check. |

A validator rejects a monotone direction on a categorical feature: a
monotone direction on a categorical feature is not testable, so
`dtype: categorical` requires `direction: none` (the default).

### excluded[]

Each entry:

| Key | Type | Required / default | Meaning |
|---|---|---|---|
| `name` | string | **required** | The variable that must play no role in scoring. It does not have to be a snapshot column — an absent variable is recorded as "not present in snapshot; nothing to screen". |
| `reason` | string | **required**, min length 3 | Why it is excluded, e.g. `protected attribute (ECOA)`. |

`covenant check exclusions` refuses to run (exit 2) when this list is
empty — there is nothing to check.

### reason_codes

| Key | Type | Required / default | Meaning |
|---|---|---|---|
| `method` | string | **required** | One of the five methods below. |
| `top_k` | int | default `4`, range 1–10 | Number of reason codes compared per applicant (the top-k set). |
| `parameters` | mapping | default `{}` | Method-specific keys, listed below. Values are paths to CSV artefacts, resolved relative to the directory containing `covenants.yaml` (absolute paths are used as-is). |

The five methods (the first four follow the Krivorotov & Richey, 2022
taxonomy; `custom` is a user-supplied file of production reason codes):

| `method` | What it declares | Required `parameters` key |
|---|---|---|
| `difference_from_mean` | Linear contribution versus the mean applicant: `coef * (x - mean) / scale` per feature, read from a coefficient table. | `coefficients` |
| `most_points_lost` | Points lost against the feature's best bin, read off the scorecard points table production actually prints (higher points = lower risk). | `points_table` |
| `univariate` | Each feature judged on its own binned bad rate (or score): the applicant's bin value minus the unweighted mean of that feature's bin values. | `bins_table` |
| `shapley` | Per-applicant attributions exported by the production pipeline, joined to the scored rows on a stable id — the file is what gets tested, not a recomputation. | `attributions_file` |
| `custom` | The production reason codes themselves, exported as a CSV of `reason_1..reason_k` and joined on a stable id. | `reasons_file` |

The exact file formats are in
[Reason-code artefact files](#reason-code-artefact-files).

**id_column rule.** When `method` is `custom` or `shapley`, the document is
invalid unless `checks.reason_codes.id_column` is set: those methods join a
per-applicant production export to the scored rows, and rows must join on a
stable key, not on row position — positional alignment breaks silently the
moment one row is filtered upstream.

### checks.reason_codes

Thresholds and sampling for Check 1 (`covenant check reason-codes`):
declared adverse-action reasons versus measured SHAP attributions of
`p_bad`, over a sample of denied applicants. The check passes when
`top1_agreement >= min_top1_agreement` **and**
`topk_jaccard >= min_topk_jaccard`.

| Key | Type | Default | Bounds | Meaning |
|---|---|---|---|---|
| `min_top1_agreement` | float | `0.75` | 0–1 | Minimum fraction of denied applicants whose declared top-1 reason matches the measured top-1 feature. |
| `min_topk_jaccard` | float | `0.60` | 0–1 | Minimum mean Jaccard similarity between declared and measured top-k reason sets. |
| `decision_threshold` | float | `0.5` | strictly between 0 and 1 | An applicant with `p_bad >= decision_threshold` counts as denied. Fewer than 10 denied applicants is a setup error (exit 2). Overridable per run with `covenant check reason-codes --threshold`. |
| `max_denied_sample` | int | `500` | >= 10 | When more rows are denied than this, a seeded random subsample of this size is evaluated. |
| `background_size` | int | `200` | >= 10 | Rows in the SHAP background sample. A second background (seeded with `random_state + 1`) measures the measured side's own stability as `background_jaccard`; below 0.8 (fixed) the record is flagged background-sensitive. |
| `id_column` | string or null | `null` | — | Stable row key joining the data snapshot to a production artefact. **Required** when `reason_codes.method` is `custom` or `shapley`; must be a column of the data snapshot at check time. |
| `placebo` | bool | `true` | — | Run the Krivorotov & Richey placebo sub-check: shuffle a declared-irrelevant feature and confirm neither side's reasons move. |
| `placebo_epsilon` | float | `1e-3` | >= 0 | Share of total mean \|attribution\| mass below which a feature counts as irrelevant enough to serve as the placebo (scale-invariant across attribution paths). The qualifying feature with the smallest share is used; if none qualifies the placebo is skipped with a note. |
| `max_placebo_shift` | float | `0.10` | 0–1 | Fraction of denied rows whose top-k set may change under the placebo shuffle before the explanation pipeline is flagged as noisy. The flag is a diagnostic of the pipeline, recorded in the details — it never fails the check on its own. |
| `random_state` | int | `0` | — | Seed for the denied subsample, backgrounds and placebo shuffle. |

For the file-based methods (`custom`, `shapley`) the declared side is
unaffected by the placebo shuffle by construction, so
`declared_topk_shift` is recorded as null; only the measured shift is
tested.

The record notes what the check itself notes: the measured side
approximates the model, not ground truth — the record names the
attribution path used and reports its stability across two seeded
backgrounds.

### checks.monotonicity

Thresholds for Check 2 (`covenant check monotonicity`). Declared directions
are tested three ways: against constraints configured on the estimator
itself (XGBoost/LightGBM `monotone_constraints`, sklearn `monotonic_cst`),
against synthetic dominance pairs, and against ICE paths swept over each
feature's empirical quantiles. The check passes when the worst per-feature
violation rate is `<= max_violation_rate` **and** no configured constraint
contradicts the declared direction — a configured contradiction is a breach
on its own.

| Key | Type | Default | Bounds | Meaning |
|---|---|---|---|---|
| `max_violation_rate` | float | `0.05` | 0–1 | Maximum tolerated violation rate, applied to the worst rate across features and across both empirical tests. |
| `n_pairs` | int | `300` | >= 10 | Synthetic dominance pairs per direction-declared feature (hold every other feature fixed, move one; `p_bad` must not move against the declared direction). |
| `n_ice_rows` | int | `30` | >= 5 | Rows whose ICE path is swept per feature. |
| `ice_grid_points` | int | `9` | >= 3 | Grid points per ICE sweep, taken from the feature's 2nd–98th percentile quantiles. |
| `tolerance` | float | `1e-4` | >= 0 | Probability moves smaller than this do not count as violations. |
| `random_state` | int | `0` | — | Seed for pair and row sampling. |

### checks.features

Thresholds for Check 3 (`covenant check features`): declared versus used
features. Two comparisons with different weight: a **structural**
comparison of the declared list against the estimator's own
`feature_names_in_` (a mismatch in either direction is a breach; when the
estimator records no input names this is recorded as unavailable, never
silently skipped), and an **attribution screen** that flags documented
features the model measurably ignores (*dead* features — a
documentation-quality warning, never a breach).

| Key | Type | Default | Bounds | Meaning |
|---|---|---|---|---|
| `dead_feature_epsilon` | float | `1e-3` | >= 0 | Share of total mean \|attribution\| mass below which a documented feature is flagged as dead. Shares — not raw magnitudes — are thresholded, so the epsilon is scale-invariant across attribution paths (logit-space linear-exact versus probability-space permutation). |
| `sample_size` | int | `300` | >= 20 | Rows in the seeded sample the attribution screen scores. |
| `background_size` | int | `100` | >= 10 | Rows in the SHAP background sample. |
| `random_state` | int | `0` | — | Seed for sampling. |

### checks.exclusions

Thresholds for Check 4 (`covenant check exclusions`): excluded variables
stay out, and obvious proxies are surfaced. Two screens: an **attribution
screen** when an excluded variable is among the model's actual inputs, and
a **proxy screen** of pairwise association (\|Spearman\|, correlation
ratio, or bias-corrected Cramér's V, chosen by dtype) between each excluded
variable and every declared feature. Proxies are *surfaced, not proven
absent*: a weak pairwise association cannot rule out a multivariate proxy.
The check passes when no excluded variable breaches the attribution
threshold and (if `fail_on_proxies` is true) no pair exceeds
`max_association`.

| Key | Type | Default | Bounds | Meaning |
|---|---|---|---|---|
| `max_excluded_attribution` | float | `1e-3` | >= 0 | If an excluded variable reaches the model anyway, its **share of total mean \|attribution\| mass** must stay below this. A share at or above the threshold is a breach. Like `dead_feature_epsilon`, the share is scale-invariant across attribution paths. |
| `max_association` | float | `0.5` | 0–1 | Association strength above which an excluded-variable/feature pair is flagged as a potential proxy. Tune to your book. |
| `fail_on_proxies` | bool | `true` | — | Whether a flagged proxy fails the check or only warns. |
| `sample_size` | int | `300` | >= 20 | Rows in the seeded sample the attribution screen scores. |
| `background_size` | int | `100` | >= 10 | Rows in the SHAP background sample. |
| `random_state` | int | `0` | — | Seed for sampling. |

When the estimator records no input names and the excluded variable is not
a declared feature, whether it reaches the model cannot be verified from
the artefact; the record says so (`reaches_model: unknown`) instead of
guessing either way.

### report

Settings for `covenant report`, the deterministic validation report (same
inputs, same bytes).

| Key | Type | Default | Bounds | Meaning |
|---|---|---|---|---|
| `target_column` | string or null | `null` | — | Name of the 0/1 outcome column in the snapshot. **Required to render a report** — `covenant report` exits 2 without it (or use `--target` to override per run). The column must contain only 0/1 labels with both classes present. |
| `time_column` | string or null | `null` | — | Optional timestamp/ordinal column enabling drift-by-slice (score PSI across time slices). When unset, the drift section says so and is skipped. |
| `n_bootstrap` | int | `500` | >= 50 | Seeded bootstrap resamples behind every confidence interval. |
| `n_bins` | int | `10` | >= 4 | Bins for calibration (ECE), PSI and CSI. |
| `challenger` | bool | `true` | — | Fit a plain logistic challenger on the same features and report its lift with confidence intervals. |
| `random_state` | int | `0` | — | Seed for bootstrap and challenger. |

### What `covenant register` validates

Registration fails (exit 2) unless all of the following hold:

1. `covenants.yaml` and `governance.yaml` validate against the schemas on
   this page (unknown keys rejected, per-field error messages).
2. Every artefact path named under `reason_codes.parameters` — any of
   `coefficients`, `points_table`, `bins_table`, `attributions_file`,
   `reasons_file` that is present — points to a file that exists. A
   covenant that points at an artefact which does not exist is a claim
   that cannot be tested, so it is rejected at registration rather than
   discovered at check time.
3. The training snapshot loads (`.csv`, `.csv.gz` or `.parquet`/`.pq`
   only) and contains every declared feature as a column.
4. The model file loads and works as a Covenant model: an estimator (or
   pipeline) exposing `predict_proba`, with `positive_class` resolvable
   against its `classes_`.

---

## governance.yaml

The governance record for a registered model version. Validated by
`covenant register` and stored inside the inventory record. Governance
fields are *not* part of the version identity — see
[version_id](#version_id-identity).

| Key | Type | Required / default | Meaning |
|---|---|---|---|
| `owner` | mapping | **required** | `name` (string, min length 1) and `email` (must be a syntactically valid email address). |
| `intended_use` | string | **required**, min length 10 | The decision this model supports and its population. |
| `limitations` | list of strings | default `[]` | Known limitations, one per entry. |
| `materiality` | mapping | **required** | See below. |
| `review_date` | date | **required** | ISO date (e.g. `2027-01-01`) of the next scheduled review. |
| `vendor` | mapping or null | default `null` | Present for third-party models. See below. |

### materiality

| Key | Type | Required | Allowed values | Meaning |
|---|---|---|---|---|
| `tier` | int | **required** | `1`, `2`, `3` | Materiality tier; `1` is highest materiality. |
| `justification` | string | **required**, min length 20 | Why this tier: exposure, purpose, portfolio share. |

The justification is mandatory by design: SR 26-2 ties governance
intensity to materiality (exposure × purpose), so the tier is a decision —
and a decision needs a stated basis, not just a number.

### vendor

Third-party models carry the same validation obligation as in-house ones
(SR 26-2; RBI FREE-AI expects regulated entities to validate vendor
models). Either `null` or:

| Key | Type | Required |
|---|---|---|
| `name` | string | **required** |
| `product` | string | **required** |
| `version` | string or null | optional |
| `contact` | string or null | optional |

---

## Reason-code artefact files

The CSV files a reason-code method reads. Each is a production artefact —
a coefficient table, a scorecard points table, a bins table, an
attributions or reasons export — because that artefact, not the model, is
what goes stale, and testing it against the model's measured behaviour is
the whole point.

Rules common to all five:

- Paths in `reason_codes.parameters` are resolved relative to the
  directory containing `covenants.yaml`; absolute paths are used as-is.
- `covenant register` fails (exit 2) when a referenced artefact file does
  not exist. At check time, a missing, malformed or incomplete artefact is
  a setup error (exit 2), never a silent skip.
- On the declared side, higher attribution = stronger reason for denial,
  matching the measured side.

### coefficients (`difference_from_mean`)

| Column | Required | Meaning |
|---|---|---|
| `feature` | yes | Feature name. The table must cover **every** model feature (extra rows are ignored). |
| `coef` | yes | Linear coefficient. No empty cells. |
| `mean` | no | Reference value; treated as `0` when the column is absent. If the column is present, every cell must be filled. |
| `scale` | no | Scaling divisor; treated as `1` when the column is absent. If the column is present, every cell must be filled. |

Declared attribution per feature: `coef * (x - mean) / scale`.

This method needs all-numeric features; a model with categorical features
should use `method: custom` and export the production reasons instead.

```csv
feature,coef,mean,scale
income,-0.8,52000,18000
dti,1.3,0.31,0.12
utilization,0.9,0.42,0.25
```

### points_table (`most_points_lost`)

| Column | Required | Meaning |
|---|---|---|
| `feature` | yes | Feature name. The table must have at least one bin for every model feature. |
| `bin_lower` | numeric bins | Lower bound (inclusive). Absent column, empty cell or `-inf` means unbounded below. |
| `bin_upper` | numeric bins | Upper bound (exclusive). Absent column, empty cell or `inf` means unbounded above. |
| `value` | categorical bins | Exact category match (string equality). A feature may not mix categorical and numeric bins. |
| `points` | yes | Scorecard points of the bin. Numeric, no empty cells. |

Classic scorecard convention: **higher points = lower risk**. Declared
attribution per feature = (max points over that feature's bins) − (points
of the applicant's bin) — "points lost", so higher = stronger denial
reason.

Numeric bin matching is `bin_lower <= x < bin_upper`. Every applicant
value must fall in exactly one bin: a value in no bin is an error naming
the feature and value, and a value in more than one bin is an error too —
bins must be disjoint so the declared reason is unambiguous.

```csv
feature,bin_lower,bin_upper,value,points
income,,30000,,10
income,30000,60000,,25
income,60000,,,40
home_status,,,RENT,12
home_status,,,OWN,30
```

### bins_table (`univariate`)

| Column | Required | Meaning |
|---|---|---|
| `feature` | yes | Feature name. The table must have at least one bin for every model feature. |
| `bin_lower` | numeric bins | Lower bound (inclusive); empty/`-inf` = unbounded. |
| `bin_upper` | numeric bins | Upper bound (exclusive); empty/`inf` = unbounded. |
| `category` | categorical bins | Exact category match. A feature may not mix categorical and numeric bins. |
| `value` | yes | The bin's bad rate (or score); higher = riskier. Numeric, no empty cells. |

Declared attribution per feature = value(applicant's bin) − unweighted
mean of that feature's bin values. The reference point is the table's own
mean, not a portfolio average, so the artefact is self-contained — the
check never needs the data snapshot to interpret it.

Bin matching follows the points-table rules (`bin_lower <= x <
bin_upper`; empty/`-inf`/`inf` cells unbounded; disjoint; full coverage)
with one deliberate divergence: categorical bins here match on a
`category` column, not `value`, because in this table `value` is the
bin-level payload. In the points table the payload column is `points`,
which leaves `value` free to hold the category. The divergence is real
and intentional; copy the header rows from the examples.

```csv
feature,bin_lower,bin_upper,category,value
income,,30000,,0.19
income,30000,60000,,0.11
income,60000,,,0.04
home_status,,,RENT,0.16
home_status,,,OWN,0.07
```

### attributions_file (`shapley`)

| Column | Required | Meaning |
|---|---|---|
| *(id column)* | yes | Named by `checks.reason_codes.id_column`; must match a column of the data snapshot. Duplicate ids are an error. |
| one column per declared feature | yes | Numeric attribution, **higher = pushes toward denial**. No empty cells; missing feature columns are an error. |

Rows are joined to the scored data on the id. The file must contain an
attribution row for **every denied row the check samples**: the check
scores the snapshot, takes the rows with `p_bad >= decision_threshold`,
and evaluates a seeded subsample of up to `max_denied_sample` of them —
any sampled id absent from the file is an error reporting the count and
the first five missing ids. Practical guidance: export attributions for
every row with `p_bad >= decision_threshold` on the same snapshot; that
is a superset of anything the check can sample.

```csv
application_id,income,dti,utilization
A-1041,0.42,1.10,0.35
A-1042,0.66,0.20,0.91
A-1043,0.05,1.72,0.44
```

### reasons_file (`custom`)

| Column | Required | Meaning |
|---|---|---|
| *(id column)* | yes | Named by `checks.reason_codes.id_column`. Duplicate ids are an error. |
| `reason_1` … `reason_k` | at least one | The reason codes production sent, most important first. Every column whose name starts with `reason_` is read. Empty cells are simply omitted from that applicant's set. |

`reason_1` is the declared top-1; the non-empty cells across the
`reason_*` columns form the declared top-k set. Coverage works as for the
attributions file: every denied row the check samples must have a row in
the file, so export reasons for every row with
`p_bad >= decision_threshold` on the same snapshot.

Spell the reasons exactly as the declared feature names: the check
compares them for equality against measured feature attributions, so a
human-readable phrasing ("debt burden too high") can never agree with the
measured side ("dti").

```csv
application_id,reason_1,reason_2,reason_3,reason_4
A-1041,dti,utilization,income,
A-1042,utilization,income,,
A-1043,dti,income,utilization,
```

---

## Exit codes and check records

### Exit codes

Exit codes are the contract — they are what lets a check sit in CI and
block a deployment:

| Code | Meaning |
|---|---|
| `0` | Pass. Every check that ran found the covenant held. |
| `1` | Covenant breach: a check ran to completion and its measured result crossed a stated threshold. |
| `2` | Usage or setup error: invalid document, missing file or column, missing/malformed artefact, too few denied applicants, unsupported data format — anything that prevented a check from producing a verdict. Unexpected exceptions are also reported as `error: …` with exit 2; `covenant --debug …` re-raises them with the full traceback. |

`covenant check all` runs every configured check and prints one summary.
Its exit code is the worst outcome, with one deliberate hardening: if any
configured check could not run (a setup error), the command exits **2
even if every check that did run passed**. A gate that silently stops
running one of its checks must not stay green — a renamed artefact or a
filtered row would otherwise switch off the adverse-action check while CI
keeps passing, which is exactly the silent degradation a governance gate
exists to prevent. Skipped checks are listed as `SKIPPED` with the error.

### Store layout

All records live as flat YAML under `.covenant/` in your own repository
(override with `--store` on any command). History is `git log`;
comparison is a diff.

```text
.covenant/
├── models/
│   └── <model_name>/
│       └── <version_id>.yaml          # inventory record from `covenant register`
└── checks/
    └── <model_name>/
        ├── <check>-<hash12>.yaml      # one record per distinct check outcome
        └── runs.log                   # append-only run timestamps
```

The inventory record (`ModelRecord`) contains: `record_schema`,
`version_id`, `model_name`, `created_at`, the three content hashes
(`hashes.model_sha256`, `hashes.data_sha256`, `hashes.covenants_sha256`),
model info (class path and library versions), data info (path, shape,
columns), and full copies of the governance record and the covenants.
`covenant show` prints it; `covenant diff` compares two versions
structurally (`~` changed, `+` added, `-` removed), suppressing
`created_at` unless `--all` is passed. Version ids may be abbreviated to
any unambiguous prefix.

### Check records and replayability

Every check writes a `CheckRecord` with these fields:

| Field | Content |
|---|---|
| `check` | Check name (`reason-codes`, `monotonicity`, `features`, `exclusions`). |
| `model_name` | From the covenants. |
| `passed` | The verdict. |
| `metrics` | The measured rates (every check produces a rate against a stated threshold, not just a verdict). |
| `thresholds` | The thresholds the verdict was judged against. |
| `n_evaluated` | How many rows/pairs the verdict rests on. |
| `inputs` | SHA-256 of the model file, the data snapshot and the (validated) covenants — the record names exactly what it was computed from. |
| `config` | The full check configuration after any CLI overrides. |
| `details` | Per-feature/per-variable breakdowns, score-band strata, worst disagreements, notes. |
| `record_sha256` | Hash stamp, see below. |

**record_sha256 convention.** The record is serialised to canonical JSON
(sorted keys, fixed separators) with the `record_sha256` field removed,
and the SHA-256 of those bytes is stamped into the field. The record's
filename is `<check>-<first 12 hex chars>.yaml`.

**Replayability.** The record contains only what the check computed from
its inputs — no timestamps, no environment noise — so the same model,
snapshot and covenants produce a byte-identical record at the same path.
Writing is idempotent: an existing record is never rewritten. Run
timestamps live instead in the append-only `runs.log` sidecar, one line
per run:

```text
2026-08-20T14:03:11+00:00 reason-codes 3f9c1a2b4d5e PASS
```

(UTC ISO timestamp, check name, 12-character record hash, `PASS` or
`BREACH`.) So the record is citable evidence and the log is the run
history; re-running a check re-cites the same record and appends a line.

### version_id identity

A model version is identified by what it is, what it was trained on, and
what it claims:

```text
version_id = sha256(canonical JSON of {model, data, covenants hashes})[:12]
```

- `model_sha256` — the model file's bytes exactly as supplied (never
  re-serialised: pickle bytes are not stable across library versions).
- `data_sha256` — a deterministic digest of the snapshot's schema and
  values via pandas' stable row hashing. Row order and column order both
  matter: a reordered snapshot is a different snapshot. The digest does
  not depend on the file format or a parquet writer version.
- `covenants_sha256` — canonical JSON of the *validated* document with
  all defaults filled in. Reformatting the YAML, editing comments, or
  writing a default value explicitly does not change it; changing any
  effective value does.

Governance is stored in the record but is **not** part of the identity:
owner, review date and tier can be amended without creating a new
version. A re-fit model, a new snapshot or a changed covenant always
creates one.

Covenant produces evidence; your validators and auditors decide. Nothing
on this page — thresholds, defaults, or a green exit code — is a claim of
compliance.
