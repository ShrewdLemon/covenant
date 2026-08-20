# Changelog

## 0.6.0 — 2026-08-20

### Added
- **`covenant compare A B DATA`**: champion vs challenger as a hash-stamped
  record — AUC/KS/Brier/ECE per model with seeded bootstrap CIs plus
  A−B paired-bootstrap deltas (identical resample rows on both score
  vectors), significance marked when the interval excludes zero. Evidence,
  not a gate; the record says which snapshot it was scored on.
- **Shapley-export provenance**: when `reason_codes.method: shapley` and the
  declared export is numerically indistinguishable from the check's own
  measured attributions, the record says so — freshness verified, not
  independent production evidence.
- `covenant checks` listing now shows each record's model version id
  (recomputed from its input hashes) and last-run timestamp from runs.log.
- `checks.reason_codes.background_stability_floor` (default 0.8) — the
  sensitivity floor is covenant policy, no longer a hardcoded constant.
- `checks.exclusions.association_sample_size` — seeded row sample for the
  proxy screen on very large books (default: full snapshot).

### Deferred (documented, not shipped)
- Exact tree-shap for HistGradientBoosting fitted on native pandas
  categoricals: the fitted category mapping is only reachable through
  sklearn private internals; the permutation fallback remains correct,
  just slower, and the attribution docstring says so.

## 0.5.0 — 2026-08-20

### Fixed
- **Confirmed misalignment bug**: `configured_directions` no longer aligns a
  bare monotone-constraint sequence read off a pipeline's final step to the
  covenant's feature order — the transformed column order need not match,
  and the guess was measured mislabeling a categorical as `increases_risk`.
  Unreadable constraints are now recorded as unreadable.

### Changed
- **Exact attribution through pipelines**: one-hot `ColumnTransformer` +
  `LogisticRegression` scorecards now take the closed-form `linear-exact`
  path (~160x faster than permutation SHAP, exact to machine precision,
  one-hot columns aggregated per raw feature), and simple pipelines around
  tree boosters take `tree-shap` (~19x). Covenant feature order no longer
  needs to match the model's fitted order (resolved by name).
- **`covenant report` is now the full evidence document**: metrics sections
  are labeled with their snapshot, `--holdout` drives out-of-sample
  discrimination/calibration as the headline (in-sample follows, labeled as
  flattering), all four check verdicts are embedded with record hashes, and
  `--governance` embeds owner, materiality tier + justification verbatim,
  review date and vendor block.
- `covenant register` fails when a declared reason-code artefact file does
  not exist (a claim that cannot be tested is rejected at write time).
- `covenant check all` now exits 2 when any configured check is skipped by
  a setup or artefact error: an incomplete gate must not pass. Found by a
  cold-start "stranger test" of the published package — a renamed
  attributions file silently switched off the adverse-action check while
  CI stayed green.

### Added
- `docs/reference.md`: the complete schema reference — every covenants.yaml
  and governance.yaml key with types/defaults/allowed values, exact column
  contracts for all five reason-code artefact files, exit-code semantics
  and the record/replayability conventions. On the docs site nav.
- `load_data` accepts `.csv.gz`.
- `check exclusions` prints and records the argmax association pair; the
  background-sensitivity flag now names the knob that fixes it
  (`checks.reason_codes.background_size`).
- **Taiwan Credit demo** (`examples/taiwan_credit/`): 30,000 real
  credit-card clients (UCI, Yeh & Lien 2009, committed). Leaky model on
  `sex`/`marriage` breaches checks 3-4; clean model passes all four with
  measured thresholds; ~60s full sequence at 30k rows, CI-asserted.
- **HMDA mortgage demo** (`examples/hmda_mortgage/`): 25,273 real 2023
  Rhode Island applications (CFPB public data, committed) with
  `derived_race`/`derived_sex`/`derived_ethnicity` excluded by covenant
  and the historical-decision framing stated explicitly. CI-asserted.
- **German Credit demo** (`examples/german_credit/`): Covenant on real,
  recognized public data (UCI / OpenML `credit-g`, committed for offline
  determinism). A model quietly fitted on `personal_status` (sex/marital
  status) and `foreign_worker` breaches `check features` and
  `check exclusions`; the clean model passes all four checks with the real
  proxy associations surfaced below threshold; the report renders real
  AUC/PSI/challenger numbers byte-identically. All asserted in CI.

## 0.4.0 — 2026-08-19

### Added
- **OptBinning adapter** (`covenant.adapters.export_scorecard_points`):
  export a fitted `optbinning.Scorecard` as a Covenant points table using
  the exact split points from the binning objects (never the display-rounded
  table strings), ready for `reason_codes.method: most_points_lost`.
  Special/Missing bins are not exported; a reversed scorecard negates
  points via `higher_points_lower_risk=False`.
- **EBM exact attribution path** (`"ebm-exact"`): InterpretML
  `ExplainableBoostingClassifier` models are explained by their own shape
  functions — logit-space `eval_terms`, centered on the background, with
  pairwise interaction terms split equally between their features. No
  sampling, no codec; categorical features consumed raw.
- **skops loading**: `.skops` model files load through `skops.io.load`,
  which refuses untrusted types instead of executing them — the
  recommended way to persist models for Covenant
  (`pip install "covenants[skops]"`).
- Optional extras: `covenants[scorecard]`, `[interpret]`, `[skops]`,
  `[boosters]`, `[integrations]` (all of them), `[docs]`.
- xgboost/lightgbm integration tests: `configured_directions` parsing on
  real boosters and the tree-shap path (skipped when the libraries are
  absent; CI runs them in a dedicated job).
- mypy in CI (strict enough to catch real defects, pydantic plugin on);
  the package ships `py.typed`.
- Documentation site (mkdocs-material) at
  https://shrewdlemon.github.io/covenant/, built `--strict` and deployed
  from CI.

## 0.3.1 — 2026-08-19

The 0.3.0 wheel uploaded to PyPI was accidentally built from a tree that
predated the fixes below, so 0.3.0 is yanked and 0.3.1 is the release to
install. The features listed under 0.3.0 are all included here.

### Fixed (from an adversarial review of the 0.3.0 diff)
- PSI computed at value level when the baseline has ≤ n_bins distinct
  values: binary flags, zero-inflated counts and constant columns now
  measure population shifts instead of silently reporting 0.0.
- `correlation_ratio`/`spearman_abs` drop non-finite pairs, so a single
  `inf` row can no longer turn a perfect proxy into an unflagged NaN.
- Check 4 consults `feature_names_in_` before snapshot presence: an
  excluded variable the model reads but the snapshot omits is a setup
  error, never a silent pass.
- `covenant check all` skips (rather than aborts) on artefact and
  registration errors and exits 2 when no check could run at all.
- Declared artefacts fail loudly on empty payload cells, overlapping bins
  and missing files, naming the table and value.
- The report's reproduce command uses file basenames, so rendered bytes
  no longer depend on how input paths were spelled.
- Dead-feature, excluded-attribution and placebo screens threshold on each
  feature's share of total attribution mass — scale-invariant across the
  logit-space linear-exact and probability-space permutation paths; checks
  3–4 record `attribution_path`.

## 0.3.0 — 2026-08-19 (yanked: the uploaded wheel predated 0.3.1's fixes)

### Added
- **Check 3 — features** (`covenant check features`): declared vs
  actually-used features. Undocumented model inputs and documented-but-unseen
  features breach; documented features with measurably zero attribution are
  surfaced as warnings.
- **Check 4 — exclusions** (`covenant check exclusions`): an excluded
  variable that still reaches the model must show ≈ 0 attribution; every
  excluded variable is screened for association (|Spearman|, correlation
  ratio, bias-corrected Cramér's V) against used features. Proxies are
  surfaced, not proven absent.
- **`covenant report`**: deterministic validation report — discrimination,
  calibration, stability (PSI/CSI vs holdout), drift by time slice, the
  monotonicity verdict, and an out-of-fold logistic challenger's lift, all
  with seeded bootstrap CIs. Byte-identical on re-render; embeds its own
  hash and each figure's hash; every section carries a "Maps to" line.
  CI proves the byte-identity claim with a double render and `diff -r`.
- All five Krivorotov & Richey declared reason-code methods:
  `most_points_lost` (scorecard points tables), `univariate` (bin-level
  score tables), and `shapley` (production attribution exports joined on
  `id_column`) join the existing `difference_from_mean` and `custom`.
- Exact attribution fast paths: closed-form contributions for logistic
  models and scaler pipelines ("linear-exact"), TreeExplainer for bare tree
  ensembles ("tree-shap"), permutation SHAP as the universal fallback —
  every check record names the path that produced its measured side.
- Placebo sub-check in Check 1 (K&R): shuffle a measurably irrelevant
  feature and report how much each side's top-k moves; flagged as noisy
  above a threshold, recorded but never a breach on its own.
- `covenant.metrics`: dependency-light AUC/KS/Gini/Brier/ECE/PSI/CSI with
  seeded percentile-bootstrap CIs and a paired bootstrap for challenger
  deltas.
- `docs/MAPPING.md` (regulatory mapping with honesty markers), SECURITY.md,
  CONTRIBUTING.md, CITATION.cff, `py.typed`.

## 0.2.0 — 2026-08-18 (git only; superseded by 0.3.0 before publishing)

### Added
- **Check 2 — monotonicity** (`covenant check monotonicity`): declared vs
  configured vs empirical monotone directions. Empirical side uses dominance
  pairs and ICE paths with per-feature violation rates against a stated
  threshold; configured side reads `monotone_constraints` / `monotonic_cst`
  off XGBoost, LightGBM and sklearn boosters, and a configured direction
  that contradicts the covenant is a breach on its own.
- `covenant check all`: every configured check, one summary, one combined
  exit code.
- `covenant show` (pretty-print a record), `covenant checks` (list check
  records with verdicts); `diff`/`show` accept version-id prefixes.
- Categorical features in Check 1: values are code-encoded for SHAP's masker
  and decoded before the model scores, so `ColumnTransformer`/`OneHotEncoder`
  pipelines work; each categorical feature gets one whole-feature attribution.
- `positive_class` covenant field, resolved against the estimator's
  `classes_`, replaces the hard-coded assumption that column 1 is "bad".
- Demo: unconstrained GBM breaches monotonicity on a U-shaped `loan_amount`
  effect the covenant declares monotone; the `monotonic_cst`-constrained GBM
  passes. CI asserts both exit codes.

### Fixed
- Check records are replayable: run timestamps moved out of the record into
  an append-only `runs.log`, so identical inputs produce byte-identical,
  hash-addressed records and re-runs never duplicate files.
- `custom` reason codes now join the data on a mandatory
  `checks.reason_codes.id_column` instead of row position, which broke
  silently the moment one row was filtered upstream.
- `covenant diff` no longer shows `created_at` noise (`--all` restores it).
- Unexpected exceptions no longer leak tracebacks: the CLI reports
  `error: …` and exits 2 (`--debug` re-raises).
- `load_model` reports both loader errors instead of swallowing the joblib
  failure.

## 0.1.0 — 2026-08-18

First release. Content-addressed model inventory (`register`, `diff`,
`list`, `init`) as flat YAML under `.covenant/`; Check 1
(`check reason-codes`): declared adverse-action reason codes vs measured
SHAP attributions, stratified by score band, with background-sensitivity
reporting and hash-stamped records; strict pydantic schemas with a mandatory
materiality-tier justification; broken/fixed scorecard demo asserted in CI.
Published to PyPI as `covenants`.
