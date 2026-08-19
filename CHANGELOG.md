# Changelog

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
