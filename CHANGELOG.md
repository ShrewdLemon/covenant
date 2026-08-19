# Changelog

## 0.3.0 — unreleased

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
