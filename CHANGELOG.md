# Changelog

## 0.2.0 — unreleased

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
