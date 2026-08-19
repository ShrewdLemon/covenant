# Contributing to Covenant

Thanks for helping. Covenant is small on purpose; the bar for a change is
that it keeps the properties below intact.

## Dev setup

```bash
uv venv
uv pip install -e ".[dev]"
```

## Test and lint

```bash
.venv/bin/python -m pytest -q          # full suite
.venv/bin/python -m pytest tests/test_check_monotonicity.py -q   # one file
.venv/bin/ruff check .                 # lint; line-length 100
.venv/bin/ruff format --check .        # formatting
```

CI also runs the broken/fixed scorecard demos in `examples/` and asserts
their exit codes — the README's central claims are themselves tested.

## Expectations

- **Behaviour changes come with tests.** A check, a schema rule, a CLI exit
  code — if it changes what the tool does, a test pins it.
- **Ruff clean.** `ruff check` and `ruff format --check` must pass with
  nothing to report.
- **Determinism is a feature, preserve it.** Check records must be
  byte-identical for identical inputs: no timestamps in records (run
  timestamps go to `runs.log`), every use of randomness takes a fixed seed,
  and hashing stays canonical (`sort_keys=True`, fixed separators). If your
  change makes two runs on the same inputs produce different bytes, it will
  be rejected.
- **Never write compliance claims** into docs, output, or error messages.
  Covenant produces evidence; validators and auditors decide. A check
  reports a rate against a stated threshold, never "compliant". Proxy
  language is always "surfaced, not proven absent".
- **Errors that mean "the user set this up wrong"** raise `CheckSetupError`
  or `DeclaredMethodError` with a message that says what to fix, not a
  traceback.
- Cite only what `docs/research-guide.md` documents; do not invent section
  numbers or quotations.

## Scope

Metrics libraries, drift monitors, and fairness toolkits already exist
(see "What Covenant is not" in the README). Contributions that re-implement
them will be redirected; contributions that make claims checkable are the
point.
