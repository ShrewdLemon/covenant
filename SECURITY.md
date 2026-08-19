# Security

## Loading a model file executes arbitrary code

**This is the most important thing on this page.** Covenant loads model
artefacts with `joblib`/`pickle`, and unpickling runs whatever code the file
contains. `covenant check ...` and `covenant register` on a malicious
`.joblib` is remote code execution on your machine or your CI runner.

Point Covenant only at artifacts you trust — in practice, at models your own
team trained and stored. Never run it against a model file received from an
untrusted source, downloaded from the internet, or attached to a bug report.
This is a property of the pickle format, not something Covenant can validate
its way around.

Safer serialization (`skops` for sklearn models, native `save_model` for
XGBoost/LightGBM) is on the roadmap.

## Reporting a vulnerability

Report vulnerabilities privately via GitHub's private security advisories on
[ShrewdLemon/covenant](https://github.com/ShrewdLemon/covenant/security/advisories/new).
Please do not open a public issue for a security problem. Include the version,
a reproduction, and what an attacker gains.

## Supported versions

Only the latest minor release receives security fixes. If you are on an
older minor, upgrade first and re-test.
