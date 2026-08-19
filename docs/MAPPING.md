# Regulator mapping

Covenant produces evidence; your validators and auditors decide — nothing in
this table is a claim of compliance. Verify every citation against the
primary texts before relying on this table: the sources below are cited as
documented in [`docs/research-guide.md`](research-guide.md), and
section-level pinpoints are marked TODO until they have been read against
the letter.

Context, as documented in the research guide: **SR 26-2** (17 Apr 2026)
supersedes SR 11-7, ties governance intensity to materiality (materiality =
exposure × purpose), expects the inventory to support enterprise-level
visibility of concentrations and dependencies, and scopes GenAI/agentic
systems out while keeping ML credit models in. The **RBI FREE-AI** committee
report (Aug 2025) surveyed regulated entities using AI and found only ~15%
used interpretation tools, ~35% validated for bias, ~18% kept audit logs and
~21% monitored drift; it recommends a board-approved AI policy (Rec 14) and
a risk-based AI audit framework (Rec 24), expects AI inventories to be
available for supervisory inspection, and expects regulated entities to
validate third-party models as rigorously as their own.

| The ask | Source (as documented) | Covenant artefact | Status |
|---|---|---|---|
| Model inventory supporting enterprise-level visibility of concentrations and dependencies | SR 26-2 (section pinpoint TODO); RBI FREE-AI — AI inventories available for supervisory inspection | `covenant register` / `diff` / `show` / `list`: one content-addressed YAML entry per model version under `.covenant/`, keyed by SHA-256 of the model artefact, training snapshot and covenants file; history is `git log` | shipped |
| Governance intensity proportionate to materiality, with the tiering justified rather than asserted | SR 26-2 — materiality = exposure × purpose; proportionate governance (section pinpoint TODO) | Governance schema requires a materiality tier **with a mandatory justification**; an empty justification or a typo'd key is rejected with a readable error, not silently accepted | shipped |
| Vendor / third-party models carry validation obligations of their own | RBI FREE-AI — regulated entities are expected to validate third-party models as rigorously as their own; SR 26-2 (section pinpoint TODO) | Vendor block in the inventory record; every check runs identically against a vendor artefact — a third-party model gets no lighter treatment than an in-house one | shipped |
| Adverse-action reasons must be specific and accurate and reflect the factors *actually* scored; checklists are insufficient, even for complex algorithms | CFPB Circular 2022-03 and the Sept 2023 CFPB guidance | Check 1 `covenant check reason-codes`: declared reasons vs measured SHAP attributions for denied applicants — top-1 agreement and top-k Jaccard against thresholds you set, stratified by score band, with background-sensitivity reported. Covenant reports the disagreement; it does not adjudicate which side is right | shipped |
| Documented behaviour must be conceptually sound: a direction the documentation declares must be one the model actually exhibits (feature-highlighting explanations silently assume monotonicity) | Barocas, Selbst & Raghavan (arXiv 1912.04930); Provenir (2022) — constraints before SHAP-based reasons; SR 26-2 conceptual-soundness pinpoint TODO | Check 2 `covenant check monotonicity`: declared vs configured (constraints read off the estimator) vs empirical (dominance pairs + ICE paths), per-feature violation rates against a stated threshold | shipped |
| Declared feature set matches the features the model actually uses | SR 26-2 (section pinpoint TODO) | Check 3 `covenant check features`: declared vs used features | shipped |
| Excluded variables stay excluded, and their obvious proxies are surfaced | CFPB / fair-lending framing as documented in the research guide (pinpoint TODO) | Check 4 `covenant check exclusions`: measured influence of declared exclusions plus an association screen — proxies are **surfaced, not proven absent** | shipped |
| Ongoing monitoring and outcome analysis | SR 26-2 (section pinpoint TODO); RBI FREE-AI survey — only ~21% of AI-using entities monitored drift | `covenant report`: discrimination, calibration, stability and drift with bootstrap CIs, rendered deterministically and mapped line-by-line to these asks | shipped |
| Drift / stability statistics | RBI FREE-AI survey — ~21% monitored drift; SR 26-2 (section pinpoint TODO) | PSI/CSI and stability sections of `covenant report`, computed in-house, deterministic bytes | shipped |
| Audit trail: runs are logged, evidence is citable and replayable | RBI FREE-AI survey — only ~18% of AI-using entities kept audit logs | Hash-stamped check records (no timestamps in the record body), append-only `runs.log` per model, and byte-identical replay: identical inputs produce identical, hash-addressed records at the same path | shipped |
| Board-approved AI policy and a risk-based AI audit framework | RBI FREE-AI Rec 14 (board-approved AI policy) and Rec 24 (risk-based AI audit framework) | Out of scope as a deliverable — a board policy is organisational, and Covenant does not write it. Covenant's records and report are evidence inputs such a policy and audit framework can point at | out of scope (evidence inputs only) |

Two honesty notes, restated because they govern how this table may be used:

- A row marked "shipped" means the artefact exists and runs; it never means
  the ask is satisfied. Passing a check is a rate above a threshold you
  chose, recorded so a validator can inspect it.
- The proxy screen in Check 4 surfaces obvious proxies; it does not and
  cannot prove their absence.
