# Covenant — Research Guide and Technical Stack
### What exists, what the papers say, where the gap really is, and what to build it with · August 2026
*(Project renamed from Pramana to Covenant, 18 Aug 2026. PyPI distribution: `covenants`; import name: `covenant`.)*

---

## 1. The landscape, in three tiers

### 1a. Commercial model-risk platforms (the incumbents you are *not* competing with on breadth)

| Who | What they do | Why it matters to you |
|---|---|---|
| **ValidMind** (Palo Alto) | Enterprise MRM platform: inventory, documentation, validation workflows, monitoring; sells "audit-ready evidence for SR 26-2, EU AI Act, SS1/23, OSFI E-23". Has an **open-source Python library** (`pip install validmind`, GPL-family licence, extras include `credit-risk`, `explainability`, `xgboost`) that runs test suites and generates documentation, but its purpose is to feed their platform. In June 2026 they shipped **Atryum**, an open control plane for AI agents. | The closest thing to a direct competitor on the *validation report* side. Do not copy or vendor their code (licence). Do read their test catalogue to see what a bank expects a report to contain. |
| **Solytics Partners** (India) | MRM Vault (inventory), Nimbus Uno (monitoring), and **MoDeVa** (model validation library, successor to PiML) — explicitly marketed against **FREE-AI** pillars in India. | Proof that the Indian market for exactly this exists and is being sold to *today*. They sell to large banks; the NBFC long tail is open. |
| SAS MRM, IBM OpenPages, Domino, Fairly, Arize / Fiddler | Enterprise governance and ML observability suites. | Reference architectures for what "inventory", "attestation", "monitoring" mean to a big bank. Nobody in this row does `pip install` and a CI exit code. |

### 1b. Open-source libraries adjacent to Covenant (what you should *use* or *learn from*, not rebuild)

| Library | Origin | What it does well | Relationship to Covenant |
|---|---|---|---|
| **PiML → MoDeVa** | Wells Fargo Corporate Model Risk (Sudjianto, Zhang) | Inherently interpretable models (GAM, EBM, GAMI-Net, XGB1/XGB2), plus model-agnostic diagnostics: weak-spot, overfit, reliability, **robustness, resilience**, fairness. Adopted by multiple banks. Philosophy: *interpret* the model (exact) beats *explain* it (post-hoc). | The best existing "conceptual soundness + outcome analysis" toolkit. Covenant should not re-implement its diagnostics; it should be able to *ingest* a PiML/MoDeVa model and add the governance layer on top. Cite it. |
| **InterpretML / EBM** | Microsoft | Explainable Boosting Machines with exact shape functions. | Ideal demo model: monotonicity and reason codes can be read *exactly* off the shape functions, so your checks have ground truth. |
| **SHAP** | Lundberg | Tree/Linear/Kernel explainers. | Your attribution engine for Check 1 — with the caveats in §3. |
| **FINRA Model Validation Toolkit** | FINRA (US regulator) | Credibility of metrics under small samples, thresholding, data-quality supervisor, interpretable nets. | A regulator wrote a validation toolkit. Cite for legitimacy; borrow the "credibility under small samples" idea for your CIs. |
| **Evidently, NannyML, deepchecks** | MLOps community | Drift, data quality, performance monitoring, HTML reports. | Do not depend on them (heavy, opinionated). Implement PSI/CSI/drift yourself (it's ~200 lines) and offer adapters later. |
| **OptBinning, scorecardpy, toad, skorecard** | credit-scoring community (ING wrote skorecard) | Optimal binning, WoE, scorecard building. | Your Brihas stack overlaps. Covenant consumes their scorecards; add an adapter for OptBinning's `Scorecard` object because it's the most-used open scorecard class. |
| **openRiskScore, openLGD, transitionMatrix** | Open Risk (Amsterdam) | PD/LGD estimation, federated scoring, rating transitions. | Cite as the open credit-risk commons; potential home for a Covenant mention. |
| **creditriskengine** | PyPI, 2026 | "scikit-learn of credit risk": Basel/CRR3 RWA, IFRS 9 / **Ind AS 109 ECL incl. RBI ECL Master Direction 2026**, IRB, model validation. | Recent, ambitious, and India-aware. Read its validation module to avoid duplicating; consider interoperability. |
| **Fairlearn, AIF360, SolasAI** | Microsoft, IBM, SolasAI | Fairness metrics and disparity testing. | Fairness is *not* Covenant's core; ship it as an optional extra via Fairlearn, and point to SolasAI for the regulatory-grade version. |
| **MLflow Model Registry** | Databricks | Model versioning. | Versioning ≠ governance. Covenant's inventory records the *claims* about a model, not just its artefact. Say this explicitly in the README. |

### 1c. Papers, guidance and practitioner writing that define the problem

The most important finding of this research: **the specific thing Covenant checks — that a credit model's stated explanations agree with its measured behaviour — is discussed in the literature as a *risk*, measured in a few papers as an *experiment*, and shipped by nobody as a *tool*.**

- **CFPB Circular 2022-03** and the **Sept 2023 CFPB guidance**: adverse-action reasons must be *specific and accurate*, must reflect the factors *actually* scored, and checklists are insufficient — even for complex algorithms. This is the legal hook for Check 1 in the US, and it is the reason "reason codes vs attributions" is a compliance question, not an academic one.
- **Nair, Sudjianto et al. (Wells Fargo, arXiv 2204.12365, 2022)** — *Explaining Adverse Actions in Credit Decisions Using Shapley Decomposition*: derives adverse-action explanation from first principles for additive and low-interaction models, generalises to Shapley and Baseline-Shapley (B-Shap). Your Check 1 should implement B-Shap as one of the "measured" attribution methods.
- **Krivorotov & Richey (SSRN 4133915, 2022)** — *Explaining Denials: Adverse Action Codes and Machine Learning*: compares four AAC methods (Shapley, Most-Points-Lost, Difference-from-Mean, Univariate binning) on XGBoost card and mortgage models, with a **placebo testing framework**; finds Univariate deviates most from Shapley, differences concentrate near the decision boundary, and Most-Points-Lost / Mean are less robust to perturbation. **This is the paper closest to Check 1** — it measures the disagreement Covenant automates. Cite it, borrow the four methods as selectable `reason_codes.method`, and borrow the placebo test as a robustness sub-check.
- **FinRegLab (2023)** — *Machine Learning Explainability & Fairness: Insights from Consumer Lending* (empirical white paper) and the companion *Policy Analysis*: defines **consistency** as "the degree to which different tools identify the same drivers for the same model", measures fidelity across SHAP/LIME/etc. on real underwriting models, and discusses "true to the model" vs "true to the data" explanations. Use their definition of consistency as your metric's citation.
- **Pace Analytics (2024)** — *Using Explainable AI to Produce ECOA Adverse Action Reasons: What Are the Risks?*: the risk is in **implementation** — KernelSHAP background/baseline choice, sampling, simplifying assumptions. Covenant must expose the background choice and report sensitivity to it, or its "measured" side is itself unreliable.
- **Sei AI (May 2026, practitioner)** — the failure mode in one line: *"the model that issues the decision should be the model that produces the reasons"*; explanations generated from a different (surrogate/stale) model drift from the actual driver. That is exactly the bug your broken-scorecard demo simulates.
- **Barocas, Selbst & Raghavan (arXiv 1912.04930)** — *The Hidden Assumptions Behind Counterfactual Explanations and Principal Reasons*: feature-highlighting explanations silently assume **stability and monotonicity**; without monotonicity constraints, "improve the flagged feature" may not improve the outcome. Cite for Check 2's rationale.
- **Provenir (2022)** — apply **both monotonic and interaction constraints** before SHAP can be used for AACs; interactions break additivity. Cite for why Check 2 and Check 1 are coupled.
- **EUI RSC 2025/24** — monotonicity-constrained XGBoost on Lending Club: restrictions change PDs and reason importance in measurable ways. Useful for the demo narrative and dataset choice.
- **arXiv 2607.04103 (July 2026)** — an SR 26-2-compatible framework for GenAI risk in banks; its consumer-credit section stresses **explanation fidelity**: polished, plausible adverse-action text can still fail to reflect the actual drivers. Cite to show the concern is live in 2026, and note SR 26-2 itself scopes GenAI out while keeping ML credit models in.
- **RBI FREE-AI report (Aug 2025)** — survey: of entities using AI, only ~15% used interpretation tools, ~35% validated for bias, ~18% kept audit logs, ~21% monitored drift; recommendations include board-approved AI policy (Rec 14), AI inventories for supervisory inspection, a risk-based AI audit framework (Rec 24), an AI Compliance Toolkit, and explicit expectations that REs validate third-party models as rigorously as their own. These numbers are your blog post's opening paragraph.
- **SR 26-2 (17 Apr 2026)** — supersedes SR 11-7; materiality = exposure × purpose; proportionate governance; inventory expected to support enterprise-level visibility of concentrations and dependencies; GenAI/agentic out of scope. Read the letter and attachment; cite by section.
- **Sudjianto & Zhang (arXiv 2111.01743)** — *Designing Inherently Interpretable ML Models*: the "interpret vs explain" argument. Covenant should be honest that post-hoc attributions are an *approximation*, and should support exact interpretation when the model allows it (EBM/GAM shape functions).

---

## 2. Where the gap really is (state it this precisely in the README)

Crowded: validation reports (ValidMind, PiML/MoDeVa, FINRA MVT, creditriskengine), drift monitoring (Evidently, NannyML), fairness (Fairlearn, SolasAI), interpretable modelling (PiML, InterpretML), scorecard building (OptBinning et al.).

Open:
1. **Explanation-consistency as a CI gate.** Reason codes vs attributions, declared vs empirical monotonicity, declared vs used features, declared exclusions vs measured influence — with a rate, a threshold, a hash-stamped record and a non-zero exit code. Measured in papers (Krivorotov & Richey; FinRegLab), warned about by regulators (CFPB), shipped by no one.
2. **Governance record as code, in the user's git.** Content-addressed YAML with a mandatory materiality-tier justification, vendor records that still carry a validation obligation, and `diff` between versions. Commercial inventories are SaaS; MLflow is an artefact registry; neither is a diffable statement of *claims*.
3. **A regulator-mapped, deterministic report for the long tail.** Not more metrics — the same metrics ValidMind and PiML compute, but byte-replayable from embedded hashes and mapped line-by-line to SR 26-2 and FREE-AI asks, installable by an NBFC with no MRM team.

Positioning sentence: *"Covenant is pytest for credit-model governance. Its one original idea is that a model's documentation is a set of testable claims about its behaviour, so it tests them."*

What **not** to claim: that nobody validates models (false), that post-hoc SHAP is ground truth (false — say it's an approximation and expose its knobs), that passing Covenant means compliance (never say this; say it produces evidence).

---

## 3. Design implications pulled from the research

**Check 1 — reason codes vs attributions.**
- Support the four AAC methods from Krivorotov & Richey as the *declared* method: `shapley` (incl. B-Shap per Nair et al.), `most_points_lost` (scorecards), `difference_from_mean` (linear contribution), `univariate`. Plus `custom` callable.
- *Measured* side: SHAP with explainer chosen by model type; **the background/baseline is a first-class parameter** (Pace Analytics): default = training-set sample of N=500 with fixed seed; report the top-k overlap under two backgrounds and flag if they disagree materially. For EBM/GAM models, use exact shape-function contributions instead of SHAP and say so in the record.
- Metric: FinRegLab-style consistency = mean top-k overlap (report both top-1 agreement rate and top-k Jaccard); disagreement concentrated near the decision boundary is expected (K&R) — stratify the report by score decile so a user can see *where* it disagrees.
- Placebo sub-check (from K&R): perturb a non-informative feature and confirm neither the reason codes nor attributions move; if they do, the explanation pipeline is noisy.

**Check 2 — monotonicity.** Dominance pairs (your Brihas generator) *and* an ICE/PDP slope test per declared feature; for tree models with `monotone_constraints` set, read the constraint from the estimator and compare to the spec (declared vs configured vs empirical — three-way). Cite Barocas et al. and Provenir for why this underpins recourse and reason codes.

**Check 3 / 4 — features & exclusions.** Straightforward; the research addition is the **proxy screen** (2511.03807 uses η² / association tests between features and protected attributes) — report association strength between excluded variables and used features, and be explicit that you surface obvious proxies rather than prove absence.

**Report.** Metrics are commodity; determinism and mapping are the product. Compute AUC/KS/Gini/Brier/ECE/PSI/CSI yourself (small, dependency-free), bootstrap CIs (FINRA MVT's "credibility under small samples" is a good citation), and put a *"how this maps"* column beside every section referencing SR 26-2 / FREE-AI.

**Interoperability.** Adapters for: sklearn `Pipeline`, XGBoost/LightGBM (read monotone constraints), OptBinning `Scorecard`, InterpretML `ExplainableBoostingClassifier`, PiML/MoDeVa registered models. That list makes Covenant the layer *above* the ecosystem instead of a competitor inside it.

**Future checks worth a roadmap line, not v1:** recourse validity (DiCE / CARLA — does following the reason code actually flip the decision?), robustness/resilience (defer to PiML), fairness (defer to Fairlearn/SolasAI).

---

## 4. Best technical stack (with reasons and the alternative you rejected)

| Layer | Choice | Why | Rejected |
|---|---|---|---|
| Language / runtime | **Python ≥ 3.11**, pure Python wheel | Where credit models live; `pip install` with no toolchain is the adoption story | Rust core (no hot loop a reader would measure; hurts adoption) |
| Model contract | `predict_proba` duck-typing + explicit adapters (`sklearn`, `xgboost`, `lightgbm`, `optbinning`, `interpret`) | Covers >95% of real credit models without forcing a wrapper | Custom model base class |
| Attributions | **`shap`** (Tree/Linear/Kernel), exact contributions for EBM/GAM/linear | Industry default; explainer chosen by type; background exposed | LIME (unstable, FinRegLab shows lower fidelity) |
| Numerics / stats | `numpy`, `pandas`, `scipy` | Bootstrap, KS, association tests | statsmodels (heavier than needed) |
| Schema / validation | **`pydantic` v2** | Readable rejection messages on missing fields; JSON-schema export for docs | dataclasses + hand-rolled checks |
| Storage | **Flat YAML** via `ruamel.yaml` (round-trip, comment-preserving) under `.covenant/` in the user's repo | Git-diffable, auditors read diffs; comments survive edits | SQLite/DB (opaque to auditors), PyYAML (drops comments) |
| Hashing / addressing | `hashlib.sha256` over (a) model file bytes as supplied, (b) canonical JSON (`sort_keys=True`, fixed separators) for records, (c) parquet bytes for snapshots; record `sklearn`/`xgboost` versions alongside | Pickle bytes are not stable across library versions — hash the artefact you were given, don't re-serialise | Re-pickling to hash (non-deterministic) |
| Model serialization (recommended to users) | **`skops`** for sklearn, native `save_model` for XGB/LGBM | Safer than pickle; version-stamped | pickle-only |
| CLI | **`typer`** (or `click`) with rich output | Subcommands, exit codes, `--json` flag for CI | argparse |
| Reporting | Markdown + PNG (matplotlib) with **deterministic rendering**: fixed `rcParams`, `savefig(..., metadata={"Software": None})` for PNG and `{"Date": None}` for SVG, no timestamps, `numpy.random.default_rng(seed)`; optional HTML via `jinja2` | Byte-identical replay is *the* property; matplotlib embeds version/date metadata by default and would break it | Plotly/HTML-first (non-deterministic bytes) |
| Testing | **`pytest` + `hypothesis`** (property tests for hashing, PSI/ECE, monotone models), golden-file tests for report bytes, `pytest-cov` | Your own house style already | unittest |
| Packaging / tooling | `pyproject.toml` with **`hatch`** or `uv`, `ruff` (lint+format), `pre-commit`, semantic versioning, `CITATION.cff` | Modern, fast, one-file config | setuptools + black + flake8 |
| CI | GitHub Actions matrix 3.11/3.12/3.13, runs the broken/fixed demos and asserts exit codes | The demo *is* the integration test | — |
| Docs | **`mkdocs-material`** with `docs/MAPPING.md` (regulatory mapping), API via `mkdocstrings` | Readable, searchable, GitHub Pages | Sphinx (fine, heavier) |
| Optional extras | `covenants[fairness]` → Fairlearn; `covenants[recourse]` → DiCE; `covenants[interpret]` → InterpretML | Keep core light; let users opt into heavy deps | Bundling everything |
| Licence | **Apache-2.0** (or MIT) | Bank-friendly; ValidMind's library is GPL-family, PiML has its own terms — never vendor their code | GPL (blocks bank adoption) |
| Datasets for demos/tests | `credit-g` (OpenML, 1k rows, CI), Give Me Some Credit (150k), Lending Club (if that's your 579k) | Public, reproducible, recognised by reviewers (EUI paper uses Lending Club) | Anything private |

---

## 5. Pitfalls the research warns about (build defensively)

- SHAP on models with strong interactions is not additive per feature — say so; recommend interaction constraints or inherently interpretable models (Provenir, Sudjianto). Your check reports disagreement; it does not adjudicate which side is "right".
- KernelSHAP is slow and background-sensitive; cap the sample, fix the seed, report the sensitivity.
- Version drift: hashes over artefact bytes + recorded library versions; a re-fit is a new version, not a diff.
- Licences: read ValidMind's and PiML/MoDeVa's before importing anything; cite freely, copy nothing.
- Never imply compliance. Every doc page: "Covenant produces evidence; your validators and auditors decide."
- Scope creep: recourse, robustness, fairness are roadmap items with pointers to the tools that already do them well.

---

## 6. Annotated reading list (in the order to read them)

1. SR 26-2 letter + attachment — federalreserve.gov/supervisionreg/srletters/SR2602.htm (read fully; cite by section)
2. RBI FREE-AI Committee Report (13 Aug 2025) — rbi.org.in (read Chapters on Governance/Protection/Assurance and Annexure IV; note the survey stats)
3. CFPB Circular 2022-03 and Sept-2023 guidance on adverse action with complex algorithms
4. Krivorotov & Richey, *Explaining Denials* — SSRN 4133915 (the closest paper to Check 1)
5. Nair et al., *Explaining Adverse Actions… Shapley Decomposition* — arXiv 2204.12365 (B-Shap)
6. FinRegLab, *ML Explainability & Fairness: Insights from Consumer Lending* (2023) + Policy Analysis (consistency/fidelity definitions)
7. Pace Analytics, *Using XAI to Produce ECOA Adverse Action Reasons: What Are the Risks?* (2024)
8. Barocas, Selbst, Raghavan, *Hidden Assumptions behind Counterfactual Explanations and Principal Reasons* — arXiv 1912.04930
9. Sudjianto et al., *PiML Toolbox* — arXiv 2305.04214; Sudjianto & Zhang, *Designing Inherently Interpretable ML Models* — arXiv 2111.01743; MoDeVa docs
10. ValidMind library README + test catalogue (docs.validmind.com) — to see what a bank expects in a report; FINRA Model Validation Toolkit docs
11. Provenir, *Constraining ML Credit Decision Models* (2022); EUI RSC 2025/24 on monotonic XGBoost
12. arXiv 2607.04103 (SR 26-2-compatible GenAI framework; explanation fidelity section)
13. OptBinning docs (Scorecard object), InterpretML EBM docs (shape functions) — for adapters
14. Bailey & López de Prado is *not* relevant here — that's Shunkan. Don't cross-pollinate for its own sake.

---

## 7. One-paragraph competitive statement for the README

> Model validation is well served: ValidMind's platform and library, Wells Fargo's PiML/MoDeVa, FINRA's Model Validation Toolkit and several credit-risk libraries compute the metrics an SR 26-2 report needs, and Evidently/NannyML monitor drift. Covenant does not compete with them on metrics. It adds the layer none of them ship: a git-native, content-addressed record of what a credit model *claims* — its features, monotone directions, exclusions and reason-code method — and a set of CI checks that fail when the model's measured behaviour contradicts those claims, plus a deterministic report mapped line-by-line to SR 26-2 and RBI FREE-AI. It wraps the ecosystem rather than replacing it.
