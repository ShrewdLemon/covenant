"""Schemas for the two documents a user writes and the record Covenant writes.

* ``ModelCovenants`` — the model's covenants (feature spec): what the model
  is supposed to do. Features and their monotone directions, excluded
  variables, and how adverse-action reason codes are derived.
* ``GovernanceRecord`` — owner, intended use, materiality tier with a
  mandatory justification, review date, vendor block for third-party models.
* ``ModelRecord`` — the inventory entry ``covenant register`` writes:
  content hashes, library versions, and the two documents above.

Validation is strict (unknown keys are rejected) so a typo in a YAML key
fails loudly at registration instead of silently weakening a check.
"""

from __future__ import annotations

import datetime as dt
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Direction(StrEnum):
    """Declared monotone direction of a feature with respect to risk."""

    INCREASES_RISK = "increases_risk"
    DECREASES_RISK = "decreases_risk"
    NONE = "none"  # no monotonicity claimed


class FeatureCovenant(StrictModel):
    name: str
    dtype: Literal["numeric", "categorical"] = "numeric"
    direction: Direction = Direction.NONE
    description: str | None = None

    @model_validator(mode="after")
    def _categorical_has_no_direction(self) -> FeatureCovenant:
        if self.dtype == "categorical" and self.direction is not Direction.NONE:
            raise ValueError(
                f"feature {self.name!r}: a monotone direction on a categorical "
                "feature is not testable; use direction: none"
            )
        return self


class ExcludedVariable(StrictModel):
    name: str
    reason: str = Field(
        min_length=3,
        description="Why the variable is excluded, e.g. 'protected attribute (ECOA)'.",
    )


class ReasonCodeMethod(StrEnum):
    """The four adverse-action code methods of Krivorotov & Richey (2022),
    plus a user-supplied file of production reason codes."""

    SHAPLEY = "shapley"
    MOST_POINTS_LOST = "most_points_lost"
    DIFFERENCE_FROM_MEAN = "difference_from_mean"
    UNIVARIATE = "univariate"
    CUSTOM = "custom"


class ReasonCodePolicy(StrictModel):
    method: ReasonCodeMethod
    top_k: int = Field(default=4, ge=1, le=10)
    parameters: dict[str, Any] = Field(default_factory=dict)


class ReasonCodeCheckConfig(StrictModel):
    """Thresholds for Check 1. These are the user's policy, not Covenant's
    verdict on what is acceptable — defaults are starting points."""

    min_top1_agreement: float = Field(default=0.75, ge=0.0, le=1.0)
    min_topk_jaccard: float = Field(default=0.60, ge=0.0, le=1.0)
    decision_threshold: float = Field(default=0.5, gt=0.0, lt=1.0)
    max_denied_sample: int = Field(default=500, ge=10)
    background_size: int = Field(default=200, ge=10)
    id_column: str | None = Field(
        default=None,
        description="Stable row key joining the data to a custom reasons file. "
        "Required when reason_codes.method is custom: positional alignment "
        "breaks silently the moment one row is filtered upstream.",
    )
    placebo: bool = Field(
        default=True,
        description="Krivorotov & Richey placebo: shuffle a declared-irrelevant "
        "feature and confirm neither side's reasons move. Skipped (with a "
        "note) when no feature is measurably irrelevant.",
    )
    placebo_epsilon: float = Field(
        default=1e-3,
        ge=0.0,
        description="Mean |attribution| below which a feature counts as "
        "irrelevant enough to serve as the placebo.",
    )
    max_placebo_shift: float = Field(
        default=0.10,
        ge=0.0,
        le=1.0,
        description="Fraction of rows whose top-k may change under the "
        "placebo before the explanation pipeline is flagged as noisy.",
    )
    random_state: int = 0


class MonotonicityCheckConfig(StrictModel):
    """Thresholds for Check 2. Violation rates are measured on synthetic
    dominance pairs and ICE paths; the threshold is the user's policy."""

    max_violation_rate: float = Field(default=0.05, ge=0.0, le=1.0)
    n_pairs: int = Field(default=300, ge=10)
    n_ice_rows: int = Field(default=30, ge=5)
    ice_grid_points: int = Field(default=9, ge=3)
    tolerance: float = Field(
        default=1e-4,
        ge=0.0,
        description="Probability moves smaller than this do not count as violations.",
    )
    random_state: int = 0


class FeaturesCheckConfig(StrictModel):
    """Thresholds for Check 3 (declared vs used features)."""

    dead_feature_epsilon: float = Field(
        default=1e-3,
        ge=0.0,
        description="Mean |attribution| below which a documented feature is "
        "flagged as dead (a warning, not a breach).",
    )
    sample_size: int = Field(default=300, ge=20)
    background_size: int = Field(default=100, ge=10)
    random_state: int = 0


class ExclusionsCheckConfig(StrictModel):
    """Thresholds for Check 4 (exclusions and proxies). The proxy screen
    surfaces obvious proxies; it never proves absence."""

    max_excluded_attribution: float = Field(
        default=1e-3,
        ge=0.0,
        description="If an excluded variable reaches the model anyway, its "
        "mean |attribution| must stay below this.",
    )
    max_association: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Association (|Spearman|, correlation ratio, or "
        "Cramer's V) between an excluded variable and a used feature above "
        "which the pair is flagged as a potential proxy. Tune to your book.",
    )
    fail_on_proxies: bool = Field(
        default=True,
        description="Whether a flagged proxy fails the check or only warns.",
    )
    sample_size: int = Field(default=300, ge=20)
    background_size: int = Field(default=100, ge=10)
    random_state: int = 0


class ChecksConfig(StrictModel):
    reason_codes: ReasonCodeCheckConfig = Field(default_factory=ReasonCodeCheckConfig)
    monotonicity: MonotonicityCheckConfig = Field(default_factory=MonotonicityCheckConfig)
    features: FeaturesCheckConfig = Field(default_factory=FeaturesCheckConfig)
    exclusions: ExclusionsCheckConfig = Field(default_factory=ExclusionsCheckConfig)


class ReportConfig(StrictModel):
    """Settings for the deterministic validation report."""

    target_column: str | None = Field(
        default=None,
        description="Name of the 0/1 outcome column in the snapshot; required "
        "to render a report.",
    )
    time_column: str | None = Field(
        default=None,
        description="Optional timestamp/ordinal column for drift-by-slice.",
    )
    n_bootstrap: int = Field(default=500, ge=50)
    n_bins: int = Field(default=10, ge=4)
    challenger: bool = Field(
        default=True,
        description="Fit a plain logistic challenger on the same features and "
        "report its lift with confidence intervals.",
    )
    random_state: int = 0


class ModelCovenants(StrictModel):
    """The model's covenants: testable claims about its behaviour."""

    covenant_schema: Literal[1] = 1
    model_name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    positive_class: int | str | None = Field(
        default=None,
        description="Label of the bad/default class as it appears in the "
        "estimator's classes_. Defaults to the class at index 1.",
    )
    features: list[FeatureCovenant] = Field(min_length=1)
    excluded: list[ExcludedVariable] = Field(default_factory=list)
    reason_codes: ReasonCodePolicy
    checks: ChecksConfig = Field(default_factory=ChecksConfig)
    report: ReportConfig = Field(default_factory=ReportConfig)

    @field_validator("features")
    @classmethod
    def _unique_feature_names(cls, v: list[FeatureCovenant]) -> list[FeatureCovenant]:
        names = [f.name for f in v]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate feature names: {sorted(dupes)}")
        return v

    @model_validator(mode="after")
    def _custom_reasons_need_id_column(self) -> ModelCovenants:
        if (
            self.reason_codes.method is ReasonCodeMethod.CUSTOM
            and not self.checks.reason_codes.id_column
        ):
            raise ValueError(
                "reason_codes.method: custom requires checks.reason_codes."
                "id_column — reasons must join the data on a stable key, "
                "not on row position"
            )
        return self

    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]

    def categorical_features(self) -> list[str]:
        return [f.name for f in self.features if f.dtype == "categorical"]

    def declared_directions(self) -> dict[str, Direction]:
        return {f.name: f.direction for f in self.features}


class Owner(StrictModel):
    name: str = Field(min_length=1)
    email: EmailStr


class Materiality(StrictModel):
    """SR 26-2 ties governance intensity to materiality (exposure x purpose).
    The tier is a decision, so the justification is mandatory."""

    tier: Literal[1, 2, 3]
    justification: str = Field(
        min_length=20,
        description="Why this tier: exposure, purpose, portfolio share. Mandatory.",
    )


class VendorRecord(StrictModel):
    """Third-party models carry the same validation obligation as in-house
    ones (SR 26-2; RBI FREE-AI expects REs to validate vendor models)."""

    name: str
    product: str
    version: str | None = None
    contact: str | None = None


class GovernanceRecord(StrictModel):
    owner: Owner
    intended_use: str = Field(min_length=10)
    limitations: list[str] = Field(default_factory=list)
    materiality: Materiality
    review_date: dt.date
    vendor: VendorRecord | None = None


class ModelInfo(StrictModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_class: str
    library_versions: dict[str, str]


class DataInfo(StrictModel):
    path: str
    n_rows: int
    n_cols: int
    columns: list[str]


class Hashes(StrictModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())

    model_sha256: str
    data_sha256: str
    covenants_sha256: str


class ModelRecord(StrictModel):
    """One inventory entry. ``version_id`` is derived from the three content
    hashes; ``created_at`` is informational and never part of the identity."""

    record_schema: Literal[1] = 1
    version_id: str
    model_name: str
    created_at: dt.datetime
    hashes: Hashes
    model: ModelInfo
    data: DataInfo
    governance: GovernanceRecord
    covenants: ModelCovenants
