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

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator


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
    random_state: int = 0


class ChecksConfig(StrictModel):
    reason_codes: ReasonCodeCheckConfig = Field(default_factory=ReasonCodeCheckConfig)


class ModelCovenants(StrictModel):
    """The model's covenants: testable claims about its behaviour."""

    covenant_schema: Literal[1] = 1
    model_name: str = Field(min_length=1, pattern=r"^[A-Za-z0-9._-]+$")
    features: list[FeatureCovenant] = Field(min_length=1)
    excluded: list[ExcludedVariable] = Field(default_factory=list)
    reason_codes: ReasonCodePolicy
    checks: ChecksConfig = Field(default_factory=ChecksConfig)

    @field_validator("features")
    @classmethod
    def _unique_feature_names(cls, v: list[FeatureCovenant]) -> list[FeatureCovenant]:
        names = [f.name for f in v]
        dupes = {n for n in names if names.count(n) > 1}
        if dupes:
            raise ValueError(f"duplicate feature names: {sorted(dupes)}")
        return v

    def feature_names(self) -> list[str]:
        return [f.name for f in self.features]


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
