from __future__ import annotations

import pytest
from pydantic import ValidationError

from covenant.schema import GovernanceRecord, ModelCovenants

BASE_GOVERNANCE = {
    "owner": {"name": "A", "email": "a@example.com"},
    "intended_use": "Consumer lending PD for the demo book.",
    "materiality": {
        "tier": 2,
        "justification": "Moderate exposure, single product line, capped ticket size.",
    },
    "review_date": "2027-01-01",
}

BASE_COVENANTS = {
    "covenant_schema": 1,
    "model_name": "m1",
    "features": [{"name": "dti", "direction": "increases_risk"}],
    "reason_codes": {"method": "difference_from_mean"},
}


def test_materiality_justification_is_mandatory() -> None:
    record = {**BASE_GOVERNANCE, "materiality": {"tier": 2}}
    with pytest.raises(ValidationError) as exc:
        GovernanceRecord.model_validate(record)
    assert "justification" in str(exc.value)


def test_materiality_justification_must_be_substantive() -> None:
    record = {**BASE_GOVERNANCE, "materiality": {"tier": 2, "justification": "because"}}
    with pytest.raises(ValidationError):
        GovernanceRecord.model_validate(record)


def test_unknown_keys_rejected() -> None:
    with pytest.raises(ValidationError) as exc:
        ModelCovenants.model_validate({**BASE_COVENANTS, "featuresz": []})
    assert "featuresz" in str(exc.value)


def test_duplicate_features_rejected() -> None:
    doc = {
        **BASE_COVENANTS,
        "features": [{"name": "dti"}, {"name": "dti"}],
    }
    with pytest.raises(ValidationError) as exc:
        ModelCovenants.model_validate(doc)
    assert "duplicate" in str(exc.value)


def test_valid_documents_parse() -> None:
    gov = GovernanceRecord.model_validate(BASE_GOVERNANCE)
    assert gov.materiality.tier == 2
    cov = ModelCovenants.model_validate(BASE_COVENANTS)
    assert cov.feature_names() == ["dti"]
    assert cov.reason_codes.top_k == 4  # default
    assert cov.checks.reason_codes.min_topk_jaccard == 0.60  # default
