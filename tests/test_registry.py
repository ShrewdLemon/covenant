from __future__ import annotations

from pathlib import Path

import pytest

from covenant.registry import RegistrationError, register
from covenant.store import Store, diff_records, read_yaml


def test_register_writes_content_addressed_record(fitted: dict, store_dir: Path) -> None:
    store = Store(store_dir)
    record = register(
        fitted["model"], fitted["data"], fitted["covenants"], fitted["governance"], store
    )
    assert record.model_name == "test-scorecard"
    path = store.record_path("test-scorecard", record.version_id)
    assert path.exists()

    on_disk = read_yaml(path)
    assert on_disk["version_id"] == record.version_id
    assert on_disk["hashes"]["model_sha256"] == record.hashes.model_sha256
    assert on_disk["governance"]["materiality"]["tier"] == 3


def test_same_inputs_same_version(fitted: dict, store_dir: Path) -> None:
    store = Store(store_dir)
    a = register(fitted["model"], fitted["data"], fitted["covenants"], fitted["governance"], store)
    b = register(fitted["model"], fitted["data"], fitted["covenants"], fitted["governance"], store)
    assert a.version_id == b.version_id
    assert store.list_versions("test-scorecard") == [a.version_id]


def test_changed_covenants_new_version_and_diff(fitted: dict, store_dir: Path) -> None:
    store = Store(store_dir)
    a = register(fitted["model"], fitted["data"], fitted["covenants"], fitted["governance"], store)
    b = register(
        fitted["model"], fitted["data"], fitted["covenants_broken"], fitted["governance"], store
    )
    assert a.version_id != b.version_id

    lines = diff_records(
        store.read_record("test-scorecard", a.version_id),
        store.read_record("test-scorecard", b.version_id),
    )
    changed = "\n".join(lines)
    assert "coefficients" in changed  # the stale table is visible in the diff
    assert "covenants_sha256" in changed


def test_register_rejects_missing_feature(fitted: dict, store_dir: Path, tmp_path: Path) -> None:
    bad = tmp_path / "covenants_bad.yaml"
    bad.write_text(
        fitted["covenants"].read_text().replace("name: dti", "name: not_a_column")
    )
    coefficients = fitted["root"] / "coefficients_live.csv"
    (tmp_path / "coefficients_live.csv").write_text(coefficients.read_text())
    with pytest.raises(RegistrationError, match="not_a_column"):
        register(fitted["model"], fitted["data"], bad, fitted["governance"], Store(store_dir))


def test_register_rejects_missing_reason_code_artefact(
    fitted: dict, store_dir: Path, tmp_path: Path
) -> None:
    """A covenant pointing at an artefact that does not exist is a claim that
    cannot be tested; registration must say so immediately (stranger-test
    finding: this used to surface only at check time)."""
    bad = tmp_path / "covenants.yaml"
    bad.write_text(fitted["covenants"].read_text())  # coefficients_live.csv absent here
    with pytest.raises(RegistrationError, match="does not exist"):
        register(fitted["model"], fitted["data"], bad, fitted["governance"], Store(store_dir))


def test_register_friendly_error_on_invalid_governance(
    fitted: dict, store_dir: Path, tmp_path: Path
) -> None:
    bad = tmp_path / "governance_bad.yaml"
    bad.write_text(fitted["governance"].read_text().replace("  justification:", "  note:"))
    with pytest.raises(RegistrationError) as exc:
        register(fitted["model"], fitted["data"], fitted["covenants"], bad, Store(store_dir))
    message = str(exc.value)
    assert "materiality" in message
    assert "justification" in message
