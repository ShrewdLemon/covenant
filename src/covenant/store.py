"""Flat-file YAML store under ``.covenant/`` in the user's own repository.

Everything an auditor needs is a text file in git: inventory records under
``models/<name>/<version_id>.yaml``, check records under
``checks/<name>/``. No database, nothing opaque; history is ``git log``
and comparison is a diff.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

from ruamel.yaml import YAML

STORE_DIR = ".covenant"

_yaml = YAML()
_yaml.default_flow_style = False
_yaml.sort_base_mapping_type_on_output = False  # type: ignore[assignment]


def yaml_dump(obj: Any) -> str:
    buf = io.StringIO()
    _yaml.dump(obj, buf)
    return buf.getvalue()


def yaml_load(text: str) -> Any:
    return _yaml.load(text)


def read_yaml(path: str | Path) -> Any:
    return yaml_load(Path(path).read_text())


def write_yaml(path: str | Path, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml_dump(obj))


class Store:
    def __init__(self, root: str | Path = STORE_DIR) -> None:
        self.root = Path(root)

    def init(self) -> None:
        (self.root / "models").mkdir(parents=True, exist_ok=True)
        (self.root / "checks").mkdir(parents=True, exist_ok=True)

    def record_path(self, model_name: str, version_id: str) -> Path:
        return self.root / "models" / model_name / f"{version_id}.yaml"

    def write_record(self, model_name: str, version_id: str, record: dict) -> Path:
        path = self.record_path(model_name, version_id)
        write_yaml(path, record)
        return path

    def read_record(self, model_name: str, version_id: str) -> dict:
        path = self.record_path(model_name, version_id)
        if not path.exists():
            known = self.list_versions(model_name)
            raise FileNotFoundError(
                f"no record {version_id} for model {model_name!r}; "
                f"known versions: {known or 'none'}"
            )
        return read_yaml(path)

    def list_versions(self, model_name: str) -> list[str]:
        model_dir = self.root / "models" / model_name
        if not model_dir.exists():
            return []
        return sorted(p.stem for p in model_dir.glob("*.yaml"))

    def list_models(self) -> list[str]:
        models_dir = self.root / "models"
        if not models_dir.exists():
            return []
        return sorted(p.name for p in models_dir.iterdir() if p.is_dir())

    def check_path(self, model_name: str, check_name: str, short_hash: str) -> Path:
        return self.root / "checks" / model_name / f"{check_name}-{short_hash}.yaml"

    def write_check(self, model_name: str, check_name: str, short_hash: str, record: dict) -> Path:
        path = self.check_path(model_name, check_name, short_hash)
        write_yaml(path, record)
        return path

    def list_checks(self, model_name: str) -> list[Path]:
        checks_dir = self.root / "checks" / model_name
        if not checks_dir.exists():
            return []
        return sorted(checks_dir.glob("*.yaml"))

    def resolve_version(self, model_name: str, prefix: str) -> str:
        """Resolve a (possibly partial) version id to the full stored one."""
        versions = self.list_versions(model_name)
        matches = [v for v in versions if v.startswith(prefix)]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            raise FileNotFoundError(
                f"no version of {model_name!r} matches {prefix!r}; "
                f"known versions: {versions or 'none'}"
            )
        raise FileNotFoundError(
            f"version prefix {prefix!r} is ambiguous for {model_name!r}: {matches}"
        )


DIFF_NOISE_KEYS = ("created_at",)


def diff_records(
    a: Any, b: Any, path: str = "", ignore: tuple[str, ...] = DIFF_NOISE_KEYS
) -> list[str]:
    """Structural diff between two record trees, as `~ changed`, `+ added`,
    `- removed` lines with dotted paths. Top-level keys in ``ignore``
    (timestamps, by default) are skipped; pass ``ignore=()`` for everything."""
    lines: list[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b), key=str):
            if not path and key in ignore:
                continue
            sub = f"{path}.{key}" if path else str(key)
            if key not in a:
                lines.append(f"+ {sub}: {b[key]!r}")
            elif key not in b:
                lines.append(f"- {sub}: {a[key]!r}")
            else:
                lines.extend(diff_records(a[key], b[key], sub))
    elif isinstance(a, list) and isinstance(b, list):
        for i in range(max(len(a), len(b))):
            sub = f"{path}[{i}]"
            if i >= len(a):
                lines.append(f"+ {sub}: {b[i]!r}")
            elif i >= len(b):
                lines.append(f"- {sub}: {a[i]!r}")
            else:
                lines.extend(diff_records(a[i], b[i], sub))
    elif a != b:
        lines.append(f"~ {path}: {a!r} -> {b!r}")
    return lines
