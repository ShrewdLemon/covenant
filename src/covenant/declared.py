"""The declared side of Check 1: the reason codes the user's production
pipeline would actually send a denied applicant.

Methods follow the Krivorotov & Richey (2022) taxonomy. Each is driven by a
production artefact (a coefficient table, a points table, a reasons export)
because that artefact — not the model — is what goes stale, and testing it
against the model's measured behaviour is the whole point.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from covenant.attribution import top_1, top_k_sets
from covenant.schema import ReasonCodeMethod, ReasonCodePolicy


class DeclaredMethodError(ValueError):
    pass


def declared_attributions(
    policy: ReasonCodePolicy,
    X: pd.DataFrame,
    covenants_dir: Path,
) -> pd.DataFrame:
    """Attribution matrix implied by the declared reason-code method.

    Higher value = stronger reason for denial, matching the measured side.
    """
    if policy.method is ReasonCodeMethod.DIFFERENCE_FROM_MEAN:
        return _difference_from_mean(policy, X, covenants_dir)
    raise DeclaredMethodError(
        f"declared method {policy.method.value!r} is not implemented in this "
        "version; implemented: difference_from_mean (attributions), "
        "custom (reason-code file). most_points_lost, univariate and "
        "shapley are on the roadmap."
    )


def declared_reason_sets(
    policy: ReasonCodePolicy,
    X: pd.DataFrame,
    covenants_dir: Path,
    ids: pd.Series | None = None,
    id_column: str | None = None,
) -> tuple[list[frozenset[str]], list[str]]:
    """(top-k reason sets, top-1 reason) per row, by the declared method."""
    if policy.method is ReasonCodeMethod.CUSTOM:
        if ids is None or id_column is None:
            raise DeclaredMethodError(
                "custom reason codes require checks.reason_codes.id_column"
            )
        return _custom_reasons(policy, ids, id_column, covenants_dir)
    attributions = declared_attributions(policy, X, covenants_dir)
    return top_k_sets(attributions, policy.top_k), top_1(attributions)


def _resolve(ref: str, covenants_dir: Path) -> Path:
    path = Path(ref)
    return path if path.is_absolute() else covenants_dir / path


def _difference_from_mean(
    policy: ReasonCodePolicy, X: pd.DataFrame, covenants_dir: Path
) -> pd.DataFrame:
    """Linear contribution vs the mean applicant: coef * (x - mean) / scale.

    The coefficient table is the artefact a production system actually uses
    to phrase adverse-action letters — which is exactly why it can go stale
    against the deployed model. Columns: feature, coef, optional mean
    (default 0), optional scale (default 1).
    """
    table_ref = policy.parameters.get("coefficients")
    if table_ref is None:
        raise DeclaredMethodError(
            "reason_codes.parameters.coefficients must point to a CSV of "
            "feature,coef[,mean][,scale] for method difference_from_mean"
        )
    table_path = _resolve(table_ref, covenants_dir)
    table = pd.read_csv(table_path)
    if "feature" not in table.columns or "coef" not in table.columns:
        raise DeclaredMethodError(
            f"{table_path} needs columns feature,coef[,mean][,scale]"
        )
    table = table.set_index("feature")
    missing = [c for c in X.columns if c not in table.index]
    if missing:
        raise DeclaredMethodError(
            f"coefficient table {table_path} lacks features: {missing}"
        )
    table = table.reindex(X.columns)
    coef = table["coef"].to_numpy(dtype=float)
    mean = table.get("mean", pd.Series(0.0, index=table.index)).to_numpy(dtype=float)
    scale = table.get("scale", pd.Series(1.0, index=table.index)).to_numpy(dtype=float)
    try:
        values_matrix = X.to_numpy(dtype=float)
    except (TypeError, ValueError) as err:
        raise DeclaredMethodError(
            "difference_from_mean needs all-numeric features; for models "
            "with categorical features use reason_codes.method: custom and "
            "export the production reasons"
        ) from err
    values = coef * (values_matrix - mean) / scale
    return pd.DataFrame(values, columns=X.columns, index=X.index)


def _custom_reasons(
    policy: ReasonCodePolicy,
    ids: pd.Series,
    id_column: str,
    covenants_dir: Path,
) -> tuple[list[frozenset[str]], list[str]]:
    """Reason codes exported from the production pipeline: a CSV with the
    id column plus reason_1..reason_k, joined to the data on the id."""
    reasons_ref = policy.parameters.get("reasons_file")
    if reasons_ref is None:
        raise DeclaredMethodError(
            "reason_codes.parameters.reasons_file must point to a CSV of "
            f"{id_column},reason_1..reason_k for method custom"
        )
    reasons_path = _resolve(reasons_ref, covenants_dir)
    table = pd.read_csv(reasons_path)
    if id_column not in table.columns:
        raise DeclaredMethodError(
            f"{reasons_path} lacks the id column {id_column!r} declared in "
            "checks.reason_codes.id_column"
        )
    reason_cols = [c for c in table.columns if c.startswith("reason_")]
    if not reason_cols:
        raise DeclaredMethodError(f"{reasons_path} has no reason_1..reason_k columns")
    if table[id_column].duplicated().any():
        dupes = table.loc[table[id_column].duplicated(), id_column].head(5).tolist()
        raise DeclaredMethodError(
            f"{reasons_path} has duplicate ids in {id_column!r}: {dupes}"
        )
    table = table.set_index(id_column)
    missing_mask = ~ids.isin(table.index)
    if missing_mask.any():
        missing_ids = ids[missing_mask].head(5).tolist()
        raise DeclaredMethodError(
            f"{reasons_path} has no reasons for {int(missing_mask.sum())} "
            f"scored rows; first missing {id_column!r} values: {missing_ids}"
        )
    rows = table.loc[ids]
    sets = [
        frozenset(str(v) for v in row if pd.notna(v))
        for row in rows[reason_cols].to_numpy()
    ]
    top1 = [str(v) for v in rows[reason_cols[0]]]
    return sets, top1
