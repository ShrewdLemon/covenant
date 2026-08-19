"""The declared side of Check 1: the reason codes the user's production
pipeline would actually send a denied applicant.

Methods follow the Krivorotov & Richey (2022) taxonomy: shapley,
most_points_lost, difference_from_mean and univariate, plus a user-supplied
file of production reason codes. Each is driven by a production artefact (a
coefficient table, a scorecard points table, a bins table, an attributions
or reasons export) because that artefact — not the model — is what goes
stale, and testing it against the model's measured behaviour is the whole
point.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
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
    if policy.method is ReasonCodeMethod.MOST_POINTS_LOST:
        return _most_points_lost(policy, X, covenants_dir)
    if policy.method is ReasonCodeMethod.UNIVARIATE:
        return _univariate(policy, X, covenants_dir)
    raise DeclaredMethodError(
        f"declared method {policy.method.value!r} joins a per-applicant "
        "production export on checks.reason_codes.id_column; call "
        "declared_reason_sets with ids and id_column instead of "
        "declared_attributions"
    )


def declared_reason_sets(
    policy: ReasonCodePolicy,
    X: pd.DataFrame,
    covenants_dir: Path,
    ids: pd.Series | None = None,
    id_column: str | None = None,
) -> tuple[list[frozenset[str]], list[str]]:
    """(top-k reason sets, top-1 reason) per row, by the declared method."""
    if policy.method in (ReasonCodeMethod.CUSTOM, ReasonCodeMethod.SHAPLEY):
        if ids is None or id_column is None:
            raise DeclaredMethodError(
                f"{policy.method.value} reason codes join a production export "
                "on a stable key and require checks.reason_codes.id_column"
            )
        if policy.method is ReasonCodeMethod.CUSTOM:
            return _custom_reasons(policy, ids, id_column, covenants_dir)
        attributions = _shapley_attributions(policy, X, ids, id_column, covenants_dir)
    else:
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
    try:
        table = pd.read_csv(table_path)
    except FileNotFoundError as err:
        raise DeclaredMethodError(f"reason-code artefact not found: {table_path}") from err
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
    for name, arr in (("coef", coef), ("mean", mean), ("scale", scale)):
        if np.isnan(arr).any():
            bad = [f for f, isnan in zip(table.index, np.isnan(arr), strict=True) if isnan]
            raise DeclaredMethodError(
                f"{table_path}: column {name!r} has empty cells for features {bad}"
            )
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


def _read_bin_table(
    policy: ReasonCodePolicy,
    covenants_dir: Path,
    param_key: str,
    payload_column: str,
    schema_hint: str,
) -> tuple[pd.DataFrame, Path]:
    """Load a binned table (points table or bins table) and validate the
    columns every row needs: feature and the payload."""
    table_ref = policy.parameters.get(param_key)
    if table_ref is None:
        raise DeclaredMethodError(
            f"reason_codes.parameters.{param_key} must point to a CSV of "
            f"{schema_hint} for method {policy.method.value}"
        )
    table_path = _resolve(table_ref, covenants_dir)
    try:
        table = pd.read_csv(table_path)
    except FileNotFoundError as err:
        raise DeclaredMethodError(f"reason-code artefact not found: {table_path}") from err
    if "feature" not in table.columns or payload_column not in table.columns:
        raise DeclaredMethodError(f"{table_path} needs columns {schema_hint}")
    return table, table_path


def _bounds(rows: pd.DataFrame, column: str, table_path: Path, fill: float) -> np.ndarray:
    """Numeric bin bounds; an absent column, empty cells, -inf and inf all
    mean unbounded."""
    if column not in rows.columns:
        return np.full(len(rows), fill)
    try:
        values = pd.to_numeric(rows[column]).to_numpy(dtype=float)
    except (TypeError, ValueError) as err:
        raise DeclaredMethodError(
            f"{table_path}: column {column!r} must be numeric, empty, -inf or inf"
        ) from err
    return np.where(np.isnan(values), fill, values)


def _matched_bin_values(
    table: pd.DataFrame,
    table_path: Path,
    X: pd.DataFrame,
    payload_column: str,
    category_column: str,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """Payload of the bin each applicant falls in, for every feature of X.

    Numeric bins match ``bin_lower <= x < bin_upper`` (empty, ``-inf`` and
    ``inf`` cells mean unbounded); categorical bins match the
    ``category_column`` cell exactly. A value falling in no bin, or a
    feature with no bins at all, is a setup error naming the feature and
    value. Returns the matched payload matrix plus each feature's full
    payload vector, so callers can form their reference point (max for
    points lost, mean for univariate) from the table alone.
    """
    has_bounds = "bin_lower" in table.columns or "bin_upper" in table.columns
    has_category = category_column in table.columns
    if not has_bounds and not has_category:
        raise DeclaredMethodError(
            f"{table_path} has no bin columns: numeric features need "
            f"bin_lower,bin_upper and categorical features need {category_column!r}"
        )
    matched = pd.DataFrame(index=X.index, columns=list(X.columns), dtype=float)
    by_feature: dict[str, np.ndarray] = {}
    for col in X.columns:
        rows = table[table["feature"] == col]
        if rows.empty:
            raise DeclaredMethodError(
                f"{table_path} has no bins for feature {col!r}; the table "
                "must cover every model feature"
            )
        try:
            payload = pd.to_numeric(rows[payload_column]).to_numpy(dtype=float)
        except (TypeError, ValueError) as err:
            raise DeclaredMethodError(
                f"{table_path}: column {payload_column!r} must be numeric"
            ) from err
        if np.isnan(payload).any():
            raise DeclaredMethodError(
                f"{table_path}: feature {col!r} has empty or non-numeric "
                f"{payload_column!r} cells; every bin needs a numeric "
                f"{payload_column}"
            )
        by_feature[col] = payload
        categories = rows[category_column] if has_category else None
        if categories is not None and categories.notna().any():
            if categories.isna().any():
                raise DeclaredMethodError(
                    f"{table_path}: feature {col!r} mixes categorical bins "
                    f"({category_column!r}) with numeric ones; use one kind "
                    "of bin per feature"
                )
            hits = (
                X[col].astype(str).to_numpy()[:, None]
                == categories.astype(str).to_numpy()[None, :]
            )
        else:
            lower = _bounds(rows, "bin_lower", table_path, fill=-np.inf)
            upper = _bounds(rows, "bin_upper", table_path, fill=np.inf)
            x = pd.to_numeric(X[col], errors="coerce").to_numpy(dtype=float)
            hits = (x[:, None] >= lower[None, :]) & (x[:, None] < upper[None, :])
        found = hits.any(axis=1)
        if not found.all():
            raw = X[col].iloc[int(np.argmin(found))]
            value = raw.item() if hasattr(raw, "item") else raw
            raise DeclaredMethodError(
                f"{table_path}: value {value!r} of feature {col!r} falls in "
                "no declared bin"
            )
        multi = hits.sum(axis=1) > 1
        if multi.any():
            raw = X[col].iloc[int(np.argmax(multi))]
            value = raw.item() if hasattr(raw, "item") else raw
            raise DeclaredMethodError(
                f"{table_path}: value {value!r} of feature {col!r} falls in "
                f"{int(hits[int(np.argmax(multi))].sum())} declared bins; "
                "bins must be disjoint so the declared reason is unambiguous"
            )
        matched[col] = payload[hits.argmax(axis=1)]
    return matched, by_feature


def _most_points_lost(
    policy: ReasonCodePolicy, X: pd.DataFrame, covenants_dir: Path
) -> pd.DataFrame:
    """Points lost against the feature's best bin, read off the scorecard
    points table production actually prints (Krivorotov & Richey 2022,
    most-points-lost).

    Classic scorecard convention: higher points = lower risk. Declared
    attribution for a feature = (max points over that feature's bins) -
    (points of the applicant's bin) — "points lost", higher = stronger
    denial reason, matching the measured side. Numeric bins match
    ``bin_lower <= x < bin_upper`` (empty, -inf and inf cells mean
    unbounded); categorical bins match the ``value`` cell exactly.
    """
    table, table_path = _read_bin_table(
        policy,
        covenants_dir,
        "points_table",
        "points",
        "feature,bin_lower,bin_upper|value,points",
    )
    matched, by_feature = _matched_bin_values(table, table_path, X, "points", "value")
    for col in X.columns:
        matched[col] = by_feature[col].max() - matched[col]
    return matched


def _univariate(
    policy: ReasonCodePolicy, X: pd.DataFrame, covenants_dir: Path
) -> pd.DataFrame:
    """Krivorotov & Richey's univariate method: each feature judged on its
    own binned bad rate (or score), higher = riskier.

    Convention, stated explicitly: declared attribution for a feature =
    value(applicant's bin) - unweighted mean of that feature's bin values.
    The reference point is the table's own mean, not a portfolio average, so
    the artefact is self-contained — the check never needs the data snapshot
    to interpret it. Bins follow the points-table schema (``bin_lower <= x <
    bin_upper``, empty/-inf/inf cells unbounded) except that categorical
    bins match exactly on a ``category`` column, because ``value`` here is
    the bin-level payload.
    """
    table, table_path = _read_bin_table(
        policy,
        covenants_dir,
        "bins_table",
        "value",
        "feature,bin_lower,bin_upper|category,value",
    )
    matched, by_feature = _matched_bin_values(table, table_path, X, "value", "category")
    for col in X.columns:
        matched[col] = matched[col] - by_feature[col].mean()
    return matched


def _shapley_attributions(
    policy: ReasonCodePolicy,
    X: pd.DataFrame,
    ids: pd.Series,
    id_column: str,
    covenants_dir: Path,
) -> pd.DataFrame:
    """Per-applicant attributions exported by the production pipeline,
    joined to the scored rows on the stable id.

    Shapley-based adverse action is a real production method (Nair et al.,
    arXiv 2204.12365), and what production sends is a file — so the file is
    what gets tested, not a recomputation. Columns: the id column plus one
    column per feature, higher = pushes toward denial.
    """
    table_ref = policy.parameters.get("attributions_file")
    if table_ref is None:
        raise DeclaredMethodError(
            "reason_codes.parameters.attributions_file must point to a CSV "
            f"of {id_column},<one column per feature> for method shapley"
        )
    table_path = _resolve(table_ref, covenants_dir)
    try:
        table = pd.read_csv(table_path)
    except FileNotFoundError as err:
        raise DeclaredMethodError(f"reason-code artefact not found: {table_path}") from err
    if id_column not in table.columns:
        raise DeclaredMethodError(
            f"{table_path} lacks the id column {id_column!r} declared in "
            "checks.reason_codes.id_column"
        )
    missing = [c for c in X.columns if c not in table.columns]
    if missing:
        raise DeclaredMethodError(
            f"{table_path} lacks attribution columns for features: {missing}"
        )
    if table[id_column].duplicated().any():
        dupes = table.loc[table[id_column].duplicated(), id_column].head(5).tolist()
        raise DeclaredMethodError(
            f"{table_path} has duplicate ids in {id_column!r}: {dupes}"
        )
    table = table.set_index(id_column)
    missing_mask = ~ids.isin(table.index)
    if missing_mask.any():
        missing_ids = ids[missing_mask].head(5).tolist()
        raise DeclaredMethodError(
            f"{table_path} has no attributions for {int(missing_mask.sum())} "
            f"scored rows; first missing {id_column!r} values: {missing_ids}"
        )
    rows = table.loc[ids]
    try:
        values = rows[list(X.columns)].to_numpy(dtype=float)
    except (TypeError, ValueError) as err:
        raise DeclaredMethodError(
            f"{table_path}: attribution columns must be numeric"
        ) from err
    if np.isnan(values).any():
        bad = [c for c, isnan in zip(X.columns, np.isnan(values).any(axis=0), strict=True) if isnan]
        raise DeclaredMethodError(
            f"{table_path}: attribution columns have empty cells for features {bad}"
        )
    return pd.DataFrame(values, columns=list(X.columns), index=X.index)


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
    try:
        table = pd.read_csv(reasons_path)
    except FileNotFoundError as err:
        raise DeclaredMethodError(f"reason-code artefact not found: {reasons_path}") from err
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
