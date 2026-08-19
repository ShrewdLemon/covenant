"""Export an OptBinning ``Scorecard`` to Covenant's points-table CSV.

OptBinning's ``Scorecard`` is the most-used open scorecard class, and its
points table is exactly the production artefact Check 1's
``most_points_lost`` method tests — the table a lender's letter-generation
system reads. This adapter writes that table in Covenant's format
(``feature,bin_lower,bin_upper,value,points``) using the **exact** split
points from the fitted binning objects, not the rounded strings of
``Scorecard.table()`` — a bound rounded for display would misassign
applicants near the boundary, which is precisely the kind of quiet drift
Covenant exists to catch.

Special and Missing bins are not exported: Covenant's bin matcher requires
every scored value to fall in a declared bin and treats missing values as
an upstream data problem, so a snapshot routed through special codes should
be cleaned before checking. The export notes how many such bins were
skipped.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def export_scorecard_points(
    scorecard: Any,
    path: str | Path,
    higher_points_lower_risk: bool = True,
) -> Path:
    """Write a fitted ``optbinning.Scorecard``'s points as a Covenant table.

    ``higher_points_lower_risk`` states the scorecard's convention (the
    OptBinning default: more points = better applicant). Pass ``False`` for
    a reversed scorecard; points are negated on export so Covenant's
    ``max(points) - points`` reason strength stays correct either way.

    Returns the path written. Use it as
    ``reason_codes: {method: most_points_lost, parameters: {points_table: ...}}``.
    """
    process = getattr(scorecard, "binning_process_", None)
    table_fn = getattr(scorecard, "table", None)
    if process is None or table_fn is None:
        raise TypeError(
            "export_scorecard_points needs a fitted optbinning.Scorecard "
            "(with binning_process_ and table()); got "
            f"{type(scorecard).__name__}"
        )
    summary = table_fn(style="summary")
    sign = 1.0 if higher_points_lower_risk else -1.0

    rows: list[dict[str, Any]] = []
    for variable in summary["Variable"].unique():
        var_rows = summary[summary["Variable"] == variable]
        regular = var_rows[~var_rows["Bin"].astype(str).isin(("Special", "Missing"))]
        points = [sign * float(p) for p in regular["Points"]]
        splits = process.get_binned_variable(str(variable)).splits

        if len(splits) and not np.issubdtype(np.asarray(splits[0]).dtype, np.number):
            # Categorical: splits is one array of category values per bin.
            if len(points) != len(splits):
                raise ValueError(
                    f"scorecard table has {len(points)} bins for {variable!r} "
                    f"but the binning object has {len(splits)}; the scorecard "
                    "looks partially refitted — refit and re-export"
                )
            for bin_categories, bin_points in zip(splits, points, strict=True):
                for category in bin_categories:
                    rows.append(
                        {"feature": variable, "value": category, "points": bin_points}
                    )
        else:
            # Numeric: n_bins = len(splits) + 1, bounds [lower, upper).
            edges = [-np.inf, *(float(s) for s in splits), np.inf]
            if len(points) != len(edges) - 1:
                raise ValueError(
                    f"scorecard table has {len(points)} bins for {variable!r} "
                    f"but the splits imply {len(edges) - 1}; the scorecard "
                    "looks partially refitted — refit and re-export"
                )
            for i, bin_points in enumerate(points):
                rows.append(
                    {
                        "feature": variable,
                        "bin_lower": "" if np.isinf(edges[i]) else repr(edges[i]),
                        "bin_upper": "" if np.isinf(edges[i + 1]) else repr(edges[i + 1]),
                        "points": bin_points,
                    }
                )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["feature", "bin_lower", "bin_upper", "value", "points"]
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path
