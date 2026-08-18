"""The model contract: anything with ``predict_proba`` over a dataframe.

Covenant does not require a wrapper class from the user; it duck-types
``predict_proba`` and reorders columns to the covenant's declared feature
order before scoring, so a dataframe with extra or shuffled columns is
handled predictably.
"""

from __future__ import annotations

import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


def load_model(path: str | Path) -> Any:
    """Load a persisted estimator (joblib or pickle).

    Loading unpickles arbitrary code: only point Covenant at model files
    you trust, i.e. your own.
    """
    path = Path(path)
    try:
        return joblib.load(path)
    except Exception as joblib_err:
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as pickle_err:
            raise ValueError(
                f"could not load {path} as joblib ({joblib_err}) "
                f"or pickle ({pickle_err})"
            ) from pickle_err


def library_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    for name in ("numpy", "pandas", "sklearn", "shap", "xgboost", "lightgbm"):
        mod = sys.modules.get(name)
        if mod is None:
            try:
                mod = __import__(name)
            except ImportError:
                continue
        versions[name] = getattr(mod, "__version__", "unknown")
    from covenant import __version__

    versions["covenant"] = __version__
    return versions


@dataclass
class CovenantModel:
    """A scored model bound to its declared feature order.

    ``positive_class`` is the label of the bad/default class as it appears
    in the estimator's ``classes_``; it selects the column of
    ``predict_proba`` that is ``p_bad``. The convention throughout Covenant
    is that higher ``p_bad`` means deny. When unset, the class at index 1
    is assumed — the sklearn convention for 0/1 targets.
    """

    estimator: Any
    feature_names: list[str]
    positive_class: int | str | None = None
    bad_class_index: int = 1

    def __post_init__(self) -> None:
        if not hasattr(self.estimator, "predict_proba"):
            raise TypeError(
                f"{type(self.estimator).__name__} has no predict_proba; "
                "Covenant's model contract is predict_proba over a dataframe."
            )
        if self.positive_class is not None:
            classes = getattr(self.estimator, "classes_", None)
            if classes is None:
                raise ValueError(
                    "positive_class is declared but the estimator exposes no "
                    "classes_; drop positive_class or use an estimator that "
                    "records its classes"
                )
            matches = [i for i, c in enumerate(classes) if c == self.positive_class]
            if not matches:
                raise ValueError(
                    f"positive_class {self.positive_class!r} not in the "
                    f"estimator's classes_ {list(classes)!r}"
                )
            self.bad_class_index = matches[0]

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"data is missing declared features: {missing}")
        return X[self.feature_names]

    def p_bad(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.estimator.predict_proba(self._align(X))
        return np.asarray(proba)[:, self.bad_class_index]
