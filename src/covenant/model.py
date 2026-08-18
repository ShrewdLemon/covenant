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
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


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

    ``bad_class_index`` selects the column of ``predict_proba`` that is the
    probability of default/bad. The convention throughout Covenant is that
    higher ``p_bad`` means deny.
    """

    estimator: Any
    feature_names: list[str]
    bad_class_index: int = 1

    def __post_init__(self) -> None:
        if not hasattr(self.estimator, "predict_proba"):
            raise TypeError(
                f"{type(self.estimator).__name__} has no predict_proba; "
                "Covenant's model contract is predict_proba over a dataframe."
            )

    def _align(self, X: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_names if c not in X.columns]
        if missing:
            raise ValueError(f"data is missing declared features: {missing}")
        return X[self.feature_names]

    def p_bad(self, X: pd.DataFrame) -> np.ndarray:
        proba = self.estimator.predict_proba(self._align(X))
        return np.asarray(proba)[:, self.bad_class_index]

    def p_bad_from_array(self, X: np.ndarray) -> np.ndarray:
        """Score a numpy array already in declared feature order (SHAP path)."""
        frame = pd.DataFrame(X, columns=self.feature_names)
        proba = self.estimator.predict_proba(frame)
        return np.asarray(proba)[:, self.bad_class_index]
