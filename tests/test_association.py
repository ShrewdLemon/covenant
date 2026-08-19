from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from covenant.association import association, correlation_ratio, cramers_v, spearman_abs


class TestSpearmanAbs:
    def test_identical_vectors(self) -> None:
        x = np.linspace(-3, 3, 200)
        assert spearman_abs(x, x) == pytest.approx(1.0)

    def test_monotone_transform_is_perfect(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.normal(size=300)
        assert spearman_abs(x, np.exp(x)) == pytest.approx(1.0)

    def test_independent_is_small(self) -> None:
        rng = np.random.default_rng(1)
        x = rng.normal(size=2000)
        y = rng.normal(size=2000)
        assert spearman_abs(x, y) < 0.1

    def test_constant_input_is_zero(self) -> None:
        rng = np.random.default_rng(2)
        assert spearman_abs(np.ones(50), rng.normal(size=50)) == 0.0

    def test_nan_pairs_dropped(self) -> None:
        x = np.linspace(0, 1, 100)
        y = x.copy()
        x[3], y[7] = np.nan, np.nan
        assert spearman_abs(x, y) == pytest.approx(1.0)


class TestCorrelationRatio:
    def test_category_determines_value(self) -> None:
        rng = np.random.default_rng(3)
        cats = rng.choice(list("abc"), size=600)
        values = pd.Series(cats).map({"a": 0.0, "b": 5.0, "c": 9.0}).to_numpy()
        assert correlation_ratio(cats, values) == pytest.approx(1.0)

    def test_independent_is_small(self) -> None:
        rng = np.random.default_rng(4)
        cats = rng.choice(list("abc"), size=2000)
        values = rng.normal(size=2000)
        assert correlation_ratio(cats, values) < 0.1

    def test_constant_values_is_zero(self) -> None:
        cats = ["a", "b", "c"] * 20
        assert correlation_ratio(cats, np.full(60, 2.5)) == 0.0

    def test_single_category_is_zero(self) -> None:
        rng = np.random.default_rng(5)
        assert correlation_ratio(["a"] * 50, rng.normal(size=50)) == 0.0


class TestCramersV:
    def test_identical_categoricals(self) -> None:
        rng = np.random.default_rng(6)
        cats = rng.choice(list("abc"), size=500)
        assert cramers_v(cats, cats) == pytest.approx(1.0, abs=0.01)

    def test_independent_is_small(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.choice(list("abc"), size=2000)
        b = rng.choice(list("xyz"), size=2000)
        assert cramers_v(a, b) < 0.1

    def test_constant_column_is_zero(self) -> None:
        rng = np.random.default_rng(8)
        assert cramers_v(["k"] * 100, rng.choice(list("ab"), size=100)) == 0.0

    def test_tiny_table_is_zero(self) -> None:
        assert cramers_v(["a", "b"], ["x", "y"]) == 0.0


class TestAssociationDispatch:
    def test_numeric_numeric_uses_spearman(self) -> None:
        rng = np.random.default_rng(9)
        x = pd.Series(rng.normal(size=200))
        strength, method = association(x, x * 2 + 1)
        assert method == "spearman_abs"
        assert strength == pytest.approx(1.0)

    def test_mixed_uses_correlation_ratio_symmetrically(self) -> None:
        rng = np.random.default_rng(10)
        cats = pd.Series(rng.choice(list("ab"), size=300))
        values = pd.Series(cats.map({"a": -1.0, "b": 1.0}) + rng.normal(0, 0.1, 300))
        strength_cv, method_cv = association(cats, values)
        strength_vc, method_vc = association(values, cats)
        assert method_cv == method_vc == "correlation_ratio"
        assert strength_cv == strength_vc
        assert strength_cv > 0.9

    def test_categorical_categorical_uses_cramers_v(self) -> None:
        rng = np.random.default_rng(11)
        a = pd.Series(rng.choice(list("abc"), size=400))
        strength, method = association(a, a)
        assert method == "cramers_v"
        assert strength == pytest.approx(1.0, abs=0.01)

    def test_dtype_decides_not_cardinality(self) -> None:
        """A 0/1 numeric coding is numeric by dtype, so the pair goes to
        Spearman — dispatch is predictable from the schema alone."""
        rng = np.random.default_rng(12)
        binary = pd.Series((rng.uniform(size=500) < 0.5).astype(int))
        _, method = association(binary, pd.Series(rng.normal(size=500)))
        assert method == "spearman_abs"
