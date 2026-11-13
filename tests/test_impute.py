# -*- coding: utf-8 -*-

"""Tests for :mod:`sincomp.impute`.

The tests verify the scikit-learn estimator lifecycle, the missing
value filling and the delegation of ``KNeighborsImputer`` and
``FactorizedImputer``, plus an integration test that imputes the
missing readings of a real wide feature table.
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"

import unittest

import numpy
import pandas
import sklearn.base

from sincomp.impute import FactorizedImputer, KNeighborsImputer


def make_missing_table():
    """Build a small wide table with one missing cell."""
    return pandas.DataFrame(
        [["b", "ei"], ["p", pandas.NA], ["b", "ɔ"]],
        columns=pandas.MultiIndex.from_tuples([
            ("d1", "initial"),
            ("d1", "final"),
        ]),
        index=["字1", "字2", "字3"],
    )


class TestKNeighborsImputer(unittest.TestCase):
    """Test the KNN imputation of the missing cells."""

    def setUp(self):
        self.X = make_missing_table()

    def test_lifecycle(self):
        # the estimator follows the scikit-learn lifecycle and fills
        # every missing cell
        est = KNeighborsImputer(n_neighbors=2)
        self.assertIsInstance(est, sklearn.base.BaseEstimator)
        self.assertEqual(est.get_params()["n_neighbors"], 2)
        sklearn.base.clone(est)

        self.assertIs(est.fit(self.X), est)
        self.assertEqual(est.n_features_in_, 2)

        out = est.fit_transform(self.X)
        self.assertEqual(out.shape, self.X.shape)
        self.assertFalse(out.isna().any().any())

    def test_numpy_input(self):
        # ndarray input marks the missing values with empty strings
        values = numpy.array(
            [["b", "ei"], ["p", ""], ["b", "ɔ"]], dtype=object
        )
        est = KNeighborsImputer().fit(values)
        out = est.transform(values)
        self.assertIsInstance(out, numpy.ndarray)
        self.assertEqual(out.shape, values.shape)
        self.assertFalse(any(value == "" for value in out.flat))

    def test_imputes_with_same_feature_only(self):
        # the d1 initial is imputed from the initials of the other
        # dialect only, not from the finals: with the finals included
        # in the distances, 字3 would be the nearest character and
        # "p" would be predicted
        X = pandas.DataFrame(
            [
                ["b", "x1", "b", "x1"],
                [pandas.NA, "x2", "b", "x2"],
                ["p", "x2", "p", "x2"],
            ],
            columns=pandas.MultiIndex.from_tuples([
                ("d1", "initial"), ("d1", "final"),
                ("d2", "initial"), ("d2", "final"),
            ]),
            index=["字1", "字2", "字3"],
        )
        out = KNeighborsImputer(n_neighbors=1).fit(X).transform(X)
        self.assertEqual(out.iloc[1, 0], "b")

    def test_transform_before_fit_raises(self):
        with self.assertRaises(ValueError):
            KNeighborsImputer().transform(self.X)

    def test_feature_names_out(self):
        est = KNeighborsImputer().fit(self.X)
        numpy.testing.assert_array_equal(
            est.get_feature_names_out(), self.X.columns.values
        )


class TestFactorizedImputer(unittest.TestCase):
    """Test the factorization-based imputation of the missing cells."""

    def setUp(self):
        self.X = make_missing_table()

    def test_lifecycle(self):
        # the estimator follows the scikit-learn lifecycle and fills
        # every missing cell
        est = FactorizedImputer(algorithm="svd", n_components=2)
        self.assertIsInstance(est, sklearn.base.BaseEstimator)
        self.assertEqual(est.get_params()["algorithm"], "svd")
        self.assertEqual(est.get_params()["n_components"], 2)
        self.assertEqual(est.get_params()["n_neighbors"], 1)
        sklearn.base.clone(est)

        self.assertIs(est.fit(self.X), est)
        self.assertEqual(est.n_features_in_, 2)

        out = est.fit_transform(self.X)
        self.assertEqual(out.shape, self.X.shape)
        self.assertFalse(out.isna().any().any())

    def test_als_fills_missing(self):
        out = FactorizedImputer(
            algorithm="als", n_components=2, random_state=0
        ).fit(self.X).transform(self.X)
        self.assertFalse(out.isna().any().any())
        self.assertIn(out.iloc[1, 1], ["ei", "ɔ"])
        # observed cells are never modified
        pandas.testing.assert_series_equal(
            out.iloc[[0, 2], 1], self.X.iloc[[0, 2], 1]
        )

    def test_svd_fills_missing(self):
        out = FactorizedImputer(
            algorithm="svd", n_components=2, random_state=0
        ).fit(self.X).transform(self.X)
        self.assertFalse(out.isna().any().any())
        self.assertIn(out.iloc[1, 1], ["ei", "ɔ"])

    def test_weak_columns_kept_unchanged(self):
        # a column with a single distinct observed reading cannot be
        # factorized and is kept unchanged
        X = self.X.copy()
        X[("d1", "tone")] = ["51", pandas.NA, "51"]

        out = FactorizedImputer(
            algorithm="als", n_components=2, random_state=0
        ).fit(X).transform(X)
        self.assertTrue(out[("d1", "tone")].isna().iloc[1])
        pandas.testing.assert_series_equal(
            out[("d1", "tone")], X[("d1", "tone")]
        )

    def test_multi_value_cells(self):
        # a cell holding several values votes for all of them but only
        # the highest voted one is predicted
        X = pandas.DataFrame(
            [["a b", "ei"], [pandas.NA, "ɔ"], ["a c", "ɔ"]],
            columns=pandas.MultiIndex.from_tuples(
                [("d1", "initial"), ("d1", "final")]
            ),
            index=["字1", "字2", "字3"],
        )
        out = FactorizedImputer(
            algorithm="als", n_components=2, random_state=0
        ).fit(X).transform(X)
        self.assertIn(out.iloc[1, 0], ["a", "b", "c"])

    def test_unknown_algorithm_raises(self):
        est = FactorizedImputer(algorithm="pca").fit(self.X)
        with self.assertRaises(ValueError):
            est.transform(self.X)

    def test_transform_before_fit_raises(self):
        with self.assertRaises(ValueError):
            FactorizedImputer().transform(self.X)

    def test_feature_names_out(self):
        est = FactorizedImputer().fit(self.X)
        numpy.testing.assert_array_equal(
            est.get_feature_names_out(), self.X.columns.values
        )


class TestImputeIntegration(unittest.TestCase):
    """Test the imputation on a real wide pronunciation table."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.data = make_missing_table()

    def test_knn_imputer_fills_all_missing_values(self):
        # every missing cell receives one of the observed readings of
        # its column and no observed cell changes
        imputed = KNeighborsImputer().fit_transform(self.data)

        self.assertListEqual(self.data.index.to_list(), imputed.index.to_list())
        self.assertListEqual(self.data.columns.to_list(), imputed.columns.to_list())
        self.assertTrue(imputed.notna().all(axis=None))

        for c in self.data.columns:
            self.assertSetEqual(
                set(" ".join(self.data[c].dropna()).split()),
                set(" ".join(imputed[c].dropna()).split()),
            )


if __name__ == "__main__":
    unittest.main()
