# -*- coding: utf-8 -*-

"""Tests for :mod:`sincomp.feature_extraction`.

The tests verify the scikit-learn estimator lifecycle and the output
shapes of ``PhoneSimilarity``, ``DialectVectorizer``,
``CharacterVectorizer`` and ``DialectCompliance``, including the
round trips through ``inverse_transform`` and the delegation of
``DialectCompliance`` to the underlying compliance computation.
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"

import unittest

import numpy
import pandas
import scipy.sparse
import sklearn.base

from sincomp.feature_extraction import (
    CharacterVectorizer,
    DialectCompliance,
    DialectVectorizer,
    PhoneSimilarity,
)

from common import make_rules, make_wide


class TestPhoneSimilarity(unittest.TestCase):
    """Test the phone similarity features of the individual dialects."""

    def setUp(self):
        self.X = numpy.array([
            ["aa aa", "bb bb", "cc cc"],
            ["aa aa", "bb", "cc cc"]
        ], dtype=object)

    def test_fit_sets_attributes(self):
        model = PhoneSimilarity(dtype=numpy.float32)
        fitted = model.fit(self.X)

        self.assertIs(fitted, model)
        self.assertEqual(model.n_features_in_, 3)
        self.assertEqual(model.n_features_out_, 9)

    def test_transform_returns_sparse(self):
        model = PhoneSimilarity(dtype=numpy.float32).fit(self.X)
        out = model.transform(self.X)

        self.assertTrue(scipy.sparse.isspmatrix(out))
        self.assertEqual(out.shape, (2, 9))

    def test_fit_transform_matches(self):
        model = PhoneSimilarity(dtype=numpy.float32)
        expect = model.fit_transform(self.X)
        self.assertTrue(scipy.sparse.isspmatrix(expect))
        self.assertEqual(expect.shape, (2, 9))

    def test_get_feature_names_out(self):
        model = PhoneSimilarity(dtype=numpy.float32).fit(self.X)
        names = model.get_feature_names_out(["i0", "i1", "i2"])

        self.assertEqual(names.shape[0], 9)
        self.assertEqual(names[0], "i0_i0")
        self.assertEqual(names[-1], "i2_i2")

    def test_inverse_transform_returns_labels(self):
        model = PhoneSimilarity(dtype=numpy.float32).fit(self.X)
        transformed = model.transform(self.X)
        recovered = model.inverse_transform(transformed)

        self.assertEqual(recovered.shape, (2, 3))
        self.assertTrue(all(isinstance(x, str) for x in recovered.ravel()))


class TestDialectVectorizer(unittest.TestCase):
    """Test the aggregation of the phone similarity features per dialect."""

    def setUp(self):
        self.df = pandas.DataFrame(
            {
                "initial": ["t k", "t"],
                "final": ["a", "a m"],
                "tone": ["1", "1"]
            },
            index=["dialect1", "dialect2"]
        )

    def test_fit_sets_attributes(self):
        vec = DialectVectorizer()
        fitted = vec.fit(self.df)

        self.assertIs(fitted, vec)
        self.assertEqual(vec.n_features_in_, 3)
        self.assertGreater(vec.n_features_out_, 0)

    def test_transform_returns_sparse(self):
        vec = DialectVectorizer().fit(self.df)
        out = vec.transform(self.df)

        self.assertTrue(scipy.sparse.isspmatrix(out))
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], vec.n_features_out_)

    def test_fit_transform_matches(self):
        vec = DialectVectorizer()
        out = vec.fit_transform(self.df)

        self.assertTrue(scipy.sparse.isspmatrix(out))
        self.assertEqual(out.shape[0], 2)

    def test_get_feature_names_out(self):
        vec = DialectVectorizer().fit(self.df)
        names = vec.get_feature_names_out(["initial", "final", "tone"])

        self.assertEqual(names.shape[0], vec.n_features_out_)

    def test_inverse_transform(self):
        vec = DialectVectorizer().fit(self.df)
        transformed = vec.transform(self.df)
        recovered = vec.inverse_transform(transformed)

        self.assertEqual(recovered.shape, self.df.shape)
        self.assertTrue(numpy.issubdtype(recovered.dtype, numpy.str_))


class TestCharacterVectorizer(unittest.TestCase):
    """Test the multi-hot encoding of the phones of every character."""

    def setUp(self):
        self.data = pandas.DataFrame(
            [
                ["t k", "a", "p"],
                ["t", "a m", "p n"]
            ],
            index=["char1", "char2"],
            columns=[
                "dialect1_initial",
                "dialect1_final",
                "dialect2_initial"
            ]
        )

    def test_fit_transform(self):
        vectorizer = CharacterVectorizer()
        vectors = vectorizer.fit_transform(self.data)

        self.assertTrue(scipy.sparse.isspmatrix(vectors))
        self.assertEqual(vectors.shape[0], 2)
        self.assertGreater(vectors.shape[1], 0)
        self.assertEqual(
            len(vectorizer.get_feature_names_out()),
            vectors.shape[1]
        )

    def test_transform_only(self):
        vectorizer = CharacterVectorizer()
        vectorizer.fit(self.data)
        transformed = vectorizer.transform(self.data)

        self.assertTrue(scipy.sparse.isspmatrix(transformed))
        self.assertEqual(transformed.shape[0], 2)

    def test_get_feature_names_out(self):
        vectorizer = CharacterVectorizer().fit(self.data)
        names = vectorizer.get_feature_names_out()

        self.assertIsInstance(names, numpy.ndarray)
        self.assertEqual(
            names.shape[0],
            vectorizer.transform(self.data).shape[1]
        )


class TestDialectCompliance(unittest.TestCase):
    """Test the rule compliance estimator."""

    def setUp(self):
        self.X = make_wide()
        self.rules = make_rules()

    def test_lifecycle(self):
        # the estimator follows the scikit-learn lifecycle and outputs
        # one compliance row per dialect and one column per rule
        est = DialectCompliance(self.rules)
        self.assertIsInstance(est, sklearn.base.BaseEstimator)
        self.assertIs(est.get_params()["rules"], self.rules)
        sklearn.base.clone(est)

        self.assertIs(est.fit(self.X), est)
        self.assertEqual(est.n_features_in_, self.X.shape[1])

        out = est.fit_transform(self.X)
        self.assertIsInstance(out, pandas.DataFrame)
        self.assertEqual(list(out.index), ["beijing", "guangzhou"])
        self.assertEqual(list(out.columns), ["r1", "r2"])

    def test_delegates_to_compliance(self):
        from sincomp._internal import compare

        est = DialectCompliance(self.rules).fit(self.X)
        out = est.transform(self.X)
        expected = compare.compliance(
            self.X, self.rules, dtype=numpy.float32, norm=2
        )
        pandas.testing.assert_frame_equal(out, expected)

    def test_invalid_rules(self):
        with self.assertRaises(ValueError):
            DialectCompliance(["not", "a", "frame"]).fit(self.X)

        bad = self.rules.drop(columns=["cid2"])
        with self.assertRaises(ValueError):
            DialectCompliance(bad).fit(self.X)

    def test_transform_before_fit_raises(self):
        with self.assertRaises(ValueError):
            DialectCompliance(self.rules).transform(self.X)


if __name__ == "__main__":
    unittest.main()
