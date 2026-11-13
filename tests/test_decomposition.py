# -*- coding: utf-8 -*-

"""Tests for :mod:`sincomp.decomposition`.

The tests verify the scikit-learn estimator lifecycle (fit /
transform / get_params), the shapes and the reproducibility of the
embeddings of ``Factorizer``, ``DialectEmbedding`` and
``CharacterEmbedding``.
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"

import unittest

import numpy
import pandas
import scipy.sparse
import sklearn.base

from sincomp.decomposition import (
    CharacterEmbedding,
    DialectEmbedding,
    Factorizer,
)
from sincomp.feature_extraction import CharacterVectorizer

from common import make_wide


class TestFactorizer(unittest.TestCase):
    """Test the masked ALS factorization estimator."""

    def setUp(self):
        self.X = make_wide()

    def test_lifecycle(self):
        # the estimator follows the scikit-learn lifecycle and exposes
        # the fitted embeddings, components and iteration count
        f = Factorizer(n_components=4, random_state=0)
        self.assertIsInstance(f, sklearn.base.BaseEstimator)
        self.assertEqual(f.get_params()["n_components"], 4)
        sklearn.base.clone(f)

        self.assertIs(f.fit(self.X), f)

        emb = f.fit_transform(self.X)
        self.assertEqual(emb.shape, (4, 4))

        # the components hold one embedding column per distinct feature
        # value, so the embedding matrix is (n_components, n_feature_values)
        names = f.get_feature_names_out()
        self.assertEqual(f.components_.shape, (4, names.shape[0]))
        self.assertEqual(f.n_features_in_, 6)
        self.assertEqual(f.n_features_out_, 4)
        self.assertGreaterEqual(f.n_iter_, 1)

    def test_delegates_to_als(self):
        # the estimator only adapts the input, so fitting must agree
        # with a direct call of the internal factorization
        from sincomp._internal import factorize, preprocessing

        f = Factorizer(n_components=3, random_state=0)
        emb = f.fit_transform(self.X)
        cells = self.X.to_numpy(dtype=object)
        vectors, _ = preprocessing.features_to_vectors(cells)
        factor_emb, components, n_iter = factorize.als(
            vectors, n_components=3, random_state=0
        )

        numpy.testing.assert_array_equal(emb, factor_emb)
        numpy.testing.assert_array_equal(f.components_, components)
        self.assertEqual(f.n_iter_, n_iter)

    def test_random_state_reproducibility(self):
        emb1 = Factorizer(n_components=3, random_state=0) \
            .fit_transform(self.X)
        emb2 = Factorizer(n_components=3, random_state=0) \
            .fit_transform(self.X)
        numpy.testing.assert_array_equal(emb1, emb2)

    def test_two_stage_training(self):
        # coverage thresholds force a dense-prefix first stage followed
        # by second-stage updates of the remaining samples and columns;
        # every kept character still receives an embedding
        f = Factorizer(
            n_components=3,
            random_state=0,
            min_samples=2,
            min_features=3,
            min_sample_coverage=0.5,
            min_feature_coverage=0.5
        )
        emb = f.fit_transform(self.X)
        self.assertEqual(emb.shape, (4, 3))

    def test_vocabularies_and_feature_names(self):
        f = Factorizer(n_components=2, random_state=0).fit(self.X)
        self.assertEqual(len(f.vocabularies_), 6)
        self.assertEqual(set(f.vocabularies_[0]), {"b", "p", "m"})
        names = f.get_feature_names_out()
        self.assertEqual(names.shape[0], f.components_.shape[1])

    def test_flat_column_index(self):
        X = pandas.DataFrame(
            [["b ei 51", "p ei 24"], ["p ɔ 213", "p ɔ 53"]],
            columns=["beijing", "guangzhou"]
        )
        f = Factorizer(n_components=2, random_state=0)
        emb = f.fit_transform(X)
        self.assertEqual(f.n_features_in_, 2)
        self.assertEqual(emb.shape, (2, 2))

    def test_transform_projection(self):
        f = Factorizer(n_components=4, random_state=0).fit(self.X)
        out = f.transform(self.X)
        self.assertEqual(out.shape, (4, 4))

        # projecting a character with an unseen feature value ignores the
        # unknown value but still uses the known ones
        X_unknown = self.X.copy()
        X_unknown.iloc[0, 0] = "z"
        self.assertEqual(f.transform(X_unknown).shape, (4, 4))

    def test_multi_valued_cells(self):
        # a cell with several feature values contributes a multi-hot
        # target, so fitting succeeds and keeps all observed values
        X = self.X.copy()
        X.iloc[0, 0] = "b p"
        f = Factorizer(n_components=3, random_state=0)
        emb = f.fit_transform(X)
        self.assertIn("p", f.vocabularies_[0])
        self.assertEqual(emb.shape, (4, 3))

    def test_missing_values_do_not_crash(self):
        # row 字4 misses the whole guangzhou dialect but still receives
        # an embedding from the observed dialect
        f = Factorizer(n_components=3, random_state=0)
        emb = f.fit_transform(self.X)
        self.assertEqual(emb.shape, (4, 3))
        self.assertTrue(numpy.isfinite(emb).all())

    def test_empty_column_raises(self):
        # a column without any observed value has an empty vocabulary
        # and cannot be factorized
        X = self.X.copy()
        X["empty"] = ""
        with self.assertRaises(ValueError):
            Factorizer(n_components=3, random_state=0).fit_transform(X)

    def test_unknown_row_raises(self):
        f = Factorizer(n_components=3, random_state=0).fit(self.X)
        X_unknown = pandas.DataFrame(
            [["x y z"]], columns=self.X.columns[:1]
        )
        with self.assertRaises(ValueError):
            f.transform(X_unknown)

    def test_transform_before_fit_raises(self):
        with self.assertRaises(ValueError):
            Factorizer().transform(self.X)

    def test_invalid_input(self):
        X_dup = self.X.copy()
        X_dup.columns = ["a", "a", "b", "c", "d", "e"]
        with self.assertRaises(ValueError):
            Factorizer().fit(X_dup)

        with self.assertRaises(ValueError):
            Factorizer().fit(self.X.iloc[[]])

        with self.assertRaises(ValueError):
            Factorizer().fit(pandas.DataFrame(index=[]))


class TestDialectEmbedding(unittest.TestCase):
    """Test the truncated SVD embedding of the dialect feature rows."""

    def setUp(self):
        self.X = scipy.sparse.csr_matrix(
            numpy.array([
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1]
            ], dtype=numpy.float32)
        )

    def test_fit_transform_returns_array(self):
        emb = DialectEmbedding(embedding_size=2)
        out = emb.fit_transform(self.X)

        self.assertIsInstance(out, numpy.ndarray)
        self.assertEqual(out.shape, (4, 2))

    def test_fit_and_transform(self):
        emb = DialectEmbedding(embedding_size=2)
        out = emb.fit_transform(self.X)
        transformed = emb.transform(self.X)

        self.assertEqual(out.shape, (4, 2))
        self.assertEqual(transformed.shape, (4, 2))
        numpy.testing.assert_allclose(out, transformed, rtol=1e-6, atol=1e-7)
        self.assertTrue(numpy.isfinite(out).all())
        self.assertTrue(numpy.isfinite(transformed).all())

    def test_inverse_transform(self):
        emb = DialectEmbedding(embedding_size=2).fit(self.X)
        reduced = emb.transform(self.X)
        reconstructed = emb.inverse_transform(reduced)

        self.assertEqual(reconstructed.shape, self.X.shape)
        self.assertTrue(numpy.isfinite(reconstructed).all())


class TestCharacterEmbedding(unittest.TestCase):
    """Test the truncated SVD embedding of the character encodings."""

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

    def test_fit_transform_returns_embeddings(self):
        char_matrix = CharacterVectorizer().fit_transform(self.data)
        embedder = CharacterEmbedding(embedding_size=2)

        embeddings = embedder.fit_transform(char_matrix)
        self.assertEqual(embeddings.shape, (2, 2))
        self.assertEqual(
            len(embedder.get_feature_names_out()),
            2
        )

    def test_transform_and_inverse_transform(self):
        char_matrix = CharacterVectorizer().fit_transform(self.data)
        embedder = CharacterEmbedding(embedding_size=2).fit(char_matrix)

        transformed = embedder.transform(char_matrix)
        reconstructed = embedder.inverse_transform(transformed)

        self.assertEqual(transformed.shape, (2, 2))
        self.assertEqual(reconstructed.shape, char_matrix.shape)

    def test_get_feature_names_out(self):
        char_matrix = CharacterVectorizer().fit_transform(self.data)
        embedder = CharacterEmbedding(embedding_size=2).fit(char_matrix)
        names = embedder.get_feature_names_out()

        self.assertEqual(names.shape[0], 2)
        self.assertEqual(names[0], "embed_0")


if __name__ == "__main__":
    unittest.main()
