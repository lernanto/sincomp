# -*- coding: utf-8 -*-

import unittest

import numpy as np
import pandas as pd
import scipy.sparse

import sincomp.embedding as embedding


class TestPhoneSimilarity(unittest.TestCase):
    def setUp(self):
        self.X = np.array([
            ['aa aa', 'bb bb', 'cc cc'],
            ['aa aa', 'bb', 'cc cc']
        ], dtype=object)

    def test_fit_sets_attributes(self):
        model = embedding.PhoneSimilarity(dtype=np.float32)
        fitted = model.fit(self.X)

        self.assertIs(fitted, model)
        self.assertEqual(model.n_features_in_, 3)
        self.assertEqual(model.n_features_out_, 9)

    def test_transform_returns_sparse(self):
        model = embedding.PhoneSimilarity(dtype=np.float32).fit(self.X)
        out = model.transform(self.X)

        self.assertTrue(scipy.sparse.isspmatrix(out))
        self.assertEqual(out.shape, (2, 9))

    def test_fit_transform_matches(self):
        model = embedding.PhoneSimilarity(dtype=np.float32)
        expect = model.fit_transform(self.X)
        self.assertTrue(scipy.sparse.isspmatrix(expect))
        self.assertEqual(expect.shape, (2, 9))

    def test_get_feature_names_out(self):
        model = embedding.PhoneSimilarity(dtype=np.float32).fit(self.X)
        names = model.get_feature_names_out(['i0', 'i1', 'i2'])

        self.assertEqual(names.shape[0], 9)
        self.assertEqual(names[0], 'i0_i0')
        self.assertEqual(names[-1], 'i2_i2')

    def test_inverse_transform_returns_labels(self):
        model = embedding.PhoneSimilarity(dtype=np.float32).fit(self.X)
        transformed = model.transform(self.X)
        recovered = model.inverse_transform(transformed)

        self.assertEqual(recovered.shape, (2, 3))
        self.assertTrue(all(isinstance(x, str) for x in recovered.ravel()))


class TestDialectVectorizer(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            {
                'initial': ['t k', 't'],
                'final': ['a', 'a m'],
                'tone': ['1', '1']
            },
            index=['dialect1', 'dialect2']
        )

    def test_fit_sets_attributes(self):
        vec = embedding.DialectVectorizer()
        fitted = vec.fit(self.df)

        self.assertIs(fitted, vec)
        self.assertEqual(vec.n_features_in_, 3)
        self.assertGreater(vec.n_features_out_, 0)

    def test_transform_returns_sparse(self):
        vec = embedding.DialectVectorizer().fit(self.df)
        out = vec.transform(self.df)

        self.assertTrue(scipy.sparse.isspmatrix(out))
        self.assertEqual(out.shape[0], 2)
        self.assertEqual(out.shape[1], vec.n_features_out_)

    def test_fit_transform_matches(self):
        vec = embedding.DialectVectorizer()
        out = vec.fit_transform(self.df)

        self.assertTrue(scipy.sparse.isspmatrix(out))
        self.assertEqual(out.shape[0], 2)

    def test_get_feature_names_out(self):
        vec = embedding.DialectVectorizer().fit(self.df)
        names = vec.get_feature_names_out(['initial', 'final', 'tone'])

        self.assertEqual(names.shape[0], vec.n_features_out_)

    def test_inverse_transform(self):
        vec = embedding.DialectVectorizer().fit(self.df)
        transformed = vec.transform(self.df)
        recovered = vec.inverse_transform(transformed)

        self.assertEqual(recovered.shape, self.df.shape)
        self.assertTrue(np.issubdtype(recovered.dtype, np.str_))


class TestDialectEmbedding(unittest.TestCase):
    def setUp(self):
        self.X = scipy.sparse.csr_matrix(
            np.array([
                [1, 0, 0, 1],
                [0, 1, 1, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1]
            ], dtype=np.float32)
        )

    def test_fit_transform_returns_array(self):
        emb = embedding.DialectEmbedding(embedding_size=2)
        out = emb.fit_transform(self.X)

        self.assertIsInstance(out, np.ndarray)
        self.assertEqual(out.shape, (4, 2))

    def test_fit_and_transform(self):
        emb = embedding.DialectEmbedding(embedding_size=2)
        out = emb.fit_transform(self.X)
        transformed = emb.transform(self.X)

        self.assertEqual(out.shape, (4, 2))
        self.assertEqual(transformed.shape, (4, 2))
        np.testing.assert_allclose(out, transformed, rtol=1e-6, atol=1e-7)
        self.assertTrue(np.isfinite(out).all())
        self.assertTrue(np.isfinite(transformed).all())

    def test_inverse_transform(self):
        emb = embedding.DialectEmbedding(embedding_size=2).fit(self.X)
        reduced = emb.transform(self.X)
        reconstructed = emb.inverse_transform(reduced)

        self.assertEqual(reconstructed.shape, self.X.shape)
        self.assertTrue(np.isfinite(reconstructed).all())


class TestCharacterVectorizerAndEmbedding(unittest.TestCase):
    def setUp(self):
        self.data = pd.DataFrame(
            [
                ['t k', 'a', 'p'],
                ['t', 'a m', 'p n']
            ],
            index=['char1', 'char2'],
            columns=[
                'dialect1_initial',
                'dialect1_final',
                'dialect2_initial'
            ]
        )

    def test_character_vectorizer_fit_transform(self):
        vectorizer = embedding.CharacterVectorizer()
        vectors = vectorizer.fit_transform(self.data)

        self.assertTrue(scipy.sparse.isspmatrix(vectors))
        self.assertEqual(vectors.shape[0], 2)
        self.assertGreater(vectors.shape[1], 0)
        self.assertEqual(
            len(vectorizer.get_feature_names_out()),
            vectors.shape[1]
        )

    def test_character_vectorizer_transform_only(self):
        vectorizer = embedding.CharacterVectorizer()
        vectorizer.fit(self.data)
        transformed = vectorizer.transform(self.data)

        self.assertTrue(scipy.sparse.isspmatrix(transformed))
        self.assertEqual(transformed.shape[0], 2)

    def test_character_vectorizer_get_feature_names_out(self):
        vectorizer = embedding.CharacterVectorizer().fit(self.data)
        names = vectorizer.get_feature_names_out()

        self.assertIsInstance(names, np.ndarray)
        self.assertEqual(
            names.shape[0],
            vectorizer.transform(self.data).shape[1]
        )

    def test_character_embedding_fit_transform(self):
        char_matrix = embedding.CharacterVectorizer().fit_transform(self.data)
        embedder = embedding.CharacterEmbedding(embedding_size=2)

        embeddings = embedder.fit_transform(char_matrix)
        self.assertEqual(embeddings.shape, (2, 2))
        self.assertEqual(
            len(embedder.get_feature_names_out()),
            2
        )

    def test_character_embedding_transform_inverse_transform(self):
        char_matrix = embedding.CharacterVectorizer().fit_transform(
            self.data
        )
        embedder = embedding.CharacterEmbedding(embedding_size=2).fit(
            char_matrix
        )

        transformed = embedder.transform(char_matrix)
        reconstructed = embedder.inverse_transform(transformed)

        self.assertEqual(transformed.shape, (2, 2))
        self.assertEqual(reconstructed.shape, char_matrix.shape)

    def test_character_embedding_get_feature_names_out(self):
        char_matrix = embedding.CharacterVectorizer().fit_transform(self.data)
        embedder = embedding.CharacterEmbedding(embedding_size=2).fit(
            char_matrix
        )
        names = embedder.get_feature_names_out()

        self.assertEqual(names.shape[0], 2)
        self.assertEqual(names[0], 'embed_0')


if __name__ == '__main__':
    unittest.main()
