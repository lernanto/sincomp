# -*- coding: utf-8 -*-

"""Tests for :mod:`sincomp._internal.factorize`.

The tests verify the ALS half-steps ``solve_embeddings`` and
``solve_components`` against analytically constructed factorizations,
and the shapes, the reproducibility, the input validation and the
masked objective of the ``factorize`` entry point.
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"

import unittest

import numpy
import scipy.sparse

from sincomp._internal.factorize import (
    als,
    solve_components,
    solve_embeddings,
)


def make_blocks(values, n_samples, n_components, seed=0, binarize=True):
    """Build sparse blocks consistent with a known factorization.

    The targets of block ``j`` are generated as the products ``u_i^T v``
    of random sample embeddings and feature-value embeddings, so the
    least-squares systems of the ALS half-steps have an exact solution.
    With ``binarize=True`` the products are thresholded into multi-hot
    targets, as consumed by :func:`factorize`.
    """

    rng = numpy.random.default_rng(seed)
    embeddings = rng.standard_normal((n_samples, n_components))
    components = [
        rng.standard_normal((n_components, vocab)) for vocab in values
    ]
    blocks = []
    for c in components:
        targets = embeddings @ c
        if binarize:
            targets = (targets > 0).astype(numpy.float32)
        blocks.append(scipy.sparse.csr_matrix(targets))
    mask = numpy.ones((n_samples, len(blocks)), dtype=bool)
    return blocks, mask, embeddings, components


class TestSolveEmbeddings(unittest.TestCase):
    """Test the least-squares solve of the sample embeddings."""

    def test_recovers_exact_embeddings(self):
        # with every cell observed and no regularization, the solve
        # reproduces the sample embeddings that generated the targets
        blocks, mask, embeddings, components = make_blocks(
            [4, 5], 20, 3, binarize=False
        )
        solved = solve_embeddings(blocks, mask, components, 0.0)
        numpy.testing.assert_allclose(solved, embeddings, atol=1e-6)

    def test_missing_blocks_exclude_their_components(self):
        # a masked-out block contributes neither its product matrix nor
        # its targets: solving only against the observed block still
        # returns finite embeddings of the right shape
        blocks, _, _, components = make_blocks([4, 5], 20, 3)
        mask = numpy.ones((20, 2), dtype=bool)
        mask[:, 1] = False
        blocks = [blocks[0], scipy.sparse.csr_matrix((20, 5))]

        solved = solve_embeddings(blocks, mask, components, 1.0)
        self.assertEqual(solved.shape, (20, 3))
        self.assertTrue(numpy.isfinite(solved).all())


class TestSolveComponents(unittest.TestCase):
    """Test the block-wise least-squares solve of the value embeddings."""

    def test_recovers_exact_components(self):
        # with every cell observed and no regularization, the solve
        # reproduces the feature-value embeddings that generated the
        # targets
        blocks, mask, embeddings, components = make_blocks(
            [4, 5], 20, 3, binarize=False
        )
        solved = solve_components(blocks, mask, embeddings, 0.0)
        self.assertEqual(len(solved), 2)
        for got, expect in zip(solved, components):
            numpy.testing.assert_allclose(got, expect, atol=1e-6)

    def test_unobserved_block_solves_to_zero(self):
        # a block with no observed sample solves its regularized system
        # to zero components
        blocks, _, embeddings, _ = make_blocks([4], 20, 3)
        empty = scipy.sparse.csr_matrix((20, 4))
        mask = numpy.ones((20, 2), dtype=bool)
        mask[:, 1] = False

        solved = solve_components(
            [blocks[0], empty], mask, embeddings, 1.0
        )
        numpy.testing.assert_array_equal(solved[1], numpy.zeros((3, 4)))


class TestFactorize(unittest.TestCase):
    """Test the masked ALS factorization entry point."""

    def test_returns_embeddings_components_and_iterations(self):
        blocks, _, _, _ = make_blocks([4, 5, 3], 30, 128, seed=1)
        emb, comps, n_iter = als(
            blocks, n_components=8, max_iter=5, random_state=0
        )

        self.assertEqual(emb.shape, (30, 8))
        self.assertEqual(comps.shape, (8, 12))
        self.assertGreaterEqual(n_iter, 1)
        self.assertTrue(numpy.isfinite(emb).all())
        self.assertTrue(numpy.isfinite(comps).all())

    def test_reproducible_with_random_state(self):
        blocks, _, _, _ = make_blocks([4, 5], 30, 128, seed=1)
        first = als(blocks, n_components=8, max_iter=3, random_state=0)
        second = als(blocks, n_components=8, max_iter=3, random_state=0)

        numpy.testing.assert_array_equal(first[0], second[0])
        numpy.testing.assert_array_equal(first[1], second[1])
        self.assertEqual(first[2], second[2])

    def test_fit_reduces_masked_error(self):
        # the ALS iterations decrease the masked prediction error of
        # the observed cells below a random initialization
        blocks, _, _, _ = make_blocks([4, 5], 30, 3, seed=2)

        def masked_rmse(emb, comps_matrix):
            # comps_matrix holds the value embeddings of the blocks
            # concatenated along the feature axis
            preds = emb @ comps_matrix
            errors = []
            column = 0
            for block in blocks:
                size = block.shape[1]
                pred = preds[:, column:column + size]
                observed = block.toarray() > 0
                errors.append(((pred - block.toarray())[observed]) ** 2)
                column += size
            return numpy.sqrt(numpy.concatenate(errors).mean())

        emb0 = numpy.random.default_rng(0).standard_normal((30, 3))
        comps0 = numpy.concatenate([
            numpy.random.default_rng(1).standard_normal((3, block.shape[1]))
            for block in blocks
        ], axis=1)
        emb, comps, _ = als(blocks, n_components=3, max_iter=10, tol=0)
        self.assertLess(masked_rmse(emb, comps), masked_rmse(emb0, comps0))

    def test_two_stage_training_covers_all_samples(self):
        # with coverage thresholds forcing a first-stage prefix, the
        # second stage still assigns embeddings to every sample
        blocks, _, _, _ = make_blocks([4, 5], 30, 128, seed=3)
        emb, _, _ = als(
            blocks,
            n_components=8,
            max_iter=3,
            min_samples=20,
            min_features=1,
            min_sample_coverage=1,
            min_feature_coverage=1,
            random_state=0,
        )
        self.assertEqual(emb.shape, (30, 8))
        self.assertTrue(numpy.isfinite(emb).all())

    def test_missing_cells_receive_embeddings(self):
        # cells whose block row is empty are missing: they never
        # contribute to the loss but still receive a finite embedding
        blocks, _, _, _ = make_blocks([4, 5], 30, 128, seed=4)
        blocks[0][5, :] = 0
        emb, _, _ = als(blocks, n_components=8, max_iter=3, random_state=0)
        self.assertTrue(numpy.isfinite(emb[5]).all())

    def test_empty_input_raises(self):
        with self.assertRaises(ValueError):
            als([])

    def test_empty_block_raises(self):
        blocks, _, _, _ = make_blocks([4, 5], 10, 8)
        blocks[1] = scipy.sparse.csr_matrix((10, 0))
        with self.assertRaises(ValueError):
            als(blocks)

    def test_no_observed_cell_raises(self):
        blocks, _, _, _ = make_blocks([4, 5], 10, 8)
        blocks = [
            scipy.sparse.csr_matrix((10, b.shape[1])) for b in blocks
        ]
        with self.assertRaises(ValueError):
            als(blocks)


if __name__ == "__main__":
    unittest.main()
