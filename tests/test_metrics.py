# -*- coding: utf-8 -*-

"""Tests for the dialect similarity dispatcher ``dialect_similarities``."""

__author__ = "黄艺华 <lernanto@foxmail.com>"


import unittest

import numpy

from sincomp import metrics
from sincomp._internal import similarity

from common import make_wide


class TestDialectSimilarities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.data = make_wide().to_numpy(dtype=object)

    def test_delegates_to_chi2(self):
        numpy.testing.assert_array_equal(
            metrics.dialect_similarities(self.data, metric="chi2"),
            similarity.chi2(self.data),
        )

    def test_delegates_to_entropy(self):
        numpy.testing.assert_array_equal(
            metrics.dialect_similarities(self.data, metric="entropy"),
            similarity.entropy(self.data),
        )

    def test_parallel_matches_sequential(self):
        numpy.testing.assert_array_equal(
            metrics.dialect_similarities(self.data, metric="chi2", n_jobs=2),
            metrics.dialect_similarities(self.data, metric="chi2"),
        )

    def test_invalid_metric(self):
        with self.assertRaises(ValueError):
            metrics.dialect_similarities(self.data, metric="cosine")

    def test_src_dest(self):
        sim = metrics.dialect_similarities(
            self.data[:, 3:], self.data, metric="entropy"
        )
        self.assertEqual(
            sim.shape,
            (self.data.shape[1] // 3 - 1, self.data.shape[1] // 3)
        )


if __name__ == "__main__":
    unittest.main()
