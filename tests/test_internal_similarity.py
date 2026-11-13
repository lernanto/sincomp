# -*- coding: utf-8 -*-

"""Tests for :mod:`sincomp._internal.similarity`.

The tests verify the shape and the finiteness of the chi-square and
entropy similarity matrices computed from a fixed wide feature table.
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"


import unittest

import numpy

from sincomp._internal import similarity

from common import make_wide


class TestSimilarity(unittest.TestCase):
    """Test the pairwise dialect similarity measures."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.data = make_wide()

    def test_chi2(self):
        # the similarity matrix is square over the dialects and finite
        sim = similarity.chi2(self.data)
        self.assertEqual(sim.shape, (self.data.columns.levels[0].shape[0],) * 2)
        self.assertTrue(numpy.all(numpy.isfinite(sim)))

    def test_entropy(self):
        sim = similarity.entropy(self.data)
        self.assertEqual(sim.shape, (self.data.columns.levels[0].shape[0],) * 2)
        self.assertTrue(numpy.all(numpy.isfinite(sim)))


if __name__ == "__main__":
    unittest.main()
