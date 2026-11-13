# -*- coding: utf-8 -*-

"""Tests for :mod:`sincomp._internal.auxiliary`.

The tests verify that ``split_data`` splits an index array into two
deterministic parts for a fixed random state.
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"


import unittest

import numpy

from sincomp._internal import auxiliary


class TestAuxiliary(unittest.TestCase):
    """Test the auxiliary helper functions."""

    def test_split_data(self):
        # the same random state produces the same split of the input
        ori = numpy.random.choice(20, size=100)
        a1, b1 = auxiliary.split_data(ori, random_state=123)
        a2, b2 = auxiliary.split_data(ori, random_state=123)
        self.assertTrue(numpy.all(a1 == a2) and numpy.all(b1 == b2))


if __name__ == "__main__":
    unittest.main()
