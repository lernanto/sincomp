# -*- coding: utf-8 -*-

"""Tests for the public preprocessing facade."""

__author__ = "黄艺华 <lernanto@foxmail.com>"

import os
import unittest

import sincomp.datasets
from sincomp.preprocessing import long_to_wide


data_dir = os.path.join(os.path.dirname(__file__), "data")


class TestLongToWide(unittest.TestCase):
    """Test the public ``sincomp.preprocessing.long_to_wide`` function."""

    def test_transform_converts_long_form_to_wide(self):
        # the wide table holds one row per character and one column
        # group per dialect feature
        dataset = sincomp.datasets.get(os.path.join(data_dir, "custom_dataset1"))
        wide = long_to_wide(
            dataset.data,
            index="cid",
            columns="did",
            values=["initial", "final", "tone"],
            aggfunc="first",
        )

        self.assertEqual(wide.shape[0], dataset.data["cid"].nunique())
        self.assertEqual(wide.columns.nlevels, 2)
        self.assertListEqual(
            sorted(set(wide.columns.get_level_values(1))),
            ["final", "initial", "tone"],
        )



if __name__ == "__main__":
    unittest.main()
