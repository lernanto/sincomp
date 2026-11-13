# -*- coding: utf-8 -*-

"""Tests for :mod:`sincomp._internal.align`.

The tests verify that ``align`` maps the characters of two datasets to
unique alignment keys with both the SVD and the factorization-machine
encoders, and that ``align_no_cid`` aligns datasets without shared
character IDs by matching the written character forms.
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"


import os
import unittest

import numpy
import pandas

from sincomp._internal import align, dataset


data_dir = os.path.join(os.path.dirname(__file__), "data")


class TestAlign(unittest.TestCase):
    """Test character alignment between two datasets."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.data1 = dataset.FileDataset(
            data_dir=os.path.join(data_dir, "custom_dataset1")
        )
        cls.data2 = dataset.FileDataset(
            data_dir=os.path.join(data_dir, "custom_dataset2")
        )

    def test_align_svd(self):
        # align returns one character table per dataset and one mapping
        # of the original character IDs to the shared alignment keys
        _, (charmap1, charmap2) = align.align(
            (self.data1, None),
            (self.data2, None),
            encoder="svd",
            embedding_size=10
        )
        self.assertEqual(charmap1.nunique(), charmap1.shape[0])
        self.assertEqual(charmap2.nunique(), charmap2.shape[0])

    def test_align_fm(self):
        _, (charmap1, charmap2) = align.align(
            (self.data1, None),
            (self.data2, None),
            encoder="fm"
        )
        self.assertEqual(charmap1.nunique(), charmap1.shape[0])
        self.assertEqual(charmap2.nunique(), charmap2.shape[0])

    def test_align_no_cid_svd(self):
        chars1 = self.data1.loc[:, ["cid", "character"]].drop_duplicates() \
            .dropna(subset="cid").set_index("cid")["character"]

        result = align.align_no_cid(
            self.data1,
            chars1,
            None,
            self.data2,
            encoder="svd"
        )
        self.assertEqual(len(result), 1)

        labels, chars2, _ = result[0][0]
        self.assertEqual(labels.shape[0], chars2.shape[0])
        self.assertTrue(
            numpy.all(
                chars1.loc[labels][pandas.notna(labels)]
                == chars2[pandas.notna(labels)]
            )
        )

    def test_align_no_cid_fm(self):
        chars1 = self.data1.loc[:, ["cid", "character"]].drop_duplicates() \
            .dropna(subset="cid").set_index("cid")["character"]

        result = align.align_no_cid(
            self.data1,
            chars1,
            None,
            self.data2,
            encoder="fm"
        )
        self.assertEqual(len(result), 1)

        labels, chars2, _ = result[0][0]
        self.assertEqual(labels.shape[0], chars2.shape[0])
        self.assertTrue(
            numpy.all(
                chars1.loc[labels][pandas.notna(labels)]
                == chars2[pandas.notna(labels)]
            )
        )


if __name__ == "__main__":
    unittest.main()
