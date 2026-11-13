# -*- coding: utf-8 -*-

"""Tests for :mod:`sincomp._internal.preprocessing`.

The tests verify the cleaning functions of the phone transcriptions,
and the romanization recognition and parsing.
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"


import os
import unittest

import numpy

from sincomp._internal import dataset, preprocessing


data_dir = os.path.join(os.path.dirname(__file__), "data")


class TestPreprocess(unittest.TestCase):
    """Test the preprocessing functions of the pronunciation records."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.data = dataset.FileDataset(
            data_dir=os.path.join(data_dir, "custom_dataset1")
        )

    def test_clean_ipa(self):
        # cleaning keeps the shape of the input column
        clean = preprocessing.clean_ipa(self.data.loc[:, "initial"])
        self.assertEqual(clean.shape, self.data.loc[:, "initial"].shape)

    def test_clean_initial(self):
        clean = preprocessing.clean_initial(self.data.loc[:, "initial"])
        self.assertEqual(clean.shape, self.data.loc[:, "initial"].shape)

    def test_clean_final(self):
        clean = preprocessing.clean_final(self.data.loc[:, "final"])
        self.assertEqual(clean.shape, self.data.loc[:, "final"].shape)

    def test_clean_tone(self):
        clean = preprocessing.clean_tone(self.data.loc[:, "tone"])
        self.assertEqual(clean.shape, self.data.loc[:, "tone"].shape)

    def test_get_romanization_type(self):
        # the romanization of the sample data is recognized as one of
        # the supported schemes
        self.assertIn(
            preprocessing.get_romanization_type(
                "".join(
                    self.data.loc[:, ["initial", "final"]].fillna("")
                    .apply("".join, axis=1)
                )
            ),
            (
                "IPA",
                "pinyin",
                "jyutping",
                "beh-oe-ji",
                "romaji",
                "romaja",
                "quoc ngu",
                "other"
            )
        )

    def test_parse_str(self):
        # parsing a syllable string splits it into initial, final and
        # tone segments
        initial, final, tone = preprocessing.parse("xu533")
        self.assertEqual(initial, "x")
        self.assertEqual(final, "u")
        self.assertEqual(tone, "533")

    def test_parse_array(self):
        segments = preprocessing.parse(["xu533", "pu533"])
        self.assertIsInstance(segments, numpy.ndarray)
        self.assertListEqual(
            segments.tolist(),
            [
                ["x", "u", "533"],
                ["p", "u", "533"]
            ]
        )


if __name__ == "__main__":
    unittest.main()
