# -*- coding: utf-8 -*-

"""Tests for the similarity CLI (:mod:`sincomp.cli.similarity`).

The tests verify the computation of the pairwise dialect similarities
from a local dataset directory: the output file naming for one or
several methods, the creation of the output parent directories, and
the rejection of an unknown method.
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"

import os
import tempfile
import unittest

import pandas

from sincomp.cli import similarity as similarity_cli


data_dir = os.path.join(os.path.dirname(__file__), "data")


class TestSimilarityCli(unittest.TestCase):
    """Test the similarity computation command."""

    def test_creates_output_parent_directories(self):
        # the output parent directories are created when missing
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "nested", "output.csv")
            self.assertEqual(
                similarity_cli.main([
                    "-m", "chi2", "-o", output,
                    os.path.join(data_dir, "custom_dataset1"),
                ]),
                0,
            )
            self.assertTrue(os.path.isfile(output))

    def test_writes_one_file_per_method(self):
        # without an explicit method both measures are computed and
        # written to one file each inside the output directory
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                similarity_cli.main([
                    "-o", tmp, os.path.join(data_dir, "custom_dataset1"),
                ]),
                0,
            )
            for method in ("chi2", "entropy"):
                path = os.path.join(tmp, f"custom_dataset1_{method}.csv")
                self.assertTrue(os.path.isfile(path), path)
                result = pandas.read_csv(path, index_col=0)
                # the similarity matrix is square over the dialects
                self.assertEqual(result.shape[0], result.shape[1])
                self.assertGreater(len(result), 0)

    def test_unknown_method_exits(self):
        # the method is restricted to the supported choices
        with self.assertRaises(SystemExit):
            similarity_cli.main([
                "-m", "cosine", os.path.join(data_dir, "custom_dataset1"),
            ])


if __name__ == "__main__":
    unittest.main()
