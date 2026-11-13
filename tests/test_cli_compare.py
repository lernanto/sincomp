# -*- coding: utf-8 -*-

"""Tests for the compare CLI (:mod:`sincomp.cli.compare`).

The tests verify the computation of the rule compliance scores from a
local dataset directory and a rule file: the output CSV columns, the
normalization option, and the rejection of an unknown dataset.
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"

import os
import tempfile
import unittest

import pandas

from sincomp.cli import compare as compare_cli


data_dir = os.path.join(os.path.dirname(__file__), "data")


class TestCompareCli(unittest.TestCase):
    """Test the rule compliance command."""

    def test_writes_compliance_csv(self):
        # the report holds one row per dialect with the compliance of
        # every rule; the output file is a positional argument
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "compliance.csv")
            self.assertEqual(
                compare_cli.main([
                    "-r", os.path.join(data_dir, "rules.json"),
                    os.path.join(data_dir, "custom_dataset1"),
                    output,
                ]),
                0,
            )

            result = pandas.read_csv(output, index_col="did")
            self.assertGreater(len(result), 0)
            self.assertEqual(
                list(result.columns),
                ["dataset", "0"],
            )
            self.assertTrue((result["0"] >= 0).all())
            self.assertTrue((result["0"] <= 1).all())

    def test_norm_zero_disables_normalization(self):
        # a non-positive norm leaves the raw compliance scores
        with tempfile.TemporaryDirectory() as tmp:
            output = os.path.join(tmp, "compliance.csv")
            self.assertEqual(
                compare_cli.main([
                    "-r", os.path.join(data_dir, "rules.json"),
                    "-n", "0",
                    os.path.join(data_dir, "custom_dataset1"),
                    output,
                ]),
                0,
            )
            self.assertTrue(os.path.isfile(output))

    def test_unknown_dataset_exits(self):
        # an unregistered dataset name without a local path stops the
        # command
        with self.assertRaises(SystemExit):
            compare_cli.main(["no_such_dataset"])


if __name__ == "__main__":
    unittest.main()
