# -*- coding: utf-8 -*-

"""Tests for :mod:`sincomp._internal.compare`.

The tests verify the output type and shape of ``compliance`` on normal
data and its edge cases: a single dialect, a rule referencing a
character absent from the data, and a rule with an empty character set.
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"


import unittest

import pandas

from sincomp._internal import compare




class TestCompare(unittest.TestCase):
    """Test the rule compliance computation of the wide feature tables."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.rules = pandas.DataFrame([
            {
                "feature": "initial",
                "cid1": ["0024", "0028", "0046"],
                "cid2": ["0054", "0059"],
            }
        ])

    def _make_features(self):
        return pandas.DataFrame(
            [
                ["p", "a", "1", "p", "a", "1"],
                ["p", "a", "1", "t", "a", "1"],
                ["t", "a", "2", "t", "a", "2"],
                ["t", "i", "1", "p", "i", "1"],
                ["k", "u", "2", "k", "u", "2"],
            ],
            columns=pandas.MultiIndex.from_tuples([
                ("d1", "initial"),
                ("d1", "final"),
                ("d1", "tone"),
                ("d2", "initial"),
                ("d2", "final"),
                ("d2", "tone"),
            ]),
            index=["0024", "0028", "0046", "0054", "0059"],
        )

    def test_compliance(self):
        # the compliance of every dialect with every rule is a value
        # inside [0, 1]
        features = self._make_features()
        compliances = compare.compliance(features, self.rules)

        self.assertIsInstance(compliances, pandas.DataFrame)
        self.assertTupleEqual(compliances.shape, (2, 1))
        self.assertGreaterEqual(compliances.min(axis=None), 0)
        self.assertLessEqual(compliances.max(axis=None), 1)

    def test_compliance_single_dialect(self):
        features = self._make_features()
        compliances = compare.compliance(features.iloc[:, :3], self.rules)

        self.assertIsInstance(compliances, pandas.DataFrame)
        self.assertTupleEqual(compliances.shape, (1, 1))

    def test_compliance_character_not_exist(self):
        # a rule character missing from the data does not raise
        features = self._make_features()
        rules = self.rules.copy()
        rules.iloc[0]["cid1"].append("0023")
        compare.compliance(features, self.rules)

    def test_compliance_empty_character(self):
        # a rule with an empty character set yields missing compliances
        features = self._make_features()
        rules = self.rules.copy()
        rules.iloc[0]["cid1"].clear()
        compliances = compare.compliance(features, self.rules)

        self.assertTrue(compliances.isna().all(axis=None))


if __name__ == "__main__":
    unittest.main()
