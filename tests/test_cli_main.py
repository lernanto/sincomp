# -*- coding: utf-8 -*-

"""Tests for the top-level CLI dispatch (:mod:`sincomp.cli`).

The tests verify the help output of the ``sincomp`` command and that
every subcommand is forwarded to its domain CLI module with the
expected arguments.
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"

import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

from sincomp import cli


class TestCliMain(unittest.TestCase):
    """Test the dispatching of the ``sincomp`` console command."""

    def test_help(self):
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(cli.main(["--help"]), 0)
        self.assertIn("align", output.getvalue())

    @mock.patch("sincomp.cli.align.main", return_value=0)
    def test_align_defaults_to_align_operation(self, main):
        # the bare align command runs the alignment operation
        self.assertEqual(cli.main(["align", "ccr"]), 0)
        main.assert_called_once_with(["align", "ccr"])

    @mock.patch("sincomp.cli.compare.main", return_value=0)
    def test_compare_is_forwarded_without_subcommand(self, main):
        self.assertEqual(cli.main(["compare", "ccr"]), 0)
        main.assert_called_once_with(["ccr"])

    @mock.patch("sincomp.cli.classifier.main", return_value=0)
    def test_classifier_is_forwarded_with_subcommand(self, main):
        self.assertEqual(
            cli.main(["classifier", "train", "rules.json", "annotations.csv"]),
            0,
        )
        main.assert_called_once_with(["train", "rules.json", "annotations.csv"])

    @mock.patch("sincomp.cli.reconstruct.main", return_value=0)
    def test_reconstruct_is_forwarded_with_subcommand(self, main):
        self.assertEqual(
            cli.main(["reconstruct", "evaluate", "--dialects", "a"]),
            0,
        )
        main.assert_called_once_with(["evaluate", "--dialects", "a"])

    @mock.patch("sincomp.cli.similarity.main", return_value=0)
    def test_similarity_is_forwarded_without_subcommand(self, main):
        self.assertEqual(cli.main(["similarity", "ccr"]), 0)
        main.assert_called_once_with(["ccr"])


if __name__ == "__main__":
    unittest.main()
