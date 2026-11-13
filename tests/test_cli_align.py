# -*- coding: utf-8 -*-

"""Tests for the align CLI (:mod:`sincomp.cli.align`).

The tests verify that the alignment evaluation command reports its
metrics as JSON for agent callers and as plain text for human callers.
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"

import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

from sincomp import cli
from sincomp.cli import align as align_cli


class TestAlignCli(unittest.TestCase):
    """Test the output formats of the align evaluation command."""

    @mock.patch(
        "sincomp.cli.align.evaluate_cmd",
        return_value={
            "cid_accuracy": 0.9,
            "cid_std": 0.1,
            "no_cid_accuracy": 0.8,
            "no_cid_std": 0.2,
        },
    )
    def test_agent_alignment_evaluation_outputs_json(self, evaluate):
        # direct module invocation prints the machine-readable report
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(align_cli.main(["evaluate", "ccr"]), 0)
        self.assertEqual(json.loads(output.getvalue())["cid_accuracy"], 0.9)
        evaluate.assert_called_once()

    @mock.patch(
        "sincomp.cli.align.evaluate_cmd",
        return_value={
            "cid_accuracy": 0.9,
            "cid_std": 0.1,
            "no_cid_accuracy": 0.8,
            "no_cid_std": 0.2,
        },
    )
    def test_human_alignment_evaluation_outputs_text(self, evaluate):
        # the human-friendly command formats the metrics as text
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(cli.main(["align", "evaluate", "ccr"]), 0)
        self.assertIn("accuracy = 0.9000", output.getvalue())
        self.assertNotIn("{", output.getvalue())
        evaluate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
