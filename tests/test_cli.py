import unittest
from contextlib import redirect_stdout
from io import StringIO
import json
from unittest import mock

from sincomp import cli
from sincomp.cli import align as align_cli


class TestCli(unittest.TestCase):
    def test_help(self):
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(cli.main(["--help"]), 0)
        self.assertIn("align", output.getvalue())

    @mock.patch("sincomp.cli.align.main", return_value=0)
    def test_align_defaults_to_align_operation(self, main):
        self.assertEqual(cli.main(["align", "ccr"]), 0)
        main.assert_called_once_with(["align", "ccr"])

    @mock.patch("sincomp.cli.compare.main", return_value=0)
    def test_compare_is_forwarded_without_subcommand(self, main):
        self.assertEqual(cli.main(["compare", "ccr"]), 0)
        main.assert_called_once_with(["ccr"])

    @mock.patch("sincomp.cli.similarity.main", return_value=0)
    def test_similarity_is_forwarded_without_subcommand(self, main):
        self.assertEqual(cli.main(["similarity", "ccr"]), 0)
        main.assert_called_once_with(["ccr"])

    @mock.patch(
        "sincomp.cli.align.alignment.evaluate",
        return_value={
            "cid_accuracy": 0.9,
            "cid_std": 0.1,
            "no_cid_accuracy": 0.8,
            "no_cid_std": 0.2,
        },
    )
    def test_agent_alignment_evaluation_outputs_json(self, evaluate):
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(align_cli.main(["evaluate", "ccr"]), 0)
        self.assertEqual(json.loads(output.getvalue())["cid_accuracy"], 0.9)
        evaluate.assert_called_once()

    @mock.patch("sincomp.cli.align.alignment.evaluate", return_value={
        "cid_accuracy": 0.9,
        "cid_std": 0.1,
        "no_cid_accuracy": 0.8,
        "no_cid_std": 0.2,
    })
    def test_human_alignment_evaluation_outputs_text(self, evaluate):
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(cli.main(["align", "evaluate", "ccr"]), 0)
        self.assertIn("accuracy = 0.9000", output.getvalue())
        self.assertNotIn("{", output.getvalue())
        evaluate.assert_called_once()


if __name__ == "__main__":
    unittest.main()