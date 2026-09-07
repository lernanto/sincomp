import unittest
from contextlib import redirect_stdout
from io import StringIO
import json
from unittest import mock

import pandas

from sincomp import cli
from sincomp.cli import align as align_cli
from sincomp.cli import dataset as dataset_cli


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

    @mock.patch(
        "sincomp.cli.dataset.dataset_api.list_datasets",
        return_value=["CCR", "MCPDict"],
    )
    def test_agent_dataset_list_outputs_json(self, list_datasets):
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(dataset_cli.main(["list"]), 0)
        self.assertEqual(json.loads(output.getvalue()), ["CCR", "MCPDict"])
        list_datasets.assert_called_once_with()

    @mock.patch(
        "sincomp.cli.dataset.dataset_api.list_datasets",
        return_value=["CCR", "MCPDict"],
    )
    def test_human_dataset_list_outputs_names_only(self, list_datasets):
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(cli.main(["dataset", "list"]), 0)
        self.assertEqual(output.getvalue(), "CCR\nMCPDict\n")
        list_datasets.assert_called_once_with()

    @mock.patch(
        "sincomp.cli.dataset.dataset_api.get",
        return_value=mock.Mock(
            dialects=pandas.DataFrame(
                {"name": ["A"], "latitude": [1.0]}, index=["027"]
            )
        ),
    )
    def test_agent_dialects_outputs_all_dialects_as_json(self, get_dataset):
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(dataset_cli.main(["dialects", "CCR"]), 0)
        self.assertEqual(
            json.loads(output.getvalue()),
            [{"did": "027", "name": "A", "latitude": 1.0}],
        )
        get_dataset.assert_called_once_with("CCR")

    @mock.patch(
        "sincomp.cli.dataset.dataset_api.get",
        return_value=mock.Mock(
            dialects=pandas.DataFrame(
                {"name": ["A"], "latitude": [1.0]}, index=["027"]
            )
        ),
    )
    def test_human_dialects_outputs_table(self, get_dataset):
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(cli.main(["dataset", "dialects", "CCR"]), 0)
        self.assertIn("027", output.getvalue())
        self.assertIn("name", output.getvalue())
        get_dataset.assert_called_once_with("CCR")

    @mock.patch(
        "sincomp.cli.dataset.dataset_api.get",
        return_value=mock.Mock(
            dialects=pandas.DataFrame(
                {"name": ["A", "B"], "province": ["广东", "福建"]},
                index=["027", "072"],
            )
        ),
    )
    def test_agent_dialects_filter_outputs_json(self, get_dataset):
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                dataset_cli.main(["dialects", "CCR", "province == '广东'"]), 0
            )
        self.assertEqual(
            json.loads(output.getvalue()),
            [{"did": "027", "name": "A", "province": "广东"}],
        )
        get_dataset.assert_called_once_with("CCR")

    @mock.patch(
        "sincomp.cli.dataset.dataset_api.get",
        return_value=mock.Mock(
            dialects=pandas.DataFrame(
                {"name": ["A", "B"], "province": ["广东", "福建"]},
                index=["027", "072"],
            )
        ),
    )
    def test_human_dialects_filter_outputs_table(self, get_dataset):
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                cli.main(["dataset", "dialects", "CCR", "did == '072'"]), 0
            )
        self.assertIn("072", output.getvalue())
        self.assertIn("B", output.getvalue())
        self.assertNotIn("027", output.getvalue())
        get_dataset.assert_called_once_with("CCR")

    @mock.patch(
        "sincomp.cli.dataset.dataset_api.get",
        return_value=mock.Mock(
            dialects=pandas.DataFrame(
                {"group": ["官话", "吴语"]}, index=["027", "072"]
            ),
            data=pandas.DataFrame(
                {
                    "did": ["027", "027", "072"],
                    "character": ["學", "學", "学"],
                    "tone_category": ["平", "去", "平"],
                }
            ),
        ),
    )
    def test_agent_query_filters_dialects_and_data(self, get_dataset):
        get_dataset.return_value.get_data.side_effect = lambda did: (
            get_dataset.return_value.data[
                get_dataset.return_value.data["did"] == did
            ]
        )
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                dataset_cli.main(
                    ["query", "CCR", "group == '官话'", "tone_category == '平'"]
                ),
                0,
            )
        self.assertEqual(
            json.loads(output.getvalue()),
            [{"did": "027", "character": "學", "tone_category": "平"}],
        )
        get_dataset.assert_called_once_with("CCR")

    @mock.patch(
        "sincomp.cli.dataset.dataset_api.get",
        return_value=mock.Mock(
            dialects=pandas.DataFrame({"province": ["广东"]}, index=["027"]),
            data=pandas.DataFrame(
                {"did": ["027"], "character": ["學"], "tone": ["平"]}
            ),
        ),
    )
    def test_human_query_outputs_table(self, get_dataset):
        get_dataset.return_value.get_data.side_effect = lambda did: (
            get_dataset.return_value.data[
                get_dataset.return_value.data["did"] == did
            ]
        )
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                cli.main(["dataset", "query", "CCR", "province == '广东'"]),
                0,
            )
        self.assertIn("character", output.getvalue())
        self.assertIn("學", output.getvalue())
        get_dataset.assert_called_once_with("CCR")


if __name__ == "__main__":
    unittest.main()