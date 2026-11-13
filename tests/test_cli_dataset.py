# -*- coding: utf-8 -*-

"""Tests for the dataset CLI (:mod:`sincomp.cli.dataset`).

The tests verify that the ``list``, ``dialects`` and ``query``
subcommands report their results as JSON for agent callers and as
plain text tables for human callers, including the dialect filter
expressions.
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"

import json
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import pandas

from sincomp import cli
from sincomp.cli import dataset as dataset_cli


class TestDatasetCli(unittest.TestCase):
    """Test the output formats of the dataset inspection commands."""

    @mock.patch(
        "sincomp.cli.dataset.dataset_api.predefined",
        return_value=[
            type(
                "_Stub",
                (),
                {
                    "name": "CCR",
                    "description": "CCR description",
                    "dialect_ids": ["a", "b"],
                },
            )(),
            type(
                "_Stub",
                (),
                {
                    "name": "MCPDict",
                    "description": "MCPDict description",
                    "dialect_ids": ["x", "y", "z"],
                },
            )(),
        ],
    )
    def test_agent_dataset_list_outputs_json(self, predefined):
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(dataset_cli.main(["list"]), 0)
        self.assertEqual(
            json.loads(output.getvalue()),
            [
                {
                    "name": "CCR",
                    "description": "CCR description",
                    "dialect_count": 2,
                },
                {
                    "name": "MCPDict",
                    "description": "MCPDict description",
                    "dialect_count": 3,
                },
            ],
        )
        predefined.assert_called_once_with()

    @mock.patch(
        "sincomp.cli.dataset.dataset_api.predefined",
        return_value=[
            type(
                "_Stub",
                (),
                {
                    "name": "CCR",
                    "description": "CCR description",
                    "dialect_ids": ["a", "b"],
                },
            )(),
            type(
                "_Stub",
                (),
                {
                    "name": "MCPDict",
                    "description": "MCPDict description",
                    "dialect_ids": ["x", "y", "z"],
                },
            )(),
        ],
    )
    def test_human_dataset_list_outputs_table(self, predefined):
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(cli.main(["dataset", "list"]), 0)

        lines = output.getvalue().splitlines()
        self.assertEqual(
            lines[0].split(), ["name", "description", "dialect_count"]
        )
        # normalize whitespace so column width changes do not break the test
        rows = {" ".join(line.split()) for line in lines[1:]}
        self.assertEqual(
            rows,
            {
                "CCR CCR description 2",
                "MCPDict MCPDict description 3",
            },
        )
        predefined.assert_called_once_with()

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
                dataset_cli.main(["dialects", "CCR", "province == '广东'"]),
                0
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
                cli.main(["dataset", "dialects", "CCR", "did == '072'"]),
                0
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
