import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import joblib
import pandas
import sklearn.compose
import sklearn.impute
import sklearn.linear_model
import sklearn.pipeline

from sincomp import cli
from sincomp.cli import align as align_cli
from sincomp.cli import classifier as classifier_cli
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

    @mock.patch("sincomp.cli.classifier.main", return_value=0)
    def test_classifier_is_forwarded_with_subcommand(self, main):
        self.assertEqual(
            cli.main(["classifier", "train", "rules.json", "annotations.csv"]),
            0,
        )
        main.assert_called_once_with(["train", "rules.json", "annotations.csv"])

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


def _build_classifier_data() -> pandas.DataFrame:
    """Build synthetic long-format pronunciations for 16 dialects.

    Characters are split into three groups: G0 (cid 0-4), G1 (cid 5-9) and
    G2 (cid 10-19). Rule r1 compares G0 with G1 and rule r2 compares G2
    with G1. A dialects are (G0 mostly x, G1 mostly x, G2 y) while B
    dialects are (G0 x, G1 mostly y, G2 y), so their rule compliances lie
    near opposite feature axes and are separable without an intercept.
    """
    rows = []
    for label in ("A", "B"):
        for i in range(8):
            did = f"{label}{i}"
            for cid in range(20):
                if cid < 5:
                    initial = "x"
                elif cid == 5:
                    # one G1 character breaks the uniform pattern
                    initial = "y" if label == "A" else "x"
                elif cid < 10:
                    initial = "x" if label == "A" else "y"
                else:
                    initial = "y"
                rows.append(
                    {"cid": str(cid), "did": did, "initial": initial}
                )
    return pandas.DataFrame(rows)


def _write_rules(path: str) -> None:
    """Write a minimal two-rule JSON rule file."""
    rules = [
        {
            "id": "r1",
            "name": "rule 1",
            "feature": "initial",
            "cid1": [str(c) for c in range(5)],
            "cid2": [str(c) for c in range(5, 10)],
        },
        {
            "id": "r2",
            "name": "rule 2",
            "feature": "initial",
            "cid1": [str(c) for c in range(10, 20)],
            "cid2": [str(c) for c in range(5, 10)],
        },
    ]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rules, f)


def _write_annotations(path: str) -> None:
    """Write the annotated dialect samples."""
    annotations = pandas.DataFrame(
        [
            ("stub", f"{label}{i}", label)
            for label in ("A", "B")
            for i in range(8)
        ],
        columns=["dataset", "did", "label"],
    )
    annotations.to_csv(
        path, index=False, encoding="utf-8", lineterminator="\n"
    )


def _make_stub_dataset() -> mock.Mock:
    """Build a dataset mock that filters pronunciations by dialect IDs."""
    data = _build_classifier_data()
    dataset = mock.Mock()

    def select(dids):
        selected = mock.Mock()
        selected.data = data[data["did"].isin(list(dids))]
        return selected

    dataset.select.side_effect = select
    return dataset


class TestClassifierCli(unittest.TestCase):
    def test_subcommand_required(self):
        with self.assertRaises(SystemExit):
            classifier_cli.build_parser().parse_args([])

    def test_train_defaults(self):
        args = classifier_cli.build_parser().parse_args(
            ["train", "rules.json", "annotations.csv"]
        )
        self.assertEqual(args.resample, 5)
        self.assertEqual(args.output_file, "dialect_classifier.bz2")

    @mock.patch(
        "sincomp.cli.classifier.datasets.get",
        return_value=_make_stub_dataset(),
    )
    def test_train_end_to_end(self, get_dataset):
        with tempfile.TemporaryDirectory() as tmp:
            rule_file = os.path.join(tmp, "rules.json")
            annotation_file = os.path.join(tmp, "annotations.csv")
            model_file = os.path.join(tmp, "classifier.bz2")
            _write_rules(rule_file)
            _write_annotations(annotation_file)

            self.assertEqual(
                classifier_cli.main([
                    "train",
                    "--resample", "0",
                    rule_file,
                    annotation_file,
                    model_file,
                ]),
                0,
            )

            self.assertTrue(os.path.isfile(model_file))
            model = joblib.load(model_file)
            self.assertIsInstance(model, sklearn.pipeline.Pipeline)
            self.assertListEqual(list(model.classes_), ["A", "B"])
        get_dataset.assert_called_with("stub")

    @mock.patch(
        "sincomp.cli.classifier.datasets.get",
        return_value=_make_stub_dataset(),
    )
    def test_evaluate_end_to_end(self, get_dataset):
        with tempfile.TemporaryDirectory() as tmp:
            rule_file = os.path.join(tmp, "rules.json")
            annotation_file = os.path.join(tmp, "annotations.csv")
            _write_rules(rule_file)
            _write_annotations(annotation_file)

            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    classifier_cli.main([
                        "evaluate",
                        "--resample", "0",
                        rule_file,
                        annotation_file,
                    ]),
                    0,
                )

            # the compliance perfectly separates the two classes
            self.assertIn("accuracy = 1.0000", output.getvalue())
        get_dataset.assert_called_with("stub")

    def test_predict_precomputed(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_file = os.path.join(tmp, "classifier.bz2")
            input_file = os.path.join(tmp, "compliances.csv")
            output_file = os.path.join(tmp, "predictions.csv")

            features = pandas.DataFrame({
                "r1": [1.0, 0.0, 1.0, 0.0],
                "r2": [1.0, 0.0, 1.0, 0.0],
            })
            model = sklearn.pipeline.make_pipeline(
                sklearn.compose.make_column_transformer(
                    ("passthrough", features.columns)
                ),
                sklearn.impute.KNNImputer(),
                sklearn.linear_model.LogisticRegression(),
            ).fit(features, ["A", "B", "A", "B"])
            joblib.dump(model, model_file)

            pandas.DataFrame(
                [("stub", "A0", 1, 1), ("stub", "B0", 0, 0)],
                columns=["dataset", "did", "r1", "r2"],
            ).to_csv(
                input_file, index=False, encoding="utf-8",
                lineterminator="\n",
            )

            self.assertEqual(
                classifier_cli.main([
                    "predict",
                    "--precomputed",
                    "--model-file", model_file,
                    input_file,
                    output_file,
                ]),
                0,
            )

            result = pandas.read_csv(output_file).set_index("did")
            self.assertGreater(result.loc["A0", "A"], result.loc["A0", "B"])
            self.assertGreater(result.loc["B0", "B"], result.loc["B0", "A"])

    @mock.patch(
        "sincomp.cli.classifier.datasets.get",
        return_value=_make_stub_dataset(),
    )
    def test_predict_from_datasets(self, get_dataset):
        with tempfile.TemporaryDirectory() as tmp:
            rule_file = os.path.join(tmp, "rules.json")
            annotation_file = os.path.join(tmp, "annotations.csv")
            model_file = os.path.join(tmp, "classifier.bz2")
            sample_file = os.path.join(tmp, "samples.csv")
            output_file = os.path.join(tmp, "predictions.csv")
            _write_rules(rule_file)
            _write_annotations(annotation_file)

            # train the model first, then predict from raw dialect samples
            classifier_cli.main([
                "train",
                "--resample", "0",
                rule_file,
                annotation_file,
                model_file,
            ])
            pandas.DataFrame(
                [("stub", "A0"), ("stub", "B0")],
                columns=["dataset", "did"],
            ).to_csv(
                sample_file, index=False, encoding="utf-8",
                lineterminator="\n",
            )

            self.assertEqual(
                classifier_cli.main([
                    "predict",
                    "--model-file", model_file,
                    "--rule-file", rule_file,
                    sample_file,
                    output_file,
                ]),
                0,
            )

            result = pandas.read_csv(output_file).set_index("did")
            self.assertGreater(result.loc["A0", "A"], result.loc["A0", "B"])
            self.assertGreater(result.loc["B0", "B"], result.loc["B0", "A"])
        get_dataset.assert_called_with("stub")


if __name__ == "__main__":
    unittest.main()