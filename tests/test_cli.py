import json
import os
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import joblib
import numpy
import pandas
import sklearn.compose
import sklearn.impute
import sklearn.linear_model
import sklearn.pipeline

from sincomp import cli, embedding
from sincomp.cli import align as align_cli
from sincomp.cli import classifier as classifier_cli
from sincomp.cli import dataset as dataset_cli
from sincomp.cli import reconstruct as reconstruct_cli


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


def _build_vector_classifier_data() -> pandas.DataFrame:
    """Build long-format pronunciations separable by pair features.

    A dialects pronounce the characters 0-9 with the same initial and
    the characters 10-19 with another one, while B dialects split the
    characters 0-4 from 5-19, so the pairwise identity features of
    the two classes differ. The character 20 is missing from a quarter
    of the dialects, exercising the empty-phone fill of the missing
    records.
    """
    rows = []
    for label in ("A", "B"):
        for i in range(8):
            did = f"{label}{i}"
            for cid in range(21):
                if cid == 20 and i >= 6:
                    # missing from 4 of the 16 dialects
                    continue
                if label == "A":
                    initial = "x" if cid < 10 else "y"
                else:
                    initial = "x" if cid < 5 else "y"
                rows.append(
                    {"cid": str(cid), "did": did, "initial": initial}
                )
    return pandas.DataFrame(rows)


def _make_vector_stub_dataset() -> mock.Mock:
    """Build a dataset mock over the pairwise separable data."""
    data = _build_vector_classifier_data()
    dataset = mock.Mock()

    def select(dids):
        selected = mock.Mock()
        selected.data = data[data["did"].isin(list(dids))]
        return selected

    dataset.select.side_effect = select
    return dataset


def _build_multiclass_vector_data() -> pandas.DataFrame:
    """Build long-format pronunciations for 24 dialects in 3 classes.

    The A, B and C dialects split the 20 characters at different
    positions — 10, 5 and 15 — so the pairwise identity features of
    the three classes differ.
    """
    rows = []
    for label, split in (("A", 10), ("B", 5), ("C", 15)):
        for i in range(8):
            did = f"{label}{i}"
            for cid in range(20):
                rows.append({
                    "cid": str(cid),
                    "did": did,
                    "initial": "x" if cid < split else "y",
                })
    return pandas.DataFrame(rows)


def _make_multiclass_vector_dataset() -> mock.Mock:
    """Build a dataset mock over the 3-class pairwise separable data."""
    data = _build_multiclass_vector_data()
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
            ["train", "annotations.csv"]
        )
        self.assertIsNone(args.rule_file)
        self.assertEqual(args.min_coverage, 0.5)
        self.assertEqual(args.embedding_size, 128)
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
                    "--rule-file", rule_file,
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
                        "--rule-file", rule_file,
                        annotation_file,
                    ]),
                    0,
                )

            # the compliance perfectly separates the two classes
            self.assertIn("precision", output.getvalue())
            self.assertIn("f1", output.getvalue())
            self.assertIn("cross validation with 16 samples, "
                          "accuracy = 1.0000", output.getvalue())
        get_dataset.assert_called_with("stub")

    @mock.patch(
        "sincomp.cli.classifier.datasets.get",
        return_value=_make_vector_stub_dataset(),
    )
    def test_train_vector_end_to_end(self, get_dataset):
        with tempfile.TemporaryDirectory() as tmp:
            annotation_file = os.path.join(tmp, "annotations.csv")
            model_file = os.path.join(tmp, "classifier.bz2")
            _write_annotations(annotation_file)

            self.assertEqual(
                classifier_cli.main([
                    "train",
                    "--resample", "1",
                    annotation_file,
                    model_file,
                ]),
                0,
            )
            model = joblib.load(model_file)
            vectorizer = model.steps[0][1]
            self.assertIsInstance(vectorizer, embedding.DialectVectorizer)
            # the character missing from a quarter of the dialects is
            # kept, its missing phones filled with empty strings
            self.assertIn("20_initial", vectorizer.feature_names_in_)
        get_dataset.assert_called_with("stub")

    def test_train_embedding_multiple_datasets(self):
        # the dialect vector features are only comparable within one
        # dataset
        annotations = pandas.DataFrame(
            [("stub", "A0", "A"), ("other", "B0", "B")],
            columns=["dataset", "did", "label"],
        )
        with self.assertRaises(ValueError):
            classifier_cli.train_embedding(annotations)

    def test_resample_data_selects_dialects_individually(self):
        data = pandas.DataFrame(
            [
                ("A", "a1"),
                ("A", "a2"),
                ("B", "b1"),
                ("B", "b2"),
            ],
            columns=["did", "cid"],
        )
        dataset = mock.Mock()

        def select(dids):
            selected = mock.Mock()
            selected.data = data[data["did"].isin(dids)]
            return selected

        dataset.select.side_effect = select
        annotations = pandas.DataFrame(
            {"did": ["A", "B"]},
            index=[10, 20],
        )

        rounds = list(classifier_cli.resample_data(
            dataset,
            annotations,
            resample=1,
            samples=1,
            min_rate=1,
            max_rate=1,
        ))

        self.assertEqual(len(rounds), 1)
        round_data, indices = rounds[0]
        self.assertEqual(set(round_data["did"]), {"A_0", "B_0"})
        self.assertEqual(list(indices), [10, 20])
        self.assertEqual(
            dataset.select.call_args_list,
            [mock.call(["A"]), mock.call(["B"])],
        )

    @mock.patch(
        "sincomp.cli.classifier.datasets.get",
        return_value=_make_vector_stub_dataset(),
    )
    def test_evaluate_vector_end_to_end(self, get_dataset):
        with tempfile.TemporaryDirectory() as tmp:
            annotation_file = os.path.join(tmp, "annotations.csv")
            _write_annotations(annotation_file)

            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    classifier_cli.main([
                        "evaluate",
                        "--resample", "0",
                        annotation_file,
                    ]),
                    0,
                )

            # the pairwise identity features separate the two classes
            self.assertIn("accuracy = 1.0000", output.getvalue())
        get_dataset.assert_called_with("stub")

    @mock.patch(
        "sincomp.cli.classifier.datasets.get",
        return_value=_make_multiclass_vector_dataset(),
    )
    def test_evaluate_multiclass(self, get_dataset):
        with tempfile.TemporaryDirectory() as tmp:
            annotation_file = os.path.join(tmp, "annotations.csv")
            dids = [
                f"{label}{i}"
                for label in ("A", "B", "C")
                for i in range(8)
            ]
            labels = [f"L{i // 8}" for i in range(24)]
            pandas.DataFrame(
                [("stub", did, label) for did, label in zip(dids, labels)],
                columns=["dataset", "did", "label"],
            ).to_csv(
                annotation_file, index=False, encoding="utf-8",
                lineterminator="\n",
            )

            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    classifier_cli.main([
                        "evaluate",
                        "--resample", "0",
                        annotation_file,
                    ]),
                    0,
                )

            # the multiclass report holds the per-fold and average
            # accuracy only
            self.assertNotIn("precision", output.getvalue())
            self.assertIn("mean", output.getvalue())
            self.assertIn("std", output.getvalue())
            self.assertIn("cross validation with 24 samples", output.getvalue())
        get_dataset.assert_called_with("stub")

    @mock.patch(
        "sincomp.cli.classifier.datasets.get",
        return_value=_make_vector_stub_dataset(),
    )
    def test_predict_vector_from_datasets(self, get_dataset):
        with tempfile.TemporaryDirectory() as tmp:
            annotation_file = os.path.join(tmp, "annotations.csv")
            model_file = os.path.join(tmp, "classifier.bz2")
            sample_file = os.path.join(tmp, "samples.csv")
            output_file = os.path.join(tmp, "predictions.csv")
            _write_annotations(annotation_file)

            classifier_cli.main([
                "train",
                "--resample", "0",
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
                    sample_file,
                    output_file,
                ]),
                0,
            )

            result = pandas.read_csv(output_file).set_index("did")
            self.assertGreater(result.loc["A0", "A"], result.loc["A0", "B"])
            self.assertGreater(result.loc["B0", "B"], result.loc["B0", "A"])
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
                "--rule-file", rule_file,
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


def _build_reconstruct_data() -> pandas.DataFrame:
    """Build long-form pronunciation data for two dialects."""
    return pandas.DataFrame(
        [
            (
                did,
                f"c{i}",
                ("p", "t", "k", "m")[i % 4],
                ("a", "i", "u", "an")[i % 4],
                str(i % 3 + 1),
            )
            for i in range(20)
            for did in ("d1", "d2")
        ],
        columns=["did", "cid", "initial", "final", "tone"],
    )


def _make_reconstruct_dataset() -> mock.Mock:
    """Build a dataset mock with two dialects for reconstruction."""
    data = _build_reconstruct_data()
    dataset = mock.Mock()
    dataset.dialects = pandas.DataFrame(
        {"name": ["Alpha", "Beta"]},
        index=pandas.Index(["d1", "d2"], name="did")
    )
    cids = sorted(data["cid"].unique())
    dataset.characters = pandas.DataFrame(
        {"character": cids},
        index=pandas.Index(cids, name="cid")
    )

    def select(dids):
        selected = mock.Mock()
        selected.data = data[data["did"].isin(list(dids))]
        return selected

    dataset.select.side_effect = select
    return dataset


def _write_segment_annotations(path: str) -> None:
    """Write per-character segment labels as annotations."""
    annotations = pandas.DataFrame(
        [
            (
                f"c{i}",
                ("p", "t", "k", "m")[i % 4],
                ("a", "i", "u", "an")[i % 4],
                str(i % 3 + 1),
            )
            for i in range(20)
        ],
        columns=["cid", "initial", "final", "tone"],
    )
    annotations.to_csv(
        path, index=False, encoding="utf-8", lineterminator="\n"
    )


class TestReconstructCli(unittest.TestCase):
    def test_parser_defaults(self):
        args = reconstruct_cli.build_parser().parse_args([])
        self.assertEqual(args.dataset, "MCPDict")
        self.assertEqual(args.embedding_size, 128)
        self.assertEqual(args.output_dir, "reconstruction")
        self.assertEqual(args.log_level, "WARNING")
        self.assertEqual(args.min_dialect_coverage, 0.5)
        self.assertEqual(args.min_cluster_size, 3)
        self.assertEqual(args.max_cluster_distance, 0.1)
        self.assertEqual(args.max_clusters, 200)
        self.assertEqual(args.min_phone_proportion, 0.1)
        self.assertEqual(args.max_merge_tolerance, 0)
        self.assertEqual(args.max_merge_distance, 0.5)

    def test_dialects_required(self):
        with self.assertRaises(SystemExit):
            reconstruct_cli.main([])

    def test_cluster_rejects_invalid_min_cluster_size(self):
        data = pandas.DataFrame(
            [("d1", "c0", "p", "a", "1")],
            columns=["did", "cid", "initial", "final", "tone"],
        )
        with self.assertRaises(ValueError):
            reconstruct_cli.cluster(
                data,
                data["cid"].unique(),
                min_cluster_size=0,
            )

    def test_assign_unclustered_characters(self):
        result = pandas.DataFrame(
            {
                "initial_cluster": [0, 0, 0, 1, 1, -1, -1],
                "final_cluster": [0, 0, 0, 0, 0, 0, 0],
                "tone_cluster": [0, 0, 0, 0, 0, 0, 0],
            },
            index=pandas.Index(
                ["c0", "c1", "c2", "c3", "c4", "c5", "c6"], name="cid"
            ),
        )
        embeddings = {
            # initial clusters along the x and y directions
            "initial": numpy.array([
                [1.0, 0.0], [1.0, 0.1], [0.9, 0.1],
                [0.0, 1.0], [0.1, 1.0],
                [1.0, 0.3], [0.2, 1.0],
            ]),
            "final": numpy.ones((7, 1)),
            "tone": numpy.ones((7, 1)),
        }

        assigned = reconstruct_cli.assign_unclustered_characters(
            result, embeddings
        )
        # each unclustered character joins the cluster of its nearest
        # clustered character, independently of the other characters
        self.assertEqual(assigned.loc["c5", "initial_cluster"], 0)
        self.assertEqual(assigned.loc["c6", "initial_cluster"], 1)
        # clustered characters keep their cluster
        numpy.testing.assert_array_equal(
            assigned["final_cluster"], result["final_cluster"]
        )
        # the input frame is not modified
        self.assertEqual(result.loc["c5", "initial_cluster"], -1)

        # fully clustered characters are returned unchanged
        pandas.testing.assert_frame_equal(
            reconstruct_cli.assign_unclustered_characters(
                assigned, embeddings
            ),
            assigned,
        )

        with self.assertRaises(ValueError):
            reconstruct_cli.assign_unclustered_characters(
                result.assign(initial_cluster=-1), embeddings
            )

    def _cluster_annotated(self, annotations):
        # c0-c3 share the frequent segments p/a; c4 has the rare
        # segments q/b and is annotated for the initial only or for
        # both segments; c5 shares the rare initial q without any
        # annotation
        data = pandas.DataFrame(
            [
                *[
                    (did, f"c{i}", "p", "a", "1")
                    for i in range(4)
                    for did in ("d1", "d2")
                ],
                ("d1", "c4", "q", "b", "1"),
                ("d2", "c4", "q", "b", "1"),
                ("d1", "c5", "q", "a", "1"),
                ("d2", "c5", "q", "a", "1"),
            ],
            columns=["did", "cid", "initial", "final", "tone"],
        )
        result, _, _ = reconstruct_cli.cluster(
            data,
            data["cid"].unique(),
            annotations,
            min_cluster_size=3,
        )
        return result

    def test_cluster_partial_annotations(self):
        # c4 is annotated for the initial only: the small initial
        # cluster q keeps the annotated character, while the small
        # final cluster b, whose annotation is empty, is dissolved
        annotations = pandas.DataFrame(
            {"initial": ["q"], "final": [None]},
            index=pandas.Index(["c4"], name="cid"),
        )
        result = self._cluster_annotated(annotations)
        # the annotated character keeps its small initial cluster
        self.assertNotEqual(
            result.loc["c4", "initial_cluster"],
            result.loc["c0", "initial_cluster"],
        )
        # the unannotated character of the small initial cluster is
        # left unclustered until the phoneme merging
        self.assertEqual(result.loc["c5", "initial_cluster"], -1)
        # the small final cluster without annotation is dissolved
        self.assertEqual(result.loc["c4", "final_cluster"], -1)
        # the initial cluster without any annotated character is
        # dissolved as well
        self.assertEqual(result.loc["c0", "initial_cluster"], -1)

    def test_cluster_complete_annotations(self):
        # with both annotations, the small initial and final clusters
        # of the annotated character are both kept
        annotations = pandas.DataFrame(
            {"initial": ["q"], "final": ["b"]},
            index=pandas.Index(["c4"], name="cid"),
        )
        result = self._cluster_annotated(annotations)
        self.assertNotEqual(
            result.loc["c4", "initial_cluster"],
            result.loc["c0", "initial_cluster"],
        )
        self.assertNotEqual(
            result.loc["c4", "final_cluster"],
            result.loc["c0", "final_cluster"],
        )
        # the clusters without any annotated character are dissolved
        self.assertEqual(result.loc["c0", "initial_cluster"], -1)
        self.assertEqual(result.loc["c0", "final_cluster"], -1)

    def test_cluster_includes_annotated_characters(self):
        # an annotated character below the coverage threshold still
        # joins the embedding fitting and the clustering
        data = pandas.DataFrame(
            [
                *[
                    (did, f"c{i}", "p", "a", "1")
                    for i in range(3)
                    for did in ("d1", "d2")
                ],
                # c3 is recorded by d1 only and carries an annotation
                ("d1", "c3", "q", "b", "1"),
            ],
            columns=["did", "cid", "initial", "final", "tone"],
        )
        covered = reconstruct_cli.covered_characters(data, 0.75)
        self.assertNotIn("c3", covered)
        annotations = pandas.DataFrame(
            {"initial": [None, None, None, "q"]},
            index=pandas.Index(["c0", "c1", "c2", "c3"], name="cid"),
        )
        result, _, _ = reconstruct_cli.cluster(
            data,
            covered,
            annotations,
            min_cluster_size=3,
        )
        # the annotated character is clustered despite its coverage
        self.assertGreaterEqual(result.loc["c3", "initial_cluster"], 0)

    def test_map_annotation_phones(self):
        result = pandas.DataFrame(
            {
                "initial_cluster": [0, 0, 1, 2],
                "final_cluster": [0, 0, 0, 0],
                "tone_cluster": [0, 0, 0, 0],
            },
            index=pandas.Index(["c0", "c1", "c2", "c3"], name="cid"),
        )
        annotations = pandas.DataFrame(
            {
                "initial": ["p", "p", "p", "k"],
                "final": ["a", "a", "a", "a"],
                "tone": ["1", "2", "1", "1"],
            },
            index=result.index,
        )
        phones = reconstruct_cli.map_annotation_phones(result, annotations)
        # clusters sharing a reading are distinguished with subscripts,
        # larger clusters first
        self.assertEqual(phones[("initial", 0)], "p₁")
        self.assertEqual(phones[("initial", 1)], "p₂")
        self.assertEqual(phones[("initial", 2)], "k")
        # a reading claimed by one cluster only carries no subscript
        self.assertEqual(phones[("final", 0)], "a")
        # a cluster with disagreeing annotations takes the mode
        self.assertEqual(phones[("tone", 0)], "1")

    def test_predict_annotated_phonemes(self):
        # the phoneme of one final cluster depends on the initial: the
        # characters following the p initial are annotated a while the
        # characters following the k initial are annotated wa
        result = pandas.DataFrame(
            {
                "initial_cluster": [0, 0, 1, 1, 0, 1],
                "final_cluster": [1, 1, 1, 1, 1, 1],
                "tone_cluster": [0, 0, 0, 0, 0, 0],
            },
            index=pandas.Index(
                ["c0", "c1", "c2", "c3", "c4", "c5"], name="cid"
            ),
        )
        annotations = pandas.DataFrame(
            {
                "initial": ["p", "p", "k", "k", None, None],
                "final": ["a", "a", "wa", "wa", None, None],
                "tone": ["1", "1", "1", "1", None, None],
            },
            index=result.index,
        )
        predicted = reconstruct_cli.predict_annotated_phonemes(
            result, annotations
        )
        # the phoneme of the same final cluster diverges with the
        # initial cluster
        self.assertEqual(predicted.loc["c4", "final_phoneme"], "a")
        self.assertEqual(predicted.loc["c5", "final_phoneme"], "wa")

    def test_extract_tree_rules(self):
        # every rule must be satisfied exactly by the training readings
        # of its own label; inverted branch directions would match the
        # readings of the other labels instead. Since the values of one
        # feature are mutually exclusive, a feature must not mix
        # positive and negative conditions in one rule
        x = pandas.DataFrame({
            "reconstructed_initial": ["p", "p", "k", "k"],
            "reconstructed_final": ["a", "a", "wa", "wa"],
        })
        y = pandas.Series(["pf", "pf", "k", "k"])
        classifier = reconstruct_cli.make_pipeline(
            reconstruct_cli.OneHotEncoder(handle_unknown="ignore"),
            reconstruct_cli.DecisionTreeClassifier(),
        ).fit(x, y)
        rules = reconstruct_cli.extract_tree_rules(classifier)

        for rule in rules:
            conditions = rule["conditions"]
            matched = x.apply(
                lambda row: all(
                    (row[column] == part)
                    if not part.startswith("!")
                    else (row[column] != part[1:])
                    for column, spec in conditions.items()
                    for part in spec.split("&")
                ),
                axis=1,
            )
            self.assertGreater(int(matched.sum()), 0)
            self.assertEqual(set(y[matched]), {rule["phone"]})

            for spec in conditions.values():
                parts = spec.split("&")
                positives = [p for p in parts if not p.startswith("!")]
                if positives:
                    self.assertEqual(positives, parts)

    def test_extract_tree_rules_rewrites_negatives(self):
        # negative conditions are rewritten as their complement among
        # the observed values whenever the complement has fewer
        # conditions
        x = pandas.DataFrame({
            "reconstructed_initial": ["p", "p", "k", "k", "t", "t"],
            "reconstructed_final": ["a", "a", "a", "a", "a", "a"],
        })
        y = pandas.Series(["A", "A", "B", "B", "C", "C"])
        classifier = reconstruct_cli.make_pipeline(
            reconstruct_cli.OneHotEncoder(handle_unknown="ignore"),
            reconstruct_cli.DecisionTreeClassifier(),
        ).fit(x, y)
        rules = reconstruct_cli.extract_tree_rules(classifier)

        observed = {"initial": {"p", "k", "t"}}
        for rule in rules:
            for feature, spec in rule["conditions"].items():
                parts = spec.split("&")
                if all(p.startswith("!") for p in parts):
                    remaining = observed[feature] - {p[1:] for p in parts}
                    self.assertGreaterEqual(len(remaining), len(parts))

        # the rule predicting the class of the last value must name it
        # positively: its path can only exclude the other two values
        c_rules = [
            rule["conditions"] for rule in rules if rule["phone"] == "C"
        ]
        self.assertTrue(c_rules)
        for conditions in c_rules:
            self.assertEqual(conditions["reconstructed_initial"], "t")

    def test_extract_tree_rules_drops_unsatisfiable(self):
        # a path excluding every observed value of a feature cannot be
        # satisfied by any sample and its rule is dropped; the stub
        # tree has the leaf '!a&!b' for the observed values a and b
        onehot = types.SimpleNamespace(
            categories_=[numpy.array(["a", "b"]), numpy.array(["a"])]
        )
        tree = types.SimpleNamespace(
            children_left=numpy.array([1, 3, -1, -1, -1]),
            children_right=numpy.array([4, 2, -1, -1, -1]),
            feature=numpy.array([0, 1, -2, -2, -2]),
            value=numpy.array([
                [[0.5, 0.5]], [[0.5, 0.5]], [[1.0, 0.0]],
                [[0.0, 1.0]], [[1.0, 0.0]],
            ]),
            n_node_samples=numpy.array([6, 3, 3, 0, 3]),
        )
        classifier = types.SimpleNamespace(
            steps=[
                ("onehot", onehot),
                ("tree", types.SimpleNamespace(
                    tree_=tree, classes_=numpy.array(["A", "B"])
                )),
            ]
        )
        rules = reconstruct_cli.extract_tree_rules(classifier)

        # the unsatisfiable leaf is dropped, the other two leaves keep
        # their rewritten single positive conditions
        self.assertEqual(len(rules), 2)
        self.assertEqual(
            [rule["conditions"]["reconstructed_initial"] for rule in rules],
            ["b", "a"],
        )
        self.assertTrue(all(rule["phone"] == "A" for rule in rules))

    def test_summarize_cluster_phones(self):
        # c2 is not recorded by d1 and has two alternate phones at d2
        data = pandas.DataFrame(
            [
                ("d1", "c0", "p"),
                ("d1", "c1", "p"),
                ("d2", "c0", "t"),
                ("d2", "c2", "p"),
                ("d2", "c2", "t"),
            ],
            columns=["did", "cid", "initial"],
        )
        data["final"] = pandas.NA
        data["tone"] = pandas.NA
        result = pandas.DataFrame(
            {
                "initial_cluster": [0, 0, 0],
                "final_cluster": [0, 0, 0],
                "tone_cluster": [0, 0, 0],
            },
            index=pandas.Index(["c0", "c1", "c2"], name="cid"),
        )

        # p occurs in 3 of 5 recorded samples, t in 2 of 5
        distributions = reconstruct_cli.cluster_phone_distributions(
            data, result
        )
        summary = reconstruct_cli.summarize_cluster_phones(
            distributions, min_phone_proportion=0.5
        )
        rows = summary.xs(0, level="cluster")
        self.assertEqual(set(rows["sample_count"]), {3})
        # t covers two samples but misses the proportion threshold
        self.assertEqual(set(rows["phone"]), {"p"})
        p_row = rows[rows["phone"] == "p"].iloc[0]
        self.assertEqual(p_row.character_count, 3)
        self.assertNotIn(("initial", 1), summary.index)
        # the display does not include the assigned phone
        self.assertNotIn("assigned_phone", summary.columns)

        summary = reconstruct_cli.summarize_cluster_phones(
            distributions, min_phone_proportion=0.4
        )
        phones = set(
            summary.xs(0, level="cluster")["phone"]
        )
        # alternate readings are counted individually, not as one cell
        self.assertEqual(phones, {"p", "t"})

    def test_assign_cluster_phones_avoids_conflicts(self):
        data = pandas.DataFrame(
            [
                ("d1", "c0", "p"), ("d2", "c0", "p"),
                ("d1", "c1", "p"), ("d2", "c1", "t"),
                ("d1", "c2", "t"), ("d2", "c2", "p"),
                ("d1", "c3", "p"), ("d2", "c3", "p"),
            ],
            columns=["did", "cid", "initial"],
        )
        data["final"] = pandas.NA
        data["tone"] = pandas.NA
        result = pandas.DataFrame(
            {
                "initial_cluster": [0, 0, 0, 1],
                "final_cluster": [0, 0, 0, 0],
                "tone_cluster": [0, 0, 0, 0],
            },
            index=pandas.Index(["c0", "c1", "c2", "c3"], name="cid"),
        )

        distributions = reconstruct_cli.cluster_phone_distributions(
            data, result
        )
        assigned = reconstruct_cli.assign_cluster_phones(distributions)
        # the larger cluster claims its most frequent reading first
        self.assertEqual(assigned.loc[("initial", 0)], "p₁")
        # the smaller cluster falls back to a distinguished form
        self.assertEqual(assigned.loc[("initial", 1)], "p₂")

    def test_assign_cluster_phones_requires_two_samples(self):
        data = pandas.DataFrame(
            [
                ("d1", "c0", "p"),
                ("d1", "c1", "p"),
                ("d1", "c2", "p"),
                # a single sample of q covers 25% but is not frequent
                ("d2", "c0", "q"),
            ],
            columns=["did", "cid", "initial"],
        )
        data["final"] = pandas.NA
        data["tone"] = pandas.NA
        result = pandas.DataFrame(
            {
                "initial_cluster": [0, 0, 0],
                "final_cluster": [0, 0, 0],
                "tone_cluster": [0, 0, 0],
            },
            index=pandas.Index(["c0", "c1", "c2"], name="cid"),
        )

        distributions = reconstruct_cli.cluster_phone_distributions(
            data, result
        )
        assigned = reconstruct_cli.assign_cluster_phones(distributions)
        # readings with fewer than two samples are ignored
        self.assertEqual(assigned.loc[("initial", 0)], "p")

    def test_claim_group_phones_keeps_most_frequent_below_threshold(self):
        groups = {("initial", 0): {"p": 1}}

        assigned = reconstruct_cli.claim_group_phones(
            groups, [("initial", 0)], min_phone_proportion=0.9
        )

        self.assertEqual(assigned, {("initial", 0): "p"})

    def test_assign_cluster_phones_keeps_most_frequent_below_threshold(self):
        distributions = pandas.DataFrame(
            {"sample_count": [1], "total": [1]},
            index=pandas.MultiIndex.from_tuples(
                [("initial", 0, "p")],
                names=["feature", "cluster", "phone"],
            ),
        )

        assigned = reconstruct_cli.assign_cluster_phones(
            distributions, min_phone_proportion=0.9
        )

        self.assertEqual(assigned.loc[("initial", 0)], "p")

    def test_merge_cluster_phonemes(self):
        data = pandas.DataFrame(
            [
                # final i only follows the ts initial
                ("d1", "c0", "ts", "i", "55"),
                ("d2", "c0", "ts", "i", "55"),
                # final ei only follows the p initial
                ("d1", "c1", "p", "ei", "55"),
                ("d2", "c1", "p", "ei", "55"),
                # in d2, ei also contrasts with i after ts
                ("d1", "c3", "ts", "ei", "55"),
            ],
            columns=["did", "cid", "initial", "final", "tone"],
        )
        result = pandas.DataFrame(
            {
                "initial_cluster": [0, 1, 0],
                "final_cluster": [1, 2, 3],
                "tone_cluster": [0, 0, 0],
            },
            index=pandas.Index(["c0", "c1", "c3"], name="cid"),
        )

        distributions = reconstruct_cli.cluster_phone_distributions(
            data, result
        )
        # mean reading vectors: cluster 1 near [1, 0], cluster 2 near
        # [0.9, 0.1] (similar), cluster 3 near [0.95, 0.05] (similar to
        # cluster 1 but contrasts with it in d1); the initial clusters
        # hold near-identical vectors so that their average pairwise
        # distance is small
        embeddings = {
            "initial": numpy.array(
                [[1.0, 0.0], [0.95, 0.312], [0.95, 0.312]]
            ),
            "final": numpy.array(
                [[1.0, 0.0], [0.9, 0.1], [0.95, 0.05]]
            ),
            "tone": numpy.array([[1.0], [1.0], [1.0]]),
        }
        phonemes = reconstruct_cli.merge_cluster_phonemes(
            result, embeddings, distributions
        )
        # final clusters 2 and 3 share the phone ei with complementary
        # environments, so they share one phoneme
        self.assertEqual(phonemes[("final", 1)], "i")
        self.assertEqual(phonemes[("final", 2)], "ei")
        self.assertEqual(phonemes[("final", 3)], "ei")
        # cluster 1 contrasts with the ei group on one character in d1
        # and keeps its own phoneme

        # tolerating one co-occurring character merges cluster 3; the
        # merged group claims its most frequent phone ei
        phonemes = reconstruct_cli.merge_cluster_phonemes(
            result, embeddings, distributions,
            max_merge_tolerance=1,
        )
        self.assertEqual(phonemes[("final", 1)], "ei")
        self.assertEqual(phonemes[("final", 2)], "ei")
        self.assertEqual(phonemes[("final", 3)], "ei")

        # a lower similarity threshold merges the initial clusters,
        # whose environments overlap by one character
        phonemes = reconstruct_cli.merge_cluster_phonemes(
            result, embeddings, distributions,
            max_merge_tolerance=1, max_merge_distance=0.3,
        )
        self.assertEqual(phonemes[("initial", 0)], "ts")
        self.assertEqual(phonemes[("initial", 1)], "ts")

    def test_merge_cluster_phonemes_label_threshold(self):
        # a rare phone below the assignment threshold must not become
        # the phoneme label of a cluster whose frequent phones are all
        # claimed by other clusters
        data = pandas.DataFrame(
            [
                ("d1", "c0", "p", "a", "55"),
                ("d2", "c0", "p", "a", "55"),
                ("d1", "c1", "pʰ", "a", "55"),
                ("d2", "c1", "pʰ", "a", "55"),
                ("d1", "c2", "t", "a", "55"),
                ("d2", "c2", "t", "a", "55"),
                ("d1", "c3", "tʰ", "a", "55"),
                ("d1", "c3", "tʰ", "n", "55"),
                ("d2", "c3", "tʰ", "a", "55"),
            ],
            columns=["did", "cid", "initial", "final", "tone"],
        )
        result = pandas.DataFrame(
            {
                "initial_cluster": [0, 1, 2, 3],
                "final_cluster": [1, 1, 2, 2],
                "tone_cluster": [0, 0, 0, 0],
            },
            index=pandas.Index(["c0", "c1", "c2", "c3"], name="cid"),
        )
        distributions = reconstruct_cli.cluster_phone_distributions(
            data, result
        )
        # orthogonal embeddings keep the final clusters unmerged
        embeddings = {
            "initial": numpy.eye(4),
            "final": numpy.array(
                [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]
            ),
            "tone": numpy.ones((4, 1)),
        }
        phonemes = reconstruct_cli.merge_cluster_phonemes(
            result, embeddings, distributions
        )
        # both clusters are dominated by a; the rare n (1 of 4 samples)
        # is not eligible as a label, so the second cluster shares a
        self.assertEqual(phonemes[("final", 1)], "a₁")
        self.assertEqual(phonemes[("final", 2)], "a₂")
        self.assertNotIn(
            "n", phonemes[phonemes.index.get_level_values(0) == "final"]
        )

    def test_merge_cluster_phonemes_includes_feature_only_clusters(self):
        data = pandas.DataFrame(
            [
                ("d1", "c0", "p", "a", "55"),
                ("d1", "c1", "t", "a", "55"),
            ],
            columns=["did", "cid", "initial", "final", "tone"],
        )
        result = pandas.DataFrame(
            {
                "initial_cluster": [0, 1],
                "final_cluster": [0, -1],
                "tone_cluster": [0, 0],
            },
            index=pandas.Index(["c0", "c1"], name="cid"),
        )
        distributions = reconstruct_cli.cluster_phone_distributions(
            data, result
        )
        embeddings = {
            "initial": numpy.array([[1.0, 0.0], [0.0, 1.0]]),
            "final": numpy.array([[1.0, 0.0], [1.0, 0.0]]),
            "tone": numpy.array([[1.0], [1.0]]),
        }

        phonemes = reconstruct_cli.merge_cluster_phonemes(
            result, embeddings, distributions
        )

        self.assertEqual(phonemes[("initial", 0)], "p")
        self.assertEqual(phonemes[("initial", 1)], "t")

    def test_merge_cluster_phonemes_greedy_across_features(self):
        # a tone pair (similarity 0.95) must be merged before an
        # initial pair (similarity 0.90); merging the tones first
        # blocks the initial merge, while merging initial first would
        # block the tone merge under a serial per-feature order
        data = pandas.DataFrame(
            [
                ("d1", "c0", "p", "a", "11"),
                ("d1", "c1", "p", "a", "11"),
                ("d1", "c2", "t", "a", "22"),
                ("d1", "c3", "t", "a", "22"),
            ],
            columns=["did", "cid", "initial", "final", "tone"],
        )
        result = pandas.DataFrame(
            {
                "initial_cluster": [0, 0, 1, 1],
                "final_cluster": [0, 0, 0, 0],
                "tone_cluster": [0, 0, 1, 1],
            },
            index=pandas.Index(["c0", "c1", "c2", "c3"], name="cid"),
        )
        distributions = reconstruct_cli.cluster_phone_distributions(
            data, result
        )
        embeddings = {
            "initial": numpy.array(
                [[1.0, 0.0], [1.0, 0.0], [0.9, 0.436], [0.9, 0.436]]
            ),
            "final": numpy.ones((4, 1)),
            "tone": numpy.array(
                [[1.0, 0.0], [1.0, 0.0], [0.95, 0.312], [0.95, 0.312]]
            ),
        }
        phonemes = reconstruct_cli.merge_cluster_phonemes(
            result, embeddings, distributions
        )
        # the tone clusters merge; the initial clusters stay separate
        # because their environments now overlap beyond the tolerance
        self.assertEqual(phonemes[("tone", 0)], "11")
        self.assertEqual(phonemes[("tone", 1)], "11")
        self.assertEqual(phonemes[("initial", 0)], "p")
        self.assertEqual(phonemes[("initial", 1)], "t")

    @mock.patch(
        "sincomp.cli.reconstruct.datasets.get",
        return_value=_make_reconstruct_dataset(),
    )
    def test_reconstruct_without_annotations(self, get_dataset):
        with tempfile.TemporaryDirectory() as tmp:
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    reconstruct_cli.main([
                        "--dataset", "stub",
                        "--dialects", "d1", "d2",
                        "--output-dir", tmp,
                        "--embedding-size", "4",
                    ]),
                    0,
                )

            result = pandas.read_csv(
                os.path.join(tmp, "reconstruction.csv"), index_col="cid"
            )
            self.assertListEqual(
                sorted(result.columns),
                [
                    "character",
                    "final",
                    "final_cluster",
                    "final_phoneme",
                    "initial",
                    "initial_cluster",
                    "initial_phoneme",
                    "tone",
                    "tone_cluster",
                    "tone_phoneme",
                ],
            )
            self.assertEqual(len(result), 20)
            # every character receives the pronunciation and phoneme of
            # its cluster
            self.assertFalse(
                result[[
                    "initial", "final", "tone",
                    "initial_phoneme", "final_phoneme", "tone_phoneme",
                ]].isna().any().any()
            )

            # cluster phone distributions and assigned phones are
            # reported, one phoneme per cluster
            cluster_phones = pandas.read_csv(
                os.path.join(tmp, "cluster_phones.csv")
            )
            self.assertListEqual(
                list(cluster_phones.columns),
                [
                    "feature",
                    "cluster",
                    "phone",
                    "character_count",
                    "sample_count",
                ],
            )
            clusters = pandas.read_csv(
                os.path.join(tmp, "clusters.csv")
            )
            self.assertListEqual(
                list(clusters.columns),
                ["feature", "cluster", "phone", "phoneme"],
            )

            # the full clustering tree of every feature is exported as
            # one file, with one node per row; leaves hold their
            # character ID and cluster ID, merges hold the two child
            # node IDs and only keep a cluster ID when both children
            # belong to one cluster
            for feature in ("initial", "final", "tone"):
                sub = pandas.read_csv(
                    os.path.join(tmp, f"{feature}_tree.csv")
                )
                self.assertListEqual(
                    list(sub.columns),
                    [
                        "node", "parent", "left", "right",
                        "distance", "size", "cid", "character",
                        "cluster",
                    ],
                )
                # the single root holds all fitted characters, and every
                # other node has a parent
                root = sub[sub["node"] == sub["node"].max()].iloc[0]
                self.assertEqual(root["size"], sub["size"].max())
                self.assertEqual(int(root["parent"]), -1)
                self.assertTrue(pandas.isna(root["cid"]))
                # leaves hold their character ID, merges hold none
                leaves = sub[sub["left"].isna()]
                self.assertEqual(len(leaves), int(sub["size"].max()))
                self.assertTrue(leaves["cid"].notna().all())
                self.assertTrue(sub[sub["left"].notna()]["cid"].isna().all())
                # the cluster ID of every leaf matches the clustering
                # result, and the root is the split point of the top
                # clusters and holds none
                merged = leaves.set_index("cid")["cluster"]
                expected = result.loc[merged.index, f"{feature}_cluster"]
                self.assertListEqual(
                    merged.astype(int).tolist(), expected.astype(int).tolist()
                )
                self.assertTrue(pandas.isna(root["cluster"]))

            # the summary is printed as plain text and no report file
            # is written
            self.assertIn("20 characters", output.getvalue())
            self.assertIn("2 dialects", output.getvalue())
            self.assertFalse(
                os.path.isfile(os.path.join(tmp, "reconstruct.json"))
            )
        get_dataset.assert_called_with("stub")

    @mock.patch("sincomp.cli.reconstruct.datasets.get")
    def test_min_dialect_coverage_excludes_sparse_characters(self, get_dataset):
        # a character observed in only one of two dialects
        data = pandas.concat(
            [
                _build_reconstruct_data(),
                pandas.DataFrame(
                    [("d1", "cx", "p", "a", "1")],
                    columns=["did", "cid", "initial", "final", "tone"],
                ),
            ],
            ignore_index=True,
        )
        dataset = mock.Mock()
        dataset.dialects = pandas.DataFrame(
            {"name": ["Alpha", "Beta"]},
            index=pandas.Index(["d1", "d2"], name="did"),
        )
        cids = sorted(data["cid"].unique())
        dataset.characters = pandas.DataFrame(
            {"character": cids},
            index=pandas.Index(cids, name="cid"),
        )

        def select(dids):
            selected = mock.Mock()
            selected.data = data[data["did"].isin(list(dids))]
            return selected

        dataset.select.side_effect = select
        get_dataset.return_value = dataset

        with tempfile.TemporaryDirectory() as tmp:
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    reconstruct_cli.main([
                        "--dataset", "stub",
                        "--dialects", "d1", "d2",
                        "--output-dir", tmp,
                        "--embedding-size", "4",
                        "--min-dialect-coverage", "0.8",
                    ]),
                    0,
                )

            result = pandas.read_csv(
                os.path.join(tmp, "reconstruction.csv"), index_col="cid"
            )
            # the sparse character stays in the output
            self.assertEqual(len(result), 21)
            self.assertEqual(result.loc["cx", "character"], "cx")
            # and it is still assigned to clusters
            for feature in ("initial", "final", "tone"):
                self.assertFalse(
                    pandas.isna(result.loc["cx", f"{feature}_cluster"])
                )
        get_dataset.assert_called_with("stub")

    @mock.patch(
        "sincomp.cli.reconstruct.datasets.get",
        return_value=_make_reconstruct_dataset(),
    )
    def test_min_cluster_size_end_to_end(self, get_dataset):
        with tempfile.TemporaryDirectory() as tmp:
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    reconstruct_cli.main([
                        "--dataset", "stub",
                        "--dialects", "d1", "d2",
                        "--output-dir", tmp,
                        "--embedding-size", "4",
                        "--min-cluster-size", "3",
                    ]),
                    0,
                )

            result = pandas.read_csv(
                os.path.join(tmp, "reconstruction.csv"), index_col="cid"
            )
            self.assertEqual(len(result), 20)
            for feature in ("initial", "final", "tone"):
                self.assertIn(f"{feature}_cluster", result.columns)
        get_dataset.assert_called_with("stub")

    @mock.patch(
        "sincomp.cli.reconstruct.datasets.get",
        return_value=_make_reconstruct_dataset(),
    )
    def test_inference_command(self, get_dataset):
        with tempfile.TemporaryDirectory() as tmp:
            # the explicit reconstruct prefix behaves like the default
            self.assertEqual(
                reconstruct_cli.main([
                    "reconstruct",
                    "--dataset", "stub",
                    "--dialects", "d1", "d2",
                    "--output-dir", tmp,
                    "--embedding-size", "4",
                ]),
                0,
            )
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    reconstruct_cli.main([
                        "inference",
                        "--dataset", "stub",
                        "--dialects", "d1", "d2",
                        "--reconstruction-file",
                        os.path.join(tmp, "reconstruction.csv"),
                        "--output-dir", tmp,
                    ]),
                    0,
                )
            # one rule file for all dialects, with leading did and
            # feature columns marking the dialect and the initial,
            # final and tone rules
            rules_file = os.path.join(tmp, "rules.csv")
            self.assertTrue(os.path.isfile(rules_file))
            all_rules = pandas.read_csv(rules_file, dtype=str)
            self.assertEqual(
                list(all_rules.columns),
                [
                    "did",
                    "feature",
                    "reconstructed_initial",
                    "reconstructed_final",
                    "reconstructed_tone",
                    "phone",
                    "support",
                ],
            )
            self.assertListEqual(
                sorted(all_rules["did"].unique().tolist()),
                ["d1", "d2"],
            )
            self.assertListEqual(
                sorted(all_rules["feature"].unique().tolist()),
                ["final", "initial", "tone"],
            )
            self.assertTrue(all_rules["phone"].notna().all())
            self.assertTrue((all_rules["support"] != "0").all())

            # across the features of one dialect at least one rule
            # carries conditions: a feature whose mapping is too
            # ambiguous may reduce to a single conditionless rule
            condition_columns = [
                column for column in all_rules.columns
                if column.startswith("reconstructed_")
            ]
            self.assertTrue(
                all_rules[condition_columns].notna().any(axis=1).any()
            )

            # the predicted readings of every character and dialect,
            # with the recorded readings for comparison
            predictions = pandas.read_csv(
                os.path.join(tmp, "predictions.csv"),
                dtype={"cid": str, "did": str},
            )
            reconstruction = pandas.read_csv(
                os.path.join(tmp, "reconstruction.csv"), index_col="cid"
            )
            self.assertEqual(
                list(predictions.columns),
                [
                    "did", "name", "cid", "character",
                    "initial", "final", "tone", "note",
                    "reconstructed_initial", "reconstructed_final",
                    "reconstructed_tone",
                    "predicted_initial", "predicted_final", "predicted_tone",
                ],
            )
            names = {"d1": "Alpha", "d2": "Beta"}
            for did in ("d1", "d2"):
                block = predictions[predictions["did"] == did]
                self.assertEqual(len(block), len(reconstruction))
                self.assertTrue((block["name"] == names[did]).all())
                for feature in ("initial", "final", "tone"):
                    self.assertTrue(block[feature].notna().all())
                for feature in (
                    "predicted_initial", "predicted_final", "predicted_tone",
                ):
                    self.assertTrue(block[feature].notna().all())

            # the run report and the prediction accuracy are reported,
            # with the average over the dialects at the end
            report = output.getvalue()
            self.assertIn("Inferred pronunciation rules of 2 dialects", report)
            self.assertIn("dialects: 2", report)
            self.assertIn("characters:", report)
            self.assertIn("samples: 40", report)
            self.assertIn("rules per dialect:", report)
            self.assertIn("phone prediction accuracy:", report)
            self.assertIn("did", report)
            for did in ("d1", "d2"):
                self.assertIn(did, report)
            self.assertIn("mean", report)
            for feature in ("initial", "final", "tone"):
                self.assertIn(feature, report)
        get_dataset.assert_called_with("stub")

    @mock.patch(
        "sincomp.cli.reconstruct.datasets.get",
        return_value=_make_reconstruct_dataset(),
    )
    def test_inference_config_dialects(self, get_dataset):
        with tempfile.TemporaryDirectory() as tmp:
            config_file = os.path.join(tmp, "config.json")
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump({"inference": {"dialects": ["d1", "d2"]}}, f)
            self.assertEqual(
                reconstruct_cli.main([
                    "reconstruct",
                    "--dataset", "stub",
                    "--dialects", "d1", "d2",
                    "--output-dir", tmp,
                    "--embedding-size", "4",
                ]),
                0,
            )
            arguments = [
                "inference",
                "--config", config_file,
                "--dataset", "stub",
                "--reconstruction-file",
                os.path.join(tmp, "reconstruction.csv"),
                "--output-dir", tmp,
            ]
            # the config file provides the descendant dialects
            self.assertEqual(reconstruct_cli.main(arguments), 0)
            self.assertTrue(
                os.path.isfile(os.path.join(tmp, "rules.csv"))
            )
            # command-line arguments take precedence over the config
            override_dir = os.path.join(tmp, "override")
            self.assertEqual(
                reconstruct_cli.main([
                    *arguments, "--dialects", "d2", "--output-dir",
                    override_dir,
                ]),
                0,
            )
            self.assertTrue(
                os.path.isfile(os.path.join(override_dir, "rules.csv"))
            )
            rules = pandas.read_csv(
                os.path.join(override_dir, "rules.csv"), dtype=str
            )
            # only the dialects of the command line are inferred
            self.assertListEqual(
                sorted(rules["did"].unique().tolist()), ["d2"]
            )
        get_dataset.assert_called_with("stub")

    @mock.patch(
        "sincomp.cli.reconstruct.datasets.get",
        return_value=_make_reconstruct_dataset(),
    )
    def test_evaluate_command(self, get_dataset):
        output = StringIO()
        with redirect_stdout(output):
            self.assertEqual(
                reconstruct_cli.main([
                    "evaluate",
                    "--dataset", "stub",
                    "--dialects", "d1", "d2",
                    "--folds", "5",
                    "--repeat", "2",
                    "--seed", "7",
                    "--embedding-size", "4",
                    "--min-cluster-size", "1",
                ]),
                0,
            )
        # the summary reports the mean and the standard deviation of
        # every feature over all trial and dialect accuracies
        text = output.getvalue()
        self.assertIn("mean", text)
        self.assertIn("std", text)
        self.assertIn("initial", text)
        self.assertIn("final", text)
        self.assertIn("tone", text)
        get_dataset.assert_called_with("stub")

    @mock.patch(
        "sincomp.cli.reconstruct.datasets.get",
        return_value=_make_reconstruct_dataset(),
    )
    def test_large_max_cluster_distance_merges_all(self, get_dataset):
        with tempfile.TemporaryDirectory() as tmp:
            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    reconstruct_cli.main([
                        "--dataset", "stub",
                        "--dialects", "d1", "d2",
                        "--output-dir", tmp,
                        "--embedding-size", "4",
                        "--max-cluster-distance", "1000",
                    ]),
                    0,
                )

            result = pandas.read_csv(
                os.path.join(tmp, "reconstruction.csv"), index_col="cid"
            )
            # a huge threshold puts every character in one cluster
            for feature in ("initial", "final", "tone"):
                self.assertEqual(
                    result[f"{feature}_cluster"].nunique(), 1
                )
        get_dataset.assert_called_with("stub")

    @mock.patch(
        "sincomp.cli.reconstruct.datasets.get",
        return_value=_make_reconstruct_dataset(),
    )
    def test_reconstruct_with_annotations(self, get_dataset):
        with tempfile.TemporaryDirectory() as tmp:
            annotation_file = os.path.join(tmp, "annotations.csv")
            _write_segment_annotations(annotation_file)

            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    reconstruct_cli.main([
                        "--dataset", "stub",
                        "--dialects", "d1", "d2",
                        "--annotation-file", annotation_file,
                        "--output-dir", tmp,
                        "--embedding-size", "4",
                    ]),
                    0,
                )

            result = pandas.read_csv(
                os.path.join(tmp, "reconstruction.csv"), index_col="cid"
            )
            self.assertIn("character", result.columns)
            for feature in ("initial", "final", "tone"):
                self.assertIn(feature, result.columns)
                self.assertIn(f"{feature}_cluster", result.columns)
                # the annotations provide the phones and the phonemes
                self.assertTrue(result[feature].notna().all())
                self.assertTrue(result[f"{feature}_phoneme"].notna().all())

            self.assertIn("20 characters", output.getvalue())

            # no pronunciation summary is written when annotations exist
            self.assertFalse(
                os.path.isfile(os.path.join(tmp, "cluster_phones.csv"))
            )
        get_dataset.assert_called_with("stub")

    @mock.patch(
        "sincomp.cli.reconstruct.datasets.get",
        return_value=_make_reconstruct_dataset(),
    )
    def test_config_provides_defaults(self, get_dataset):
        with tempfile.TemporaryDirectory() as tmp:
            config_file = os.path.join(tmp, "config.json")
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump({
                    "dataset": "stub",
                    "dialects": ["d1", "d2"],
                    "embedding_size": 4,
                }, f)

            output = StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    reconstruct_cli.main([
                        "--config", config_file,
                        "--output-dir", tmp,
                    ]),
                    0,
                )

            self.assertTrue(
                os.path.isfile(os.path.join(tmp, "reconstruction.csv"))
            )
            self.assertIn("20 characters", output.getvalue())
            self.assertIn("stub", output.getvalue())
        get_dataset.assert_called_with("stub")


if __name__ == "__main__":
    unittest.main()