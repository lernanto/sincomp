# -*- coding: utf-8 -*-

"""Tests for the classifier CLI (:mod:`sincomp.cli.classifier`).

The tests verify the parser defaults and the config file handling of
the ``train``, ``evaluate`` and ``predict`` subcommands, the dialect
resampling, and the end-to-end training, evaluation and prediction of
the rule-compliance and vector-based dialect classifiers on stub
datasets with mocked data access.
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"

import json
import os
import sys
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

from sincomp.cli import classifier as classifier_cli
from sincomp.feature_extraction import DialectVectorizer


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
    """Test the classifier training, evaluation and prediction commands."""

    def test_subcommand_required(self):
        with self.assertRaises(SystemExit):
            classifier_cli.build_parser().parse_args([])

    def test_train_defaults(self):
        args = classifier_cli.build_parser().parse_args(
            ["train", "--annotation-file", "annotations.csv"]
        )
        self.assertIsNone(args.rule_file)
        self.assertEqual(args.min_coverage, 0.5)
        self.assertEqual(args.embedding_size, 128)
        self.assertEqual(args.resample, 5)
        self.assertEqual(args.samples, 1000)
        self.assertEqual(args.model_file, "dialect_classifier.bz2")

        args = classifier_cli.build_parser().parse_args([
            "train", "--annotation-file", "annotations.csv", "--model-file",
            os.path.join("custom", "model.bz2"),
        ])
        self.assertEqual(args.model_file, os.path.join("custom", "model.bz2"))

    def test_train_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config_file = os.path.join(tmp, "config.json")
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump({
                    "resample": 2,
                    "min-coverage": 0.6,
                    "samples": 300,
                    "input": "unused.csv",
                    "unknown-option": "ignored",
                }, f)

            # A flat config provides defaults for matching arguments;
            # keys unused by this command are ignored.
            args = classifier_cli.build_parser(
                classifier_cli.load_config(config_file)
            ).parse_args([
                "train", "--annotation-file", "annotations.csv",
            ])
            self.assertEqual(args.resample, 2)
            self.assertEqual(args.samples, 300)
            self.assertEqual(args.min_coverage, 0.6)
            self.assertFalse(hasattr(args, "input"))

            # the command-line values take precedence over the config
            args = classifier_cli.build_parser(
                classifier_cli.load_config(config_file)
            ).parse_args([
                "train", "--resample", "3",
                "--annotation-file", "annotations.csv",
            ])
            self.assertEqual(args.resample, 3)
            self.assertEqual(args.samples, 300)
            self.assertEqual(args.min_coverage, 0.6)

    @mock.patch("sincomp.cli.classifier.train")
    def test_train_config_arguments(self, train):
        with tempfile.TemporaryDirectory() as tmp:
            config_file = os.path.join(tmp, "config.json")
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump({
                    "annotation_file": "annotations.csv",
                    "model_file": "model.bz2",
                    "predict": {"input": "unused.csv"},
                }, f)

            self.assertEqual(
                classifier_cli.main(["train", "--config", config_file]), 0
            )
            args = train.call_args.args[0]
            self.assertEqual(args.annotation_file, "annotations.csv")
            self.assertEqual(args.model_file, "model.bz2")

            self.assertEqual(
                classifier_cli.main([
                    "train", "--config", config_file,
                    "--annotation-file", "override.csv",
                ]),
                0,
            )
            args = train.call_args.args[0]
            self.assertEqual(args.annotation_file, "override.csv")
            self.assertEqual(args.model_file, "model.bz2")

    def test_predict_config_streams(self):
        args = classifier_cli.build_parser({
            "input_file": "-",
            "prediction_file": "-",
        }).parse_args(["predict"])
        self.assertIs(args.input_file, sys.stdin)
        self.assertIs(args.prediction_file, sys.stdout)

    @mock.patch(
        "sincomp.cli.classifier.datasets.get",
        return_value=_make_stub_dataset(),
    )
    def test_train_end_to_end(self, get_dataset):
        with tempfile.TemporaryDirectory() as tmp:
            rule_file = os.path.join(tmp, "rules.json")
            annotation_file = os.path.join(tmp, "annotations.csv")
            model_file = os.path.join(tmp, "nested", "model", "classifier.bz2")
            _write_rules(rule_file)
            _write_annotations(annotation_file)

            with self.assertLogs(
                classifier_cli.logger, level="INFO"
            ) as output:
                self.assertEqual(
                    classifier_cli.main([
                        "--log-level", "INFO",
                        "train",
                        "--resample", "0",
                        "--rule-file", rule_file,
                        "--annotation-file", annotation_file,
                        "--model-file", model_file,
                    ]),
                    0,
                )

            self.assertRegex(
                "\n".join(output.output),
                r"cross-validation accuracy = \d+\.\d{4}.*\d+\.\d{4}",
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
                        "--annotation-file", annotation_file,
                    ]),
                    0,
                )

            # the compliance perfectly separates the two classes
            self.assertIn("precision", output.getvalue())
            self.assertIn("f1", output.getvalue())
            self.assertIn("cross validation with 16 samples", output.getvalue())
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

            with self.assertLogs(
                classifier_cli.logger, level="INFO"
            ) as output:
                self.assertEqual(
                    classifier_cli.main([
                        "--log-level", "INFO",
                        "train",
                        "--resample", "1",
                        "--annotation-file", annotation_file,
                        "--model-file", model_file,
                    ]),
                    0,
                )
            self.assertRegex(
                "\n".join(output.output),
                r"cross-validation accuracy = \d+\.\d{4}.*\d+\.\d{4}",
            )
            model = joblib.load(model_file)
            vectorizer = model.steps[0][1]
            self.assertIsInstance(vectorizer, DialectVectorizer)
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
                        "--annotation-file", annotation_file,
                    ]),
                    0,
                )

            # the pairwise identity features separate the two classes
            self.assertIn("cross validation with 16 samples", output.getvalue())
            self.assertIn("mean", output.getvalue())
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
                        "--annotation-file", annotation_file,
                    ]),
                    0,
                )

            # the multiclass report holds the per-fold and average
            # accuracy only
            self.assertNotIn("precision", output.getvalue())
            self.assertIn("mean", output.getvalue())
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
            output_file = os.path.join(tmp, "nested", "predictions.csv")
            _write_annotations(annotation_file)

            classifier_cli.main([
                "train",
                "--resample", "0",
                "--annotation-file", annotation_file,
                "--model-file", model_file,
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
                    "--input-file", sample_file,
                    "--prediction-file", output_file,
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
                    "--input-file", input_file,
                    "--prediction-file", output_file,
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
                "--annotation-file", annotation_file,
                "--model-file", model_file,
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
                    "--input-file", sample_file,
                    "--prediction-file", output_file,
                ]),
                0,
            )

            result = pandas.read_csv(output_file).set_index("did")
            self.assertGreater(result.loc["A0", "A"], result.loc["A0", "B"])
            self.assertGreater(result.loc["B0", "B"], result.loc["B0", "A"])
        get_dataset.assert_called_with("stub")


if __name__ == "__main__":
    unittest.main()
