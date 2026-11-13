# -*- coding: utf-8 -*-

"""Tests for the reconstruct CLI (:mod:`sincomp.cli.reconstruct`).

The tests verify the parser defaults and the subcommand structure,
the character clustering helpers, the annotation handling, the tree
rule extraction and the cluster phoneme assignment and merging, plus
the end-to-end ``reconstruct``, ``inference`` and ``evaluate``
commands on stub datasets with mocked data access.
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"

import json
import os
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest import mock

import numpy
import pandas

from sincomp.cli import reconstruct as reconstruct_cli


def _reconstruct_output_args(directory):
    filenames = [
        "reconstruction.csv",
        "clusters.csv",
        "cluster_phones.csv",
        "initial_tree.csv",
        "final_tree.csv",
        "tone_tree.csv",
    ]
    options = [
        "--reconstruction-file",
        "--clusters-file",
        "--cluster-phones-file",
        "--initial-tree-file",
        "--final-tree-file",
        "--tone-tree-file",
    ]
    return [
        value
        for option, filename in zip(options, filenames)
        for value in (option, os.path.join(directory, filename))
    ]


def _inference_output_args(directory, reconstruction_file=None):
    if reconstruction_file is None:
        reconstruction_file = os.path.join(directory, "reconstruction.csv")
    return [
        "--reconstruction-file",
        reconstruction_file,
        "--rules-file",
        os.path.join(directory, "rules.csv"),
        "--predictions-file",
        os.path.join(directory, "predictions.csv"),
    ]


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
    """Test the proto-language reconstruction commands."""

    def test_subcommand_help_is_scoped(self):
        output = StringIO()
        with redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                reconstruct_cli.main(["reconstruct", "--help"])

        self.assertEqual(raised.exception.code, 0)
        help_text = output.getvalue()
        self.assertIn(
            "Reconstruct proto-language phonemes from the selected dialects",
            help_text,
        )
        self.assertNotIn("Subcommands:", help_text)
        self.assertNotIn("infer the pronunciation rules", help_text)
        self.assertNotIn("evaluate the pipeline on held-out", help_text)

    def test_top_level_help_lists_all_subcommands(self):
        output = StringIO()
        with redirect_stdout(output):
            with self.assertRaises(SystemExit) as raised:
                reconstruct_cli.main(["--help"])

        self.assertEqual(raised.exception.code, 0)
        help_text = output.getvalue()
        self.assertIn("Subcommands:", help_text)
        self.assertIn("reconstruct", help_text)
        self.assertIn("inference", help_text)
        self.assertIn("evaluate", help_text)
        self.assertIn("Reconstruct proto-language phonemes", help_text)
        self.assertIn("produced by ``reconstruct``", help_text)
        self.assertIn("held-out dialects and characters", help_text)

    def test_parser_defaults(self):
        parser = reconstruct_cli.build_parser()
        args = parser.parse_args(["reconstruct"])
        self.assertEqual(args.dataset, "MCPDict")
        self.assertEqual(args.embedding_size, 128)
        self.assertEqual(args.reconstruction_file, "reconstruction.csv")
        self.assertEqual(args.clusters_file, "clusters.csv")
        self.assertEqual(args.cluster_phones_file, "cluster_phones.csv")
        self.assertEqual(args.tone_tree_file, "tone_tree.csv")
        self.assertFalse(hasattr(args, "output_dir"))
        inference_args = parser.parse_args(["inference"])
        self.assertEqual(
            inference_args.reconstruction_file,
            args.reconstruction_file,
        )
        self.assertEqual(inference_args.rules_file, "rules.csv")
        self.assertEqual(inference_args.predictions_file, "predictions.csv")
        self.assertFalse(hasattr(inference_args, "output_dir"))
        self.assertEqual(args.log_level, "WARNING")
        self.assertEqual(args.min_dialect_coverage, 0.5)
        self.assertEqual(args.min_cluster_size, 3)
        self.assertEqual(args.max_cluster_distance, 0.1)
        self.assertEqual(args.max_clusters, 200)
        self.assertEqual(args.min_phone_proportion, 0.1)
        self.assertEqual(args.max_merge_tolerance, 0)
        self.assertEqual(args.max_merge_distance, 0.5)

    def test_all_reconstruct_subcommands_use_options(self):
        parser = reconstruct_cli.build_parser()
        subparsers = next(
            action for action in parser._actions
            if action.dest == "command"
        )
        for name, command_parser in subparsers.choices.items():
            with self.subTest(parser=command_parser.prog):
                self.assertTrue(all(
                    action.option_strings
                    for action in command_parser._actions
                    if action.dest != "help"
                ))

    def test_embedding_size_is_common_argument(self):
        parser = reconstruct_cli.build_parser()
        for command in ("reconstruct", "evaluate"):
            with self.subTest(command=command):
                self.assertEqual(
                    parser.parse_args([command]).embedding_size, 128
                )
                self.assertEqual(
                    parser.parse_args([
                        command, "--embedding-size", "64",
                    ]).embedding_size,
                    64,
                )

    def test_evaluate_accepts_annotation_file_without_loading_it(self):
        args = reconstruct_cli.build_parser().parse_args([
            "evaluate", "--annotation-file", "annotations.csv",
        ])
        self.assertEqual(args.annotation_file, "annotations.csv")

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
                        *_reconstruct_output_args(tmp),
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
                    "dialect_count",
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
            self.assertEqual(result.columns[-1], "dialect_count")
            self.assertEqual(len(result), 20)
            self.assertTrue((result["dialect_count"] == 2).all())
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
            self.assertIn("character_count: 20", output.getvalue())
            self.assertIn("dialect_count: 2", output.getvalue())
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
                        *_reconstruct_output_args(tmp),
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
            self.assertEqual(result.loc["cx", "dialect_count"], 1)
            self.assertEqual(result.loc["c0", "dialect_count"], 2)
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
                        *_reconstruct_output_args(tmp),
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
                    *_reconstruct_output_args(tmp),
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
                        *_inference_output_args(tmp),
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
            self.assertIn("Data summary:", report)
            self.assertIn("dialect_count: 2", report)
            self.assertIn("character_count: 20", report)
            self.assertIn("sample_count: 40", report)
            self.assertIn("Rules per dialect:", report)
            self.assertIn("Phone prediction accuracy:", report)
            self.assertIn("did", report)
            self.assertIn("Alpha", report)
            self.assertIn("Beta", report)
            for did in ("d1", "d2"):
                self.assertIn(did, report)
            self.assertIn("mean", report)
            self.assertRegex(
                report,
                r"(?m)^mean\s+\s+1\.0000\s+1\.0000\s+0\.4000",
            )
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
                json.dump({"dialects": ["d1", "d2"]}, f)
            self.assertEqual(
                reconstruct_cli.main([
                    "reconstruct",
                    "--dataset", "stub",
                    "--dialects", "d1", "d2",
                    *_reconstruct_output_args(tmp),
                    "--embedding-size", "4",
                ]),
                0,
            )
            arguments = [
                "inference",
                "--config", config_file,
                "--dataset", "stub",
                *_inference_output_args(tmp),
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
                    *arguments,
                    "--dialects", "d2",
                    "--rules-file", os.path.join(override_dir, "rules.csv"),
                    "--predictions-file",
                    os.path.join(override_dir, "predictions.csv"),
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
        with self.assertLogs(
            "sincomp.cli.reconstruct", level="DEBUG"
        ) as logs:
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
                        "--log-level", "DEBUG",
                    ]),
                    0,
                )
        self.assertTrue(any("d1 accuracy" in message for message in logs.output))
        self.assertTrue(any("d2 accuracy" in message for message in logs.output))
        self.assertRegex(
            "\n".join(logs.output),
            r"accuracy: \{'initial': '[0-9]\.\d{4}'",
        )
        # the summary reports the mean and the standard deviation of
        # every feature over all trial and dialect accuracies
        text = output.getvalue()
        self.assertIn("mean", text)
        self.assertIn("std", text)
        self.assertIn("initial", text)
        self.assertIn("final", text)
        self.assertIn("tone", text)
        self.assertIn("initial_clusters", text)
        self.assertIn("initial_phonemes", text)
        self.assertIn("final_clusters", text)
        self.assertIn("final_phonemes", text)
        self.assertIn("tone_clusters", text)
        self.assertIn("tone_phonemes", text)
        self.assertIn("Data summary:", text)
        self.assertIn("min_dialect_coverage: 0.5", text)
        cluster_report = text.split(
            "Cluster and phoneme counts per trial:\n", 1
        )[1].split(
            "Phone prediction accuracy of held-out dialects and characters",
            1,
        )[0]
        self.assertRegex(cluster_report, r"(?m)^0\s+\d+\s+\d+")
        self.assertRegex(cluster_report, r"(?m)^mean\s")
        self.assertRegex(cluster_report, r"(?m)^std\s")
        self.assertRegex(cluster_report, r"(?m)^mean\s+\d+\.\d{2}")
        self.assertRegex(cluster_report, r"(?m)^std\s+\d+\.\d{2}")
        self.assertIn(
            "(per-trial dialect mean, overall mean and std):", text
        )
        accuracy_report = text.split(
            "(per-trial dialect mean, overall mean and std):\n", 1
        )[1]
        self.assertRegex(accuracy_report, r"(?m)^0\s+")
        self.assertRegex(accuracy_report, r"(?m)^1\s+")
        self.assertRegex(accuracy_report, r"(?m)^mean\s+")
        self.assertRegex(accuracy_report, r"(?m)^std\s+")
        self.assertRegex(accuracy_report, r"\d\.\d{4}")
        self.assertLess(
            text.index("Cluster and phoneme counts per trial:"),
            text.index(
                "Phone prediction accuracy of held-out dialects and characters"
            ),
        )
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
                        *_reconstruct_output_args(tmp),
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
                        *_reconstruct_output_args(tmp),
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

            self.assertIn("character_count: 20", output.getvalue())
            self.assertIn("dialect_count: 2", output.getvalue())

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
                        *_reconstruct_output_args(tmp),
                    ]),
                    0,
                )

            self.assertTrue(
                os.path.isfile(os.path.join(tmp, "reconstruction.csv"))
            )
            self.assertIn("character_count: 20", output.getvalue())
            self.assertIn("embedding_size: 4", output.getvalue())
            self.assertIn("stub", output.getvalue())
        get_dataset.assert_called_with("stub")


if __name__ == "__main__":
    unittest.main()
