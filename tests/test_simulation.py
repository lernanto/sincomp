# -*- coding: utf-8 -*-
"""Tests for the dialect formation simulation.

The tests only assert deterministic constraints that hold for every
seed and every supported numpy version: structural invariants of the
generated tree and records, membership relations and input
validation. The random draws are seeded for reproducibility of the
inputs, but no assertion depends on the sampled values themselves —
the structural tests run over several seeds to enforce that.
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"


import re
import unittest

import numpy
import pandas
from pandas.testing import assert_frame_equal

from sincomp import datasets
from sincomp.simulation import _mutate, gen_dialects, gen_dataset, sample_chars


COLUMNS = ["did", "cid", "character", "initial", "final", "tone"]
TREE_COLUMNS = ["did", "parent", "left", "right", "change"]
CHANGE_PATTERN = re.compile(r"(.+) > (.+)")
# the structural invariants must hold for every seed, not only for a
# lucky one, so the tree and sampling tests run over several seeds
SEEDS = (0, 1, 42)


def small(**kwargs):
    """Run a small fast simulation with a fixed seed."""

    options = {
        "max_dialects": 8,
        "n_iter": 3,
        "seed": 42,
    }
    if "characters" not in kwargs:
        options["n_characters"] = 300
    options.update(kwargs)
    return gen_dataset(**options)


def levels_of(tree):
    """Return the generation level of every recorded dialect."""

    return tree["did"].str[1:].str.split("_").str[0].astype(int)


def modern_of(tree):
    """Return the tree rows of the deepest-level dialects."""

    levels = levels_of(tree)
    return tree.loc[levels == levels.max()]


def rows_of(data, did):
    """Return the records of one dialect indexed by character ID."""

    return data[data["did"] == did].set_index("cid").sort_index()


class MutateTest(unittest.TestCase):
    def test_zero_rate_keeps_values(self):
        indices = numpy.array([0, 1, 2, 3, 4])
        _mutate(indices, 0.0, numpy.random.default_rng(42))

        numpy.testing.assert_array_equal(indices, [0, 1, 2, 3, 4])

    def test_values_stay_in_present_values(self):
        # whatever the draws are, every mutated value is one of the
        # values the dialect already holds
        for seed in SEEDS:
            indices = numpy.arange(100, dtype=int) % 5
            _mutate(indices, 0.3, numpy.random.default_rng(seed))
            self.assertTrue(((indices >= 0) & (indices < 5)).all())

    def test_reproducible_with_seed(self):
        first = numpy.arange(100, dtype=int) % 5
        second = numpy.arange(100, dtype=int) % 5
        _mutate(first, 0.3, numpy.random.default_rng(42))
        _mutate(second, 0.3, numpy.random.default_rng(42))

        numpy.testing.assert_array_equal(first, second)

    def test_high_rate_raises(self):
        # when k * rate / (k - 1) exceeds one for the k present
        # values, they cannot support the rate
        with self.assertRaises(ValueError):
            _mutate(
                numpy.array([0, 1] * 50, dtype=int), 0.9,
                numpy.random.default_rng(42),
            )

    def test_single_value_never_mutates(self):
        indices = numpy.zeros(100, dtype=int)
        _mutate(indices, 0.5, numpy.random.default_rng(42))
        self.assertTrue((indices == 0).all())


class GenDialectsTest(unittest.TestCase):
    def test_returns_complete_records_and_tree(self):
        for seed in SEEDS:
            data, tree = gen_dialects(
                n_vocabularies=[5, 4], n_characters=100,
                max_dialects=4, n_iter=3, seed=seed,
            )

            self.assertIsInstance(data, pandas.DataFrame)
            self.assertIsInstance(tree, pandas.DataFrame)
            self.assertListEqual(
                list(data.columns),
                ["did", "cid", "character", "feature1", "feature2"],
            )
            self.assertListEqual(list(tree.columns), TREE_COLUMNS)
            # the population grows 1 -> 2 -> 4 and is capped at 4, so
            # exactly 11 dialects are generated and recorded whatever
            # the random draws are
            self.assertEqual(len(tree), 11)
            self.assertEqual(data["did"].nunique(), 11)
            self.assertEqual(len(data), 11 * 100)
            cids = set(data["cid"])
            self.assertEqual(len(cids), 100)
            for did in tree["did"]:
                sub = rows_of(data, did)
                # every dialect holds the full character set exactly
                # once
                self.assertEqual(len(sub), 100)
                self.assertEqual(set(sub.index), cids)

    def test_population_per_level(self):
        # every live dialect conceives a left and a right child, and
        # the cap cuts the conceived children down to max_dialects, so
        # level k holds exactly min(2^k, max_dialects) dialects
        for seed in SEEDS:
            _, tree = gen_dialects(
                n_vocabularies=[5, 4], n_characters=100,
                max_dialects=4, n_iter=3, seed=seed,
            )

            counts = levels_of(tree).value_counts()
            for level in range(4):
                self.assertEqual(counts.get(level, 0), min(2 ** level, 4))
            # the deepest level holds the modern dialects
            self.assertEqual(len(modern_of(tree)), 4)

    def test_reproducible_with_seed(self):
        first, first_tree = gen_dialects(
            n_vocabularies=[5, 4], n_characters=100,
            max_dialects=4, n_iter=3, seed=42,
        )
        second, second_tree = gen_dialects(
            n_vocabularies=[5, 4], n_characters=100,
            max_dialects=4, n_iter=3, seed=42,
        )

        assert_frame_equal(first, second)
        assert_frame_equal(first_tree, second_tree)

    def test_auto_encoding(self):
        data, _ = gen_dialects(
            n_vocabularies=[5, 4], n_characters=100,
            max_dialects=4, n_iter=3, seed=42,
        )

        self.assertTrue(data["cid"].str.fullmatch(r"c\d+").all())
        self.assertEqual(data["cid"].iloc[0], "c1")
        self.assertTrue((data["cid"] == data["character"]).all())
        # unnamed features auto-encode as feature1 ... with values
        # f1_1 ...
        self.assertTrue(data["feature1"].str.fullmatch(r"f1_\d+").all())
        self.assertTrue(data["feature2"].str.fullmatch(r"f2_\d+").all())

    def test_named_features_first_letter_encoding(self):
        data, _ = gen_dialects(
            features=["onset", "coda"],
            n_vocabularies=[4, 3],
            n_characters=100, max_dialects=4, n_iter=2, seed=42,
        )

        self.assertListEqual(
            list(data.columns),
            ["did", "cid", "character", "onset", "coda"],
        )
        # named features auto-encode with their first letter
        self.assertTrue(data["onset"].str.fullmatch(r"o\d+").all())
        self.assertTrue(data["coda"].str.fullmatch(r"c\d+").all())

    def test_explicit_values(self):
        for seed in SEEDS:
            data, _ = gen_dialects(
                features=["a", "b"],
                vocabularies=[["x", "y", "z"], ["1", "2", "3"]],
                n_characters=100, max_dialects=4, n_iter=2, seed=seed,
            )

            self.assertTrue(set(data["a"]) <= {"x", "y", "z"})
            self.assertTrue(set(data["b"]) <= {"1", "2", "3"})

    def test_explicit_characters(self):
        # explicit character names survive the simulation; the
        # balanced ancestor values keep every feature eligible and
        # the tiny mutation rate keeps the present values stable
        characters = ["之", "乎", "者", "也", "天", "地", "人", "和"]
        values = numpy.array([
            ["x", "1"],
            ["y", "2"],
            ["z", "3"],
            ["x", "1"],
            ["y", "2"],
            ["z", "3"],
            ["x", "1"],
            ["y", "2"],
        ])
        for seed in SEEDS:
            data, _ = gen_dialects(
                characters=characters, values=values,
                max_dialects=4, n_iter=2, mutation_rate=0.001, seed=seed,
            )

            self.assertTrue(set(data["character"]) <= set(characters))
            self.assertTrue(data["cid"].str.fullmatch(r"c\d+").all())

    def test_values_seed_the_ancestor(self):
        # the given values are the ancestor records: the root dialect
        # reads exactly them, whatever the seed is
        characters = ["之", "乎", "者", "也"]
        values = numpy.array([
            ["x", "1"],
            ["y", "2"],
            ["x", "1"],
            ["z", "3"],
        ])
        for seed in SEEDS:
            data, _ = gen_dialects(
                features=["initial", "final"],
                characters=characters, values=values,
                max_dialects=4, n_iter=2, seed=seed,
            )

            root = rows_of(data, "d0_0")
            self.assertListEqual(list(root.index), ["c1", "c2", "c3", "c4"])
            self.assertListEqual(
                list(root["initial"]), ["x", "y", "x", "z"],
            )
            self.assertListEqual(
                list(root["final"]), ["1", "2", "1", "3"],
            )
            self.assertListEqual(list(root["character"]), characters)

    def test_values_may_not_cover_the_vocabulary(self):
        # the vocabulary may be larger than the observed values; the
        # two features hold two distinct values each, so one merge
        # can never exhaust both of them
        values = numpy.array([
            ["x", "1"],
            ["y", "2"],
            ["x", "1"],
            ["y", "2"],
        ])
        for seed in SEEDS:
            data, _ = gen_dialects(
                features=["a", "b"],
                vocabularies=[["x", "y", "z", "w"], ["1", "2", "3", "4"]],
                values=values, max_dialects=4, n_iter=2, seed=seed,
            )

            self.assertTrue(set(data["a"]) <= {"x", "y", "z", "w"})
            self.assertTrue(set(data["b"]) <= {"1", "2", "3", "4"})

    def test_infer_from_values(self):
        # every structural argument is inferred from the values: the
        # feature names, the vocabularies and the character IDs
        values = numpy.array([
            ["x", "1"],
            ["y", "2"],
            ["z", "3"],
            ["w", "4"],
        ])
        data, _ = gen_dialects(
            values=values, max_dialects=4, n_iter=1, seed=42,
        )

        self.assertListEqual(
            list(data.columns),
            ["did", "cid", "character", "feature1", "feature2"],
        )
        root = rows_of(data, "d0_0")
        self.assertListEqual(list(root.index), ["c1", "c2", "c3", "c4"])
        self.assertListEqual(list(root["feature1"]), ["x", "y", "z", "w"])
        self.assertListEqual(list(root["feature2"]), ["1", "2", "3", "4"])

    def test_n_features_mismatch_raises(self):
        with self.assertRaises(ValueError):
            gen_dialects(n_features=3, n_vocabularies=[3, 4], seed=42)
        with self.assertRaises(ValueError):
            gen_dialects(
                features=["a", "b"], n_features=3,
                n_vocabularies=[3, 4], seed=42,
            )

    def test_values_disagreement_raises(self):
        with self.assertRaises(ValueError):
            gen_dialects(
                n_characters=5,
                values=numpy.array([["x"], ["y"]]),
                max_dialects=4, n_iter=1, seed=42,
            )
        with self.assertRaises(ValueError):
            gen_dialects(
                characters=["之", "乎", "者"],
                values=numpy.array([["x"], ["y"]]),
                max_dialects=4, n_iter=1, seed=42,
            )
        with self.assertRaises(ValueError):
            gen_dialects(
                vocabularies=[["x", "y"]],
                values=numpy.array([["x"], ["q"]]),
                max_dialects=4, n_iter=1, seed=42,
            )

    def test_dialect_id_reflects_lineage(self):
        for seed in SEEDS:
            _, tree = gen_dialects(
                n_vocabularies=[5, 4], n_characters=100,
                max_dialects=4, n_iter=3, seed=seed,
            )

            rows = tree.set_index("did")
            dids = set(rows.index)

            def parts(did):
                level, number = did[1:].split("_")
                return int(level), int(number)

            for did, row in rows.iterrows():
                level, number = parts(did)
                if pandas.isna(row["parent"]):
                    # the root is d0_0
                    self.assertEqual((level, number), (0, 0))
                    continue
                # the parent is a recorded dialect of the previous
                # level
                self.assertIn(row["parent"], dids)
                parent_level, parent_number = parts(row["parent"])
                self.assertEqual(level, parent_level + 1)
                # the number of a child is 2 * the number of its
                # parent, plus one for the right child
                self.assertIn(
                    number,
                    (2 * parent_number, 2 * parent_number + 1),
                )
                # the recorded children exist in the tree
                for child in (row["left"], row["right"]):
                    if not pandas.isna(child):
                        self.assertIn(child, dids)

    def test_change_records(self):
        for seed in SEEDS:
            _, tree = gen_dialects(
                n_vocabularies=[5, 4], n_characters=100,
                max_dialects=4, n_iter=3, seed=seed,
            )

            # every recorded dialect of the deepest level never split
            # again
            self.assertTrue((modern_of(tree)["change"] == "").all())
            for row in tree.itertuples(index=False):
                both_children = not pandas.isna(row.left) and not pandas.isna(
                    row.right
                )
                # a full split always applies one rule change, so a
                # dialect with both children recorded carries a change;
                # a single surviving child may or may not apply one
                if both_children:
                    self.assertNotEqual(row.change, "", row.did)
                if row.change:
                    match = CHANGE_PATTERN.fullmatch(row.change)
                    self.assertIsNotNone(match, row.change)
                    self.assertNotEqual(match.group(1), match.group(2))

    def test_changes_take_effect(self):
        # the source and the target of a rule change are drawn from
        # the values the dialect actually holds, so every recorded
        # change alters the readings of the child
        for seed in SEEDS:
            data, tree = gen_dialects(
                n_vocabularies=[5, 4], n_characters=100,
                max_dialects=4, n_iter=3, seed=seed,
            )
            feature_columns = list(data.columns[3:])
            rows = data.set_index(["did", "cid"]).sort_index()
            for row in tree.itertuples(index=False):
                if not row.change:
                    continue
                source = CHANGE_PATTERN.fullmatch(row.change).group(1)
                # the source value may name a value of several
                # features, so count the changed characters over all
                # of them: the merged feature alone guarantees at
                # least one
                affected = 0
                for child in (row.left, row.right):
                    if pandas.isna(child):
                        continue
                    for c in feature_columns:
                        parent_vals = rows.loc[
                            (row.did, slice(None)), c
                        ].to_numpy()
                        child_vals = rows.loc[
                            (child, slice(None)), c
                        ].to_numpy()
                        affected = max(
                            affected,
                            int((parent_vals != child_vals).sum()),
                        )
                self.assertGreater(affected, 0, row.change)

    def test_change_carried_by_odd_child(self):
        # the rule change of a split is carried by the odd child:
        # every character that the merge touched reads a different
        # value from the parent, so the feature of a recorded change
        # is always among the odd child's differences
        features = ["initial", "final"]
        feature_of_letter = {"i": "initial", "f": "final"}
        for seed in SEEDS:
            data, tree = gen_dialects(
                features=features, n_vocabularies=[5, 4], n_characters=100,
                max_dialects=4, n_iter=3, seed=seed,
            )

            tree_rows = tree.set_index("did")
            for did, row in tree_rows.iterrows():
                if not row["change"]:
                    continue
                parent_rows = rows_of(data, did)
                f = feature_of_letter[row["change"][0]]
                for child_did in (row["left"], row["right"]):
                    if pandas.isna(child_did):
                        continue
                    number = int(child_did[1:].split("_")[1])
                    if number % 2 == 0:
                        continue
                    child_rows = rows_of(data, child_did)
                    self.assertTrue(
                        (
                            parent_rows[f].to_numpy()
                            != child_rows[f].to_numpy()
                        ).any(),
                        child_did,
                    )

    def test_max_dialects_cap(self):
        for seed in SEEDS:
            data, tree = gen_dialects(
                n_vocabularies=[5, 4], n_characters=100,
                max_dialects=5, n_iter=3, seed=seed,
            )

            # the population grows 1 -> 2 -> 4 and is capped at 5
            counts = levels_of(tree).value_counts()
            for level in range(4):
                self.assertEqual(counts.get(level, 0), min(2 ** level, 5))
            # every recorded dialect carries records
            self.assertEqual(data["did"].nunique(), len(tree))

    def test_cap_extinction_keeps_population(self):
        data, tree = gen_dialects(
            n_vocabularies=[7, 5], n_characters=100,
            max_dialects=6, n_iter=4, seed=42,
        )

        # the population grows 1 -> 2 -> 4 -> 6 and stays at 6 with
        # the surplus children deleted at every split, whatever the
        # draws are
        counts = levels_of(tree).value_counts()
        for level in range(5):
            self.assertEqual(counts.get(level, 0), min(2 ** level, 6))
        self.assertEqual(len(tree), 19)
        self.assertEqual(len(data), len(tree) * 100)

    def test_single_dialect(self):
        data, tree = gen_dialects(
            n_vocabularies=[5, 4], n_characters=100,
            max_dialects=1, n_iter=3, seed=42,
        )

        # with a population cap of one, a single lineage continues
        # and one of the two children of every split dies out
        self.assertEqual(len(tree), 4)
        levels = levels_of(tree)
        self.assertEqual((levels == levels.max()).sum(), 1)
        self.assertEqual(data["did"].nunique(), 4)
        self.assertEqual(len(data), 4 * 100)

    def test_requires_inventories(self):
        with self.assertRaises(ValueError):
            gen_dialects(n_characters=100, seed=42)
        with self.assertRaises(ValueError):
            gen_dialects(features=["a", "b"], n_vocabularies=[3, 4, 5], seed=42)
        with self.assertRaises(ValueError):
            gen_dialects(
                features=["a", "b"],
                vocabularies=[["x", "y"], ["1", "2", "3"]],
                n_vocabularies=[2, 2],
                seed=42,
            )

    def test_validation(self):
        with self.assertRaises(ValueError):
            gen_dialects(n_vocabularies=[1, 4], seed=42)
        with self.assertRaises(ValueError):
            gen_dialects(n_vocabularies=[5, 4], max_dialects=0, seed=42)
        with self.assertRaises(ValueError):
            gen_dialects(n_vocabularies=[5, 4], n_iter=0, seed=42)
        with self.assertRaises(ValueError):
            gen_dialects(n_vocabularies=[5, 4], n_iter=5, seed=42)
        with self.assertRaises(ValueError):
            gen_dialects(n_vocabularies=[5, 4], mutation_rate=0.0, seed=42)
        with self.assertRaises(ValueError):
            gen_dialects(n_vocabularies=[5, 4], mutation_rate=1.0, seed=42)
        with self.assertRaises(ValueError):
            gen_dialects(n_vocabularies=[5, 4], n_characters=0, seed=42)

    def test_parameter_warnings(self):
        # the value inventories nearly run out within the generations
        with self.assertWarns(UserWarning):
            gen_dialects(
                n_vocabularies=[5, 4], n_characters=100, n_iter=3, seed=42
            )
        # most characters lose their ancestral readings
        with self.assertWarns(UserWarning):
            gen_dialects(
                n_vocabularies=[5, 4], n_characters=100,
                n_iter=2, mutation_rate=0.3, seed=42,
            )


class SampleCharsTest(unittest.TestCase):
    def setUp(self):
        self.data, self.tree = gen_dialects(
            features=["initial", "final"],
            n_vocabularies=[5, 4], n_characters=200,
            max_dialects=4, n_iter=2, seed=42,
        )
        self.modern = set(modern_of(self.tree)["did"])
        # the sampling parameters are given explicitly so the test
        # never depends on the defaults
        self.options = {
            "beta_alpha": 2.0, "beta_beta": 6.0, "dirichlet_alpha": 2.0,
        }

    def test_samples_modern_records_only(self):
        # the historical dialects are untouched and every sampled row
        # is a row of the modern records, whatever the draws are
        for seed in SEEDS:
            sampled = sample_chars(self.data, self.tree, seed=seed,
                                   **self.options)

            self.assertTrue(set(sampled["did"]) <= self.modern)
            self.assertNotIn("d0_0", set(sampled["did"]))
            self.assertFalse(sampled.duplicated(["did", "cid"]).any())
            for did, sub in sampled.groupby("did"):
                full_rows = set(
                    map(tuple, self.data[self.data["did"] == did]
                        .itertuples(index=False, name=None))
                )
                for row in sub.itertuples(index=False, name=None):
                    self.assertIn(row, full_rows)

    def test_sampled_counts_within_records(self):
        # each dialect records at most its own characters, and every
        # character is recorded by at most the dialects that hold it
        for seed in SEEDS:
            sampled = sample_chars(self.data, self.tree, seed=seed,
                                   **self.options)

            for did, sub in sampled.groupby("did"):
                self.assertLessEqual(len(sub), 200)
            coverage = sampled.groupby("cid")["did"].nunique()
            self.assertTrue((coverage <= len(self.modern)).all())

    def test_reproducible_with_seed(self):
        first = sample_chars(self.data, self.tree, seed=7, **self.options)
        second = sample_chars(self.data, self.tree, seed=7, **self.options)

        assert_frame_equal(first, second)

    def test_validation(self):
        with self.assertRaises(ValueError):
            sample_chars(self.data, self.tree, beta_alpha=0.0)
        with self.assertRaises(ValueError):
            sample_chars(self.data, self.tree, beta_beta=0.0)
        with self.assertRaises(ValueError):
            sample_chars(self.data, self.tree, dirichlet_alpha=-0.1)

    def test_parameter_warnings(self):
        # the expected coverage of the beta distribution is very low
        with self.assertWarns(UserWarning):
            sample_chars(
                self.data, self.tree, beta_alpha=0.1, beta_beta=1.0,
                seed=7,
            )


class GenDatasetTest(unittest.TestCase):
    def test_returns_dataset_and_tree(self):
        dataset, tree = small()

        self.assertIsInstance(dataset, datasets.Dataset)
        self.assertIsInstance(tree, pandas.DataFrame)
        self.assertListEqual(list(tree.columns), TREE_COLUMNS)

    def test_long_form_format(self):
        for seed in SEEDS:
            dataset, tree = small(seed=seed)

            data = dataset._data
            self.assertListEqual(list(data.columns), COLUMNS)
            # only the modern dialects are sampled and no character is
            # recorded twice by one dialect
            self.assertTrue(set(data["did"]) <= set(modern_of(tree)["did"]))
            for did, sub in data.groupby("did"):
                self.assertLessEqual(len(sub), 300)
                self.assertFalse(sub.duplicated("cid").any())

    def test_reproducible_with_seed(self):
        first, first_tree = small()
        second, second_tree = small()

        assert_frame_equal(first._data, second._data)
        assert_frame_equal(first_tree, second_tree)

    def test_auto_encoding(self):
        dataset, _ = small()

        data = dataset._data
        self.assertTrue(data["cid"].str.fullmatch(r"c\d+").all())
        self.assertTrue((data["cid"] == data["character"]).all())
        # phones auto-encode as i1 ... / f1 ... / t1 ... since the
        # initial, final and tone values are real phones
        self.assertTrue(data["initial"].str.fullmatch(r"i\d+").all())
        self.assertTrue(data["final"].str.fullmatch(r"f\d+").all())
        self.assertTrue(data["tone"].str.fullmatch(r"t\d+").all())

    def test_explicit_names(self):
        characters = ["之", "乎", "者", "也"]
        dataset, _ = small(
            characters=characters,
            initials=["p", "pʰ", "b"],
            finals=["a", "o", "ə"],
            tones=["1", "2", "3", "4"],
            n_iter=2,
        )

        data = dataset._data
        self.assertLessEqual(set(data["character"]), set(characters))
        self.assertTrue(data["cid"].str.fullmatch(r"c\d+").all())
        self.assertTrue(set(data["initial"]) <= {"p", "pʰ", "b"})
        self.assertTrue(set(data["final"]) <= {"a", "o", "ə"})
        self.assertTrue(set(data["tone"]) <= {"1", "2", "3", "4"})

    def test_count_disagreement_raises(self):
        with self.assertRaises(ValueError):
            small(n_initial=2, initials=["p", "pʰ", "b"])
        with self.assertRaises(ValueError):
            small(n_characters=5, characters=["之", "乎"])

    def test_max_dialects_cap(self):
        dataset, tree = small(max_dialects=5, n_iter=4)

        dialects = dataset._data["did"].unique()
        self.assertLessEqual(len(dialects), 5)
        self.assertEqual(set(dialects), set(modern_of(tree)["did"]))

    def test_n_dialects_subsets_generated_dialects(self):
        # same seed, same generated dialects: the drawn dialects are a
        # subset of the generated ones whatever the draw is
        for seed in SEEDS:
            full_tree = small(seed=seed)[1]
            dataset, tree = small(n_dialects=5, seed=seed)

            self.assertEqual(len(tree), 5)
            self.assertTrue(set(tree["did"]) <= set(full_tree["did"]))
            # the dataset holds the records of the deepest sampled
            # level only
            levels = levels_of(tree)
            self.assertEqual(
                set(dataset._data["did"]),
                set(tree.loc[levels == levels.max(), "did"]),
            )

        # drawing every generated dialect keeps them all
        _, tree = small(n_dialects=15)
        self.assertEqual(len(tree), 15)

    def test_n_dialects_validation(self):
        # zero or negative draws are rejected
        with self.assertRaises(ValueError):
            small(n_dialects=0)

    def test_n_dialects_exceeds_generated_dialects(self):
        # a draw larger than the generated dialects keeps them all
        # with a warning
        with self.assertWarns(UserWarning):
            dataset, tree = small(n_dialects=10_000)

        self.assertEqual(len(tree), 15)

    def test_validation(self):
        with self.assertRaises(ValueError):
            small(n_initial=1)
        with self.assertRaises(ValueError):
            small(max_dialects=0)
        with self.assertRaises(ValueError):
            small(n_iter=0)
        with self.assertRaises(ValueError):
            small(n_iter=8, n_tone=8)
        with self.assertRaises(ValueError):
            small(mutation_rate=0.0)
        with self.assertRaises(ValueError):
            small(mutation_rate=1.0)
        with self.assertRaises(ValueError):
            small(n_characters=0)
        with self.assertRaises(ValueError):
            small(sample_beta_alpha=0.0)


if __name__ == "__main__":
    unittest.main()
