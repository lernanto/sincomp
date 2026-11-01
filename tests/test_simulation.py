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


import unittest

import numpy
import pandas
from pandas.testing import assert_frame_equal

from sincomp import datasets
from sincomp.simulation import (
    _generate,
    _mutate,
    gen_dialects,
    gen_dataset,
    sample_chars,
)


COLUMNS = ["did", "cid", "character", "initial", "final", "tone"]
TREE_COLUMNS = [
    "did", "parent", "left", "right",
    "changed_feature", "source", "target",
]
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
    if "n_characters" not in kwargs and "characters" not in kwargs:
        options["n_characters"] = 300
    for count, names, size in (
        ("n_initial", "initials", 20),
        ("n_final", "finals", 20),
        ("n_tone", "tones", 8),
    ):
        if count not in kwargs and names not in kwargs:
            options[count] = size
    options.update(kwargs)
    return gen_dataset(**options)


def levels_of(tree):
    """Return the generation level of every recorded dialect."""

    return tree["did"].str[1:].str.split("_").str[0].astype(int)


def modern_of(tree):
    """Return the tree rows of the deepest-level dialects."""

    levels = levels_of(tree)
    return tree.loc[levels == levels.max()]


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

    def test_out_of_range_rate_raises(self):
        # any rate outside [0, 1] is rejected
        with self.assertRaises(ValueError):
            _mutate(
                numpy.array([0, 1] * 50, dtype=int), 1.5,
                numpy.random.default_rng(42),
            )

    def test_single_value_never_mutates(self):
        indices = numpy.zeros(100, dtype=int)
        _mutate(indices, 0.5, numpy.random.default_rng(42))
        self.assertTrue((indices == 0).all())

    def test_mutations_follow_matrix_weights(self):
        # with a transition matrix, every candidate redraws from the
        # other present values with the weights of the row of its
        # current value: the row that puts all weight on another
        # present value moves every character there, while the row
        # without mass on another present value cannot mutate
        indices = numpy.zeros(100, dtype=int)
        indices[::2] = 1
        matrix = numpy.array([[0.0, 1.0], [0.0, 1.0]])
        _mutate(indices, 1.0, numpy.random.default_rng(42), matrix)

        self.assertTrue((indices == 1).all())


class GenerateTest(unittest.TestCase):
    def test_no_change_returns_none_feature_source_and_target(self):
        values = {0: numpy.array([0, 1, 0, 1])}
        matrices = [numpy.array([[0.0, 1.0], [1.0, 0.0]])]

        child, f, source, target = _generate(
            values, [2], matrices, 0.0,
            numpy.random.default_rng(42), change=False,
        )

        self.assertTrue((child[0] == values[0]).all())
        self.assertIsNone(f)
        self.assertIsNone(source)
        self.assertIsNone(target)

    def test_change_returns_feature_source_and_target(self):
        values = {0: numpy.array([0, 1, 0, 1])}
        matrices = [numpy.array([[0.0, 1.0], [1.0, 0.0]])]

        child, f, source, target = _generate(
            values, [2], matrices, 0.0,
            numpy.random.default_rng(42), change=True,
        )

        self.assertEqual(f, 0)
        self.assertNotEqual(source, target)
        # the source value is replaced with the target wherever held
        self.assertTrue((child[0] != source).all())

    def test_target_equal_source_raises(self):
        values = {0: numpy.zeros(4, dtype=int)}
        # the row of the only held value points back to itself
        matrices = [numpy.array([[1.0, 0.0], [1.0, 0.0]])]

        with self.assertRaises(ValueError):
            _generate(
                values, [2], matrices, 0.0,
                numpy.random.default_rng(42), change=True,
            )


class GenDialectsTest(unittest.TestCase):
    def test_returns_dialect_numbers_and_survivor_arrays(self):
        for seed in SEEDS:
            (
                dialect_numbers, numbers, values, changes,
            ) = gen_dialects(
                100, 2, n_categories=[5, 4],
                n_iter=3, max_dialects=4, seed=seed,
            )

            # every conceived dialect is returned in one ordered
            # array, cut children included: the population grows
            # 1 -> 2 -> 4 and is capped at 4
            self.assertIsInstance(dialect_numbers, numpy.ndarray)
            self.assertEqual(len(dialect_numbers), 11)
            self.assertTrue((numpy.diff(dialect_numbers) > 0).all())
            # the generation of a dialect follows from its number, so
            # no per-generation information is lost: level g holds
            # min(2^g, 4) conceived dialects
            levels = [
                (number + 1).bit_length() - 1
                for number in dialect_numbers.tolist()
            ]
            for level in range(4):
                self.assertEqual(levels.count(level), min(2 ** level, 4))
            # every conceived dialect descends from an earlier one by
            # the parent number rule
            for number in dialect_numbers.tolist()[1:]:
                self.assertIn(
                    (number - 1) // 2, set(dialect_numbers.tolist())
                )

            self.assertIsInstance(numbers, numpy.ndarray)
            self.assertIsInstance(values, numpy.ndarray)
            # only the lineages surviving to the deepest level are
            # generated, so the count varies with the draw: at least
            # the modern dialects and at most all the conceived ones
            self.assertTrue(4 <= len(numbers) <= 11)
            self.assertEqual(values.shape, (len(numbers), 100, 2))
            self.assertTrue(numpy.issubdtype(numbers.dtype, numpy.integer))
            self.assertEqual(changes.shape, (len(numbers), 3))
            self.assertTrue(numpy.issubdtype(changes.dtype, numpy.integer))
            self.assertEqual(numbers[0], 0)
            self.assertListEqual(changes[0].tolist(), [-1, -1, -1])

            number_set = set(numbers.tolist())
            self.assertEqual(len(number_set), len(numbers))
            # the parent number of a node follows from its own
            for number in numbers.tolist()[1:]:
                self.assertIn((number - 1) // 2, number_set)
            # the survivors are among the conceived dialects
            self.assertLessEqual(number_set, set(dialect_numbers.tolist()))
            # the deepest-level dialects fill the population cap and
            # every right child carries a rule change while the root
            # and the left children inherit unchanged
            modern = [
                number for number in numbers.tolist()
                if (number + 1).bit_length() - 1 == 3
            ]
            self.assertEqual(len(modern), 4)
            for number, change in zip(numbers.tolist(), changes.tolist()):
                if number > 0 and number % 2 == 0:
                    self.assertGreaterEqual(change[0], 0, number)
                    self.assertNotEqual(change[1], change[2], number)
                else:
                    self.assertEqual(change, [-1, -1, -1], number)

    def test_population_per_level(self):
        # only the lineages surviving to the deepest level are
        # generated, so level k holds at most min(2^k, max_dialects)
        # dialects, and the deepest level fills the cap
        for seed in SEEDS:
            _, numbers, _, _ = gen_dialects(
                100, 2, n_categories=[5, 4],
                n_iter=3, max_dialects=4, seed=seed,
            )

            levels = numpy.floor(numpy.log2(numbers + 1)).astype(int)
            for level in range(4):
                self.assertLessEqual(
                    (levels == level).sum(), min(2 ** level, 4)
                )
            self.assertEqual((levels == 3).sum(), 4)

    def test_reproducible_with_seed(self):
        first = gen_dialects(
            100, 2, n_categories=[5, 4],
            n_iter=3, max_dialects=4, seed=42,
        )
        second = gen_dialects(
            100, 2, n_categories=[5, 4],
            n_iter=3, max_dialects=4, seed=42,
        )

        for first_array, second_array in zip(first, second):
            numpy.testing.assert_array_equal(first_array, second_array)

    def test_explicit_values(self):
        # the caller owns the code translation: the generated value
        # codes decode into the names of the caller's own vocabulary
        vocabularies = [["x", "y", "z"], ["1", "2", "3"]]
        for seed in SEEDS:
            _, _, values, _ = gen_dialects(
                100, 2, n_categories=[3, 3],
                n_iter=2, max_dialects=4, seed=seed,
            )
            decoded = [
                numpy.asarray(vocab)[values[:, :, f]]
                for f, vocab in enumerate(vocabularies)
            ]

            self.assertTrue(set(decoded[0].reshape(-1)) <= {"x", "y", "z"})
            self.assertTrue(set(decoded[1].reshape(-1)) <= {"1", "2", "3"})

    def test_values_seed_the_ancestor(self):
        # the given codes are the ancestor values: the root dialect
        # holds exactly them, whatever the seed is
        values = numpy.array([
            [0, 0],
            [1, 1],
            [0, 0],
            [2, 2],
        ])
        for seed in SEEDS:
            _, _, survived, _ = gen_dialects(
                4, 2, values=values,
                n_iter=2, max_dialects=4, seed=seed,
            )

            numpy.testing.assert_array_equal(survived[0], values)

    def test_values_may_not_cover_the_vocabulary(self):
        # the vocabulary may be larger than the observed values; the
        # two features hold two distinct values each, so one merge
        # can never exhaust both of them
        values = numpy.array([
            [0, 0],
            [1, 1],
            [0, 0],
            [1, 1],
        ])
        for seed in SEEDS:
            _, _, survived, _ = gen_dialects(
                4, 2, n_categories=[4, 4],
                values=values, n_iter=2, max_dialects=4, seed=seed,
            )

            for f in range(2):
                self.assertTrue(
                    ((survived[:, :, f] >= 0)
                     & (survived[:, :, f] < 4)).all()
                )

    def test_infer_from_values(self):
        # every structural argument is inferred from the values: the
        # number of features and the vocabulary sizes, and the root
        # holds the given codes exactly
        values = numpy.array([
            [0, 0],
            [1, 1],
            [2, 2],
            [3, 3],
        ])
        _, numbers, survived, _ = gen_dialects(
            4, values=values, n_iter=1, max_dialects=4, seed=42,
        )

        self.assertEqual(survived.shape[1:], (4, 2))
        numpy.testing.assert_array_equal(survived[0], values)

    def test_n_features_inferred(self):
        # n_features is inferred from whichever argument carries it
        _, _, survived, _ = gen_dialects(
            100, n_categories=[5, 4],
            n_iter=2, max_dialects=4, seed=42,
        )
        self.assertEqual(survived.shape[2], 2)
        _, _, survived, _ = gen_dialects(
            100, n_categories=[3, 3],
            n_iter=2, max_dialects=4, seed=42,
        )
        self.assertEqual(survived.shape[2], 2)

    def test_n_features_mismatch_raises(self):
        with self.assertRaises(ValueError):
            gen_dialects(100, 3, n_categories=[3, 4], seed=42)
        with self.assertRaises(ValueError):
            gen_dialects(
                100, 3, values=numpy.array([[0, 1], [1, 0]]), seed=42,
            )
        with self.assertRaises(ValueError):
            gen_dialects(
                100, 2,
                values=numpy.array([[0, 1, 2], [1, 0, 1]]),
                seed=42,
            )

    def test_n_features_disagreement_raises(self):
        # two arguments that carry different feature counts are
        # rejected when n_features does not resolve the conflict
        with self.assertRaises(ValueError):
            gen_dialects(
                100,
                values=numpy.array([[0, 1, 2], [1, 0, 1]]),
                n_categories=[3, 4],
                seed=42,
            )

    def test_values_disagreement_raises(self):
        with self.assertRaises(ValueError):
            gen_dialects(
                5, 1,
                values=numpy.array([[0], [1]]),
                n_iter=1, max_dialects=4, seed=42,
            )
        with self.assertRaises(ValueError):
            gen_dialects(
                2, 1, n_categories=[1],
                values=numpy.array([[0], [1]]),
                n_iter=1, max_dialects=4, seed=42,
            )

    def test_changes_take_effect(self):
        # the source of a rule change is drawn from the values the
        # parent holds and is replaced everywhere in the carrying
        # dialect: mutations only redraw from present values and can
        # never reintroduce the source
        for seed in SEEDS:
            _, numbers, values, changes = gen_dialects(
                100, 2, n_categories=[5, 4],
                n_iter=3, max_dialects=4, seed=seed,
            )
            positions = {
                number: i for i, number in enumerate(numbers.tolist())
            }
            for i, (f, source, target) in enumerate(changes):
                if f < 0:
                    continue
                number = int(numbers[i])
                parent = positions[(number - 1) // 2]
                self.assertIn(
                    source, set(values[parent, :, f].tolist()), number
                )
                self.assertNotIn(
                    source, set(values[i, :, f].tolist()), number
                )
                self.assertIn(
                    target, set(values[i, :, f].tolist()), number
                )

    def test_max_dialects_cap(self):
        for seed in SEEDS:
            _, numbers, _, _ = gen_dialects(
                100, 2, n_categories=[5, 4],
                n_iter=3, max_dialects=5, seed=seed,
            )

            # only the surviving lineages are generated, so level k
            # holds at most min(2^k, max_dialects) of them, and the
            # deepest level fills the cap
            levels = numpy.floor(numpy.log2(numbers + 1)).astype(int)
            self.assertEqual((levels == 3).sum(), 5)
            for level in range(3):
                self.assertLessEqual(
                    (levels == level).sum(), min(2 ** level, 5)
                )

    def test_cap_extinction_keeps_population(self):
        _, numbers, _, _ = gen_dialects(
            100, 2, n_categories=[7, 5],
            n_iter=4, max_dialects=6, seed=42,
        )

        # the conceived children are cut to the cap before they are
        # settled, so the deepest level always holds 6 modern
        # dialects whatever the earlier lineages do, and the upper
        # levels hold their surviving ancestors only
        levels = numpy.floor(numpy.log2(numbers + 1)).astype(int)
        self.assertEqual((levels == 4).sum(), 6)
        for level in range(4):
            self.assertLessEqual(
                (levels == level).sum(), min(2 ** level, 6)
            )

    def test_single_dialect(self):
        _, numbers, survived, _ = gen_dialects(
            100, 2, n_categories=[5, 4],
            n_iter=3, max_dialects=1, seed=42,
        )

        # with a population cap of one, a single lineage continues
        # and one of the two children of every split dies out; only
        # the surviving lineage is generated, holding the full
        # character set once per ancestor level plus the modern one
        self.assertEqual(len(numbers), 4)
        self.assertEqual(survived.shape, (4, 100, 2))

    def test_requires_inventories(self):
        # the number of features and the vocabularies must be given
        # or inferable
        with self.assertRaises(ValueError):
            gen_dialects(100, seed=42)
        with self.assertRaises(ValueError):
            gen_dialects(100, 2, seed=42)
        with self.assertRaises(ValueError):
            gen_dialects(100, 2, n_categories=[3, 4, 5], seed=42)

    def test_validation(self):
        with self.assertRaises(ValueError):
            gen_dialects(100, 2, n_categories=[1, 4], seed=42)
        with self.assertRaises(ValueError):
            gen_dialects(100, 2, n_categories=[5, 4], max_dialects=0, seed=42)
        with self.assertRaises(ValueError):
            gen_dialects(100, 2, n_categories=[5, 4], n_iter=0, seed=42)
        with self.assertRaises(ValueError):
            gen_dialects(100, 2, n_categories=[5, 4], mutation_rate=0.0,
                         seed=42)
        with self.assertRaises(ValueError):
            gen_dialects(100, 2, n_categories=[5, 4], mutation_rate=1.0,
                         seed=42)
        with self.assertRaises(ValueError):
            gen_dialects(0, 2, n_categories=[5, 4], seed=42)

    def test_matrices_validation(self):
        # one matrix per feature is required, square with the
        # vocabulary size, finite, non-negative and row-normalized;
        # a non-zero diagonal only warns
        with self.assertRaises(ValueError):
            gen_dialects(
                100, 2, n_categories=[5, 4],
                transition_matrices=[numpy.eye(5)], seed=42,
            )
        with self.assertRaises(ValueError):
            gen_dialects(
                100, 2, n_categories=[5, 4],
                transition_matrices=[numpy.eye(5), numpy.eye(3)], seed=42,
            )
        with self.assertRaises(ValueError):
            gen_dialects(
                100, 2, n_categories=[5, 4],
                transition_matrices=[numpy.eye(5), -numpy.eye(4)], seed=42,
            )
        with self.assertRaises(ValueError):
            gen_dialects(
                100, 2, n_categories=[5, 4],
                transition_matrices=[numpy.eye(5), numpy.full((4, 4), 0.2)],
                seed=42,
            )

    def test_matrix_diagonal_warns_and_rejects_source_draw(self):
        # a non-zero diagonal warns, and a rule change that draws the
        # source itself as its target is rejected with an error
        with self.assertLogs("sincomp.simulation", level="WARNING") as seen:
            with self.assertRaises(ValueError):
                gen_dialects(
                    100, 2, n_categories=[5, 4],
                    n_iter=3, max_dialects=4,
                    transition_matrices=[numpy.eye(5), numpy.eye(4)],
                    seed=42,
                )
        self.assertTrue(
            any("assigns probability to the source" in message
                for message in seen.output)
        )

    def test_rule_change_target_may_leave_the_inventory(self):
        # the target of a rule change is drawn from the whole
        # vocabulary, so a change may replace one present value with
        # another one the dialect does not hold yet and leave the
        # number of distinct values untouched
        values = numpy.array([[0], [1], [0], [1]])
        matrices = [numpy.array([
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])]
        _, numbers, survived, changes = gen_dialects(
            4, 1, n_categories=[3],
            values=values, n_iter=1, max_dialects=2,
            mutation_rate=0.001, transition_matrices=matrices, seed=42,
        )

        self.assertTrue(((survived >= 0) & (survived < 3)).all())
        self.assertTrue((changes[:, 0] >= 0).any())
        for i, (f, source, target) in enumerate(changes):
            if f < 0:
                continue
            # the forced matrix only allows 0 > 2 and 1 > 0, and the
            # carrying node applies the change, so it holds the
            # target value
            self.assertEqual(f, 0)
            self.assertIn((source, target), ((0, 2), (1, 0)))
            self.assertIn(target, survived[i, :, f].tolist())

    def test_merge_down_to_one_value_warns(self):
        # a rule change may merge a feature down to a single value,
        # which is logged instead of raising
        with self.assertLogs("sincomp.simulation", level="WARNING") as seen:
            gen_dialects(
                100, 2, n_categories=[2, 4],
                n_iter=3, max_dialects=4, seed=0,
            )

        self.assertTrue(
            any("single value" in message for message in seen.output)
        )

    def test_parameter_warnings(self):
        # most characters lose their ancestral values
        with self.assertLogs("sincomp.simulation", level="WARNING") as seen:
            gen_dialects(
                100, 2, n_categories=[5, 4],
                n_iter=2, mutation_rate=0.3, seed=42,
            )
        self.assertTrue(
            any("keep their ancestral" in message for message in seen.output)
        )


class SampleCharsTest(unittest.TestCase):
    def test_returns_one_unique_index_array_per_dialect(self):
        for seed in SEEDS:
            sampled = sample_chars(200, 4, seed=seed)

            self.assertIsInstance(sampled, list)
            self.assertEqual(len(sampled), 4)
            for indices in sampled:
                self.assertIsInstance(indices, numpy.ndarray)
                self.assertEqual(indices.ndim, 1)
                self.assertTrue(numpy.issubdtype(indices.dtype, numpy.integer))
                self.assertTrue(((0 <= indices) & (indices < 200)).all())
                self.assertEqual(len(indices), len(numpy.unique(indices)))
                numpy.testing.assert_array_equal(indices, numpy.sort(indices))

    def test_reproducible_with_seed(self):
        first = sample_chars(200, 4, seed=7)
        second = sample_chars(200, 4, seed=7)

        self.assertEqual(len(first), len(second))
        for first_indices, second_indices in zip(first, second):
            numpy.testing.assert_array_equal(first_indices, second_indices)

    def test_validation(self):
        with self.assertRaises(ValueError):
            sample_chars(200, 4, beta_alpha=0.0)
        with self.assertRaises(ValueError):
            sample_chars(200, 4, beta_beta=0.0)
        with self.assertRaises(ValueError):
            sample_chars(200, 4, dirichlet_alpha=-0.1)
        with self.assertRaises(ValueError):
            sample_chars(0, 4)
        with self.assertRaises(ValueError):
            sample_chars(200, 0)

    def test_parameter_warnings(self):
        # the expected coverage of the beta distribution is very low
        with self.assertLogs("sincomp.simulation", level="WARNING"):
            sample_chars(
                200, 4, beta_alpha=0.1, beta_beta=1.0, seed=7,
            )


class GenDatasetTest(unittest.TestCase):
    def test_returns_dataset_and_tree(self):
        dataset, tree = small()

        self.assertIsInstance(dataset, datasets.Dataset)
        self.assertIsInstance(tree, pandas.DataFrame)
        self.assertListEqual(list(tree.columns), TREE_COLUMNS)

    def test_tree_reflects_lineage(self):
        for seed in SEEDS:
            _, tree = small(seed=seed)

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
                # within a level, the left child takes twice the
                # parent's position and the right child one more
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
            _, tree = small(seed=seed)

            # the change of a tree row is the rule change the dialect
            # carries from its parent: the root and the left children
            # inherit unchanged while every right child applies one
            for row in tree.itertuples(index=False):
                level, pos = row.did[1:].split("_")
                number = 2 ** int(level) - 1 + int(pos)
                if number > 0 and number % 2 == 0:
                    self.assertFalse(pandas.isna(row.changed_feature), row.did)
                    self.assertFalse(pandas.isna(row.source), row.did)
                    self.assertFalse(pandas.isna(row.target), row.did)
                    self.assertNotEqual(row.source, row.target, row.did)
                else:
                    self.assertTrue(pandas.isna(row.changed_feature), row.did)
                    self.assertTrue(pandas.isna(row.source), row.did)
                    self.assertTrue(pandas.isna(row.target), row.did)

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

        # the sampled dialects are modern ones and fill the cap; the
        # cut children of the last split share the deepest level
        # without records
        dialects = set(dataset._data["did"])
        levels = levels_of(tree)
        deepest = set(tree.loc[levels == levels.max(), "did"])
        self.assertLessEqual(len(dialects), 5)
        self.assertLessEqual(dialects, deepest)
        self.assertEqual(len(dialects & deepest), 5)

    def test_n_dialects_subsets_generated_dialects(self):
        # same seed, same generated dialects: the drawn dialects are a
        # subset of the generated ones whatever the draw is
        for seed in SEEDS:
            full_tree = small(seed=seed)[1]
            dataset, tree = small(n_dialects=5, seed=seed)

            assert_frame_equal(tree, full_tree)
            self.assertTrue(
                set(dataset._data["did"]).issubset(
                    set(modern_of(tree)["did"])
                )
            )
            self.assertEqual(dataset._data["did"].nunique(), 5)

        # asking for more modern dialects than were generated keeps all
        _, tree = small(n_dialects=15)
        self.assertEqual(len(modern_of(tree)), 8)

    def test_n_dialects_validation(self):
        # zero or negative draws are rejected
        with self.assertRaises(ValueError):
            small(n_dialects=0)

    def test_n_dialects_exceeds_generated_dialects(self):
        # a draw larger than the generated dialects keeps them all
        # with a logged warning
        with self.assertLogs("sincomp.simulation", level="WARNING"):
            dataset, tree = small(n_dialects=10_000)

        self.assertEqual(len(tree), 15)
        self.assertEqual(dataset._data["did"].nunique(), 8)

    def test_count_or_name_list_is_required(self):
        options = {
            "n_characters": 10,
            "max_dialects": 2,
            "n_iter": 2,
        }
        for missing_count, other_counts in (
            ("n_initial", {"n_final": 5, "n_tone": 5}),
            ("n_final", {"n_initial": 5, "n_tone": 5}),
            ("n_tone", {"n_initial": 5, "n_final": 5}),
        ):
            with self.subTest(feature=missing_count):
                with self.assertRaises(ValueError):
                    gen_dataset(**options, **other_counts)
        with self.assertRaises(ValueError):
            gen_dataset(
                n_initial=5, n_final=5, n_tone=5,
                max_dialects=2, n_iter=2,
            )

    def test_explicit_name_lists_infer_counts(self):
        characters = ["甲", "乙", "丙", "丁"]
        dataset, _ = gen_dataset(
            characters=characters,
            initials=["p", "b", "m", "t", "d"],
            finals=["a", "o", "i", "u", "e"],
            tones=["1", "2", "3", "4", "5"],
            max_dialects=2, n_iter=2, seed=42,
        )

        data = dataset._data
        self.assertLessEqual(set(data["character"]), set(characters))
        self.assertTrue(set(data["initial"]) <= {"p", "b", "m", "t", "d"})
        self.assertTrue(set(data["final"]) <= {"a", "o", "i", "u", "e"})
        self.assertTrue(set(data["tone"]) <= {"1", "2", "3", "4", "5"})

    def test_validation(self):
        with self.assertRaises(ValueError):
            small(n_initial=1)
        with self.assertRaises(ValueError):
            small(max_dialects=0)
        with self.assertRaises(ValueError):
            small(n_iter=0)
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
