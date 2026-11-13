# -*- coding: utf-8 -*-

"""Tests for the public dataset facade :mod:`sincomp.datasets`.

The tests verify the registry of the predefined datasets, loading a
custom dataset from a directory or a single CSV file through ``get``,
the random dataset generation facade ``make_dataset``, and the cache
directory attributes derived from the ``SINCOMP_CACHE`` environment
variable. No test requires network access.
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"

import os
import tempfile
import unittest
import unittest.mock

import sincomp.datasets


data_dir = os.path.join(os.path.dirname(__file__), "data")


def setUpModule():
    """Point the cache to a temporary directory and reload the module."""
    global env_patcher

    env_patcher = unittest.mock.patch.dict(os.environ, {
        "SINCOMP_CACHE": os.path.join(tempfile.TemporaryDirectory().name, "cache"),
    })
    env_patcher.start()

    import importlib
    importlib.reload(sincomp.datasets)


def tearDownModule():
    env_patcher.stop()

    import importlib
    importlib.reload(sincomp.datasets)


class TestPredefined(unittest.TestCase):
    """Test the registry and the lookup of the predefined datasets."""

    def test_predefined_lists_all_registered_datasets(self):
        # every predefined dataset exposes a unique name
        datasets = sincomp.datasets.predefined()
        self.assertEqual(len(datasets), 4)
        self.assertEqual(len({d.name for d in datasets}), 4)

    def test_get_returns_predefined_datasets_by_name(self):
        # the lookup accepts the registered names and their aliases
        self.assertIs(sincomp.datasets.get("CCR"), sincomp.datasets.ccr)
        self.assertIs(sincomp.datasets.get("ccr"), sincomp.datasets.ccr)
        self.assertIs(sincomp.datasets.get("xiaoxue"), sincomp.datasets.ccr)
        self.assertIs(sincomp.datasets.get("MCPDict"), sincomp.datasets.mcpdict)
        self.assertIs(
            sincomp.datasets.get("beidazihui"), sincomp.datasets.beidazihui
        )
        self.assertIs(
            sincomp.datasets.get("zhongguoyuyan"), sincomp.datasets.zhongguoyuyan
        )

    def test_get_unknown_name_returns_none(self):
        # a name that is neither registered nor an existing path warns
        # and returns None
        self.assertIsNone(sincomp.datasets.get("no_such_dataset"))


class TestGetCustom(unittest.TestCase):
    """Test loading custom datasets from local files and directories."""

    def test_get_directory(self):
        # a directory of per-dialect CSV files loads as a file dataset
        # whose records hold the standard pronunciation columns
        dataset = sincomp.datasets.get(os.path.join(data_dir, "custom_dataset1"))
        self.assertIsNotNone(dataset)
        self.assertListEqual(
            sorted(dataset.data.columns),
            ["character", "cid", "did", "final", "initial", "tone"],
        )

    def test_get_csv_file(self):
        # a single CSV file loads as a dataset named after the file
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "custom.csv")
            with open(path, "w", encoding="utf-8", newline="\n") as f:
                f.write("did,cid,initial,final,tone\n")
                f.write("d1,c1,p,a,1\n")
                f.write("d2,c1,b,an,2\n")

            dataset = sincomp.datasets.get(path)
            self.assertEqual(dataset.name, "custom")
            self.assertEqual(len(dataset.data), 2)

    def test_get_missing_path_returns_none(self):
        self.assertIsNone(
            sincomp.datasets.get(os.path.join(data_dir, "no_such_dir"))
        )


class TestMakeDataset(unittest.TestCase):
    """Test the random dataset generation facade."""

    def test_make_dataset_returns_long_form_records(self):
        # the simulated dataset holds the standard long-form columns
        dataset = sincomp.datasets.make_dataset(
            n_initial=3, n_final=3, n_tone=2, n_characters=10,
            max_dialects=4, n_iter=2, seed=42,
        )
        self.assertListEqual(
            sorted(dataset.data.columns),
            ["character", "cid", "did", "final", "initial", "tone"],
        )
        self.assertGreater(len(dataset.data), 0)

    def test_make_dataset_requires_feature_inventories(self):
        # without any feature inventory or character list the counts
        # cannot be inferred and the call is rejected
        with self.assertRaises(ValueError):
            sincomp.datasets.make_dataset(seed=42)


class TestCacheDirectories(unittest.TestCase):
    """Test the cache directory attributes set at import time."""

    def test_cache_dir_follows_environment(self):
        # cache_dir is derived from SINCOMP_CACHE and dataset_dir sits
        # inside it
        self.assertEqual(
            sincomp.datasets.dataset_dir,
            os.path.join(sincomp.datasets.cache_dir, "datasets"),
        )
        self.assertEqual(
            sincomp.datasets.cache_dir,
            os.environ["SINCOMP_CACHE"],
        )


if __name__ == "__main__":
    unittest.main()
