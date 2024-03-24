# -*- coding: utf-8 -*-

"""
测试 SinComp 数据集相关功能
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import unittest
import unittest.mock
import os
import pandas

from common import data_dir, setUpModule, tearDownModule


class TestFileDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        import sincomp.datasets
        cls.dataset = sincomp.datasets.FileDataset(
            path=os.path.join(data_dir, 'custom_dataset1')
        )

    def test_file_dataset(self):
        self.assertEqual(len(self.dataset), 2)

    def test_data(self):
        data = self.dataset.data
        self.assertTrue(isinstance(data, pandas.DataFrame))
        for col in 'did', 'cid', 'initial', 'final', 'tone':
            self.assertTrue(col in data.columns)

    def test_filter(self):
        data = self.dataset.filter(['01']).data
        self.assertTrue(isinstance(data, pandas.DataFrame))
        for col in 'did', 'cid', 'initial', 'final', 'tone':
            self.assertTrue(col in data.columns)

    def test_sample(self):
        data = self.dataset.sample(n=1).data
        self.assertTrue(isinstance(data, pandas.DataFrame))
        for col in 'did', 'cid', 'initial', 'final', 'tone':
            self.assertTrue(col in data.columns)

    def test_shuffle(self):
        data = self.dataset.shuffle().data
        self.assertTrue(isinstance(data, pandas.DataFrame))
        for col in 'did', 'cid', 'initial', 'final', 'tone':
            self.assertTrue(col in data.columns)

    def test_append(self):
        import sincomp.datasets

        other = sincomp.datasets.FileDataset(
            path=os.path.join(data_dir, 'custom_dataset2')
        )
        output = self.dataset.append(other)

        self.assertTrue(isinstance(output, sincomp.datasets.FileDataset))
        self.assertEqual(len(output), len(self.dataset) + len(other))
        self.assertEqual(
            output.data.shape[0],
            self.dataset.data.shape[0] + other.shape[0]
        )


class TestCCRDataset(unittest.TestCase):
    def test_dialect_info(self):
        import sincomp.datasets

        info = sincomp.datasets.ccr.metadata['dialect_info']
        self.assertTrue(isinstance(info, pandas.DataFrame))
        self.assertGreater(info.shape[0], 0)
        for col in 'group', 'subgroup', 'cluster', 'subcluster', 'spot':
            self.assertTrue(col in info.columns)

    def test_char_info(self):
        import sincomp.datasets

        info = sincomp.datasets.ccr.metadata['char_info']
        self.assertTrue(isinstance(info, pandas.DataFrame))
        self.assertGreater(info.shape[0], 0)

    def test_load_data(self):
        import sincomp.datasets

        _, data = sincomp.datasets.ccr.load_data('C027')[0]
        self.assertTrue(isinstance(data, pandas.DataFrame))
        self.assertEqual(data.shape[0], 20)
        for col in 'did', 'cid', 'character', 'initial', 'final', 'tone':
            self.assertTrue(col in data.columns)


class TestMCPDictDataset(unittest.TestCase):
    def test_dialect_info(self):
        import sincomp.datasets

        info = sincomp.datasets.mcpdict.metadata['dialect_info']
        self.assertTrue(isinstance(info, pandas.DataFrame))
        self.assertEqual(info.shape[0], 2)
        for col in 'group', 'subgroup', 'cluster', 'subcluster', 'spot':
            self.assertTrue(col in info.columns)

    def test_data(self):
        import sincomp.datasets

        data = sincomp.datasets.mcpdict.data
        self.assertTrue(isinstance(data, pandas.DataFrame))
        self.assertEqual(data.shape[0], 40)
        for col in 'did', 'character', 'initial', 'final', 'tone':
            self.assertTrue(col in data.columns)


class TestZhongguoyuyanDataset(unittest.TestCase):
    def test_dialect_info(self):
        import sincomp.datasets

        info = sincomp.datasets.zhongguoyuyan.metadata['dialect_info']
        self.assertTrue(isinstance(info, pandas.DataFrame))
        self.assertEqual(info.shape[0], 2)
        for col in 'group', 'subgroup', 'cluster', 'subcluster', 'spot':
            self.assertTrue(col in info.columns)

    def test_char_info(self):
        import sincomp.datasets

        info = sincomp.datasets.zhongguoyuyan.metadata['char_info']
        self.assertTrue(isinstance(info, pandas.DataFrame))
        self.assertGreater(info.shape[0], 0)

    def test_data(self):
        import sincomp.datasets

        data = sincomp.datasets.zhongguoyuyan.data
        self.assertTrue(isinstance(data, pandas.DataFrame))
        self.assertEqual(data.shape[0], 40)
        for col in 'did', 'character', 'initial', 'final', 'tone':
            self.assertTrue(col in data.columns)
