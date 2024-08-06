# -*- coding: utf-8 -*-

"""
测试对齐多个数据集中的多音字
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import unittest
import os
import pandas
import sincomp.preprocess
import sincomp.align

from common import data_dir, setUpModule, tearDownModule


class TestSimilarity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        import sincomp.datasets
        cls.data1 = sincomp.datasets.FileDataset(
            path=os.path.join(data_dir, 'custom_dataset1')
        )
        cls.data2 = sincomp.datasets.FileDataset(
            path=os.path.join(data_dir, 'custom_dataset2')
        )

    def test_align(self):
        chars1, chars2 = sincomp.align.align((self.data1, None), (self.data2, None))
        self.assertTrue(chars1['label'].nunique() == chars1.shape[0])
        self.assertTrue(chars2['label'].nunique() == chars2.shape[0])
        cid = pandas.Index(chars1['label']).intersection(chars2['label'])
        self.assertTrue((chars1.set_index('label')['character'].reindex(cid)
            == chars2.set_index('label')['character'].reindex(cid)).all())
