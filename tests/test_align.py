# -*- coding: utf-8 -*-

"""
测试对齐多个数据集中的多音字
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import numpy
import os
import pandas
import unittest
import sincomp.preprocess
import sincomp.align

from common import data_dir, setUpModule, tearDownModule


class TestAlign(unittest.TestCase):
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
        self.assertEqual(chars1['label'].nunique(), chars1.shape[0])
        self.assertEqual(chars2['label'].nunique(), chars2.shape[0])
        cid = pandas.Index(chars1['label']).intersection(chars2['label'])
        self.assertTrue((chars1.set_index('label')['simplified'].reindex(cid)
            == chars2.set_index('label')['simplified'].reindex(cid)).all())

    def test_align_no_cid(self):
        chars1 = self.data1[['cid', 'character']].drop_duplicates() \
            .dropna(subset='cid').set_index('cid')['character']

        result = sincomp.align.align_no_cid(
            pandas.pivot_table(
                self.data1.data,
                values=['initial', 'final', 'tone'],
                index='cid',
                columns='did',
                aggfunc='first'
            ),
            chars1,
            None,
            self.data2
        )
        self.assertEqual(len(result), 1)

        labels, chars2, _ = result[0][0]
        self.assertEqual(labels.shape[0], chars2.shape[0])
        self.assertTrue(
            numpy.all(chars1.loc[labels][labels != None] == chars2[labels != None])
        )
