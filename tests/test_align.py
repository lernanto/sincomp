# -*- coding: utf-8 -*-

"""
测试对齐多个数据集中的多音字
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import numpy
import os
import pandas
import unittest

import sincomp.align
import sincomp.datasets


data_dir = os.path.join(os.path.dirname(__file__), 'data')


class TestAlign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.data1 = sincomp.datasets.FileDataset(
            data_dir=os.path.join(data_dir, 'custom_dataset1')
        )
        cls.data2 = sincomp.datasets.FileDataset(
            data_dir=os.path.join(data_dir, 'custom_dataset2')
        )

    def test_align_svd(self):
        chars, (charmap1, charmap2) = sincomp.align.align(
            (self.data1, None),
            (self.data2, None),
            encoder='svd',
            embedding_size=10
        )
        self.assertEqual(charmap1.nunique(), charmap1.shape[0])
        self.assertEqual(charmap2.nunique(), charmap2.shape[0])

    def test_align_fm(self):
        chars, (charmap1, charmap2) = sincomp.align.align(
            (self.data1, None),
            (self.data2, None),
            encoder='fm'
        )
        self.assertEqual(charmap1.nunique(), charmap1.shape[0])
        self.assertEqual(charmap2.nunique(), charmap2.shape[0])

    def test_align_no_cid(self):
        chars1 = self.data1.loc[:, ['cid', 'character']].drop_duplicates() \
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


if __name__ == '__main__':
    unittest.main()