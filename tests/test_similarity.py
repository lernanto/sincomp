# -*- coding: utf-8 -*-

"""
测试计算方言之间相似度
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import unittest
import os
import numpy
import sincomp.preprocess

from common import data_dir, setUpModule, tearDownModule


class TestSimilarity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        import sincomp.datasets
        cls.data = sincomp.preprocess.transform(
            sincomp.datasets.FileDataset(
                path=os.path.join(data_dir, 'custom_dataset1')
            ),
            index='cid',
            values=['initial', 'final', 'tone'],
            aggfunc='first'
        )

    def test_chi2(self):
        import sincomp.similarity

        sim = sincomp.similarity.chi2(self.data)
        self.assertEqual(sim.shape, (self.data.columns.levels[0].shape[0],) * 2)
        self.assertTrue(numpy.all(numpy.isfinite(sim)))

    def test_entropy(self):
        import sincomp.similarity

        sim = sincomp.similarity.entropy(self.data)
        self.assertEqual(sim.shape, (self.data.columns.levels[0].shape[0],) * 2)
        self.assertTrue(numpy.all(numpy.isfinite(sim)))
