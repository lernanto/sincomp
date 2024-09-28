# -*- coding: utf-8 -*-

"""
测试预处理方言读音数据函数
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import os
import pandas
import unittest
import sincomp.preprocess

from common import data_dir, setUpModule, tearDownModule


class TestPreprocess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        import sincomp.datasets
        cls.data = sincomp.datasets.FileDataset(
            path=os.path.join(data_dir, 'custom_dataset1')
        )

    def test_clean_ipa(self):
        clean = sincomp.preprocess.clean_ipa(self.data['initial'])
        self.assertEqual(clean.shape, self.data['initial'].shape)

    def test_clean_initial(self):
        clean = sincomp.preprocess.clean_initial(self.data['initial'])
        self.assertEqual(clean.shape, self.data['initial'].shape)

    def test_clean_final(self):
        clean = sincomp.preprocess.clean_final(self.data['final'])
        self.assertEqual(clean.shape, self.data['final'].shape)

    def test_clean_tone(self):
        clean = sincomp.preprocess.clean_tone(self.data['tone'])
        self.assertEqual(clean.shape, self.data['tone'].shape)

    def test_transform(self):
        output = sincomp.preprocess.transform(
            self.data,
            index='cid',
            columns='did'
        )
        self.assertEqual(
            output.shape[0],
            self.data['cid'].value_counts().shape[0]
        )
        self.assertTrue(output.notna().any(axis=0).all())

    def test_impute(self):
        data = sincomp.preprocess.transform(
            self.data.data.sample(frac=0.7),
            index='cid',
            columns='did',
            values=['initial', 'final', 'tone'],
            aggfunc=lambda x: ' '.join(x.dropna())
        ).replace('', pandas.NA)
        imputed = sincomp.preprocess.impute(data)

        self.assertListEqual(data.index.to_list(), imputed.index.to_list())
        self.assertListEqual(data.columns.to_list(), imputed.columns.to_list())
        self.assertTrue(imputed.notna().all(axis=None))

        for c in data.columns:
            self.assertSetEqual(
                set(' '.join(data[c].dropna()).split()),
                set(' '.join(imputed[c].dropna()).split()),
            )
