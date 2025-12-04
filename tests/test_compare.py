# -*- coding: utf-8 -*-

"""
本模块包含对 sincomp.compare 模块的单元测试。
测试内容包括：
    - 正常数据输出类型和形状
    - 单方言、单字符等极端情况
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import os
import pandas
import unittest

import sincomp.compare
import sincomp.datasets
import sincomp.preprocess


data_dir = os.path.join(os.path.dirname(__file__), 'data')


class TestCompare(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.data = sincomp.datasets.FileDataset(
            data_dir=os.path.join(data_dir, 'custom_dataset1')
        ).data[['did', 'cid', 'initial', 'final', 'tone']]
        cls.rules = pandas.read_json(
            os.path.join(data_dir, 'rules.json'),
            orient='records'
        )

    def test_compare(self):
        data = sincomp.preprocess.transform(
            self.data,
            index='cid',
            columns='did',
            values=['initial', 'final', 'tone'],
            aggfunc=lambda s: ' '.join(s.dropna())
        )
        compliances = sincomp.compare.compliance(data, self.rules)

        self.assertIsInstance(compliances, pandas.DataFrame)
        self.assertTupleEqual(compliances.shape, (2, 1))
        self.assertGreaterEqual(compliances.min(axis=None), 0)
        self.assertLessEqual(compliances.max(axis=None), 1)

    def test_compare_single_dialect(self):
        data = sincomp.preprocess.transform(
            self.data,
            index='cid',
            columns='did',
            values=['initial', 'final', 'tone'],
            aggfunc=lambda s: ' '.join(s.dropna())
        )
        compliances = sincomp.compare.compliance(data.iloc[:, :3], self.rules)

        self.assertIsInstance(compliances, pandas.DataFrame)
        self.assertTupleEqual(compliances.shape, (1, 1))

    def test_compare_characer_not_exist(self):
        data = sincomp.preprocess.transform(
            self.data,
            index='cid',
            columns='did',
            values=['initial', 'final', 'tone'],
            aggfunc=lambda s: ' '.join(s.dropna())
        )
        rules = self.rules.copy()
        rules.iloc[0]['cid1'].append('0023')
        sincomp.compare.compliance(data, self.rules)

    def test_compare_empty_character(self):
        data = sincomp.preprocess.transform(
            self.data,
            index='cid',
            columns='did',
            values=['initial', 'final', 'tone'],
            aggfunc=lambda s: ' '.join(s.dropna())
        )
        rules = self.rules.copy()
        rules.iloc[0]['cid1'].clear()
        compliances = sincomp.compare.compliance(data, self.rules)

        self.assertTrue(compliances.isna().all(axis=None))


if __name__ == '__main__':
    unittest.main()