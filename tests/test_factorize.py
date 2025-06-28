# -*- coding: utf-8 -*-

"""
本模块包含对 sincomp.factorize 模块的单元测试。
测试内容包括：
    - 正常数据输出类型和形状
    - 空 DataFrame 的异常处理
    - 单方言、单字符等极端情况
    - 嵌入维度变化的行为
    - 非法输入的异常处理
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import numpy
import os
import unittest

import sincomp.factorize
import sincomp.datasets


data_dir = os.path.join(os.path.dirname(__file__), 'data')


class TestFactorize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        cls.data = sincomp.datasets.FileDataset(
            data_dir=os.path.join(data_dir, 'custom_dataset1')
        ).data[['did', 'cid', 'initial', 'final', 'tone']]

    def test_factorize_output(self):
        char_num = self.data['cid'].nunique()
        dialect_num = self.data['did'].nunique()

        char_embs, phone_embs, chars, dialects, phones = sincomp.factorize.factorize(
            self.data, embedding_size=8, max_iter=2,
            min_dialects=1, min_characters=1
        )
        self.assertIsInstance(char_embs, numpy.ndarray)
        self.assertEqual(char_embs.shape[0], char_num)
        self.assertEqual(char_embs.shape[1], 8)
        self.assertIsInstance(phone_embs, numpy.ndarray)
        self.assertGreaterEqual(phone_embs.shape[0], 1)
        self.assertEqual(phone_embs.shape[1], 8)
        self.assertIsInstance(chars, numpy.ndarray)
        self.assertEqual(chars.shape[0], char_num)
        self.assertIsInstance(dialects, numpy.ndarray)
        self.assertEqual(dialects.shape[0], dialect_num)
        self.assertIsInstance(phones, list)
        self.assertEqual(len(phones), dialect_num)
        self.assertIsInstance(phones[0], list)
        self.assertEqual(len(phones[0]), 3)
        self.assertIsInstance(phones[0][0], numpy.ndarray)
        self.assertGreater(phones[0][0].shape[0], 0)

    def test_empty_dataframe(self):
        empty = self.data.iloc[0:0]
        with self.assertRaises(Exception):
            sincomp.factorize.factorize(empty)

    def test_single_dialect(self):
        single = self.data[self.data['did'] == self.data.iloc[0]['did']]
        char_embs, phone_embs, chars, dialects, phones = sincomp.factorize.factorize(
            single, embedding_size=2, max_iter=1,
            min_dialects=1, min_characters=1
        )
        self.assertEqual(dialects.shape[0], 1)

    def test_single_character(self):
        single = self.data[self.data['cid'] == self.data.iloc[0]['cid']]
        char_embs, phone_embs, chars, dialects, phones = sincomp.factorize.factorize(
            single, embedding_size=2, max_iter=1,
            min_dialects=1, min_characters=1
        )
        self.assertEqual(char_embs.shape[0], 1)
        self.assertEqual(chars.shape[0], 1)

    def test_minimal_embedding_size(self):
        char_embs, phone_embs, chars, dialects, phones = sincomp.factorize.factorize(
            self.data, embedding_size=1, max_iter=2,
            min_dialects=1, min_characters=1
        )
        self.assertEqual(char_embs.shape[1], 1)
        self.assertEqual(phone_embs.shape[1], 1)

    def test_high_embedding_size(self):
        char_embs, phone_embs, chars, dialects, phones = sincomp.factorize.factorize(
            self.data, embedding_size=1024, max_iter=1,
            min_dialects=1, min_characters=1
        )
        self.assertEqual(char_embs.shape[1], 1024)
        self.assertEqual(phone_embs.shape[1], 1024)

    def test_invalid_input(self):
        with self.assertRaises(Exception):
            sincomp.factorize.factorize(
                numpy.ndarray({'a': [1, 2], 'b': [3, 4]})
            )


if __name__ == '__main__':
    unittest.main()