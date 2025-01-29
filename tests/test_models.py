# -*- coding: utf-8 -*-

"""
测试编码器模型
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import os
import unittest

import numpy
import pandas
import tensorflow as tf

import sincomp.models

from common import data_dir, setUpModule, tearDownModule, tmp_dir


class TestProcessor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        import sincomp.datasets

        cls.data = sincomp.datasets.FileDataset(
            path=os.path.join(data_dir, 'custom_dataset1')
        ).data

    def setUp(self):
        super().setUp()

        self.processor = sincomp.models.Processor(
            [self.data['did'].drop_duplicates()],
            [self.data['cid'].drop_duplicates()],
            [
                self.data['initial'].drop_duplicates(),
                self.data['final'].drop_duplicates(),
                self.data['tone'].drop_duplicates()
            ]
        )

    def test_call(self):
        dialects = self.data[['did']]
        chars = self.data[['cid']]
        targets = self.data[['initial', 'final', 'tone']]
        dialect_ids, char_ids, target_ids = self.processor(dialects, chars, targets)

        self.assertIsInstance(dialect_ids, numpy.ndarray)
        self.assertTupleEqual(dialect_ids.shape, dialects.shape)
        self.assertIsInstance(dialect_ids[0, 0], numpy.integer)
        self.assertIsInstance(char_ids, numpy.ndarray)
        self.assertTupleEqual(char_ids.shape, chars.shape)
        self.assertIsInstance(char_ids[0, 0], numpy.integer)
        self.assertIsInstance(target_ids, numpy.ndarray)
        self.assertTupleEqual(target_ids.shape, targets.shape)
        self.assertIsInstance(target_ids[0, 0], numpy.integer)

    def test_call_na(self):
        dialect_ids, char_ids, target_ids = self.processor(
            [[pandas.NA]],
            [[pandas.NA]],
            [[pandas.NA, pandas.NA, pandas.NA]]
        )

        self.assertTrue(numpy.all(dialect_ids == 0))
        self.assertTrue(numpy.all(char_ids == 0))
        self.assertTrue(numpy.all(target_ids == 0))

class TestBilinearEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        import sincomp.datasets

        data = sincomp.datasets.FileDataset(
            path=os.path.join(data_dir, 'custom_dataset1')
        ).data
        cls.processor = sincomp.models.Processor(
            [data['did'].drop_duplicates()],
            [
                data['cid'].drop_duplicates(),
                data['character'].drop_duplicates()
            ],
            [
                data['initial'].drop_duplicates(),
                data['final'].drop_duplicates(),
                data['tone'].drop_duplicates()
            ]
        )
        dialects, chars, targets = cls.processor(
            data[['did']],
            data[['cid', 'character']],
            data[['initial', 'final', 'tone']]
        )
        cls.dataset = tf.data.Dataset.from_tensor_slices((
            (dialects, chars),
            targets
        ))

    def setUp(self):
        super().setUp()

        self.model = sincomp.models.BilinearEncoder(
            self.processor.dialect_vocab_sizes,
            self.processor.char_vocab_sizes,
            self.processor.target_vocab_sizes,
            missing_id=0
        )
        self.optimizer = tf.keras.optimizers.SGD()

    def test_call(self):
        outputs = self.model(
            tf.convert_to_tensor([[1]]),
            tf.convert_to_tensor([[1, 0]])
        )

        self.assertEqual(len(outputs), 3)
        for o, n in zip(outputs, self.processor.target_vocab_sizes):
            self.assertIsInstance(o, tf.Tensor)
            self.assertListEqual(o.shape.as_list(), [1, n])

    def test_train(self):
        loss, acc = self.model.train(self.optimizer, self.dataset.batch(10))

        self.assertIsInstance(loss, tf.Tensor)
        self.assertListEqual(loss.shape.as_list(), [])
        self.assertIsInstance(acc, tf.Tensor)
        self.assertListEqual(
            acc.shape.as_list(),
            [len(self.processor.target_vocab_sizes)]
        )

    def test_evaluate(self):
        loss, acc = self.model.evaluate(self.dataset.batch(10))

        self.assertIsInstance(loss, tf.Tensor)
        self.assertListEqual(loss.shape.as_list(), [])
        self.assertIsInstance(acc, tf.Tensor)
        self.assertListEqual(
            acc.shape.as_list(),
            [len(self.processor.target_vocab_sizes)]
        )

    def test_fit(self):
        self.model.fit(
            self.optimizer,
            self.dataset,
            self.dataset,
            epochs=2,
            batch_size=10,
            checkpoint_dir=os.path.join(tmp_dir, 'checkpoints'),
            log_dir=os.path.join(tmp_dir, 'log')
        )