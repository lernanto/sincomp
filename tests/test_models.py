# -*- coding: utf-8 -*-

"""
测试编码器模型
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import os
import unittest

import numpy
import sklearn.preprocessing
import tensorflow as tf

import sincomp.models

from common import data_dir, setUpModule, tearDownModule, tmp_dir


def encode_data(data) -> tuple[
    numpy.ndarray[int],
    numpy.ndarray[int],
    list[int],
    list[int],
    list[int]
]:
    encoder = sklearn.preprocessing.OrdinalEncoder(
        dtype=numpy.int32,
        encoded_missing_value=-1
    )
    data = encoder.fit_transform(
        data[['did', 'cid', 'character', 'initial', 'final', 'tone']]
    ) + 1

    return (
        data[:, :3],
        data[:, 3:],
        [len(c) + 1 for c in encoder.categories_[:1]],
        [len(c) + 1 for c in encoder.categories_[1:3]],
        [len(c) + 1 for c in encoder.categories_[3:]]
    )


class TestBilinearEncoder(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        import sincomp.datasets

        data = sincomp.datasets.FileDataset(
            path=os.path.join(data_dir, 'custom_dataset1')
        )
        (
            features,
            targets,
            cls.dialect_vocab_sizes,
            cls.char_vocab_sizes,
            cls.target_vocab_sizes
        ) = encode_data(data)

        cls.dataset = tf.data.Dataset.from_tensor_slices((
            (features[:, :1], features[:, 1:]),
            targets
        ))

    def setUp(self):
        super().setUp()

        self.model = sincomp.models.BilinearEncoder(
            self.dialect_vocab_sizes,
            self.char_vocab_sizes,
            self.target_vocab_sizes,
            missing_id=0
        )
        self.optimizer = tf.keras.optimizers.SGD()

    def test_call(self):
        outputs = self.model(tf.convert_to_tensor([[1]]), tf.convert_to_tensor([[1, 0]]))

        self.assertEqual(len(outputs), 3)
        for o, n in zip(outputs, self.target_vocab_sizes):
            self.assertIsInstance(o, tf.Tensor)
            self.assertListEqual(o.shape.as_list(), [1, n])

    def test_train(self):
        loss, acc = self.model.train(self.optimizer, self.dataset.batch(10))

        self.assertIsInstance(loss, tf.Tensor)
        self.assertListEqual(loss.shape.as_list(), [])
        self.assertIsInstance(acc, tf.Tensor)
        self.assertListEqual(acc.shape.as_list(), [len(self.target_vocab_sizes)])

    def test_evaluate(self):
        loss, acc = self.model.evaluate(self.dataset.batch(10))

        self.assertIsInstance(loss, tf.Tensor)
        self.assertListEqual(loss.shape.as_list(), [])
        self.assertIsInstance(acc, tf.Tensor)
        self.assertListEqual(acc.shape.as_list(), [len(self.target_vocab_sizes)])

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