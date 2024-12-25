# -*- coding: utf-8 -*-

"""
测试编码器模型
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import os
import unittest

import numpy
import sklearn.preprocessing
import torch
import torch.utils
import torch.utils.data

import sincomp.models

from common import data_dir, setUpModule, tearDownModule, tmp_dir


def encode_data(data) -> tuple[
    numpy.ndarray[int],
    numpy.ndarray[int],
    list[int],
    list[int],
    list[int]
]:
    feature_encoder = sklearn.preprocessing.OrdinalEncoder(
        dtype=int,
        encoded_missing_value=-1
    )
    target_encoder = sklearn.preprocessing.OrdinalEncoder(
        dtype=int,
        encoded_missing_value=-1
    )
    features = feature_encoder.fit_transform(data[['did', 'cid', 'character']]) + 1
    targets = target_encoder.fit_transform(data[['initial', 'final', 'tone']])

    return (
        features,
        targets,
        [len(c) + 1 for c in feature_encoder.categories_[:1]],
        [len(c) + 1 for c in feature_encoder.categories_[1:]],
        list(map(len, target_encoder.categories_))
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

        cls.dataset = torch.utils.data.StackDataset(
            torch.utils.data.TensorDataset(
                torch.as_tensor(features[:, :1]),
                torch.as_tensor(features[:, 1:])
            ),
            torch.as_tensor(targets, dtype=torch.long)
        )

    def setUp(self):
        super().setUp()

        self.model = sincomp.models.BilinearEncoder(
            self.dialect_vocab_sizes,
            self.char_vocab_sizes,
            self.target_vocab_sizes,
            missing_id=0,
            dropout=0.2
        )

    def test_forward(self):
        outputs = self.model.forward(torch.as_tensor([[1]]), torch.as_tensor([[1, 0]]))

        self.assertEqual(len(outputs), 3)
        for o, n in zip(outputs, self.target_vocab_sizes):
            self.assertIsInstance(o, torch.Tensor)
            self.assertTupleEqual(o.size(), (1, n))

    if torch.cuda.is_available():
        def test_to_cuda(self):
            device = torch.device('cuda:0')
            self.model.to(device)

            for param in self.model.parameters():
                self.assertEqual(param.device, device)


class TestMultiTargetLoss(unittest.TestCase):
    def test_forward(self):
        logits = [
            torch.as_tensor([[2.0, 1.0], [1.0, 2.0]]),
            torch.as_tensor([[1.0, 3.0], [2.0, 3.0]])
        ]
        targets = torch.as_tensor([[0, -1], [1, 0]])

        loss = sincomp.models.MultiTargetLoss()(logits, targets)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTupleEqual(loss.size(), ())

    def test_forward_weight(self):
        logits = [
            torch.as_tensor([[2.0, 1.0], [1.0, 2.0]]),
            torch.as_tensor([[1.0, 3.0], [2.0, 3.0]])
        ]
        targets = torch.as_tensor([[0, -1], [1, 0]])

        loss = sincomp.models.MultiTargetLoss(weight=[0.4, 0.6])(logits, targets)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTupleEqual(loss.size(), ())

    def test_forward_reduction(self):
        logits = [
            torch.as_tensor([[2.0, 1.0], [1.0, 2.0]]),
            torch.as_tensor([[1.0, 3.0], [2.0, 3.0]])
        ]
        targets = torch.as_tensor([[0, -1], [1, 0]])

        loss = sincomp.models.MultiTargetLoss(reduction='none')(logits, targets)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTupleEqual(loss.size(), (2,))

    def test_forward_weight_reduction(self):
        logits = [
            torch.as_tensor([[2.0, 1.0], [1.0, 2.0]]),
            torch.as_tensor([[1.0, 3.0], [2.0, 3.0]])
        ]
        targets = torch.as_tensor([[0, -1], [1, 0]])

        loss = sincomp.models.MultiTargetLoss(
            weight=[0.4, 0.6],
            reduction='none'
        )(logits, targets)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertTupleEqual(loss.size(), (2,))


class TestPronunciationPredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()

        import sincomp.datasets

        data = sincomp.datasets.FileDataset(
            path=os.path.join(data_dir, 'custom_dataset1')
        )
        (
            cls.features,
            cls.targets,
            cls.dialect_vocab_sizes,
            cls.char_vocab_sizes,
            cls.target_vocab_sizes
        ) = encode_data(data)

    def setUp(self):
        super().setUp()

        self.model = sincomp.models.BilinearEncoder(
            self.dialect_vocab_sizes,
            self.char_vocab_sizes,
            self.target_vocab_sizes,
            missing_id=0,
            dropout=0.2
        )
        self.predictor = sincomp.models.PronunciationPredictor(
            self.model,
            batch_size=10
        )

    def test_make_data_loader(self):
        loader = self.predictor._make_data_loader(self.features, self.targets)
        (dialects, chars), targets = next(iter(loader))

        self.assertIsInstance(dialects, torch.Tensor)
        self.assertTupleEqual(dialects.size(), (10, 1))
        self.assertIsInstance(chars, torch.Tensor)
        self.assertTupleEqual(chars.size(), (10, 2))
        self.assertIsInstance(targets, torch.Tensor)
        self.assertTupleEqual(targets.size(), (10, 3))

    def test_fit(self):
        predictor = self.predictor.fit(self.features, self.targets)
        self.assertIs(predictor, self.predictor)

    def test_partial_fit(self):
        predictor = self.predictor.partial_fit(self.features, self.targets)
        self.assertIs(predictor, self.predictor)

    def test_predict(self):
        pred = self.predictor.predict(self.features)

        self.assertTupleEqual(pred.shape, self.targets.shape)
        self.assertTrue((pred >= 0).all())
        self.assertTrue((pred < numpy.asarray(self.target_vocab_sizes)).all())

    def test_predict_proba(self):
        probs = self.predictor.predict_proba(self.features)

        for p, n in zip(probs, self.target_vocab_sizes):
            self.assertIsInstance(p, numpy.ndarray)
            self.assertTupleEqual(p.shape, (self.targets.shape[0], n))
            self.assertTrue((p >= 0).all())
            self.assertTrue((p <= 1).all())


class TestModels(unittest.TestCase):
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

        cls.dataset = torch.utils.data.StackDataset(
            torch.utils.data.TensorDataset(
                torch.as_tensor(features[:, :1]),
                torch.as_tensor(features[:, 1:])
            ),
            torch.as_tensor(targets, dtype=torch.long)
        )

    def setUp(self):
        super().setUp()

        self.model = sincomp.models.BilinearEncoder(
            self.dialect_vocab_sizes,
            self.char_vocab_sizes,
            self.target_vocab_sizes,
            missing_id=0,
            dropout=0.2
        )
        self.loss = sincomp.models.MultiTargetLoss(
            weight=[0.3, 0.4, 0.3],
            ignore_index=-1,
            reduction='none'
        )
        self.optimizer = torch.optim.SGD(self.model.parameters())
        self.data_loader = torch.utils.data.DataLoader(self.dataset, batch_size=10)

    def test_train(self):
        loss, acc = sincomp.models.train(
            self.model,
            self.loss,
            self.optimizer,
            self.data_loader
        )

        self.assertIsInstance(loss, torch.Tensor)
        self.assertTupleEqual(loss.size(), ())
        self.assertIsInstance(acc, torch.Tensor)
        self.assertTupleEqual(acc.size(), (len(self.target_vocab_sizes),))

    def test_evaluate(self):
        loss, acc = sincomp.models.evaluate(
            self.model,
            self.loss,
            self.data_loader
        )

        self.assertIsInstance(loss, torch.Tensor)
        self.assertTupleEqual(loss.size(), ())
        self.assertIsInstance(acc, torch.Tensor)
        self.assertTupleEqual(acc.size(), (len(self.target_vocab_sizes),))

    if torch.cuda.is_available():
        def test_train_cuda(self):
            device = torch.device('cuda:0')
            self.model.to(device)
            self.loss.to(device)
            loss, acc = sincomp.models.train(
                self.model,
                self.loss,
                self.optimizer,
                self.data_loader
            )

            self.assertIsInstance(loss, torch.Tensor)
            self.assertEqual(loss.device, device)
            self.assertTupleEqual(loss.size(), ())
            self.assertIsInstance(acc, torch.Tensor)
            self.assertEqual(acc.device, device)
            self.assertTupleEqual(acc.size(), (len(self.target_vocab_sizes),))

        def test_evaluate_cuda(self):
            device = torch.device('cuda:0')
            self.model.to(device)
            self.loss.to(device)
            loss, acc = sincomp.models.evaluate(
                self.model,
                self.loss,
                self.data_loader
            )

            self.assertIsInstance(loss, torch.Tensor)
            self.assertEqual(loss.device, device)
            self.assertTupleEqual(loss.size(), ())
            self.assertIsInstance(acc, torch.Tensor)
            self.assertEqual(acc.device, device)
            self.assertTupleEqual(acc.size(), (len(self.target_vocab_sizes),))


    def test_fit(self):
        sincomp.models.fit(
            self.model,
            self.dataset,
            self.dataset,
            loss=self.loss,
            optimizer=self.optimizer,
            lr_scheduler=torch.optim.lr_scheduler.ExponentialLR(self.optimizer, 0.9),
            epochs=2,
            batch_size=10,
            num_workers=2,
            checkpoint_dir=os.path.join(tmp_dir, 'checkpoints'),
            log_dir=os.path.join(tmp_dir, 'log')
        )