# -*- coding: utf-8 -*-

"""
方言字音编码器模型.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import numpy
import tensorflow as tf


class ContrastiveEncoder:
    """
    使用孪生网络 + 对比损失训练方言音系 embedding.

    从包含多个方言声母、韵母、声调的同一个样本中随机抽样两份、每份包含部分方言数据的输入，
    输入孪生网络编码成 embedding 后，采用对比损失，即相同样本距离越近越好，不同样本距离必须大于预先指定的边界。
    """

    def __init__(self, input_size, emb_size):
        self.embedding = tf.Variable(tf.random_normal_initializer()(
            shape=(input_size, emb_size),
            dtype=tf.float32)
        )
        self.trainable_variables = (self.embedding,)

    @tf.function
    def encode(self, inputs):
        weight = tf.linalg.normalize(
            tf.cast(inputs != -1, tf.float32),
            ord=1,
            axis=1
        )[0]
        return tf.reduce_sum(
            tf.nn.embedding_lookup(self.embedding, tf.maximum(inputs, 0)) \
                * tf.expand_dims(weight, -1),
                axis=1
            )

    @tf.function
    def loss(self, inputs, targets):
        margin = 1
        d2 = tf.reduce_sum(tf.square(
            tf.expand_dims(self.encode(inputs), 1) \
                - tf.expand_dims(self.encode(targets), 0)
        ), axis=1)
        d = tf.sqrt(d2)
        return tf.reduce_mean(
            tf.eye(inputs.shape[0], dtype=tf.float32) * d2 \
                + (1 - tf.eye(inputs.shape[0], dtype=tf.float32)) \
                * tf.square(tf.maximum(0, margin - d)),
            axis=1
        )

    @tf.function
    def update(self, inputs, targets, optimizer):
        with tf.GradientTape() as tape:
            loss = self.loss(inputs, targets)
        grad = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return loss


def gen_sample(data, sample=None):
    """
    生成孪生网络的输入.
    """

    def gen():
        input_shapes = [len(d.categories) for d in data.dtypes]
        base = numpy.cumsum(input_shapes) - input_shapes
        for indices in zip(*[c.cat.codes for _, c in data.iteritems()]):
            indices = tuple(idx + base[i] for i, idx in enumerate(indices) if idx != -1)
            if sample is None:
                yield indices
            elif len(indices) > sample:
                yield (
                    numpy.random.choice(indices, sample, replace=False),
                    numpy.random.choice(indices, sample, replace=False)
                )

    return gen


class AutoEncoder:
    """
    孪生网络 + 随机负采样自编码器.

    从同一个样本中随机采样两路输入孪生网络，其中一路作为真正的输入，另一路作为预测目标及负采样的负例。
    """

    def __init__(self, input_size, emb_size):
        self.embedding = tf.Variable(tf.random_normal_initializer()(
            shape=(input_size, emb_size), dtype=tf.float32)
        )
        self.trainable_variables = (self.embedding,)

    @tf.function
    def encode(self, *inputs):
        embeddings = []
        for i, input in enumerate(inputs):
            weight = tf.cast(input != -1, tf.float32)
            weight = weight / tf.maximum(
                tf.reduce_sum(weight, axis=1, keepdims=True),
                1e-9
            )
            embeddings.append(tf.reduce_sum(
                tf.nn.embedding_lookup(self.embedding, tf.maximum(input, 0)) \
                    * tf.expand_dims(weight, -1),
                axis=1
            ))

        return embeddings[0] if len(embeddings) == 1 \
            else tf.reduce_sum(embeddings, axis=0)

    @tf.function
    def loss(self, inputs, targets):
        logits = tf.matmul(
            self.encode(*inputs),
            self.encode(*targets),
            transpose_b=True
        )
        labels = tf.range(logits.shape[0], dtype=tf.int32)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, logits)
        acc = tf.cast(
            tf.argmax(logits, axis=1, output_type=tf.int32) == labels,
            tf.float32
        )
        return loss, acc

    @tf.function
    def update(self, inputs, targets, optimizer):
        with tf.GradientTape() as tape:
            loss, acc = self.loss(inputs, targets)
        grad = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return loss, acc


class ContrastiveGenerator:
    """
    随机样本生成器.

    从同一个样本根据指定额采样率随机采样两路输入作为孪生网络的输入。可以对声母、韵母、声调分别指定采样率。
    """

    def __init__(self, *data):
        self.data = data

    def contrastive(self, sample):
        try:
            samples = list(sample)
        except:
            samples = [sample] * len(self.data)

        for i, s in enumerate(samples):
            if s < 1:
                s = int(s * self.data[i].shape[1])
            samples[i] = numpy.clip(s, 1, self.data[i].shape[1] - 1)

        def gen():
            for i in range(self.data[0].shape[0]):
                inputs = []
                targets = []

                for j in range(len(self.data)):
                    indices = self.data[j][i][self.data[j][i] != -1]
                    if indices.shape[0] >= 2:
                        m = numpy.random.randint(1, min(samples[j] + 1, indices.shape[0]))
                        n = numpy.random.randint(1, min(samples[j] + 1, indices.shape[0]))
                        inputs.append(numpy.random.choice(indices, m, replace=False))
                        targets.append(numpy.random.choice(indices, n, replace=False))
                    else:
                        inputs.append(numpy.empty(0))
                        targets.append(numpy.empty(0))

                if sum(ip.shape[0] for ip in inputs) > 0 \
                    and sum(t.shape[0] for t in targets) > 0:
                    yield tuple(inputs), tuple(targets)

        return gen

    def input(self):
        for i in range(self.data[0].shape[0]):
            inputs = []
            for j in range(len(self.data)):
                indices = self.data[j][i][self.data[j][i] != -1]
                inputs.append(indices)

            if sum(ip.shape[0] for ip in inputs) > 0:
                yield tuple(inputs)


class DenoisingAutoEncoder:
    """
    降噪自编码器进行方言音系编码.

    输入为多个方言点的声母、韵母、声调，随机删除其中部分数据，输出为还原的数据。
    分声母、韵母、声调分别求 embedding，多个方言的声母、韵母、声调分别求平均，
    然后把声母、韵母、声调 embedding 相加得到音节 embedding。
    使用音节 embedding 预测各方言的声母、韵母、声调。
    """

    def __init__(self, input_shapes, emb_size, symetric=True):
        self.input_shapes = tuple(input_shapes)
        self.limits = numpy.cumsum((0,) + self.input_shapes)
        self.embedding = tf.Variable(tf.random_normal_initializer()(
            shape=(self.limits[-1], emb_size),
            dtype=tf.float32
        ))
        if symetric:
            self.output_embedding = self.embedding
            self.trainable_variables = (self.embedding,)
        else:
            self.output_embedding = tf.Variable(tf.random_normal_initializer()(
                shape=(self.limits[-1], emb_size),
                dtype=tf.float32
            ))
            self.trainable_variables = (self.embedding, self.output_embedding)

    @tf.function
    def encode(self, inputs):
        # -1 为删除的数据，使用 ragged tensor 表示不定长的数据
        emb = tf.reduce_mean(tf.nn.embedding_lookup(
            self.embedding,
            tf.ragged.boolean_mask(inputs, inputs >= 0)
        ), axis=2).to_tensor()
        # 对方言求平均，再把声母、韵母、声调相加
        return tf.reduce_sum(
            tf.where(tf.math.is_nan(emb), tf.zeros_like(emb), emb),
            axis=1
        )

    @tf.function
    def predict(self, inputs):
        logits = tf.matmul(
            self.encode(inputs),
            self.output_embedding,
            transpose_b=True
        )

        preds = []
        for i in range(self.limits.shape[0] - 1):
            preds.append(tf.math.argmax(
                logits[:, self.limits[i]:self.limits[i + 1]],
                axis=1
            ) + self.limits[i])

        return tf.stack(preds, axis=1)

    @tf.function
    def predict_proba(self, inputs):
        logits = tf.matmul(
            self.encode(inputs),
            self.output_embedding,
            transpose_b=True
        )

        probs = []
        for i in range(self.limits.shape[0] - 1):
            probs.append(
                tf.nn.softmax(logits[:, self.limits[i]:self.limits[i + 1]])
            )

        return tf.concat(probs, axis=1)

    @tf.function
    def loss(self, inputs, targets):
        logits = tf.matmul(
            self.encode(inputs),
            self.output_embedding,
            transpose_b=True
        )

        loss = []
        for i in range(targets.shape[1]):
            loss.append(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.maximum(targets[:, i] - self.limits[i], 0),
                logits=logits[:, self.limits[i]:self.limits[i + 1]]
            ))
        loss = tf.stack(loss)
        return tf.reduce_sum(tf.reduce_mean(
            tf.ragged.boolean_mask(loss, tf.transpose(targets >= 0)),
            axis=1
        ))

    @tf.function
    def update(self, inputs, targets, optimizer):
        with tf.GradientTape() as tape:
            loss = self.loss(inputs, targets)
        grad = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return loss


class NoiseGenerator:
    """
    生成降噪自编码器的训练样本.

    可以为声母、韵母、声调分别指定采样的比例，按指定的比例随机选择若干个方言的声母、韵母、声调作为输入。
    """

    def __init__(self, *data):
        self.data = data

    def noise(self, sample):
        try:
            samples = list(sample)
        except:
            samples = [sample] * len(self.data)

        for i, s in enumerate(samples):
            if isinstance(s, float):
                s = int(s * self.data[i].shape[1])
            samples[i] = numpy.clip(s, 1, self.data[i].shape[1] - 1)

        def gen():
            for i in range(self.data[0].shape[0]):
                inputs = []

                for j in range(len(self.data)):
                    indices = self.data[j][i][self.data[j][i] >= 0]
                    if indices.shape[0] >= 2:
                        m = numpy.random.randint(1, min(samples[j] + 1, indices.shape[0]))
                        n = numpy.random.randint(1, min(samples[j] + 1, indices.shape[0]))
                        inputs.append(numpy.random.choice(indices, m, replace=False))
                    else:
                        inputs.append(numpy.empty(0))

                if sum(ip.shape[0] for ip in inputs) > 0:
                    yield tuple(inputs), numpy.concatenate([self.data[j][i] for j in range(len(self.data))])

        return gen

    def input(self):
        for i in range(self.data[0].shape[0]):
            inputs = []
            for j in range(len(self.data)):
                indices = self.data[j][i][self.data[j][i] >= 0]
                inputs.append(indices)

            if sum(ip.shape[0] for ip in inputs) > 0:
                yield tuple(inputs)