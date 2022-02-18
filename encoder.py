#!/usr/bin/python3 -O
# -*- encoding: utf-8 -*-

'''
使用编码器训练方言音系 embedding.
'''

__author__ = '黄艺华 <lernanto@foxmail.com>'


import sys
import numpy as np
import tensorflow as tf
import datetime
import recon


input_file = sys.argv[1]

data = recon.clean(recon.load(input_file))

for c in data.columns:
    if c.partition('_')[2] in ('聲母', '韻母', '調值'):
        data[c] = data[c].astype('category')

class ContrastiveEncoder:
    '''
    使用孪生网络 + 对比损失训练方言音系 embedding.

    从包含多个方言声母、韵母、声调的同一个样本中随机抽样两份、每份包含部分方言数据的输入，
    输入孪生网络编码成 embedding 后，采用对比损失，即相同样本距离越近越好，不同样本距离必须大于预先指定的边界。
    '''

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
    '''
    生成孪生网络的输入.
    '''

    def gen():
        input_shapes = [len(d.categories) for d in data.dtypes]
        base = np.cumsum(input_shapes) - input_shapes
        for indices in zip(*[c.cat.codes for _, c in data.iteritems()]):
            indices = tuple(idx + base[i] for i, idx in enumerate(indices) if idx != -1)
            if sample is None:
                yield indices
            elif len(indices) > sample:
                yield (
                    np.random.choice(indices, sample, replace=False),
                    np.random.choice(indices, sample, replace=False)
                )

    return gen

columns = [c for c in data.columns if c.endswith('聲母')]
categories = [t.categories for t in data.dtypes[columns]]

dataset = tf.data.Dataset.from_generator(
    gen_sample(data[columns], sample=30),
    output_types=(tf.int32, tf.int32)
).batch(100, drop_remainder=True)

emb_size = 100
encoder = ContrastiveEncoder(sum(len(c) for c in categories), emb_size)
optimizer = tf.optimizers.Adam()

log_dir = 'tensorboard/{}'.format(
    datetime.datetime.now().strftime('%Y%m%d%H%M')
)
summary_writer = tf.summary.create_file_writer(log_dir)

loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)

for epoch in range(20):
    for inputs, targets in dataset:
        loss(encoder.update(inputs, targets, optimizer))

    with summary_writer.as_default():
        tf.summary.scalar('loss', loss.result(), step=epoch)
    loss.reset_states()
