#!/usr/bin/python3 -O
# -*- encoding: utf-8 -*-

import sys
import numpy as np
import tensorflow as tf
import datetime
import recon


input_file = sys.argv[1]

data = recon.clean(recon.load(input_file)

for c in data.columns:
    if c.partition('_')[2] in ('聲母', '韻母', '調值'):
        data[c] = data[c].astype('category')

class ContrastiveEncoder:
    def __init__(self, input_size, emb_size):
        self.embedding = tf.Variable(tf.random_normal_initializer()(shape=(input_size, emb_size), dtype=tf.float32))
        self.trainable_variables = (self.embedding,)

    def encode(self, inputs):
        weight = tf.linalg.normalize(tf.cast(inputs != -1, tf.float32), ord=1, axis=1)[0]
        return tf.reduce_sum(tf.nn.embedding_lookup(self.embedding, tf.maximum(inputs, 0)) * tf.expand_dims(weight, -1), axis=1)

    def loss(self, inputs, targets):
        margin = 1
        d2 = tf.reduce_sum(tf.square(tf.expand_dims(self.encode(inputs), 1) - tf.expand_dims(self.encode(targets), 0)), axis=1)
        d = tf.sqrt(d2)
        return tf.reduce_mean(tf.eye(inputs.shape[0], dtype=tf.float32) * d2 + (1 - tf.eye(inputs.shape[0], dtype=tf.float32)) * tf.square(tf.maximum(0, margin - d)), axis=1)

    def update(self, inputs, targets, optimizer):
        with tf.GradientTape() as tape:
            loss = self.loss(inputs, targets)
        grad = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return loss

def gen_sample(data, sample=None):
    def gen():
        input_shapes = [len(d.categories) for d in data.dtypes]
        base = np.cumsum(input_shapes) - input_shapes
        for indices in zip(*[c.cat.codes for _, c in data.iteritems()]):
            indices = tuple(idx + base[i] for i, idx in enumerate(indices) if idx != -1)
            if sample is None:
                yield indices
            elif len(indices) > sample:
                yield np.random.choice(indices, sample, replace=False), np.random.choice(indices, sample, replace=False)

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
