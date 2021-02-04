#!/usr/bin/python3 -O
# -*- encoding: utf-8 -*-

import sys
import numpy as np
import tensorflow as tf
import datetime


class AutoEncoder:
    def __init__(self, input_shape, emb_size):
        try:
            self.input_shape = tuple(input_shape)
        except:
            self.input_shape = (int(input_shape),)

        embeddings = []
        for size in self.input_shape:
            embeddings.append(tf.Variable(tf.random_normal_initializer()(shape=(size, emb_size), dtype=tf.float32)))
        self.embeddings = tuple(embeddings)
        self.trainable_variables = self.embeddings

    def encode(self, inputs):
        assert len(inputs) == len(self.embeddings)

        embeddings = []
        for i, input in enumerate(inputs):
            weight = tf.cast(input != -1, tf.float32)
            weight = weight / tf.maximum(tf.reduce_sum(weight, axis=1, keepdims=True), 1e-9)
            embeddings.append(tf.reduce_sum(tf.nn.embedding_lookup(self.embeddings[i], tf.maximum(input, 0)) * tf.expand_dims(weight, -1), axis=1))
            
        return tuple(embeddings)

    def loss(self, inputs, targets):
        logits = tf.matmul(tf.reduce_sum(self.encode(inputs), axis=0), tf.reduce_sum(self.encode(targets), axis=0), transpose_b=True)
        return tf.nn.sparse_softmax_cross_entropy_with_logits(tf.range(logits.shape[0], dtype=tf.int32), logits)

    def update(self, inputs, targets, optimizer):
        with tf.GradientTape() as tape:
            loss = self.loss(inputs, targets)
        grad = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return loss

class ContrastiveGenerator:
    def __init__(self, data):
        self.data = data
        self.input_nums = [d.shape[1] for d in data]
        self.input_shapes = [[len(t.categories) for t in d.dtypes] for d in data]
        self.bases = [np.cumsum(shapes) - shapes for shapes in self.input_shapes]

    def contrastive(self, sample):
        try:
            samples = list(sample)
        except:
            samples = [sample] * len(self.input_nums)
            
        for i, s in enumerate(samples):
            if s < 1:
                s = int(s * self.data[i].shape[1])
            samples[i] = np.clip(s, 1, self.data[i].shape[1] - 1)

        def gen():
            for sample in zip(*[zip(*[c.cat.codes for _, c in d.iteritems()]) for d in self.data]):
                inputs = []
                targets = []

                for i, indices in enumerate(sample):
                    indices = [idx + self.bases[i][j] for j, idx in enumerate(indices) if idx != -1]
                    if len(indices) >= 2:
                        m = np.random.randint(1, min(samples[i] + 1, len(indices)))
                        n = np.random.randint(1, min(samples[i] + 1, len(indices)))
                        inputs.append(np.random.choice(indices, m, replace=False))
                        targets.append(np.random.choice(indices, n, replace=False))
                    else:
                        inputs.append(np.empty(0))
                        targets.append(np.empty(0))
                
                if sum(i.shape[0] for i in inputs) > 0 and sum(t.shape[0] for t in targets) > 0:
                    yield tuple(inputs), tuple(targets)

        return gen

    def input(self):
        for sample in zip(*[zip(*[c.cat.codes for _, c in d.iteritems()]) for d in self.data]):
            inputs = []
            for i, indices in enumerate(sample):
                indices = [idx + self.bases[i][j] for j, idx in enumerate(indices) if idx != -1]
                if len(indices) > 0:
                    inputs.append(indices)
                else:
                    inputs.append([])
                    
            if sum(len(i) for i in inputs) > 0:
                yield tuple(inputs)


input_file = sys.argv[1]

data = recon.clean(recon.load(input_file))

for c in data.columns:
    if c.partition('_')[2] in ('聲母', '韻母', '調值'):
        data[c] = data[c].astype('category')

initials = [c for c in data.columns if c.endswith('聲母')]
finals = [c for c in data.columns if c.endswith('韻母')]
tones = [c for c in data.columns if c.endswith('調值')]
initial_categories = [t.categories for t in data.dtypes[initials]]
final_categories = [t.categories for t in data.dtypes[finals]]
tone_categories = [t.categories for t in data.dtypes[tones]]
input_shapes = [[len(c) for c in initial_categories], [len(c) for c in final_categories], [len(c) for c in tone_categories]]

generator = ContrastiveGenerator([data[initials], data[finals], data[tones]])

dataset = tf.data.Dataset.from_generator(
    generator.contrastive(0.5),
    output_types=((tf.int32, tf.int32, tf.int32), (tf.int32, tf.int32, tf.int32))
).shuffle(1000).padded_batch(100, padded_shapes=(((None,), (None,), (None,)), ((None,), (None,), (None,))), padding_values=-1, drop_remainder=True)

emb_size = 10
encoder = AutoEncoder([sum(shape) for shape in input_shapes], emb_size)
optimizer = tf.optimizers.Adam()

output_prefix = 'tensorboard/{}'.format(
    datetime.datetime.now().strftime('%Y%m%d%H%M')
)

log_dir = output_prefix
summary_writer = tf.summary.create_file_writer(log_dir)

loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)

for epoch in range(20):
    for inputs, targets in dataset:
        loss(encoder.update(inputs, targets, optimizer))

    with summary_writer.as_default():
            tf.summary.scalar('loss', loss.result(), step=epoch)
    loss.reset_states()


log_dir = 'tensorboard'
data[['id'] + columns].to_csv('{}/{}'.format(log_dir, 'metadata.tsv'), sep='\t', index=False)

initial_emb, final_emb, tone_emb = tuple(tf.concat(emb, axis=0).numpy() for emb in zip(*[encoder.encode(inputs) for inputs in tf.data.Dataset.from_generator(
    generator.input,
    output_types=(tf.int32, tf.int32, tf.int32)
).padded_batch(100, padded_shapes=((None,), (None,), (None,)), padding_values=-1)]))

cp = tf.train.Checkpoint(initial_embedding=tf.Variable(initial_emb), final_embedding=tf.Variable(final_emb), tone_embedding=tf.Variable(final_emb))
cp.save('{}/{}'.format(log_dir, 'embedding.ckpt'))

config = projector.ProjectorConfig()
for key in ('initial', 'final', 'tone'):
    embedding = config.embeddings.add()
    embedding.tensor_name = '{}_embedding/.ATTRIBUTES/VARIABLE_VALUE'.format(key)
    embedding.metadata_path = 'metadata.tsv'

projector.visualize_embeddings(log_dir, config)
