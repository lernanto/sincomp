#!/usr/bin/python3 -O
# -*- encoding: utf-8 -*-

import sys
import os
import datetime
import numpy as np
import tensorflow as tf
import recon


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


class ContrastiveEncoder:
    def __init__(self, input_size, emb_size):
        self.embedding = tf.Variable(tf.random_normal_initializer()(shape=(input_size, emb_size), dtype=tf.float32))
        self.trainable_variables = (self.embedding,)

    def encode(self, *inputs):
        embeddings = []
        for i, input in enumerate(inputs):
            weight = tf.cast(input != -1, tf.float32)
            weight = weight / tf.maximum(tf.reduce_sum(weight, axis=1, keepdims=True), 1e-9)
            embeddings.append(tf.reduce_sum(tf.nn.embedding_lookup(self.embedding, tf.maximum(input, 0)) * tf.expand_dims(weight, -1), axis=1))

        return embeddings[0] if len(embeddings) == 1 else tf.reduce_sum(embeddings, axis=0)

    def loss(self, inputs, targets):
        logits = tf.matmul(self.encode(*inputs), self.encode(*targets), transpose_b=True)
        return tf.nn.sparse_softmax_cross_entropy_with_logits(tf.range(logits.shape[0], dtype=tf.int32), logits)

    def update(self, inputs, targets, optimizer):
        with tf.GradientTape() as tape:
            loss = self.loss(inputs, targets)
        grad = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return loss


class ContrastiveGenerator:
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
            samples[i] = np.clip(s, 1, self.data[i].shape[1] - 1)

        def gen():
            for i in range(self.data[0].shape[0]):
                inputs = []
                targets = []

                for j in range(len(self.data)):
                    indices = self.data[j][i][self.data[j][i] != -1]
                    if indices.shape[0] >= 2:
                        m = np.random.randint(1, min(samples[j] + 1, indices.shape[0]))
                        n = np.random.randint(1, min(samples[j] + 1, indices.shape[0]))
                        inputs.append(np.random.choice(indices, m, replace=False))
                        targets.append(np.random.choice(indices, n, replace=False))
                    else:
                        inputs.append(np.empty(0))
                        targets.append(np.empty(0))

                if sum(ip.shape[0] for ip in inputs) > 0 and sum(t.shape[0] for t in targets) > 0:
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
    def __init__(self, input_shapes, emb_size):
        self.input_shapes = tuple(input_shapes)
        self.limits = np.cumsum((0,) + self.input_shapes)
        self.embedding = tf.Variable(tf.random_normal_initializer()(shape=(self.limits[-1], emb_size), dtype=tf.float32))
        self.output_embedding = tf.Variable(tf.random_normal_initializer()(shape=(self.limits[-1], emb_size), dtype=tf.float32))
        self.trainable_variables = (self.embedding, self.output_embedding)

    def encode(self, *inputs):
        embeddings = []
        for i, input in enumerate(inputs):
            weight = tf.cast(input >= 0, tf.float32)
            weight = weight / tf.maximum(tf.reduce_sum(weight, axis=1, keepdims=True), 1e-9)
            embeddings.append(tf.reduce_sum(tf.nn.embedding_lookup(self.embedding, tf.maximum(input, 0)) * tf.expand_dims(weight, -1), axis=1))

        return embeddings[0] if len(embeddings) == 1 else tf.reduce_sum(embeddings, axis=0)

    def predict(self, *inputs):
        logits = tf.matmul(self.encode(*inputs), self.output_embedding, transpose_b=True)

        preds = []
        for i in range(self.limits.shape[0] - 1):
            preds.append(tf.math.argmax(logits[:, self.limits[i]:self.limits[i + 1]], axis=1) + self.limits[i])

        return tf.stack(preds, axis=1)

    def predict_proba(self, *inputs):
        logits = tf.matmul(self.encode(*inputs), self.output_embedding, transpose_b=True)

        probs = []
        for i in range(self.limits.shape[0] - 1):
            probs.append(tf.nn.softmax(logits[:, self.limits[i]:self.limits[i + 1]]))

        return tf.concat(probs, axis=1)

    def loss(self, inputs, targets):
        logits = tf.matmul(self.encode(*inputs), self.output_embedding, transpose_b=True)

        loss = []
        for i in range(targets.shape[1]):
            loss.append(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=tf.maximum(targets[:, i] - self.limits[i], 0),
                logits=logits[:, self.limits[i]:self.limits[i + 1]]
            ))
        loss = tf.stack(loss, axis=1)

        weight = tf.cast(targets >= 0, tf.float32)
        weight = weight / tf.maximum(tf.reduce_sum(weight, axis=0, keepdims=True), 1e-9)

        return tf.reduce_sum(loss * weight)

    def update(self, inputs, targets, optimizer):
        with tf.GradientTape() as tape:
            loss = self.loss(inputs, targets)
        grad = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return loss


class NoiseGenerator:
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
            samples[i] = np.clip(s, 1, self.data[i].shape[1] - 1)

        def gen():
            for i in range(self.data[0].shape[0]):
                inputs = []

                for j in range(len(self.data)):
                    indices = self.data[j][i][self.data[j][i] >= 0]
                    if indices.shape[0] >= 2:
                        m = np.random.randint(1, min(samples[j] + 1, indices.shape[0]))
                        n = np.random.randint(1, min(samples[j] + 1, indices.shape[0]))
                        inputs.append(np.random.choice(indices, m, replace=False))
                    else:
                        inputs.append(np.empty(0))

                if sum(ip.shape[0] for ip in inputs) > 0:
                    yield tuple(inputs), np.concatenate([self.data[j][i] for j in range(len(self.data))])

        return gen

    def input(self):
        for i in range(self.data[0].shape[0]):
            inputs = []
            for j in range(len(self.data)):
                indices = self.data[j][i][self.data[j][i] >= 0]
                inputs.append(indices)

            if sum(ip.shape[0] for ip in inputs) > 0:
                yield tuple(inputs)


if __name__ == '__main__':
    input_file = sys.argv[1]

    data = recon.clean(recon.load(input_file))

    for c in data.columns:
        if c.partition('_')[2] in ('聲母', '韻母', '調值'):
            data[c] = data[c].astype('category')

    initials = [c for c in data.columns if c.endswith('聲母')]
    finals = [c for c in data.columns if c.endswith('韻母')]
    tones = [c for c in data.columns if c.endswith('調值')]
    columns = initials + finals + tones

    data.dropna(how='all', subset=columns, inplace=True)

    for c in columns:
        data[c] = data[c].astype('category')

    initial_categories = [t.categories for t in data.dtypes[initials]]
    final_categories = [t.categories for t in data.dtypes[finals]]
    tone_categories = [t.categories for t in data.dtypes[tones]]
    categories = initial_categories + final_categories + tone_categories

    bases = np.insert(np.cumsum([len(c) for c in categories])[:-1], 0, 0)
    codes = np.empty(data[columns].shape, dtype=np.int32)
    for i, c in enumerate(columns):
        codes[:, i] = data[c].cat.codes

    codes = pd.DataFrame(columns=columns, data=np.where(codes >= 0, codes + bases, -1))

    generator = ContrastiveGenerator(codes[initials].values, codes[finals].values, codes[tones].values)

    dataset = tf.data.Dataset.from_generator(
        generator.contrastive(0.5),
        output_types=((tf.int32, tf.int32, tf.int32), (tf.int32, tf.int32, tf.int32))
    ).shuffle(1000).padded_batch(100, padded_shapes=(((None,), (None,), (None,)), ((None,), (None,), (None,))), padding_values=-1, drop_remainder=True)

    emb_size = 10
    encoder = AutoEncoder([sum(shape) for shape in input_shapes], emb_size)
    optimizer = tf.optimizers.Adam()

    output_prefix = os.path.join(
        'tensorboard', 
        datetime.datetime.now().strftime('%Y%m%d%H%M')
    )

    log_dir = output_prefix
    summary_writer = tf.summary.create_file_writer(log_dir)
    loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)

    checkpoint = tf.train.Checkpoint(embedding=encoder.embedding, optimizer=optimizer)
    manager = tf.train.CheckpointManager(
        checkpoint,
        os.path.join(output_prefix, 'checkpoints'),
        max_to_keep=10
    )

    for epoch in range(100):
        for inputs, targets in dataset:
            loss(encoder.update(inputs, targets, optimizer))

        with summary_writer.as_default():
                tf.summary.scalar('loss', loss.result(), step=epoch)
        loss.reset_states()

        if epoch % 10 == 9:
            manager.save()

    log_dir = 'tensorboard'
    data[['id'] + columns].to_csv(os.path.join(log_dir, 'metadata.tsv'), sep='\t', index=False)

    initial_emb = encoder.encode(codes[initials].values).numpy()
    final_emb = encoder.encode(codes[finals].values).numpy()
    tone_emb = encoder.encode(codes[tones].values).numpy()

    cp = tf.train.Checkpoint(initial_embedding=tf.Variable(initial_emb), final_embedding=tf.Variable(final_emb), tone_embedding=tf.Variable(final_emb))
    cp.save(os.path.join(log_dir, 'embedding.ckpt'))

    config = projector.ProjectorConfig()
    for key in ('initial', 'final', 'tone'):
        embedding = config.embeddings.add()
        embedding.tensor_name = '{}_embedding/.ATTRIBUTES/VARIABLE_VALUE'.format(key)
        embedding.metadata_path = 'metadata.tsv'

    projector.visualize_embeddings(log_dir, config)
