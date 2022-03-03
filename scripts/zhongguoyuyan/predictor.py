#!/usr/bin/python3 -O
# -*- coding: utf-8 -*-

'''
方言字音编码器.
'''

__author__ = '黄艺华 <lernanto@foxmail.com>'


import sys
import os
import logging
import datetime
import pandas
import tensorflow as tf
from util import clean_data


class DialectPredictor:
    def __init__(
        self,
        dialects,
        chars,
        targets,
        emb_size=20,
        dialect_emb_size=None,
        char_emb_size=None,
        target_emb_size=None,
        transform_heads=1,
        transform_size=100,
        activation=tf.nn.softmax,
        target_bias=False,
        emb_l2=0.01,
        l2=0,
        optimizer=None
    ):
        if dialect_emb_size is None:
            dialect_emb_size = emb_size
        if char_emb_size is None:
            char_emb_size = emb_size
        if target_emb_size is None:
            target_emb_size = char_emb_size

        self.dialects = tf.convert_to_tensor(dialects)
        self.chars = tf.convert_to_tensor(chars)
        self.targets = [tf.convert_to_tensor(o) for o in targets]
        self.activation = activation
        self.target_bias = target_bias
        self.emb_l2 = emb_l2
        self.l2 = l2

        self.dialect_table = tf.lookup.StaticVocabularyTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=self.dialects,
                values=tf.range(dialects.shape[0], dtype=tf.int64)
            ),
            num_oov_buckets=1
        )
        self.char_table = tf.lookup.StaticVocabularyTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=self.chars,
                values=tf.range(chars.shape[0], dtype=tf.int64)
            ),
            num_oov_buckets=1
        )

        self.dialect_emb = tf.Variable(tf.random_normal_initializer()(
            shape=(self.dialect_table.size(), dialect_emb_size),
            dtype=tf.float32
        ), name='dialect_emb')
        self.char_emb = tf.Variable(tf.random_normal_initializer()(
            shape=(self.char_table.size(), char_emb_size),
            dtype=tf.float32
        ), name='char_emb')

        self.dialect_att_weight = tf.Variable(tf.random_normal_initializer()(
            shape=(dialect_emb_size, transform_heads, transform_size),
            dtype=tf.float32
        ), name='dialect_att_weight')
        self.dialect_att_bias = tf.Variable(tf.random_normal_initializer()(
            shape=(transform_heads, transform_size),
            dtype=tf.float32
        ), name='dialect_att_bias')
        self.char_att_weight = tf.Variable(tf.random_normal_initializer()(
            shape=(char_emb_size, transform_heads, transform_size),
            dtype=tf.float32
        ), name='char_att_weight')
        self.char_att_bias = tf.Variable(tf.random_normal_initializer()(
            shape=(transform_heads, transform_size),
            dtype=tf.float32
        ), name='char_bias')
        self.trans_weight = tf.Variable(tf.random_normal_initializer()(
            shape=(transform_heads, transform_size, char_emb_size),
            dtype=tf.float32
        ), name='trans_weight')

        self.target_tables = []
        for target in self.targets:
            self.target_tables.append(tf.lookup.StaticVocabularyTable(
                tf.lookup.KeyValueTensorInitializer(
                    keys=target,
                    values=tf.range(target.shape[0], dtype=tf.int64)
                ),
                num_oov_buckets=1
            ))

        self.target_embs = []
        for i, target in enumerate(self.targets):
            self.target_embs.append(tf.Variable(tf.random_normal_initializer()(
                shape=(target.shape[0], target_emb_size),
                dtype=tf.float32
            ), name='target_emb{}'.format(i)))

        self.trainable_variables = [
            self.dialect_emb,
            self.char_emb,
            self.dialect_att_weight,
            self.dialect_att_bias,
            self.char_att_weight,
            self.char_att_bias,
            self.trans_weight
        ] + self.target_embs

        if self.target_bias:
            self.target_biases = []
            for target in self.targets:
                self.target_biases.append(tf.Variable(tf.random_normal_initializer()(
                    shape=(target.shape[0],),
                    dtype=tf.float32
                )))

            self.trainable_variables += self.target_biases

        self.optimizer = tf.optimizers.Adam() if optimizer is None else optimizer

    def dialect_to_id(self, dialect):
        return self.dialect_table.lookup(tf.convert_to_tensor(dialect))

    def id_to_dialect(self, dialect_id):
        return tf.gather(self.dialects, dialect_id)

    def char_to_id(self, char):
        return self.char_table.lookup(tf.convert_to_tensor(char))

    def id_to_char(self, char_id):
        return tf.gather(self.chars, char_id)

    def target_to_id(self, index, target):
        return self.target_tables[index].lookup(tf.convert_to_tensor(target))

    def id_to_target(self, index, target_id):
        return tf.gather(self.targets[index], target_id)

    def get_dialect_emb(self, dialect):
        return tf.nn.embedding_lookup(self.dialect_emb, self.dialect_to_id(dialect))

    def get_char_emb(self, char):
        return tf.nn.embedding_lookup(self.char_emb, self.char_to_id(char))

    def get_target_emb(self, index, target):
        return tf.nn.embedding_lookup(
            self.target_embs[index],
            self.target_to_id(index, target)
        )

    def dialect_att(self, dialect_emb):
        return tf.nn.sigmoid(tf.tensordot(
            dialect_emb,
            self.dialect_att_weight,
            [1, 0]
        ) + self.dialect_att_bias[None, :, :])

    def char_att(self, char_emb):
        return self.activation(tf.tensordot(
            char_emb,
            self.char_att_weight,
            [1, 0]
        ) + self.char_att_bias[None, :, :])

    @tf.function
    def transform(self, dialect_emb, char_emb):
        att = self.dialect_att(dialect_emb) * self.char_att(char_emb)
        return char_emb + tf.tensordot(att, self.trans_weight, [[1, 2], [0, 1]])

    def logits(self, dialect_emb, char_emb):
        emb = self.transform(dialect_emb, char_emb)
        logits = [tf.matmul(emb, e, transpose_b=True) for e in self.target_embs]

        if self.target_bias:
            logits = [l + b for l, b in zip(logits, self.target_biases)]

        return logits

    @tf.function
    def predict_id(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        dialect_emb = self.get_dialect_emb(inputs[:, 0])
        char_emb = self.get_char_emb(inputs[:, 1])
        logits = self.logits(dialect_emb, char_emb)
        return tf.stack(
            [tf.argmax(l, axis=1, output_type=tf.int32) for l in logits],
            axis=1
        )

    @tf.function
    def predict(self, inputs):
        ids = self.predict_id(inputs)
        return tf.stack(
            [self.id_to_target(i, ids[:, i]) for i in range(ids.shape[1])],
            axis=1
        )

    @tf.function
    def predict_proba(self, inputs):
        inputs = tf.convert_to_tensor(inputs)
        dialect_emb = self.get_dialect_emb(inputs[:, 0])
        char_emb = self.get_char_emb(inputs[:, 1])
        logits = self.logits(dialect_emb, char_emb)
        return [tf.nn.softmax(l) for l in logits]

    @tf.function
    def predict_id_emb(self, dialect_emb, char_emb):
        logits = self.logits(dialect_emb, char_emb)
        return tf.stack(
            [tf.argmax(l, axis=1, output_type=tf.int32) for l in logits],
            axis=1
        )

    @tf.function
    def predict_emb(self, dialect_emb, char_emb):
        ids = self.predict_id_emb(dialect_emb, char_emb)
        return tf.stack(
            [self.id_to_target(i, ids[:, i]) for i in range(ids.shape[1])],
            axis=1
        )

    @tf.function
    def predict_proba_emb(self, dialect_emb, char_emb):
        logits = self.logits(dialect_emb, char_emb)
        return [tf.nn.softmax(l) for l in logits]

    @tf.function
    def loss(self, inputs, targets):
        inputs = tf.convert_to_tensor(inputs)
        targets = tf.convert_to_tensor(targets)

        dialect_emb = self.get_dialect_emb(inputs[:, 0])
        char_emb = self.get_char_emb(inputs[:, 1])
        logits = self.logits(dialect_emb, char_emb)

        pred_ids = tf.stack(
            [tf.argmax(l, axis=1, output_type=tf.int32) for i, l in enumerate(logits)],
            axis=1
        )

        target_ids = tf.stack(
            [self.target_to_id(i, targets[:, i]) for i in range(targets.shape[1])],
            axis=1
        )

        loss = []
        for i, l in enumerate(logits):
            loss.append(tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=target_ids[:, i],
                logits=l
            ))
        loss = tf.reduce_sum(tf.stack(loss, axis=1), axis=1)

        if self.emb_l2 > 0:
            emb = self.transform(dialect_emb, char_emb)
            loss += self.emb_l2 * tf.reduce_sum(
                tf.square(emb - char_emb),
                axis=1
            )

        if self.l2 > 0:
            for v in self.trainable_variables:
                loss += self.l2 * tf.reduce_sum(tf.square(v))

        return loss, target_ids, pred_ids

    @tf.function
    def update(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss, target_ids, pred_ids = self.loss(inputs, targets)
        grad = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grad, self.trainable_variables))
        return loss, target_ids, pred_ids

def load_data(prefix, ids, suffix='mb01dz.csv'):
    '''加载方言字音数据'''

    logging.info(f'loading {len(ids)} files ...')

    data = []
    for id in ids:
        fname = os.path.join(prefix, id + suffix)
        try:
            d = pandas.read_csv(fname, encoding='utf-8', dtype=str)
            d['oid'] = id
            data.append(d)
        except Exception as e:
            logging.error(f'cannot load file {fname}: {e}')

    dialects = len(data)
    data = clean_data(pandas.concat(data, ignore_index=True), minfreq=2)
    data = data[(data['initial'] != '') | (data['finals'] != '') | (data['tone'] != '')]

    logging.info(f'done. loaded {dialects} dialects {data.shape[0]} records')
    return data


if __name__ == '__main__':
    prefix = sys.argv[1]

    dialect_path = os.path.join(prefix, 'dialect')
    location = pandas.read_csv(
        os.path.join(dialect_path, 'location.csv'),
        index_col=0
    ).sample(100)
    char = pandas.read_csv(os.path.join(prefix, 'words.csv'), index_col=0)
    data = load_data(dialect_path, location.index)

    emb_size = 20
    encoder = DialectPredictor(
        data['oid'].unique(),
        data['iid'].unique(),
        (data['initial'].unique(), data['finals'].unique(), data['tone'].unique()),
        emb_size=emb_size,
        transform_layer=1,
        transform_size=100,
        activation=tf.nn.softmax,
        target_bias=False,
        l2=0
    )

    dataset = tf.data.Dataset.from_tensor_slices(data[['oid', 'iid', 'initial', 'finals', 'tone']].values).shuffle(100000).map(lambda x: (x[:2], x[2:]))
    train_dataset = dataset.skip(10000)
    test_dataset = dataset.take(10000)

    output_prefix = os.path.join(
        'tensorboard',
        'encoder',
        datetime.datetime.now().strftime('%Y%m%d%H%M')
    )

    log_dir = output_prefix
    summary_writer = tf.summary.create_file_writer(log_dir)

    loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    acc = tf.keras.metrics.Accuracy('acc', dtype=tf.float32)
    eval_loss = tf.keras.metrics.Mean('eval_loss', dtype=tf.float32)
    eval_acc = tf.keras.metrics.Accuracy('eval_acc', dtype=tf.float32)

    checkpoint = tf.train.Checkpoint(
        dialect_emb=encoder.dialect_emb,
        char_emb=encoder.char_emb,
        optimizer=encoder.optimizer
    )
    manager = tf.train.CheckpointManager(checkpoint, os.path.join(output_prefix, 'checkpoints'), max_to_keep=10)

    epochs = 100
    batch_size = 100

    for epoch in range(epochs):
        for inputs, targets in train_dataset.batch(batch_size):
            l, target_ids, pred_ids = encoder.update(inputs, targets)
            loss.update_state(l)
            acc.update_state(target_ids, pred_ids)

        for inputs, targets in test_dataset.batch(batch_size):
            l, target_ids, pred_ids = encoder.loss(inputs, targets)
            eval_loss.update_state(l)
            eval_acc.update_state(target_ids, pred_ids)

        with summary_writer.as_default():
                tf.summary.scalar('loss', loss.result(), step=epoch)
                tf.summary.scalar('acc', acc.result(), step=epoch)
                tf.summary.scalar('eval_loss', eval_loss.result(), step=epoch)
                tf.summary.scalar('eval_acc', eval_acc.result(), step=epoch)

        loss.reset_states()
        acc.reset_states()
        eval_loss.reset_states()
        eval_acc.reset_states()

        manager.save()