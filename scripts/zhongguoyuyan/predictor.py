#!/usr/bin/python3 -O
# -*- coding: utf-8 -*-

'''
方言字音编码器.
'''

__author__ = '黄艺华 <lernanto@foxmail.com>'


import os
import logging
import datetime
import pandas
import numpy
import tensorflow as tf


class DialectPredictor:
    def __init__(
        self,
        dialects,
        chars,
        outputs,
        emb_size=20,
        dialect_emb_size=None,
        char_emb_size=None,
        residual_layer=2,
        residual_size=100,
        activation=tf.nn.relu,
        output_bias=False,
        emb_l2=0.01,
        l2=0,
        optimizer=None
    ):
        if dialect_emb_size is None:
            dialect_emb_size = emb_size
        if char_emb_size is None:
            char_emb_size = emb_size

        self.dialects = tf.convert_to_tensor(dialects)
        self.chars = tf.convert_to_tensor(chars)
        self.outputs = [tf.convert_to_tensor(o) for o in outputs]
        self.activation = activation
        self.output_bias = output_bias
        self.emb_l2 = emb_l2
        self.l2 = l2

        self.dialect_table = tf.lookup.StaticVocabularyTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=dialects,
                values=tf.range(dialects.shape[0], dtype=tf.int64)
            ),
            num_oov_buckets=1
        )
        self.char_table = tf.lookup.StaticVocabularyTable(
            initializer=tf.lookup.KeyValueTensorInitializer(
                keys=chars,
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

        self.res_weights1 = []
        self.res_weights2 = []
        for i in range(residual_layer):
            self.res_weights1.append(tf.Variable(tf.random_normal_initializer()(
                shape=(dialect_emb_size + char_emb_size, residual_size),
                dtype=tf.float32
            ), name='res_weight{}_1'.format(i)))
            self.res_weights2.append(tf.Variable(tf.random_normal_initializer()(
                shape=(residual_size, char_emb_size),
                dtype=tf.float32
            ), name='res_weight{}_2'.format(i)))

        self.output_tables = []
        for output in outputs:
            self.output_tables.append(tf.lookup.StaticVocabularyTable(
                tf.lookup.KeyValueTensorInitializer(
                    keys=output,
                    values=tf.range(output.shape[0], dtype=tf.int64)
                ),
                num_oov_buckets=1
            ))

        self.output_embs = []
        self.output_biases = []
        for i, output in enumerate(outputs):
            self.output_embs.append(tf.Variable(tf.random_normal_initializer()(
                shape=(output.shape[0], char_emb_size),
                dtype=tf.float32
            ), name='output{}'.format(i)))

        self.trainable_variables = [self.dialect_emb, self.char_emb] \
            + self.res_weights1 + self.res_weights2 + self.output_embs

        if self.output_bias:
            self.output_biases = []
            for output in outputs:
                self.output_biases.append(tf.Variable(tf.random_normal_initializer()(
                    shape=(output.shape[0],),
                    dtype=tf.float32
                )))

            self.trainable_variables += self.output_biases

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
        return self.output_tables[index].lookup(tf.convert_to_tensor(target))

    def id_to_target(self, index, target_id):
        return tf.gather(self.outputs[index], target_id)

    def get_dialect_emb(self, dialect):
        return tf.nn.embedding_lookup(self.dialect_emb, self.dialect_to_id(dialect))

    def get_char_emb(self, char):
        return tf.nn.embedding_lookup(self.char_emb, self.char_to_id(char))

    def get_target_emb(self, index, target):
        return tf.nn.embedding_lookup(
            self.output_embs[index],
            self.target_to_id(index, target)
        )

    @tf.function
    def transform(self, dialect_emb, char_emb):
        emb = char_emb
        for w1, w2 in zip(self.res_weights1, self.res_weights2):
            x = tf.concat([dialect_emb, emb], axis=1)
            x = self.activation(tf.matmul(x, w1))
            emb += tf.matmul(x, w2)

        return emb

    def logits(self, dialect_emb, char_emb):
        emb = self.transform(dialect_emb, char_emb)
        logits = [tf.matmul(emb, e, transpose_b=True) for e in self.output_embs]
        if self.output_bias:
            logits = [l + b for l, b in zip(logits, self.output_biases)]

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
        output_emb = self.transform(dialect_emb, char_emb)

        logits = [tf.matmul(output_emb, e, transpose_b=True) for e in self.output_embs]
        if self.output_bias:
            logits = [l + b for l, b in zip(logits, self.output_biases)]

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
            loss += self.emb_l2 * tf.reduce_sum(
                tf.square(output_emb - char_emb),
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

    data = []
    for id in ids:
        fname = os.path.join(prefix, id + suffix)
        try:
            d = pandas.read_csv(fname, encoding='utf-8', dtype=str)
            d['oid'] = id
            data.append(d)
        except Exception as e:
            logging.error(f'cannot load file {fname}: {e}')

    data = pandas.concat(data, ignore_index=True)

    # 读音数据中有些输入错误，清洗
    ipa = 'A-Za-z\u00c0-\u03ff\u1d00-\u1dbf\u1e00-\u1eff\u2205\u2c60-\u2c7f\ua720-\ua7ff\uab30-\uab6f\ufb00-\ufb4f\U00010780-\U000107ba\U0001df00-\U0001df1e'

    # 有些符号使用了多种写法，统一成较常用的一种
    clean = data['initial'].fillna('').str.lower() \
        .str.replace(f'[^{ipa}]', '') \
        .str.replace('[\u00f8\u01ff]', '\u2205') \
        .str.replace('\u02a3', 'dz') \
        .str.replace('\u02a4', 'dʒ') \
        .str.replace('\u02a5', 'dʑ') \
        .str.replace('\u02a6', 'ts') \
        .str.replace('\u02a7', 'tʃ') \
        .str.replace('\u02a8', 'tɕ') \
        .str.replace('[\u02b0\u02b1]', 'h') \
        .str.replace('g', 'ɡ')

    # 删除出现次数少的读音
    clean[clean.groupby(clean).transform('count') <= 2] = ''
    mask = data['initial'] != clean
    if numpy.count_nonzero(mask):
        for (r, c), cnt in pandas.DataFrame({
            'raw': data.loc[mask, 'initial'],
            'clean': clean[mask]
        }).value_counts().iteritems():
            logging.warn(f'replace {r} -> {c} {cnt}')

    data['initial'] = clean

    clean = data['finals'].fillna('').str.lower().str.replace(f'[^{ipa}]', '')
    clean[clean.groupby(clean).transform('count') <= 2] = ''
    mask = data['finals'] != clean
    if numpy.count_nonzero(mask):
        for (r, c), cnt in pandas.DataFrame({
            'raw': data.loc[mask, 'finals'],
            'clean': clean[mask]
        }).value_counts().iteritems():
            logging.warn(f'replace {r} -> {c} {cnt}')

    data['finals'] = clean

    clean = data['tone'].fillna('').str.lower().str.replace(f'[^1-5]', '')
    clean[clean.groupby(clean).transform('count') <= 2] = ''
    mask = data['tone'] != clean
    if numpy.count_nonzero(mask):
        for (r, c), cnt in pandas.DataFrame({
            'raw': data.loc[mask, 'tone'],
            'clean': clean[mask]
        }).value_counts().iteritems():
            logging.warn(f'replace {r} -> {c} {cnt}')

    data['tone'] = clean

    return data


if __name__ == '__main__':
    prefix = r'D:\git\zhongguoyuyan\csv\dialect'
    location = pandas.read_csv(os.path.join(prefix, 'location.csv'), index_col=0)
    char = pandas.read_csv(r'D:\git\zhongguoyuyan\csv\words.csv', index_col=0)
    sample = location.sample(100)

    data = load_data(prefix, sample.index)

    emb_size = 20
    encoder = DialectPredictor(
        data['oid'].unique(),
        data['iid'].unique(),
        (data['initial'].unique(), data['finals'].unique(), data['tone'].unique()),
        emb_size=emb_size,
        residual_layer=2,
        residual_size=100,
        activation=tf.nn.relu,
        output_bias=False,
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