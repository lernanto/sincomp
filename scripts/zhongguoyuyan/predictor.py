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
import numpy
import tensorflow as tf


class Predictor(tf.train.Checkpoint):
    '''预测指定方言点的字音.'''

    def __init__(
        self,
        dialects,
        chars,
        targets,
        dialect_emb_size=20,
        char_emb_size=20,
        target_sim='inner_product',
        l2=0,
        optimizer=None,
        name='predictor'
    ):
        super().__init__(epoch=tf.Variable(0, dtype=tf.int64))

        self.dialects = tf.convert_to_tensor(dialects)
        self.chars = tf.convert_to_tensor(chars)
        self.targets = [tf.convert_to_tensor(t) for t in targets]
        self.dialect_emb_size = dialect_emb_size
        self.char_emb_size = char_emb_size
        self.target_sim = target_sim
        self.l2 = l2
        self.name = name
        self.variables = []

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

        init = tf.random_normal_initializer()

        self.dialect_emb = tf.Variable(init(
            shape=(self.dialect_table.size(), self.dialect_emb_size),
            dtype=tf.float32
        ), name='dialect_emb')
        self.char_emb = tf.Variable(init(
            shape=(self.char_table.size(), self.char_emb_size),
            dtype=tf.float32
        ), name='char_emb')

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
            self.target_embs.append(tf.Variable(init(
                shape=(target.shape[0], self.char_emb_size),
                dtype=tf.float32
            ), name=f'target_emb{i}'))

        self.add_variable(self.dialect_emb, self.char_emb, *self.target_embs)

        if self.target_sim == 'inner_product':
            self.target_biases = []
            for i, target in enumerate(self.targets):
                self.target_biases.append(tf.Variable(
                    init(shape=(target.shape[0],), dtype=tf.float32),
                    name=f'target_bias{i}'
                ))

            self.add_variable(*self.target_biases)

        self.optimizer = tf.optimizers.Adam(0.02) if optimizer is None else optimizer

    def add_variable(self, *args):
        self.variables.extend(args)

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

    def logits(self, dialect_emb, char_emb):
        emb = self.transform(dialect_emb, char_emb)

        if self.target_sim == 'inner_product':
            logits = [tf.matmul(emb, e, transpose_b=True) + b[None, :] \
                for e, b in zip(self.target_embs, self.target_biases)]
        elif self.target_sim == 'euclidean_distance':
            logits = [-tf.reduce_sum(
                tf.square(emb[:, None] - e[None, :]),
                axis=-1
            ) for e in self.target_embs]

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

        if self.l2 > 0:
            for v in self.variables:
                loss += self.l2 * tf.reduce_sum(tf.square(v))

        return loss, target_ids, pred_ids

    @tf.function
    def update(self, inputs, targets):
        with tf.GradientTape() as tape:
            loss, target_ids, pred_ids = self.loss(inputs, targets)
        grad = tape.gradient(loss, self.variables)
        self.optimizer.apply_gradients(zip(grad, self.variables))
        return loss, target_ids, pred_ids

    def train(
        self,
        train_data,
        eval_data,
        epochs=20,
        batch_size=100,
        output_path=None
    ):
        if output_path is None:
            output_path = os.path.join(
                self.name,
                f'{datetime.datetime.now():%Y%m%d%H%M}'
            )

        train_loss = tf.keras.metrics.Mean('train_loss', dtype=tf.float32)
        train_acc = tf.keras.metrics.Accuracy('train_acc', dtype=tf.float32)
        eval_loss = tf.keras.metrics.Mean('eval_loss', dtype=tf.float32)
        eval_acc = tf.keras.metrics.Accuracy('eval_acc', dtype=tf.float32)

        writer = tf.summary.create_file_writer(output_path)
        manager = tf.train.CheckpointManager(
            self,
            os.path.join(output_path, 'checkpoints'),
            max_to_keep=epochs
        )

        while self.epoch < epochs:
            for inputs, targets in train_data.shuffle(10000).batch(batch_size):
                loss, target_ids, pred_ids = self.update(inputs, targets)
                train_loss.update_state(loss)
                train_acc.update_state(target_ids, pred_ids)

            for inputs, targets in eval_data.batch(batch_size):
                loss, target_ids, pred_ids = self.loss(inputs, targets)
                eval_loss.update_state(loss)
                eval_acc.update_state(
                    target_ids,
                    pred_ids,
                    target_ids < tf.cast(tf.stack([t.shape[0] for t in self.targets]), tf.int64)
                )

            with writer.as_default():
                tf.summary.scalar(train_loss.name, train_loss.result(), step=self.epoch)
                tf.summary.scalar(train_acc.name, train_acc.result(), step=self.epoch)
                tf.summary.scalar(eval_loss.name, eval_loss.result(), step=self.epoch)
                tf.summary.scalar(eval_acc.name, eval_acc.result(), step=self.epoch)

            train_loss.reset_states()
            train_acc.reset_states()
            eval_loss.reset_states()
            eval_acc.reset_states()

            self.epoch.assign_add(1)
            manager.save()

class LinearPredictor(Predictor):
    '''方言 embedding 和字 embedding 直接相加.'''

    def __init__(self, *args, emb_size=20, name='linear_predictor', **kwargs):
        super().__init__(
            *args,
            dialect_emb_size=emb_size,
            char_emb_size=emb_size,
            name=name,
            **kwargs
        )

    def transform(self, dialect_emb, char_emb):
        return dialect_emb + char_emb

class MLPPredictor(Predictor):
    '''使用 MLP 作为字音变换.'''

    def __init__(
        self,
        *args,
        hidden_layer=2,
        hidden_size=100,
        activation=tf.nn.relu,
        name='mlp_predictor',
        **kwargs
    ):
        super().__init__(*args, name=name, **kwargs)

        self.activation = activation

        init = tf.random_normal_initializer()
        self.weights = []
        self.biases = []
        for i in range(hidden_layer):
            input_shape = self.dialect_emb_size + self.char_emb_size if i == 0 else hidden_size
            output_shape = self.char_emb_size if i == hidden_layer - 1 else hidden_size

            self.weights.append(tf.Variable(
                init(shape=(input_shape, output_shape), dtype=tf.float32),
                name=f'weight{i}')
            )
            self.biases.append(tf.Variable(
                init(shape=(output_shape,), dtype=tf.float32),
                name=f'bias{i}'
            ))

        self.add_variable(*self.weights, *self.biases)

    def transform(self, dialect_emb, char_emb):
        x = tf.concat([dialect_emb, char_emb], axis=1)
        for w, b in zip(self.weights, self.biases):
            x = self.activation(tf.matmul(x, w) + b[None, :])

        return x

class ResidualPredictor(Predictor):
    '''预测方言字音变换的残差.'''

    def __init__(
        self,
        *args,
        hidden_size=100,
        activation=tf.nn.relu,
        name='residual_predictor',
        **kwargs
    ):
        super().__init__(*args, name=name, **kwargs)

        self.activation = activation

        init = tf.random_normal_initializer()
        self.weight0 = tf.Variable(init(
            shape=(self.dialect_emb_size + self.char_emb_size, hidden_size),
            dtype=tf.float32
        ), name='weight0')
        self.bias = tf.Variable(
            init(shape=(hidden_size,), dtype=tf.float32),
            name='bias'
        )
        self.weight1 = tf.Variable(
            init(shape=(hidden_size, self.char_emb_size), dtype=tf.float32),
            name='weight1'
        )

        self.add_variable(self.weight0, self.bias, self.weight1)

    def transform(self, dialect_emb, char_emb):
        return tf.matmul(
            self.activation(tf.matmul(
                tf.concat([dialect_emb, char_emb], axis=1),
                self.weight0
            ) + self.bias[None, :]),
            self.weight1
        )

class AttentionPredictor(Predictor):
    def __init__(
        self,
        *args,
        transform_heads=5,
        transform_size=20,
        dialect_att_activation=tf.nn.sigmoid,
        char_att_activation=tf.nn.softmax,
        name='attention_predictor',
        **kwargs
    ):
        super().__init__(*args, name=name, **kwargs)

        self.dialect_att_activation = dialect_att_activation
        self.char_att_activation = char_att_activation

        init = tf.random_normal_initializer()
        self.dialect_att_weight = tf.Variable(init(
            shape=(self.dialect_emb_size, transform_heads, transform_size),
            dtype=tf.float32
        ), name='dialect_att_weight')
        self.dialect_att_bias = tf.Variable(
            init(shape=(transform_heads, transform_size), dtype=tf.float32),
            name='dialect_att_bias'
        )
        self.char_att_weight = tf.Variable(init(
            shape=(self.char_emb_size, transform_heads, transform_size),
            dtype=tf.float32
        ), name='char_att_weight')
        self.char_att_bias = tf.Variable(
            init(shape=(transform_heads, transform_size), dtype=tf.float32),
            name='char_bias'
        )
        self.trans_weight = tf.Variable(init(
            shape=(transform_heads, transform_size, self.char_emb_size),
            dtype=tf.float32
        ), name='trans_weight')

        self.add_variable(
            self.dialect_att_weight,
            self.dialect_att_bias,
            self.char_att_weight,
            self.char_att_bias,
            self.trans_weight
        )

    def dialect_att(self, dialect_emb):
        return self.dialect_att_activation(tf.tensordot(
            dialect_emb,
            self.dialect_att_weight,
            [1, 0]
        ) + self.dialect_att_bias[None, :, :])

    def char_att(self, char_emb):
        return self.char_att_activation(tf.tensordot(
            char_emb,
            self.char_att_weight,
            [1, 0]
        ) + self.char_att_bias[None, :, :])

    def transform(self, dialect_emb, char_emb):
        att = self.dialect_att(dialect_emb) * self.char_att(char_emb)
        return char_emb + tf.tensordot(att, self.trans_weight, [[1, 2], [0, 1]])

def load_data(prefix, ids, suffix='mb01dz.csv'):
    '''加载方言字音数据'''

    data = []
    for id in ids:
        fname = os.path.join(prefix, id + suffix)
        try:
            d = pandas.read_csv(
                fname,
                encoding='utf-8',
                dtype={'iid': int, 'initial': str, 'finals': str, 'tone': str},
            )
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

def benchmark(data):
    data = data.astype(str)
    dialects = data['oid'].unique()
    chars = data['iid'].unique()
    initials = data['initial'].unique()
    finals = data['finals'].unique()
    tones = data['tone'].unique()

    dataset = tf.data.Dataset.from_tensor_slices((
        data[['oid', 'iid']].values,
        data[['initial', 'finals', 'tone']].values
    ))
    eval_size = int(data.shape[0] * 0.1)
    train_data = dataset.skip(eval_size)
    eval_data = dataset.take(eval_size)

    LinearPredictor(dialects, chars, (initials, finals, tones)) \
        .train(train_data, eval_data)
    MLPPredictor(dialects, chars, (initials, finals, tones)) \
        .train(train_data, eval_data)
    ResidualPredictor(dialects, chars, (initials, finals, tones)) \
        .train(train_data, eval_data)
    AttentionPredictor(dialects, chars, (initials, finals, tones)) \
        .train(train_data, eval_data)


if __name__ == '__main__':
    prefix = sys.argv[1]

    dialect_path = os.path.join(prefix, 'dialect')
    location = pandas.read_csv(
        os.path.join(dialect_path, 'location.csv'),
        index_col=0
    ).sample(100)
    char = pandas.read_csv(os.path.join(prefix, 'words.csv'), index_col=0)
    data = load_data(dialect_path, location.index).sample(frac=0.8)

    benchmark(data)