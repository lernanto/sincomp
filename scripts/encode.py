#!/usr/bin/env -S python3 -O
# -*- coding: utf-8 -*-

"""
使用编码器训练方言音系 embedding.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import argparse
import os
import datetime
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorboard.plugins import projector

import sincomp.datasets
import sincomp.auxiliary
import sincomp.models


if __name__ == '__main__':
    parser = argparse.ArgumentParser(globals().get('__doc__'))
    parser.add_argument('-o', '--output', default='.', help='输出目录')
    parser.add_argument('--embedding-size', type=int, default=10, help='向量大小')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    parser.add_argument('--batch-size', type=int, default=100, help='批大小')
    parser.add_argument(
        '--sample-rate',
        type=float,
        default=0.5,
        help='输入降噪自编码器的特征采样率'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.01,
        help='学习率'
    )
    parser.add_argument('dataset', help='使用的方言数据集')
    parser.add_argument('group', nargs='+', help='要建模的方言大类')
    args = parser.parse_args()

    # 加载方言数据
    dataset = getattr(sincomp.datasets, args.dataset)
    char = dataset.metadata['char_info'][['character']]
    info = dataset.metadata['dialect_info']
    dataset = dataset.filter(info[info['group'].isin(args.group)].index)

    # 展开成字为行、方言点为列的字音矩阵
    data = sincomp.preprocess.transform(
        dataset,
        index='cid',
        values=['initial', 'final', 'tone'],
        aggfunc='first'
    )
    char = char.reindex(data.index)

    data = data.swaplevel(axis=1).reindex(columns=pd.MultiIndex.from_product((
        ['initial', 'final', 'tone'],
        data.columns.levels[0]
    )))

    # 方言声韵调编码，缺失值为 -1
    codes, categories = sincomp.auxiliary.encode(data)
    bases = np.insert(
        np.cumsum([c.shape[0] for c in categories])[:-1],
        0,
        0
    )
    codes = pd.DataFrame(
        data=np.where(codes >= 0, codes + bases, -1),
        index=data.index,
        columns=data.columns
    )

    generator = sincomp.models.ContrastiveGenerator(
        codes.loc[:, 'initial'].values,
        codes.loc[:, 'final'].values,
        codes.loc[:, 'tone'].values
    )

    dataset = tf.data.Dataset.from_generator(
        generator.contrastive(args.sample_rate),
        output_types=(
            (tf.int32, tf.int32, tf.int32),
            (tf.int32, tf.int32, tf.int32)
        )
    ).shuffle(1000).padded_batch(
        args.batch_size,
        padded_shapes=(((None,), (None,), (None,)), ((None,), (None,), (None,))),
        padding_values=-1,
        drop_remainder=True
    )

    encoder = sincomp.models.AutoEncoder(
        sum(c.shape[0] for c in categories),
        args.embedding_size
    )
    optimizer = tf.optimizers.Adam(args.learning_rate)

    output_prefix = os.path.join(
        args.output,
        datetime.datetime.now().strftime('%Y%m%d%H%M')
    )

    summary_writer = tf.summary.create_file_writer(output_prefix)
    loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)

    checkpoint = tf.train.Checkpoint(
        embedding=encoder.embedding,
        optimizer=optimizer
    )
    manager = tf.train.CheckpointManager(
        checkpoint,
        os.path.join(output_prefix, 'checkpoints'),
        max_to_keep=10
    )

    for epoch in range(args.epochs):
        for inputs, targets in dataset:
            loss(encoder.update(inputs, targets, optimizer))

        with summary_writer.as_default():
                tf.summary.scalar('loss', loss.result(), step=epoch)
        loss.reset_states()

        if epoch % 10 == 9:
            manager.save()

    # 保存声韵调向量及相关数据
    variables = {}
    config = projector.ProjectorConfig()
    for col in ('initial', 'final', 'tone'):
        name = f'{col}_embedding'
        fname = f'{col}.tsv'
        variables[name] = tf.Variable(encoder.encode(codes.loc[:, col].values))

        pd.concat([char, data.loc[:, col]], axis=1).to_csv(
            os.path.join(output_prefix, fname),
            sep='\t',
            encoding='utf-8',
            lineterminator='\n'
        )

        emb = config.embeddings.add()
        emb.tensor_name = f'{name}/.ATTRIBUTES/VARIABLE_VALUE'
        emb.metadata_path = fname

    tf.train.Checkpoint(**variables).save(
        os.path.join(output_prefix, 'embedding.ckpt')
    )
    projector.visualize_embeddings(output_prefix, config)
