#!/usr/bin/python3 -O
# -*- coding: utf-8 -*-

"""
训练和评估方言字音编码器.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import logging
import argparse
import os
import json
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorboard.plugins import projector

import sinetym.models
import sinetym.auxiliary
from sinetym.datasets import xiaoxuetang, zhongguoyuyan


def load_dictionaries(prefix):
    """
    加载词典.

    Parameters:
        prefix (str): 词典路径前缀

    Returns:
        did, cid, intitial, final, tone (pd.Series): 词典列表
    """

    logging.info(f'load dictionaries from {prefix}')

    dics = {}
    for name in ('did', 'cid', 'character', 'initial', 'final', 'tone'):
        dics[name] = pd.read_csv(
            os.path.join(prefix, f'{name}.csv'),
            index_col=0,
            encoding='utf-8'
        )

    return (
        dics['did'],
        dics['cid'],
        dics['character'],
        dics['initial'],
        dics['final'],
        dics['tone']
    )

def load_datasets(config):
    """
    加载数据集.

    Parameters:
        config (dict): 数据集配置

    Returns:
        data (`pandas.DataFrame`): 所有数据集汇总成的数据表
    """

    logging.info(f'loading datasets...')

    data = []
    for d in config:
        if isinstance(d, str):
            dataset = getattr(sinetym.datasets, d)

        else:
            d = d.copy()
            dataset = getattr(sinetym.datasets, d.pop('name')).filter(**d)

        # 如果数据集包含方言变体，把变体编号加入为方言 ID 的一部分
        if 'variant' in dataset.columns:
            dataset = dataset.assign(did=dataset['did'] + dataset['variant'])

        data.append(dataset[['did', 'cid', 'character', 'initial', 'final', 'tone']])

    data = pd.concat(data, axis=0, ignore_index=True)

    logging.info(f'done, {data.shape[0]} data loaded.')
    return data

def build_model(config):
    """
    根据配置创建模型和优化器.

    Parameters:
        config (dict): 多层级的配置字典，分析配置文件获得

    Returns:
        model (sinetym.models.EncoderBase): 创建的编码器模型
        optimizer (`tensorflow.optimizers.Optimizer`): 创建的优化器
    """

    # 创建模型
    model_config = config.pop('model').copy()
    model_class = getattr(sinetym.models, model_config.pop('class'))
    model = model_class(**model_config)

    # 创建优化器
    optimizer_config = config.pop('optimizer').copy()
    optimizer_class = getattr(tf.optimizers, optimizer_config.pop('class'))

    # 学习率可以是一个对象，根据配置创建
    lr_config = optimizer_config.pop('learning_rate').copy()
    if isinstance(lr_config, object):
        lr_class = getattr(tf.optimizers.schedules, lr_config.pop('class'))
        lr = lr_class(**lr_config)
    else:
        lr = lr_config

    optimizer = optimizer_class(learning_rate=lr, **optimizer_config)

    return model, optimizer

def make_embeddings(
    location,
    char,
    initial,
    final,
    tone,
    encoder,
    output_path=''
):
    """
    为 TensorBoard embedding projector 显示向量创建需要的数据.

    Parameters:
        location (`pd.DataFrame`): 方言点信息数据表
        char (`pd.DataFrame`): 字信息数据表
        initial (`pd.Series`): 模型的声母表
        final (`pd.Series`): 模型的韵母表
        tone (`pd.Series`): 模型的声调表
        encoder (`PredictorBase`): 方言字音预测模型
        output_path (str): 输出向量数据的路径前缀
    """

    logging.info(f'save embeddings to {output_path} ...')

    os.makedirs(output_path, exist_ok=True)

    # 创建向量字典
    location[['name', 'dialect']].fillna('').to_csv(
        os.path.join(output_path, 'dialect.tsv'),
        sep='\t',
        encoding='utf-8',
        lineterminator='\n'
    )

    char[['item']].fillna('').to_csv(
        os.path.join(output_path, 'char.tsv'),
        sep='\t',
        encoding='utf-8',
        lineterminator='\n'
    )

    for name in ('initial', 'final', 'tone'):
        locals()[name].to_csv(
            os.path.join(output_path, f'{name}.tsv'),
            header=False,
            index=False,
            encoding='utf-8',
            lineterminator='\n'
        )

    # 保存向量值
    embeddings = {
        'dialect': encoder.dialect_emb,
        'char': encoder.input_embs[0],
        'initial': encoder.output_embs[0],
        'final': encoder.output_embs[1],
        'tone': encoder.output_embs[2]
    }

    checkpoint = tf.train.Checkpoint(**embeddings)
    checkpoint.save(os.path.join(output_path, 'embeddings.ckpt'))

    # 创建向量元数据
    config = projector.ProjectorConfig()
    for name in embeddings:
        emb = config.embeddings.add()
        emb.tensor_name = f'{name}/.ATTRIBUTES/VARIABLE_VALUE'
        emb.metadata_path = f'{name}.tsv'

    projector.visualize_embeddings(output_path, config)

def benchmark(config, train_data, eval_data=None):
    """
    评估各种配置的模型效果.

    Parameters:
        config (dict): 用于创建模型的配置字典
        train_data (`tensorflow.data.Dataset`): 训练数据
        eval_data (`tensorflow.data.Dataset`): 评估数据

    从 config 读取所有模型配置，为每一组配置创建一个模型，训练并评估效果。
    """

    for conf in config['models']:
        conf = conf.copy()
        name = conf.pop('name')
        model, optimizer = build_model(conf)
        output_path = os.path.join(config['output_dir'], name)

        logging.info(f'start training model {name}, output path = {output_path} ...')
        model.fit(
            optimizer,
            train_data,
            eval_data,
            output_path=output_path,
            **conf
        )
        logging.info('done.')

def train(config, name, train_data, eval_data=None):
    """
    训练模型.

    Parameters:
        config (dict): 多层级的配置字典，分析配置文件获得
        name (str): 用于训练的模型配置名称，用于从 config 中读取指定配置
        train_data (`tensorflow.data.Dataset`): 训练数据
        eval_data (`tensorflow.data.Dataset`): 评估数据

    从 config 中读取由 name 指定的配置，创建并训练模型。
    """

    # 从模型列表中查找待训练的模型
    conf = next((c for c in config['models'] if c['name'] == name)).copy()
    conf.pop('name')
    model, optimizer = build_model(conf)
    output_path = os.path.join(config['output_dir'], name)

    logging.info(f'training model {name}, output path = {output_path}...')
    model.fit(
        optimizer,
        train_data,
        eval_data=eval_data,
        output_path=output_path,
        **conf
    )
    logging.info('done.')

def evaluate(config, name, eval_data):
    """
    评估训练完成的模型效果.

    Parameters:
        config (dict): 多层级的配置字典，分析配置文件获得
        name (str): 用于训练的模型配置名称，用于从 config 中读取指定配置
        eval_data (`tensorflow.data.Dataset`): 评估数据
    """

    conf = next((c for c in config['models'] if c['name'] == name)).copy()
    model, _ = build_model(conf)

    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(
        checkpoint,
        os.path.join(config['output_dir'], name, 'checkpoints'),
        None
    )
    logging.info(f'restore model from checkpoint {manager.latest_checkpoint} ...')
    checkpoint.restore(manager.latest_checkpoint)
    logging.info('done.')

    logging.info('evaluating model ...')
    loss, acc = model.evaluate(eval_data.batch(conf.get('batch_size', 100)))
    logging.info(f'done. loss = {loss}, accuracy = {acc}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(globals().get('__doc__'))
    parser.add_argument('command', choices=['train', 'evaluate', 'benchmark'], help='执行操作')
    parser.add_argument('-D', '--debug', action='store_true', default=False, help='显示调试信息')
    parser.add_argument('config', type=argparse.FileType('r'), help='配置文件')
    parser.add_argument('model', nargs='?', help='指定训练或评估的模型')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)

    logging.info(f'load configuration from {args.config}')
    config = json.load(args.config)

    did, cid, character, initial, final, tone = load_dictionaries(config['dictionary_dir'])
    did = did[:config['dialect_num'] - 1]
    cid = cid[:config['input_nums'][0] - 1]
    character = character[:config['input_nums'][1] - 1]
    initial = initial[:config['output_nums'][0] - 1]
    final = final[:config['output_nums'][1] - 1]
    tone = tone[:config['output_nums'][2] - 1]

    data = load_datasets(config['datasets'])
    data = sinetym.auxiliary.OrdinalEncoder(
        categories=[
            did.index,
            cid.index,
            character.index,
            initial.index,
            final.index,
            tone.index
        ],
        dtype=np.int32
    ).fit(data[:1]).transform(data)

    data = tf.data.Dataset.from_tensor_slices((
        data[:, 0],
        data[:, 1:3],
        data[:, 3:6]
    ))
    data = data.shuffle(data.cardinality(), seed=10273)

    if args.command == 'train':
        train(config, args.model, data)

    elif args.command == 'evaluate':
        evaluate(config, args.model, data)

    elif args.command == 'benchmark':
        eval_size = data.cardinality() // 10
        train_data = data.skip(eval_size)
        eval_data = data.take(eval_size)
        benchmark(config, train_data, eval_data)
