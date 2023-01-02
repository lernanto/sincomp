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
import pandas
import numpy
from sklearn.preprocessing import OrdinalEncoder
import tensorflow as tf
from tensorboard.plugins import projector

import sinetym


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
    model = model_class(
        location.shape[0],
        (char.shape[0],),
        (initial.shape[0], final.shape[0], tone.shape[0]),
        **model_config
    )

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
        location (`pandas.DataFrame`): 方言点信息数据表
        char (`pandas.DataFrame`): 字信息数据表
        initial (`pandas.Series`): 模型的声母表
        final (`pandas.Series`): 模型的韵母表
        tone (`pandas.Series`): 模型的声调表
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

def train(
    config,
    name,
    location,
    char,
    initial,
    final,
    tone,
    train_data,
    eval_data=None
):
    """
    训练模型.

    Parameters:
        config (dict): 多层级的配置字典，分析配置文件获得
        name (str): 用于训练的模型配置名称，用于从 config 中读取指定配置
        location (`pandas.DataFrame`): 方言点信息数据表
        char (`pandas.DataFrame`): 字信息数据表
        initial (`pandas.Series`): 模型的声母表
        final (`pandas.Series`): 模型的韵母表
        tone (`pandas.Series`): 模型的声调表
        train_data (`tensorflow.data.Dataset`): 训练数据
        eval_data (`tensorflow.data.Dataset`): 评估数据

    从 config 中读取由 name 指定的配置，创建并训练模型。
    """

    # 从模型列表中查找待训练的模型
    conf = next((c for c in config['models'] if c['name'] == name)).copy()
    conf.pop('name')
    model, optimizer = build_model(conf)
    output_path = os.path.join(config['output_dir'], name)

    logging.info('start training ...')
    model.fit(
        optimizer,
        train_data,
        eval_data=eval_data,
        output_path=output_path,
        **conf
    )
    logging.info('done.')

    # 为 TensorBoard embedding projector 显示向量创建需要的数据
    make_embeddings(
        location,
        char,
        initial,
        final,
        tone,
        model,
        output_path=os.path.join(output_path, 'embeddings')
    )

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
        max_to_keep=20
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

    # 加载词典
    logging.info(f'load dictionaries from {config["dictionary_dir"]}')
    for name in ('lid', 'cid', 'initial', 'final', 'tone'):
        locals()[name] = pandas.Series(open(
            os.path.join(config['dictionary_dir'], f'{name}.csv'),
            encoding='utf-8'
        )).str.rstrip('\r\n')

    cid = cid.astype(int)

    logging.info(f'load data from {config["data_dir"]}')
    dialect_path = os.path.join(config['data_dir'], 'dialect')
    location = sinetym.datasets.zhongguoyuyan.load_location(
        os.path.join(config['data_dir'], 'location.csv')
    ).reindex(lid)
    char = pandas.read_csv(
        os.path.join(config['data_dir'], 'words.csv'),
        index_col=0
    ).reindex(cid)
 
    data = sinetym.datasets.load_data(dialect_path, suffix='mb01dz.csv')
    data = OrdinalEncoder(
        categories=[lid, cid, initial, final, tone],
        dtype=numpy.int32,
        handle_unknown='use_encoded_value',
        unknown_value=-1
    ).fit_transform(data[['lid', 'cid', 'initial', 'final', 'tone']])

    dataset = tf.data.Dataset.from_tensor_slices((
        data[:, 0],
        data[:, 1:2],
        data[:, 2:5]
    )).shuffle(data.shape[0])

    if args.command == 'train':
        logging.info(f'train {args.model}, input = {config["data_dir"]}')
        train(config, args.model, location, char, initial, final, tone, dataset)

    elif args.command == 'evaluate':
        logging.info(f'evaluate {args.model}, input = {config["data_dir"]}')
        evaluate(config, args.model, dataset)

    elif args.command == 'benchmark':
        logging.info(f'benchmark, input = {config["data_dir"]}')
        eval_size = int(data.shape[0] * 0.1)
        train_data = dataset.skip(eval_size)
        eval_data = dataset.take(eval_size)
        benchmark(config, train_data, eval_data)