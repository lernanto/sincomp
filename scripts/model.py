#!/usr/bin/env -S python3 -O
# -*- coding: utf-8 -*-

"""
训练和评估方言字音编码器
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import argparse
import json
import logging
import numpy as np
import os
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf

import sincomp.datasets
import sincomp.models
import sincomp.auxiliary
import sincomp.preprocess


def load_data(config: dict) -> sincomp.datasets.Dataset:
    """
    加载数据集

    Parameters:
        config: 数据集配置

    Returns:
        data: 所有数据集拼接成的数据集
    """

    logging.info(f'loading datasets...')

    for d in config:
        if isinstance(d, str):
            dataset = sincomp.datasets.get(d)
        else:
            dataset = sincomp.datasets.get(d['name']).filter(d['did'])

        try:
            data += dataset
        except UnboundLocalError:
            data = dataset

    logging.info(f'done, {len(data)} dialects loaded.')
    return data.data

def split_data(
    data: pd.DataFrame,
    validation_size: float | int,
    test_size: float | int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    把数据集划分成训练集、验证集、测试集

    Parameters:
        data: 待划分数据
        validation_size: 验证集大小，浮点数表示比例，整数表示数量
        test_size: 测试集大小，浮点数表示比例，整数表示数量

    Returns:
        train_data, val_data, test_data: 训练集、验证集、测试集
    """

    if isinstance(validation_size, float):
        validation_size = int(data.shape[0] * validation_size)

    if isinstance(test_size, float):
        test_size = int(data.shape[0] * test_size)

    # 按字分层随机划分
    groups=data[config['columns']['char'][0]]
    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        stratify=np.where(
            groups.groupby(groups).transform('count') >= 2,
            groups,
            ''
        )
    )
    groups=train_data[config['columns']['char'][0]]
    train_data, val_data = train_test_split(
        train_data,
        test_size=validation_size,
        stratify=np.where(
            groups.groupby(groups).transform('count') >= 2,
            groups,
            ''
        )
    )

    return train_data, val_data, test_data

def encode_data(
    processor: sincomp.models.Processor,
    dialects: np.ndarray[str],
    chars: np.ndarray[str],
    targets: np.ndarray[str]
) -> tf.data.Dataset:
    """
    把方言数据编码成模型识别的 TensorFlow 格式数据集

    Parameters:
        processor: 预处理器
        dialects, chars, targets: 待编码的方言特征、字特征、预测目标字符串

    Returns:
        编码后的 TensorFlow Dataset

    如 dialects, chars, targets 任一个的所有列均为缺失值或未知值，丢弃该条记录
    """

    dialect_ids, char_ids, target_ids = processor(dialects, chars, targets)
    mask = np.any(dialect_ids != 0, axis=1) & np.any(char_ids != 0, axis=1) \
        & np.any(target_ids != 0, axis=1)
    
    return tf.data.Dataset.from_tensor_slices(
        ((dialect_ids[mask], char_ids[mask]), target_ids[mask])
    )
 
def get_model(config: dict) -> tuple[type, dict, tf.optimizers.Optimizer]:
    """
    从配置获取创建模型的类型和参数，并创建优化器

    Parameters:
        config: 多层级的配置字典，分析配置文件获得

    Returns:
        model_class: 模型类型
        model_args: 构造模型的参数
        optimizer: 创建的优化器
    """

    # 获取模型类型和参数
    model_config = config.pop('model').copy()
    model_class = getattr(sincomp.models, model_config.pop('class'))

    # 创建优化器
    optimizer_config = config.pop('optimizer').copy()
    optimizer_class = getattr(tf.optimizers, optimizer_config.pop('class'))

    # 学习率可以是一个对象，根据配置创建
    lr_config = optimizer_config.pop('learning_rate')
    if isinstance(lr_config, dict):
        lr_config = lr_config.copy()
        lr_class = getattr(tf.optimizers.schedules, lr_config.pop('class'))
        lr = lr_class(**lr_config)
    else:
        lr = lr_config

    optimizer = optimizer_class(learning_rate=lr, **optimizer_config)

    return model_class, model_config, optimizer

def prepare(args: argparse.Namespace, config: dict) -> None:
    """
    模型训练前的预处理工作

    1. 根据方言数据集构建词典
    2. 把数据集划分成训练集、验证集、测试集
    """

    # 根据方言数据构建词典
    data = load_data(config['datasets'])

    minfreq = config.get('min_frequency')
    maxcat = config.get('max_categories')

    vocabs = {}
    for name, columns in config['columns'].items():
        voc = []
        for c in columns:
            dic = sincomp.auxiliary.make_dict(
                data[c],
                # 当 minfreq, maxcat 为字典时，可以为不同列指定不同的值
                min_frequency=minfreq.get(name) if isinstance(minfreq, dict) \
                    else minfreq,
                max_categories=maxcat.get(name) if isinstance(maxcat, dict) \
                    else maxcat,
            )
            logging.info(f'{c} vocabulary size = f{dic.shape[0]}')
            voc.append(dic.index.tolist())

        vocabs[f'{name}_vocabs'] = voc

    # 保存词汇表
    vocab_file = config.get('vocab_file', 'vocabs.json')
    logging.info(f'save vocabularies to {vocab_file} .')
    os.makedirs(os.path.dirname(vocab_file), exist_ok=True)
    with open(vocab_file, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(vocabs, f, ensure_ascii=False, indent=4)

    # 划分数据集
    train_data, val_data, test_data = split_data(
        data,
        config.get('validation_size', 0.1),
        config.get('test_size', 0.1)
    )

    output_dir = config.get('split_dir', '.')
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, 'train.csv')
    logging.info(f'save {train_data.shape[0]} train data to {path} .')
    train_data.to_csv(path, index=False, encoding='utf-8', lineterminator='\n')
    path = os.path.join(output_dir, 'validation.csv')
    logging.info(f'save {val_data.shape[0]} validation data to {path} .')
    val_data.to_csv(path, index=False, encoding='utf-8', lineterminator='\n')
    path = os.path.join(output_dir, 'test.csv')
    logging.info(f'save {test_data.shape[0]} test data to {path} .')
    test_data.to_csv(path, index=False, encoding='utf-8', lineterminator='\n')

def compare(args: argparse.Namespace, config: dict) -> None:
    """
    对比各种配置的模型效果

    从 config 读取所有模型配置，为每一组配置创建一个模型，训练并评估效果。
    """

    # 载入训练和评估数据
    train_data = sincomp.datasets.get(
        os.path.join(config.get('split_dir', '.'), 'train.csv')
    )
    val_data = sincomp.datasets.get(
        os.path.join(config.get('split_dir', '.'), 'validation.csv')
    )
    test_data = sincomp.datasets.get(
        os.path.join(config.get('split_dir', '.'), 'test.csv')
    )

    # 载入词典
    with open(config.get('vocab_file', 'vocabs.json'), encoding='utf-8') as f:
        vocabs = json.load(f)

    processor = sincomp.models.Processor(**vocabs)

    dialect_col = config['columns']['dialect']
    char_col = config['columns']['char']
    target_col = config['columns']['target']
    train_data = encode_data(
        processor,
        train_data[dialect_col],
        train_data[char_col],
        train_data[target_col]
    )
    val_data = encode_data(
        processor,
        val_data[dialect_col],
        val_data[char_col],
        val_data[target_col]
    )
    test_data = encode_data(
        processor,
        test_data[dialect_col],
        test_data[char_col],
        test_data[target_col]
    )
    logging.info(
        f'compare models, train data size = {train_data.cardinality()}, '
        f'validation data size = {val_data.cardinality()}, '
        f'test data size = {test_data.cardinality()}.'
    )
    train_data = train_data.shuffle(
        train_data.cardinality(),
        reshuffle_each_iteration=True
    )

    for conf in config['models']:
        # 根据配置文件创建模型并训练
        conf = conf.copy()
        name = conf.pop('name')
        model_class, model_args, optimizer = get_model(conf)
        model_args.update(
            dialect_vocab_sizes=processor.dialect_vocab_sizes,
            char_vocab_sizes=processor.char_vocab_sizes,
            target_vocab_sizes=processor.target_vocab_sizes,
            missing_id=0
        )

        output_dir = os.path.join(
            config.get('output_dir', '.'),
            'compare',
            name
        )
        log_dir = os.path.join(config.get('log_dir', '.'), 'compare', name)
        logging.info(
            f'start training model {name}, '
            f'output directory = {output_dir}, '
            f'log directory = {log_dir} ...'
        )

        # 保存词汇表和模型参数
        os.makedirs(output_dir, exist_ok=True)
        with open(
            os.path.join(output_dir, 'vocabs.json'),
            'w',
            encoding='utf-8',
            newline='\n'
        ) as f:
            json.dump(vocabs, f, ensure_ascii=False, indent=4)

        with open(os.path.join(output_dir, 'model.json'), 'w', newline='\n') as f:
            json.dump(model_args, f, indent=4)

        model = model_class(**model_args)
        model.fit(
            optimizer,
            train_data,
            val_data,
            checkpoint_dir=output_dir,
            log_dir=log_dir,
            **conf
        )
        logging.info('done.')

        # 使用测试数据集评估模型效果
        logging.info('evaluating model ...')
        loss, acc = model.evaluate(test_data.batch(conf.get('batch_size', 100)))
        logging.info(f'done. loss = {loss}, accuracy = {acc}')
        print(f'{name}: loss = {loss}, accuracy = {acc}')

def train(args: argparse.Namespace, config: dict) -> None:
    """
    训练模型

    从 config 中读取由 args.model 指定的配置，创建并训练模型。
    """

    # 载入词典及数据
    with open(config.get('vocab_file', 'vocabs.json'), encoding='utf-8') as f:
        vocabs = json.load(f)

    processor = sincomp.models.Processor(**vocabs)

    data = load_data(config['datasets'])
    data = encode_data(
        processor,
        data[config['columns']['dialect']],
        data[config['columns']['char']],
        data[config['columns']['target']]
    )
    logging.info(f'train data size = {data.cardinality()}.')
    data = data.shuffle(data.cardinality(), reshuffle_each_iteration=True)

    # 从模型列表中查找待训练的模型
    conf = next((c for c in config['models'] if c['name'] == args.model)).copy()
    conf.pop('name')
    model_class, model_args, optimizer = get_model(conf)
    model_args.update(
        dialect_vocab_sizes=processor.dialect_vocab_sizes,
        char_vocab_sizes=processor.char_vocab_sizes,
        target_vocab_sizes=processor.target_vocab_sizes,
        missing_id=0
    )

    output_dir = os.path.join(config.get('output_dir', '.'), args.model)
    log_dir = os.path.join(config.get('log_dir', '.'), args.model)
    logging.info(
        f'start training model {args.model}, '
        f'output directory = {output_dir}, '
        f'log directory = {log_dir} ...'
    )

    # 保存词汇表和模型参数
    os.makedirs(output_dir, exist_ok=True)
    with open(
        os.path.join(output_dir, 'vocabs.json'),
        'w',
        encoding='utf-8',
        newline='\n'
    ) as f:
        json.dump(vocabs, f, ensure_ascii=False, indent=4)

    with open(os.path.join(output_dir, 'model.json'), 'w', newline='\n') as f:
        json.dump(model_args, f, indent=4)

    model = model_class(**model_args)
    model.fit(
        optimizer,
        data,
        checkpoint_dir=output_dir,
        log_dir=log_dir,
        **conf
    )
    logging.info('done.')

    # 评估模型效果
    logging.info('evaluating model ...')
    loss, acc = model.evaluate(data.batch(conf.get('batch_size', 100)))
    logging.info(f'done. loss = {loss}, accuracy = {acc}')
    print(f'loss = {loss}, accuracy = {acc}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(globals().get('__doc__'))
    parser.add_argument('-c', '--config', type=argparse.FileType('r'), help='配置文件')

    subparsers = parser.add_subparsers()

    prepare_parser = subparsers.add_parser('prepare')
    prepare_parser.set_defaults(func=prepare)

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(func=train)
    train_parser.add_argument('model', help='指定训练的模型')

    compare_parser = subparsers.add_parser('compare')
    compare_parser.set_defaults(func=compare)

    args = parser.parse_args()

    config = json.load(args.config)
    logging.getLogger().setLevel(
        getattr(logging, config.get('log_level', 'WARNING'))
    )
    args.func(args, config)
