#!/usr/bin/env -S python3 -O
# -*- coding: utf-8 -*-

"""
训练和评估方言字音编码器
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import logging
import argparse
import os
import itertools
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
import tensorflow as tf

import sincomp.datasets
import sincomp.models
import sincomp.auxiliary


def load_dictionaries(prefix='.'):
    """
    加载词典

    Parameters:
        prefix: 词典路径前缀

    Returns:
        dicts: 词典列表
    """

    logging.info(f'load dictionaries from {prefix}')

    dicts = {}
    for e in os.scandir(prefix):
        if e.is_file() and e.name.endswith('.csv'):
            dicts[os.path.splitext(e.name)[0]] \
                = pd.read_csv(e.path, index_col=0, encoding='utf-8')

    return dicts

def load_datasets(config: dict) -> sincomp.datasets.Dataset:
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
    return data

def build_model(config: dict, **kwargs) -> tuple[
    tf.Module,
    tf.optimizers.Optimizer
]:
    """
    根据配置创建模型和优化器

    Parameters:
        config: 多层级的配置字典，分析配置文件获得
        kwargs: 透传给模型构造函数的其他参数

    Returns:
        model: 创建的编码器模型
        optimizer: 创建的优化器
    """

    # 创建模型
    model_config = config.pop('model').copy()
    model_class = getattr(sincomp.models, model_config.pop('class'))
    model = model_class(**model_config, **kwargs)

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

    return model, optimizer

def encode_data(
    data: pd.DataFrame,
    dialect_indeces: list[str],
    char_indeces: list[str],
    target_indeces: list[str],
    dialect_dicts: list[pd.Series],
    char_dicts: list[pd.Series],
    target_dicts: list[pd.Series]
) -> tf.data.Dataset:
    """
    把方言数据集编码成模型识别的 TensorFlow 格式数据集

    Parameters:
        data: 待编码数据集
        dialect_indeces: `data` 中作为方言特征的列
        char_indeces: `data` 中作为子特征的列
        target_indeces: `data` 中作为预测目标的列
        dialect_dicts: 方言特征词典，顺序和 `dialect_indeces` 相同
        char_dicts: 方言特征词典，顺序和 `char_indeces` 相同
        target_dicts: 方言特征词典，顺序和 `target_indeces` 相同

    Returns:
        dataset: `data` 编码后转换成的 TensorFlow 数据集
    """

    indeces = dialect_indeces + char_indeces + target_indeces
    data = OrdinalEncoder(
        categories=[d.index for d in dialect_dicts + char_dicts + target_dicts],
        dtype=np.int32,
        handle_unknown='use_encoded_value',
        unknown_value=-1,
        encoded_missing_value=-1
    ).fit(data[:1][indeces]).transform(data[indeces]) + 1

    limits = np.cumsum([
        0,
        len(dialect_indeces),
        len(char_indeces),
        len(target_indeces)
    ])

    return tf.data.Dataset.from_tensor_slices((
        (data[:, limits[0]:limits[1]], data[:, limits[1]:limits[2]]),
        data[:, limits[2]:limits[3]]
    ))

def make_data(
    data: pd.DataFrame,
    dialect_indeces: list[str],
    char_indeces: list[str],
    target_indeces: list[str],
    target_dicts: list[pd.Series],
    dialect_dicts: list[pd.Series] | None = None,
    char_dicts: list[pd.Series] | None = None,
    test_size: float = 0.2,
    minfreq: float | int = 0.00001,
    random_state: np.random.RandomState | None = None
) -> tuple[
    tf.data.Dataset,
    tf.data.Dataset,
    list[pd.Series],
    list[pd.Series]
]:
    """
    为模型构造训练及测试数据集

    Parameters:
        data: 输入的方言数据
        dialect_indeces: data 中代表方言特征的列
        char_indeces: data 中代表子特征的列
        target_indeces: data 中代表目标输出的列
        target_dicts: 输出数据的词典列表
        dialect_dicts: 方言数据的词典列表
        char_dicts: 字数据的词典列表
        test_size: 切分评估数据的比例
        minfreq: 为数据构造编码器时只保留出现频次或比例不小于该值的值
        random_state: 用于复现划分数据结果

    Returns:
        train_data: 训练数据集
        test_data: 测试数据集
        dialect_dicts: 最终的方言数据词典列表
        char_dicts: 最终的字数据词典列表

    把原始数据按 test_size 随机分为训练集和评估集，并用 dialect_dicts、
    char_dicts 和 target_dicts 编码数据。如果 dialect_dicts 或 char_dicts 为空，
    根据相应的训练数据构建。
    """

    # 分割数据为训练集和测试集
    train_data, test_data = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state
    )

    logging.info(
        f'split {data.shape[0]} data into {train_data.shape[0]} train data '
        f'and {test_data.shape[0]} test data.'
    )

    # 如果输入词典为空，根据训练数据创建
    if dialect_dicts is None:
        dialect_dicts = [sincomp.auxiliary.make_dict(c, minfreq=minfreq) \
            for _, c in train_data[dialect_indeces].items()]

    if char_dicts is None:
        char_dicts = [sincomp.auxiliary.make_dict(c, minfreq=minfreq) \
            for _, c in train_data[char_indeces].items()]

    if any(d.shape[0] <= 0 for d in itertools.chain(
            dialect_dicts, char_dicts, target_dicts
        )) or (train_data.shape[0] <= 0):
        raise RuntimeError('Empty dictionary or training data, check your data!')

    train_data = encode_data(
        train_data,
        dialect_indeces,
        char_indeces,
        target_indeces,
        dialect_dicts,
        char_dicts,
        target_dicts
    )
    test_data = encode_data(
        test_data,
        dialect_indeces,
        char_indeces,
        target_indeces,
        dialect_dicts,
        char_dicts,
        target_dicts
    )
    return train_data, test_data, dialect_dicts, char_dicts

def split(config: dict) -> None:
    """
    把数据集划分成训练集、验证集、测试集

    Parameters:
        config: 全局配置字典

    划分后的数据保存到 config['split_dir']
    """

    data = load_datasets(config['datasets']).data

    validation_size = config.get('validation_size', 0.1)
    if isinstance(validation_size, float):
        validation_size = int(data.shape[0] * validation_size)

    test_size = config.get('test_size', 0.1)
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

    output_dir = config.get('split_dir', '.')
    os.makedirs(output_dir, exist_ok=True)

    path = os.path.join(output_dir, 'train.csv')
    logging.info(f'save train data to {path} .')
    train_data.to_csv(path, index=False, encoding='utf-8', lineterminator='\n')
    path = os.path.join(output_dir, 'validation.csv')
    logging.info(f'save validation data to {path} .')
    val_data.to_csv(path, index=False, encoding='utf-8', lineterminator='\n')
    path = os.path.join(output_dir, 'test.csv')
    logging.info(f'save test data to {path} .')
    test_data.to_csv(path, index=False, encoding='utf-8', lineterminator='\n')

def mkdict(config: dict) -> None:
    """
    根据方言数据构建词典

    Parameters:
        config: 全局配置字典
    """

    prefix = config.get('dictionary_dir', '.')
    logging.info(f'make dictionaries to {prefix}...')

    data = load_datasets(config['datasets'])

    os.makedirs(prefix, exist_ok=True)

    for name in sum(config['columns'].values(), []):
        dic = sincomp.auxiliary.make_dict(
            data[name],
            minfreq=config.get('min_freq'),
            sort='value'
        )
        dic.index.rename(name, inplace=True)

        fname = os.path.join(prefix, name + '.csv')
        logging.info(f'save {dic.shape[0]} values to {fname}')
        dic.to_csv(fname, encoding='utf-8', lineterminator='\n')

    logging.info('done.')

def benchmark(config: dict, data: pd.DataFrame) -> None:
    """
    评估各种配置的模型效果

    Parameters:
        config: 全局配置字典
        data: 用于训练及评估的数据集

    从 config 读取所有模型配置，为每一组配置创建一个模型，训练并评估效果。
    """

    # 输出数据必须全部编码
    target_dicts = [sincomp.auxiliary.make_dict(
        data[c],
        minfreq=config.get('min_freq')
    ) for c in config['columns']['target']]

    # 切分数据用于训练及不同项目的评估
    random_state = np.random.RandomState(37511)
    train_dialect, _ = sincomp.auxiliary.split_data(
        data[config['columns']['dialect']].fillna('').apply(' '.join, axis=1),
        return_mask=True,
        random_state=random_state
    )
    train_char, _ = sincomp.auxiliary.split_data(
        data[config['columns']['char']].fillna('').apply(' '.join, axis=1),
        return_mask=True,
        random_state=random_state
    )
    train_data, validate_data, dialect_dicts, char_dicts = make_data(
        data[train_dialect & train_char],
        config['columns']['dialect'],
        config['columns']['char'],
        config['columns']['target'],
        target_dicts,
        test_size=0.1,
        minfreq=config.get('min_freq'),
        random_state=random_state
    )

    dialect_vocab_sizes = [len(d) + 1 for d in dialect_dicts]
    char_vocab_sizes = [len(d) + 1 for d in char_dicts]
    target_vocab_sizes = [len(d) + 1 for d in target_dicts]
    logging.info(
        f'dialect_vocab_sizes = {dialect_vocab_sizes}, '
        f'char_vocab_sizes = {char_vocab_sizes}, '
        f'target_vocab_sizes = {target_vocab_sizes}.'
    )

    # 保存词典
    dict_dir = config.get('dictionary_dir', '.')
    os.makedirs(dict_dir, exist_ok=True)
    logging.info(f'saving dictionaries to {dict_dir}...')

    for name, dic in zip(
        config['columns']['dialect'] + config['columns']['char'] \
            + config['columns']['target'],
        dialect_dicts + char_dicts + target_dicts
    ):
        dic.to_csv(
            os.path.join(dict_dir, name + '.csv'),
            encoding='utf-8',
            lineterminator='\n'
        )

    logging.info('done.')

    for conf in config['models']:
        # 根据配置文件创建模型并训练
        conf = conf.copy()
        name = conf.pop('name')
        model, optimizer = build_model(
            conf,
            dialect_vocab_sizes=dialect_vocab_sizes,
            char_vocab_sizes=char_vocab_sizes,
            target_vocab_sizes=target_vocab_sizes,
            missing_id=0
        )

        checkpoint_dir = os.path.join(config.get('checkpoint_dir', '.'), name)
        log_dir = os.path.join(config.get('log_dir', '.'), name)
        logging.info(
            f'start training model {name}, '
            f'checkpoint directory = {checkpoint_dir}, '
            f'log directory = {log_dir} ...'
        )

        model.fit(
            optimizer,
            train_data,
            validate_data,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            **conf
        )
        logging.info('done.')

def train(
    config: dict,
    name: str,
    dialect_vocab_sizes: list[int],
    char_vocab_sizes: list[int],
    target_vocab_sizes: list[int],
    train_data: tf.data.Dataset,
    validate_data: tf.data.Dataset | None = None
) -> None:
    """
    训练模型

    Parameters:
        config: 多层级的配置字典，分析配置文件获得
        name: 用于训练的模型配置名称，用于从 config 中读取指定配置
        dialect_vocab_sizes, char_vocab_sizes, target_vocab_sizes: 传给模型构造函数的参数
        train_data: 训练数据
        validate_data: 评估数据

    从 config 中读取由 name 指定的配置，创建并训练模型。
    """

    # 从模型列表中查找待训练的模型
    conf = next((c for c in config['models'] if c['name'] == name)).copy()
    conf.pop('name')
    model, optimizer = build_model(
        conf,
        dialect_vocab_sizes=dialect_vocab_sizes,
        char_vocab_sizes=char_vocab_sizes,
        target_vocab_sizes=target_vocab_sizes,
        missing_id=0
    )

    checkpoint_dir = os.path.join(config.get('checkpoint_dir', '.'), name)
    log_dir = os.path.join(config.get('log_dir', '.'), name)
    logging.info(
        f'start training model {name}, '
        f'checkpoint directory = {checkpoint_dir}, '
        f'log directory = {log_dir} ...'
    )

    model.fit(
        optimizer,
        train_data,
        validate_data=validate_data,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        **conf
    )
    logging.info('done.')

def evaluate(
    config: dict,
    name: str,
    dialect_vocab_sizes: list[int],
    char_vocab_sizes: list[int],
    target_vocab_sizes: list[int],
    data: tf.data.Dataset
) -> None:
    """
    评估训练完成的模型效果

    Parameters:
        config: 多层级的配置字典，分析配置文件获得
        name: 用于训练的模型配置名称，用于从 config 中读取指定配置
        dialect_vocab_sizes, char_vocab_sizes, target_vocab_sizes: 传给模型构造函数的参数
        data: 评估数据
    """

    conf = next((c for c in config['models'] if c['name'] == name)).copy()
    model, _ = build_model(
        conf,
        dialect_vocab_sizes=dialect_vocab_sizes,
        char_vocab_sizes=char_vocab_sizes,
        target_vocab_sizes=target_vocab_sizes,
        missing_id=0
    )

    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(
        checkpoint,
        os.path.join(config['output_dir'], name, 'checkpoints'),
        None
    )
    checkpoint.restore(manager.latest_checkpoint)
    logging.info(f'restored model from checkpoint {manager.latest_checkpoint} .')

    logging.info('evaluating model ...')
    loss, acc = model.evaluate(data.batch(conf.get('batch_size', 100)))
    logging.info(f'done. loss = {loss}, accuracy = {acc}')

def export(
    config: dict,
    name: str,
    dicts: dict[str,
    pd.DataFrame],
    prefix: str = '.'
) -> None:
    """
    从模型导出向量及其他权重到文件

    Parameters:
        config: 多层级的配置字典，分析配置文件获得
        name: 用于训练的模型配置名称，用于从 config 中读取指定配置
        dicts: 模型输入输出字典
        prefix: 输出路径前缀
    """

    conf = next((c for c in config['models'] if c['name'] == name)).copy()
    model, _, _ = build_model(
        conf,
        dialect_vocab_sizes=[dicts[c].shape[0] + 1 for c in config['columns']['dialect']],
        char_vocab_sizes=[dicts[c].shape[0] + 1 for c in config['columns']['char']],
        target_vocab_sizes=[dicts[c].shape[0] + 1 for c in config['columns']['target']],
        missing_id=0
    )

    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(
        checkpoint,
        os.path.join(config['output_dir'], name, 'checkpoints'),
        None
    )
    checkpoint.restore(manager.latest_checkpoint)
    logging.info(f'restore model from checkpoint {manager.latest_checkpoint} .')

    output_dir = os.path.join(prefix, name)
    logging.info(f'exporting {name} weights to {output_dir} ...')
    os.makedirs(output_dir, exist_ok=True)

    # 导出输入输出向量，保存为 CSV 格式
    for name in ('dialect', 'char', 'target'):
        columns = config['columns'][name]

        for c, e in zip(columns, (getattr(model, f'{name}_embs'))):
            fname = os.path.join(output_dir, c + '.csv')
            logging.info(f'save {fname}')
            idx = dicts[c].index
            pd.DataFrame(e.numpy(), index=idx.insert(idx.shape[0], '')).to_csv(
                fname,
                header=False,
                encoding='utf-8',
                lineterminator='\n'
            )

        for c, b in zip(columns, (getattr(model, f'{name}_biases', []))):
            fname = os.path.join(output_dir, c + '_bias.csv')
            logging.info(f'save {fname}')
            idx = dicts[c].index
            pd.Series(b.numpy(), index=idx.insert(idx.shape[0], '')).to_csv(
                fname,
                header=False,
                encoding='utf-8',
                lineterminator='\n'
            )

    # 导出所有模型权重
    for v in model.variables:
        fname = os.path.join(output_dir, v.name.partition(':')[0] + '.txt')
        logging.info(f'save {fname}')
        a = v.numpy()
        np.savetxt(fname, np.reshape(a, (-1, a.shape[-1])) if a.ndim > 2 else a)

    logging.info('done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(globals().get('__doc__'))
    parser.add_argument(
        'command',
        choices=['split', 'mkdict', 'train', 'evaluate', 'benchmark', 'export'],
        help='执行操作'
    )
    parser.add_argument('-D', '--debug', action='store_true', default=False, help='显示调试信息')
    parser.add_argument('config', type=argparse.FileType('r'), help='配置文件')
    parser.add_argument('model', nargs='?', help='指定训练或评估的模型')
    parser.add_argument('output', nargs='?', help='指定输出路径前缀')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)

    logging.info(f'load configuration from {args.config.name}')
    config = json.load(args.config)

    if args.command in ('train', 'evaluate', 'export'):
        dicts = load_dictionaries(config.get('dictionary_dir', '.'))
        dialect_dicts, char_dicts, target_dicts \
            = [[dicts[c] for c in config['columns'][name]] \
                for name in ('dialect', 'char', 'target')]

    if args.command in ('train', 'evaluate', 'benchmark'):
        data = load_datasets(config['datasets']).shuffle(random_state=98101)

        # 删除不含有效方言、字、读音信息的记录
        for name in 'dialect', 'char', 'target':
            data = data.dropna(how='all', subset=config['columns'][name])

        if args.command in ('train', 'evaluate'):
            data = encode_data(
                data,
                config['columns']['dialect'],
                config['columns']['char'],
                config['columns']['target'],
                dialect_dicts,
                char_dicts,
                target_dicts
            )

            if args.command == 'train':
                data = data.shuffle(
                    100000,
                    seed=10273,
                    reshuffle_each_iteration=True
                )

    if args.command == 'split':
        split(config)

    elif args.command == 'mkdict':
        mkdict(config)

    elif args.command == 'train':
        train(
            config,
            args.model,
            *[[d.shape[0] + 1 for d in dicts] \
                for dicts in (dialect_dicts, char_dicts, target_dicts)],
            data
        )

    elif args.command == 'evaluate':
        evaluate(
            config,
            args.model,
            *[[d.shape[0] + 1 for d in dicts] \
                for dicts in (dialect_dicts, char_dicts, target_dicts)],
            data
        )

    elif args.command == 'benchmark':
        benchmark(config, data)

    elif args.command == 'export':
        export(config, args.model, dicts, args.output)
