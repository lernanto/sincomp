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
            dic = pd.read_csv(e.path, dtype=str, encoding='utf-8')
            dic.set_index(dic.iloc[:, 0], inplace=True)
            dicts[os.path.splitext(e.name)[0]] = dic

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

def split(args: argparse.Namespace, config: dict) -> None:
    """
    把数据集划分成训练集、验证集、测试集

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

def mkdict(args: argparse.Namespace, config: dict) -> None:
    """根据方言数据构建词典"""

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

    # 载入词典
    dicts = load_dictionaries(config.get('dictionary_dir', '.'))
    dialect_dicts, char_dicts, target_dicts = \
        [[dicts[c] for c in config['columns'][name]] \
            for name in ('dialect', 'char', 'target')]
    dialect_vocab_sizes, char_vocab_sizes, target_vocab_sizes = \
        [[d.shape[0] + 1 for d in ds] \
            for ds in (dialect_dicts, char_dicts, target_dicts)]
    logging.info(
        f'dialect_vocab_sizes = {dialect_vocab_sizes}, '
        f'char_vocab_sizes = {char_vocab_sizes}, '
        f'target_vocab_sizes = {target_vocab_sizes}.'
    )

    train_data = encode_data(
        train_data,
        config['columns']['dialect'],
        config['columns']['char'],
        config['columns']['target'],
        dialect_dicts,
        char_dicts,
        target_dicts
    )
    train_data = train_data.shuffle(
        train_data.cardinality(),
        reshuffle_each_iteration=True
    )
    val_data = encode_data(
        val_data,
        config['columns']['dialect'],
        config['columns']['char'],
        config['columns']['target'],
        dialect_dicts,
        char_dicts,
        target_dicts
    )

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

        checkpoint_dir = os.path.join(
            config.get('checkpoint_dir', '.'),
            'compare',
            name
        )
        log_dir = os.path.join(config.get('log_dir', '.'), 'compare', name)
        logging.info(
            f'start training model {name}, '
            f'checkpoint directory = {checkpoint_dir}, '
            f'log directory = {log_dir} ...'
        )

        model.fit(
            optimizer,
            train_data,
            val_data,
            checkpoint_dir=checkpoint_dir,
            log_dir=log_dir,
            **conf
        )
        logging.info('done.')

def train(args: argparse.Namespace, config: dict) -> None:
    """
    训练模型

    从 config 中读取由 args.model 指定的配置，创建并训练模型。
    """

    # 载入词典及数据
    dicts = load_dictionaries(config.get('dictionary_dir', '.'))
    dialect_dicts, char_dicts, target_dicts = \
        [[dicts[c] for c in config['columns'][name]] \
            for name in ('dialect', 'char', 'target')]
    dialect_vocab_sizes, char_vocab_sizes, target_vocab_sizes = \
        [[d.shape[0] + 1 for d in ds] \
            for ds in (dialect_dicts, char_dicts, target_dicts)]

    data = encode_data(
        load_datasets(config['datasets']),
        config['columns']['dialect'],
        config['columns']['char'],
        config['columns']['target'],
        dialect_dicts,
        char_dicts,
        target_dicts
    )
    data = data.shuffle(data.cardinality(), reshuffle_each_iteration=True)

    # 从模型列表中查找待训练的模型
    conf = next((c for c in config['models'] if c['name'] == args.model)).copy()
    conf.pop('name')
    model, optimizer = build_model(
        conf,
        dialect_vocab_sizes=dialect_vocab_sizes,
        char_vocab_sizes=char_vocab_sizes,
        target_vocab_sizes=target_vocab_sizes,
        missing_id=0
    )

    checkpoint_dir = os.path.join(config.get('checkpoint_dir', '.'), args.model)
    log_dir = os.path.join(config.get('log_dir', '.'), args.model)
    logging.info(
        f'start training model {args.model}, '
        f'checkpoint directory = {checkpoint_dir}, '
        f'log directory = {log_dir} ...'
    )

    model.fit(
        optimizer,
        data,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        **conf
    )
    logging.info('done.')

def evaluate(args: argparse.Namespace, config: dict) -> None:
    """评估模型效果"""

    # 载入词典及数据
    dicts = load_dictionaries(config.get('dictionary_dir', '.'))
    dialect_dicts, char_dicts, target_dicts = \
        [[dicts[c] for c in config['columns'][name]] \
            for name in ('dialect', 'char', 'target')]
    dialect_vocab_sizes, char_vocab_sizes, target_vocab_sizes = \
        [[d.shape[0] + 1 for d in ds] \
            for ds in (dialect_dicts, char_dicts, target_dicts)]

    data = encode_data(
        load_datasets(config['datasets']),
        config['columns']['dialect'],
        config['columns']['char'],
        config['columns']['target'],
        dialect_dicts,
        char_dicts,
        target_dicts
    )

    conf = next((c for c in config['models'] if c['name'] == args.model)).copy()
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
        os.path.join(config['checkpoint_dir'], args.model),
        None
    )
    ckpt = manager.checkpoints[args.checkpoint]
    logging.info(f'restored model {args.model} from checkpoint {ckpt} .')
    checkpoint.restore(ckpt)

    logging.info('evaluating model ...')
    loss, acc = model.evaluate(data.batch(conf.get('batch_size', 100)))
    logging.info(f'done. loss = {loss}, accuracy = {acc}')
    print(f'loss = {loss}, accuracy = {acc}')

def export(args: argparse.Namespace, config: dict) -> None:
    """从模型导出向量及其他权重到文件"""

    dicts = load_dictionaries(config.get('dictionary_dir', '.'))

    conf = next((c for c in config['models'] if c['name'] == args.model)).copy()
    model, _ = build_model(
        conf,
        dialect_vocab_sizes=[dicts[c].shape[0] + 1 for c in config['columns']['dialect']],
        char_vocab_sizes=[dicts[c].shape[0] + 1 for c in config['columns']['char']],
        target_vocab_sizes=[dicts[c].shape[0] + 1 for c in config['columns']['target']],
        missing_id=0
    )

    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(
        checkpoint,
        os.path.join(config['checkpoint_dir'], args.model),
        None
    )
    ckpt = manager.checkpoints[args.checkpoint]
    logging.info(f'restore {args.model} from checkpoint {ckpt} .')
    checkpoint.restore(ckpt)

    output_dir = os.path.join(args.output, args.model)
    logging.info(f'exporting model {args.model} weights to {args.output} ...')
    os.makedirs(args.output, exist_ok=True)

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
    parser.add_argument('-c', '--config', type=argparse.FileType('r'), help='配置文件')

    subparsers = parser.add_subparsers()

    split_parser = subparsers.add_parser('split')
    split_parser.set_defaults(func=split)

    mkdict_parser = subparsers.add_parser('mkdict')
    mkdict_parser.set_defaults(func=mkdict)

    train_parser = subparsers.add_parser('train')
    train_parser.set_defaults(func=train)
    train_parser.add_argument('model', help='指定训练的模型')

    evaluate_parser = subparsers.add_parser('evaluate')
    evaluate_parser.set_defaults(func=evaluate)
    evaluate_parser.add_argument('model', help='指定评估的模型')
    evaluate_parser.add_argument(
        'checkpoint',
        nargs='?',
        type=int,
        default=-1,
        help='指定导出的检查点'
    )

    compare_parser = subparsers.add_parser('compare')
    compare_parser.set_defaults(func=compare)

    export_parser = subparsers.add_parser('export')
    export_parser.set_defaults(func=export)
    export_parser.add_argument('-o', '--output', default='.', help='指定输出路径前缀')
    export_parser.add_argument('model', help='指定导出的模型')
    export_parser.add_argument(
        'checkpoint',
        nargs='?',
        type=int,
        default=-1,
        help='指定导出的检查点'
    )

    args = parser.parse_args()

    config = json.load(args.config)
    logging.getLogger().setLevel(
        getattr(logging, config.get('log_level', 'WARNING'))
    )
    args.func(args, config)
