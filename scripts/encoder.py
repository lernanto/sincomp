#!/usr/bin/python3 -O
# -*- coding: utf-8 -*-

"""
训练和评估方言字音编码器.
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
from tensorboard.plugins import projector

import sinetym.datasets
import sinetym.models
import sinetym.auxiliary


def load_dictionaries(prefix='.'):
    """
    加载词典.

    Parameters:
        prefix (str): 词典路径前缀

    Returns:
        dialect_dicts, input_dicts, output_dicts (array-like): 词典列表
    """

    logging.info(f'load dictionaries from {prefix}')

    dicts = {}
    for e in os.scandir(prefix):
        if e.is_file() and e.name.endswith('.csv'):
            dicts[os.path.splitext(e.name)[0]] \
                = pd.read_csv(e.path, index_col=0, encoding='utf-8')

    return dicts

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

        data.append(dataset)

    data = sinetym.datasets.concat(*data)

    logging.info(f'done, {data.shape[0]} data loaded.')
    return data

def build_model(config, **kwargs):
    """
    根据配置创建模型和优化器.

    Parameters:
        config (dict): 多层级的配置字典，分析配置文件获得
        kwargs (dict): 透传给模型构造函数的其他参数

    Returns:
        model (sinetym.models.EncoderBase): 创建的编码器模型
        optimizer (`tensorflow.optimizers.Optimizer`): 创建的优化器
    """

    # 创建模型
    model_config = config.pop('model').copy()
    model_class = getattr(sinetym.models, model_config.pop('class'))
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

def build_new_model(model, dialect_nums=None, input_nums=None):
    """
    基于已训练的模型创建新模型.

    Parameters:
        model (sinetym.models.EncoderBase): 已训练的基线模型
        dialect_nums (array-like of int): 新方言数量，为空时沿用旧方言
        input_nums (array-like of int): 新字数量，为空时沿用旧字数

    Returns:
        new_model: 新模型，该模型为以 model.__class__ 为基类的新类型

    针对非空的 dialect_nums 和 input_nums 在 model 内创建新变量，其他变量沿用 model 现有的。
    """

    class NewModel(model.__class__):
        """
        为新模型动态构造新类型，继承 model.
        """

        def __init__(self, model, dialect_nums=None, input_nums=None):
            """
            为 dialect_nums 和 input_nums 创建相应的变量，其他变量沿用 model 的.
            """

            self.__dict__.update(model.__dict__)
            self._trainable_variables = []

            init = tf.random_normal_initializer()

            if dialect_nums is not None:
                self.dialect_nums = dialect_nums
                self.dialect_embs = [tf.Variable(
                    init(shape=(n + 1, self.dialect_emb_size), dtype=tf.float32),
                    name=f'dialect_emb{i}'
                ) for i, n in enumerate(dialect_nums)]
                self._trainable_variables.extend(self.dialect_embs)

            if input_nums is not None:
                self.input_nums = input_nums
                self.input_embs = [tf.Variable(
                    init(shape=(n + 1, self.input_emb_size), dtype=tf.float32),
                    name=f'input_emb{i}'
                ) for i, n in enumerate(input_nums)]
                self._trainable_variables.extend(self.input_embs)

        @property
        def trainable_varialbes(self):
            """
            改写 trainable_variables 属性使训练时只更新新变量.
            """

            return self._trainable_variables

    logging.info(
        f'build new model from {model.name}, '
        f'dialect numbers = {dialect_nums}, input numbers = {input_nums}.'
    )
    return NewModel(model, dialect_nums, input_nums)

def make_data(
    dialect,
    inputs,
    outputs,
    output_dicts,
    dialect_dicts=None,
    input_dicts=None,
    test_size=0.2,
    minfreq=0.00001,
    random_state=None
):
    """
    为模型构造训练及测试数据集.

    Parameters:
        dialect (`pandas.DataFrame`): 输入的方言数据
        inputs (`pandas.DataFrame`): 输入的字数据
        outputs (`pandas.DataFrame`): 目标输出数据
        output_dicts (array-like): 输出数据的词典列表
        dialect_dicts (array-like): 方言数据的词典列表
        input_dicts (array-like): 字数据的词典列表
        test_size (float): 切分评估数据的比例
        minfreq (float or int): 为数据构造编码器时只保留出现频次或比例不小于该值的值
        random_state (int or `numpy.random.RandomState`): 用于复现划分数据结果

    Returns:
        train_data (`tensorflow.data.Dataset`): 训练数据集
        test_data (`tensorflow.data.Dataset`): 测试数据集
        dialect_dicts (array-like): 最终的方言数据词典列表
        input_dicts (array-like): 最终的字数据词典列表

    把原始数据按 test_size 随机分为训练集和评估集，并用 dialect_dicts、
    input_dicts 和 output_dicts 编码数据。如果 dialect_dicts 或 input_dicts 为空，
    根据相应的训练数据构建。
    """

    logging.info(f'making train/test data from {dialect.shape[0]} data...')

    # 分割数据为训练集和测试集
    (
        train_dialect, test_dialect,
        train_input, test_input,
        train_output, test_output
    ) = train_test_split(
        dialect,
        inputs,
        outputs,
        test_size=test_size,
        random_state=random_state
    )

    # 如果输入词典为空，根据训练数据创建
    if dialect_dicts is None:
        dialect_dicts = [sinetym.auxiliary.make_dict(c, minfreq=minfreq) \
            for _, c in train_dialect.items()]

    if input_dicts is None:
        input_dicts = [sinetym.auxiliary.make_dict(c, minfreq=minfreq) \
            for _, c in train_input.items()]

    if any(d.shape[0] <= 0 for d in itertools.chain(
            dialect_dicts, input_dicts, output_dicts
        )) or any(d.shape[0] <= 0 for d in (train_dialect, train_input, train_output)):
        raise RuntimeError('Empty dictionary or training data, check your data!')

    # 根据词典编码训练和测试数据
    dialect_encoder, input_encoder, output_encoder = [OrdinalEncoder(
        categories=[d.index for d in dicts],
        dtype=np.int32,
        handle_unknown='use_encoded_value',
        unknown_value=-1,
        encoded_missing_value=-1
    ).fit(data[:1]) for dicts, data in (
        (dialect_dicts, train_dialect),
        (input_dicts, train_input),
        (output_dicts, train_output)
    )]

    train_data = tf.data.Dataset.from_tensor_slices((
        dialect_encoder.transform(train_dialect),
        input_encoder.transform(train_input),
        output_encoder.transform(train_output)
    ))
    train_data = train_data.shuffle(train_data.cardinality())
    test_data = tf.data.Dataset.from_tensor_slices((
        dialect_encoder.transform(test_dialect),
        input_encoder.transform(test_input),
        output_encoder.transform(test_output)
    ))

    logging.info(
        f'done, train data size = {train_data.cardinality()}, '
        f'test data size = {test_data.cardinality()}, '
        f'dialect numbers = {[d.shape[0] for d in dialect_dicts]}, '
        f'input numbers = {[d.shape[0] for d in input_dicts]}'
    )

    return train_data, test_data, dialect_dicts, input_dicts

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

def mkdict(config):
    """
    根据方言数据构建词典.

    Parameters:
        config (dict): 全局配置字典
    """

    prefix = config.get('dictionary_dir', '.')
    logging.info(f'make dictionaries to {prefix}...')

    data = load_datasets(config['datasets'])

    os.makedirs(prefix, exist_ok=True)

    for name in sum(config['columns'].values(), []):
        dic = sinetym.auxiliary.make_dict(
            data[name],
            minfreq=config.get('min_freq'),
            sort='value'
        )
        dic.index.rename(name, inplace=True)

        fname = os.path.join(prefix, name + '.csv')
        logging.info(f'save {dic.shape[0]} values to {fname}')
        dic.to_csv(fname, encoding='utf-8', lineterminator='\n')

    logging.info('done.')

def benchmark(config, data):
    """
    评估各种配置的模型效果.

    Parameters:
        config (dict): 全局配置字典
        data (`pandas.DataFrame`): 用于训练及评估的数据集

    从 config 读取所有模型配置，为每一组配置创建一个模型，训练并评估效果。
    每个模型除使用常规的评估集评估效果以外，还额外评估：
        - 对基础模型没见过的方言的适应能力
        - 对基础模型没见过的字的适应能力
        - 对方言和字基础模型都没见过的数据的适应能力
    评估方法为保持基础模型参数不变的基础上，从包含新方言或新字的数据集随机划分一部分数据
    来训练新的方言向量和字向量，然后评估预测剩余另一部分数据的准确率。
    """

    # 输出数据必须全部编码
    output_dicts = [sinetym.auxiliary.make_dict(
        data[c],
        minfreq=config.get('min_freq')
    ) for c in config['columns']['output']]

    # 切分数据用于训练及不同项目的评估：
    #   - 包含训练方言 ID 和训练字 ID
    #   - 包含测试方言 ID 和训练字 ID
    #   - 包含训练方言 ID 和测试字 ID
    #   - 包含测试方言 ID 和测试字 ID
    random_state = np.random.RandomState(37511)
    train_dialect, test_dialect = sinetym.auxiliary.split_data(
        data[config['columns']['dialect']].apply(tuple, axis=1).map(hash),
        return_mask=True,
        random_state=random_state
    )
    train_input, test_input = sinetym.auxiliary.split_data(
        data[config['columns']['input']].apply(tuple, axis=1).map(hash),
        return_mask=True,
        random_state=random_state
    )
    data1 = data[train_dialect & train_input]
    train_data1, validate_data1, dialect_dicts1, input_dicts1 = make_data(
        data1[config['columns']['dialect']],
        data1[config['columns']['input']],
        data1[config['columns']['output']],
        output_dicts,
        test_size=0.1,
        minfreq=config.get('min_freq'),
        random_state=random_state
    )
    dialect_nums, input_nums, output_nums = [[d.shape[0] for d in dicts] \
        for dicts in (dialect_dicts1, input_dicts1, output_dicts)]
    logging.info(
        f'dialect numbers = {dialect_nums}, input numbers = {input_nums}, '
        f'output numbers = {output_nums}'
    )

    # 保存词典
    prefix = config.get('output_dir', '.')
    dict_dir = config.get('dictionary_dir', os.path.join(prefix, 'dictionaries'))
    os.makedirs(prefix, exist_ok=True)
    os.makedirs(dict_dir, exist_ok=True)
    logging.info(f'saving dictionaries to {dict_dir}...')

    for name, dic in zip(
        config['columns']['dialect'] + config['columns']['input'] \
            + config['columns']['output'],
        dialect_dicts1 + input_dicts1 + output_dicts
    ):
        dic.to_csv(
            os.path.join(dict_dir, name + '.csv'),
            encoding='utf-8',
            lineterminator='\n'
        )

    logging.info('done.')

    # 剩余数据预处理成数据集用于评估
    datasets = []
    for n, d, dd, id in (
        ('new_dialect', data[test_dialect & train_input], None, input_dicts1),
        ('new_input', data[train_dialect & test_input], dialect_dicts1, None),
        ('new_dialect_input', data[test_dialect & test_input], None, None)
    ):
        if d.shape[0] > 0:
            train_data, validate_data, dialect_dicts, input_dicts = make_data(
                d[config['columns']['dialect']],
                d[config['columns']['input']],
                d[config['columns']['output']],
                output_dicts,
                dialect_dicts=dd,
                input_dicts=id,
                test_size=0.5,
                minfreq=config.get('min_freq'),
                random_state=random_state
            )
            datasets.append((n,
                train_data,
                validate_data,
                dialect_dicts if dd is None else None,
                input_dicts if id is None else None
            ))

            new_dict_dir = os.path.join(dict_dir, n)
            os.makedirs(new_dict_dir, exist_ok=True)
            for nm, dic in zip(
                config['columns']['dialect'] + config['columns']['input'],
                dialect_dicts + input_dicts
            ):
                dic.to_csv(
                    os.path.join(new_dict_dir, nm + '.csv'),
                    encoding='utf-8',
                    lineterminator='\n'
                )

    for conf in config['models']:
        # 根据配置文件创建模型并训练
        conf = conf.copy()
        name = conf.pop('name')
        model, optimizer = build_model(
            conf,
            dialect_nums=dialect_nums,
            input_nums=input_nums,
            output_nums=output_nums
        )

        output_path = os.path.join(prefix, name)
        logging.info(f'start training model {name}, output path = {output_path} ...')
        model.fit(
            optimizer,
            train_data1,
            validate_data1,
            output_path=output_path,
            **conf
        )
        logging.info('done.')

        # 使用剩余的数据集评估模型效果
        for n, train_data, validate_data, dialect_dicts, input_dicts in datasets:
            logging.info(
                f'evaluate {n}, training size = {train_data.cardinality()}, '
                f'validation size = {validate_data.cardinality()}.'
            )

            # 从上面训练完成的模型复制一个副本，针对数据微调然后评估
            new_model = build_new_model(
                model,
                dialect_nums=None if dialect_dicts is None \
                    else [d.shape[0] for d in dialect_dicts],
                input_nums=None if input_dicts is None \
                    else [d.shape[0] for d in input_dicts]
            )
            optimizer = tf.optimizers.SGD(
                tf.optimizers.schedules.ExponentialDecay(0.1, 100000, 0.9)
            )
            new_model.fit(
                optimizer,
                train_data,
                validate_data,
                epochs=10,
                batch_size=100,
                output_path=os.path.join(prefix, name, n)
            )

def train(
    config,
    name,
    dialect_nums,
    input_nums,
    output_nums,
    train_data,
    validate_data=None
):
    """
    训练模型.

    Parameters:
        config (dict): 多层级的配置字典，分析配置文件获得
        name (str): 用于训练的模型配置名称，用于从 config 中读取指定配置
        dialect_nums, input_nums, output_nums: 传给模型构造函数的参数
        train_data (`tensorflow.data.Dataset`): 训练数据
        validate_data (`tensorflow.data.Dataset`): 评估数据

    从 config 中读取由 name 指定的配置，创建并训练模型。
    """

    # 从模型列表中查找待训练的模型
    conf = next((c for c in config['models'] if c['name'] == name)).copy()
    conf.pop('name')
    model, optimizer = build_model(
        conf,
        dialect_nums=dialect_nums,
        input_nums=input_nums,
        output_nums=output_nums
    )
    output_path = os.path.join(config['output_dir'], name)

    logging.info(f'training model {name}, output path = {output_path}...')
    model.fit(
        optimizer,
        train_data,
        validate_data=validate_data,
        output_path=output_path,
        **conf
    )
    logging.info('done.')

def evaluate(config, name, dialect_nums, input_nums, output_nums, data):
    """
    评估训练完成的模型效果.

    Parameters:
        config (dict): 多层级的配置字典，分析配置文件获得
        name (str): 用于训练的模型配置名称，用于从 config 中读取指定配置
        dialect_nums, input_nums, output_nums: 传给模型构造函数的参数
        data (`tensorflow.data.Dataset`): 评估数据
    """

    conf = next((c for c in config['models'] if c['name'] == name)).copy()
    model, _ = build_model(
        conf,
        dialect_nums=dialect_nums,
        input_nums=input_nums,
        output_nums=output_nums
    )

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
    loss, acc = model.evaluate(data.batch(conf.get('batch_size', 100)))
    logging.info(f'done. loss = {loss}, accuracy = {acc}')

def export(config, name, dicts, prefix='.'):
    """
    从模型导出向量及其他权重到文件.

    Parameters:
        config (dict): 多层级的配置字典，分析配置文件获得
        name (str): 用于训练的模型配置名称，用于从 config 中读取指定配置
        dicts (dict): 模型输入输出字典
        prefix (str): 输出路径前缀
    """

    conf = next((c for c in config['models'] if c['name'] == name)).copy()
    model, _ = build_model(
        conf,
        dialect_nums=[dicts[c].shape[0] for c in config['columns']['dialect']],
        input_nums=[dicts[c].shape[0] for c in config['columns']['input']],
        output_nums=[dicts[c].shape[0] for c in config['columns']['output']]
    )

    checkpoint = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(
        checkpoint,
        os.path.join(config['output_dir'], name, 'checkpoints'),
        None
    )
    logging.info(f'restore model from checkpoint {manager.latest_checkpoint}...')
    checkpoint.restore(manager.latest_checkpoint)
    logging.info('done.')

    output_dir = os.path.join(prefix, name)
    logging.info(f'exporting {name} weights to {output_dir}...')
    os.makedirs(output_dir, exist_ok=True)

    # 导出输入输出向量，保存为 CSV 格式
    for name in ('dialect', 'input', 'output'):
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
        choices=['mkdict', 'train', 'evaluate', 'benchmark', 'export'],
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
        dialect_dicts, input_dicts, output_dicts \
            = [[dicts[c] for c in config['columns'][name]] \
                for name in ('dialect', 'input', 'output')]

    if args.command in ('train', 'evaluate'):
        data = load_datasets(config['datasets'])
        data = tuple([OrdinalEncoder(
            categories=[dicts[c].index for c in config['columns'][name]],
            dtype=np.int32,
            handle_unknown='use_encoded_value',
            unknown_value=-1,
            encoded_missing_value=-1
        ).fit(data[:1][config['columns'][name]]) \
            .transform(data[config['columns'][name]]) \
            for name in ('dialect', 'input', 'output')])

        data = tf.data.Dataset.from_tensor_slices(data) \
            .shuffle(data[0].shape[0], seed=10273)

    if args.command == 'mkdict':
        mkdict(config)

    elif args.command == 'train':
        train(
            config,
            args.model,
            *[[d.shape[0] for d in dicts] \
                for dicts in (dialect_dicts, input_dicts, output_dicts)],
            data
        )

    elif args.command == 'evaluate':
        evaluate(
            config,
            args.model,
            *[[d.shape[0] for d in dicts] \
                for dicts in (dialect_dicts, input_dicts, output_dicts)],
            data
        )

    elif args.command == 'benchmark':
        benchmark(config, load_datasets(config['datasets']))

    elif args.command == 'export':
        export(config, args.model, dicts, args.output)
