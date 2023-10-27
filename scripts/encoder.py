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
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import tensorflow as tf
from tensorboard.plugins import projector

import sinetym.datasets
import sinetym.models
import sinetym.auxiliary


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

def split_data(data, test_did_size=0.1, test_cid_size=0.1, random_state=None):
    """
    根据方言 ID 和字 ID 切分数据集.

    Parameters:
        data (`pandas.DataFrame`): 原始数据集
        test_did_size (float): 测试方言 ID 占总样本的比例
        test_cid_size (float): 测试字 ID 占总样本的比例
        random_state (int or `numpy.random.RandomState`): 用于复现划分结果

    Returns:
        data1, data2, data3, data4 (`pandas.DataFrame`): 切分后的数据集

    从 data 中随机选择一批方言 ID 和字 ID，作为测试 ID，把数据集切分为如下部分：
        - data1: 不包含测试方言 ID 和测试字 ID
        - data2: 只包含测试方言 ID 且不包含测试字 ID
        - data3: 不包含测试方言 ID 且只包含测试字 ID
        - data4: 只包含测试方言 ID 和测试字 ID
    """

    logging.info(f'split data, size = {data.shape[0]}')

    if random_state is None:
        random_state = np.random.mtrand._rand
    elif isinstance(random_state, int):
        random_state = np.random.RandomState(random_state)

    # 随机选择测试方言 ID 和字 ID，使满足在样本中占指定的比例，但尽量选择常见的方言和字加入训练集
    did = data['did'].value_counts() / data.shape[0]
    weight = did + random_state.normal(scale=did.std(), size=did.shape[0])
    did = did[weight.sort_values().index].cumsum()
    test_did = pd.Series(True, index=did[did < test_did_size].index) \
        .reindex(data['did']).fillna(False).values

    cid = data['cid'].value_counts() / data.shape[0]
    weight = cid + random_state.normal(scale=cid.std(), size=cid.shape[0])
    cid = cid[weight.sort_values().index].cumsum()
    test_cid = pd.Series(True, index=cid[cid < test_cid_size].index) \
        .reindex(data['cid']).fillna(False).values

    data = (
        data[~(test_did | test_cid)],
        data[test_did & ~test_cid],
        data[~test_did & test_cid],
        data[test_did & test_cid]
    )

    logging.info(f'done, result data size = {[d.shape[0] for d in data]}')
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

def build_new_model(model, new_dialect_num=None, new_input_num=None):
    """
    基于已训练的模型创建新模型.

    Parameters:
        model (sinetym.models.EncoderBase): 已训练的基线模型
        new_dialect_num (int): 新增方言数量，为空时无新增方言
        new_input_num (int): 新增字数量，为空时无新增字

    Returns:
        new_model: 新模型，该模型为以 model.__class__ 为基类的新类型

    沿用 model 内部所有变量，只针对 new_dialect_num 和 new_input_num 在 new_model 内创建新变量。
    new_model 预测时，对于输入方言 ID 和字 ID 优先检索新变量，不命中才使用 model 的旧变量。
    new_model 训练时只更新新变量，但 model 的变量由于其他原因改变时，也会影响 new_model 的预测结果。
    """

    logging.info(
        f'build new model from {model.name}, '
        f'with {new_dialect_num} new dialects, {new_input_num} new inputs.'
    )

    class NewModel(model.__class__):
        """
        为新模型动态构造新类型，继承 model.
        """

        def __init__(self, model, new_dialect_num=None, new_input_num=None):
            """
            为 new_dialect_num 和 new_input_num 创建相应的变量，其他变量沿用 model 的.
            """

            self.__dict__.update(model.__dict__)

            if new_dialect_num is not None:
                self.new_dialect_emb = tf.Variable(
                    tf.random_normal_initializer()(
                        shape=(new_dialect_num, self.dialect_emb_size),
                        dtype=tf.float32
                    ), name='new_dialect_emb'
                )

            if new_input_num is not None:
                self.new_input_emb = tf.Variable(
                    tf.random_normal_initializer()(
                        shape=(new_input_num, self.input_emb_size),
                        dtype=tf.float32
                    ), name='new_input_emb'
                )

        def encode_dialect(self, dialect):
            """
            方言 ID 在原模型中找不到尝试在 new_dialect_emb 中查找.
            """

            if not hasattr(self, 'new_dialect_emb'):
                return super().encode_dialect(dialect)

            return tf.where(
                tf.logical_or(dialect[:, 0:1] > 0, dialect[:, 1:2] == 0),
                super().encode_dialect(dialect),
                tf.nn.embedding_lookup(self.new_dialect_emb, dialect[:, 1])
            )

        def encode(self, inputs):
            """
            字 ID 在原模型中找不到尝试在 new_input_emb 中查找.
            """

            if not hasattr(self, 'new_input_emb'):
                return super().encode(inputs)

            n = len(self.input_embs)
            return tf.where(
                tf.logical_or(inputs[:, 0:1] > 0, inputs[:, n:n + 1] == 0),
                super().encode(inputs),
                tf.nn.embedding_lookup(self.new_input_emb, inputs[:, n])
            )

        @property
        def trainable_varialbes(self):
            """
            改写 trainable_variables 属性使训练时只更新新变量.
            """

            return tuple(getattr(self, n) for n in (
                'new_dialect_emb',
                'new_input_emb'
            ) if hasattr(self, n))

    return NewModel(model, new_dialect_num, new_input_num)

def make_data(
    data,
    output_encoder,
    test_size=0.2,
    encoder=None,
    new_dialect=False,
    new_input=False,
    random_state=None
):
    """
    为模型构造训练及测试数据集.

    Parameters:
        data (`pandas.DataFrame`): 原始数据
        output_encoder (`sklearn.preprocessing.OrdinalEncoder`): 编码输出数据的编码器
        test_size (float): 切分评估数据的比例
        encoder (sinetym.auxiliary.OrdinalEncoder): 用于编码数据的基础编码器
        new_dialect (bool): 是否为新方言构造数据
        new_input (bool): 是否为新字构造数据
        random_state (int or `numpy.random.RandomState`): 用于复现划分数据结果

    Returns:
        train_data (`tensorflow.data.Dataset`): 训练数据集
        test_data (`tensorflow.data.Dataset`): 测试数据集
        encoder (sinetym.auxiliary.OrdinalEncoder): 针对训练数据新建的编码器
        new_dialect_num (int): 训练集中包含的方言 ID 数，
            如果 new_dialect 为假，返回 None
        new_input_num (int): 训练集中包含的字 ID 数，
            如果 new_input 为假，返回 None

    把原始数据按 test_size 随机分为训练集和评估集，并用 encoder 和 output_encoder 编码数据。
    当 new_dialect 为真时，对训练和测试集中的方言 ID 作额外的新编码，
    当 new_input 为真时，对训练和测试集中的字 ID 作额外的新编码，
    把新编码作为新列插入原始编码数据。
    """

    logging.info(f'making train/test data from {data.shape[0]} data...')

    tri, tei, train_output, test_output = train_test_split(
        data[['did', 'cid', 'character']],
        output_encoder.transform(data[['initial', 'final', 'tone']]),
        test_size=test_size,
        random_state=random_state
    )

    # 计算需要插入新编码的位置
    columns = []
    indeces = []
    limits = np.asarray([0, 1, 3])

    if new_dialect:
        columns.append('did')
        indeces.append(1)
        limits[1:] += 1

    if new_input:
        columns.append('cid')
        indeces.append(3)
        limits[2:] += 1

    # 为新方言 ID 和字 ID 构建编码器，并把新编码插入现有编码的后一列
    minfreq = max(int(0.00001 * tri.shape[0]), 2)
    if encoder is None:
        encoder = sinetym.auxiliary.OrdinalEncoder(
            dtype=np.int32,
            min_frequency=minfreq
        ).fit(pd.DataFrame(
            SimpleImputer(missing_values='', strategy='most_frequent') \
                .fit_transform(tri),
            index=tri.index,
            columns=tri.columns
        ))

    train_input = encoder.transform(tri)
    test_input = encoder.transform(tei)

    if len(columns) > 0:
        encoder = sinetym.auxiliary.OrdinalEncoder(
            dtype=np.int32,
            min_frequency=minfreq
        ).fit(pd.DataFrame(
            SimpleImputer(missing_values='', strategy='most_frequent') \
                .fit_transform(tri[columns]),
            index=tri.index,
            columns=columns
        ))
        train_input = np.insert(
            train_input,
            indeces,
            encoder.transform(tri[columns]),
            axis=1
        )
        test_input = np.insert(
            test_input,
            indeces,
            encoder.transform(tei[columns]),
            axis=1
        )

    train_data = tf.data.Dataset.from_tensor_slices(
        tuple(train_input[:, limits[i]:limits[i + 1]] \
            for i in range(limits.shape[0] - 1)) + (train_output,)
    )
    train_data = train_data.shuffle(train_data.cardinality())
    test_data = tf.data.Dataset.from_tensor_slices(
        tuple(test_input[:, limits[i]:limits[i + 1]] \
            for i in range(limits.shape[0] - 1)) + (test_output,)
    )

    new_dialect_num = encoder.categories_[0].shape[0] + 1 if new_dialect else None
    new_input_num = encoder.categories_[-1].shape[0] + 1 if new_input else None

    logging.info(
        f'done, train data size = {train_data.cardinality()}, '
        f'test data size = {test_data.cardinality()}, '
        f'new dialect number = {new_dialect_num}, '
        f'new input number = {new_input_num}.'
    )

    return (train_data, test_data, encoder, new_dialect_num, new_input_num)

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

def mkdict(data, prefix='.', minfreq=2):
    """
    根据方言数据构建词典.

    Parameters:
        data (`pandas.DataFrame`): 方言读音数据表
        prefix (str): 保存词典的目录
        minfreq (int): 出现频次不小于该值才计入词典
    """

    logging.info(f'make dictionaries to {prefix}...')

    for name in ('did', 'cid', 'character', 'initial', 'final', 'tone'):
        dic = data.loc[data[name] != '', name].value_counts().rename('count')
        dic.index.rename(name, inplace=True)

        if minfreq is not None and minfreq > 1:
            dic = dic[dic >= minfreq]

        fname = f'{os.path.join(prefix, name)}.csv'
        logging.info(f'save {fname}')
        dic.sort_values(ascending=False).to_csv(
            fname,
            encoding='utf-8',
            lineterminator='\n'
        )

    logging.info('done.')

def benchmark(config, data):
    """
    评估各种配置的模型效果.

    Parameters:
        config (dict): 用于创建模型的配置字典
        data (`pandas.DataFrame`): 用于训练及评估的数据集

    从 config 读取所有模型配置，为每一组配置创建一个模型，训练并评估效果。
    每个模型除使用常规的评估集评估效果以外，还额外评估：
        - 对基础模型没见过的方言的适应能力
        - 对基础模型没见过的字的适应能力
        - 对方言和字基础模型都没见过的数据的适应能力
    评估方法为保持基础模型参数不变的基础上，从包含新方言或新字的数据集随机划分一部分数据
    来训练新的方言向量和字向量，然后评估预测剩余另一部分数据的准确率。
    """

    # 输出数据必须全部编码，编码前剔除缺失值
    columns = ['initial', 'final', 'tone']
    output_encoder = sinetym.auxiliary.OrdinalEncoder(
        dtype=np.int32,
        min_frequency=2
    ).fit(pd.DataFrame(
        SimpleImputer(missing_values='', strategy='most_frequent') \
            .fit_transform(data[columns]),
        index=data.index,
        columns=columns
    ))

    # 切分数据用于训练及不同项目的评估
    data1, data2, data3, data4 = split_data(data, random_state=37511)
    train_data1, eval_data1, encoder, _, _ = make_data(
        data1,
        output_encoder,
        test_size=0.1,
        random_state=37511
    )
    dialect_num = encoder.categories_[0].shape[0] + 1
    input_nums = [c.shape[0] + 1 for c in encoder.categories_[1:]]
    output_nums = [c.shape[0] + 1 for c in output_encoder.categories_]
    logging.info(
        f'dialect number = {dialect_num}, input numbers = {input_nums}, '
        f'output numbers = {output_nums}'
    )

    # 保存词典
    prefix = os.path.join(config['output_dir'], 'benchmark')
    dict_dir = os.path.join(prefix, 'dictionaries')
    os.makedirs(dict_dir, exist_ok=True)
    logging.info(f'saving dictionaries to {dict_dir}...')

    for e in (encoder, output_encoder):
        for i, name in enumerate(e.feature_names_in_):
            pd.Series(e.categories_[i], name=name).to_csv(
                os.path.join(dict_dir, name + '.csv'),
                index=False,
                encoding='utf-8',
                lineterminator='\n'
            )

    logging.info('done.')

    # 剩余数据预处理成数据集用于评估
    data = []
    for n, d, new_dialect, new_input in (
        ('new_dialect', data2, True, False),
        ('new_input', data3, False, True),
        ('new_dialect_input', data4, True, True)
    ):
        train_data, eval_data, new_encoder, new_dialect_num, new_input_num = make_data(
            d,
            output_encoder,
            test_size=0.5,
            encoder=encoder,
            new_dialect=new_dialect,
            new_input=new_input,
            random_state=37511
        )
        data.append((n, train_data, eval_data, new_dialect_num, new_input_num))

        if new_encoder is not None:
            new_dict_dir = os.path.join(dict_dir, n)
            os.makedirs(new_dict_dir, exist_ok=True)
            for i, f in enumerate(new_encoder.feature_names_in_):
                pd.Series(new_encoder.categories_[i], name=f).to_csv(
                    os.path.join(new_dict_dir, f + '.csv'),
                    index=False,
                    encoding='utf-8',
                    lineterminator='\n'
                )

    for conf in config['models']:
        # 根据配置文件创建模型并训练
        conf = conf.copy()
        name = conf.pop('name')
        model, optimizer = build_model(
            conf,
            dialect_num=dialect_num,
            input_nums=input_nums,
            output_nums=output_nums
        )

        output_path = os.path.join(prefix, name)
        logging.info(f'start training model {name}, output path = {output_path} ...')
        model.fit(
            optimizer,
            train_data1,
            eval_data1,
            output_path=output_path,
            **conf
        )
        logging.info('done.')

        # 使用剩余的数据集评估模型效果
        for n, train_data, eval_data, new_dialect_num, new_input_num in data:
            logging.info(
                f'evaluate {n}, train size = {train_data.cardinality()}, '
                f'evaluation size = {eval_data.cardinality()}.'
            )

            # 从上面训练完成的模型复制一个副本，针对数据微调然后评估
            new_model = build_new_model(
                model,
                new_dialect_num=new_dialect_num,
                new_input_num=new_input_num
            )
            optimizer = tf.optimizers.SGD(
                tf.optimizers.schedules.ExponentialDecay(0.1, 100000, 0.9)
            )
            new_model.fit(
                optimizer,
                train_data,
                eval_data,
                epochs=10,
                batch_size=100,
                output_path=os.path.join(config['output_dir'], 'benchmark', 'name', 'n')
            )

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
    model, optimizer = build_model(
        conf,
        dialect_num=config['dialect_num'],
        input_nums=config['input_nums'],
        output_nums=config['output_nums']
    )
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
    model, _ = build_model(
        conf,
        dialect_num=config['dialect_num'],
        input_nums=config['input_nums'],
        output_nums=config['output_nums']
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
    loss, acc = model.evaluate(eval_data.batch(conf.get('batch_size', 100)))
    logging.info(f'done. loss = {loss}, accuracy = {acc}')

def export(config, name, prefix='.'):
    """
    从模型导出权重到文件.

    Parameters:
        config (dict): 多层级的配置字典，分析配置文件获得
        name (str): 用于训练的模型配置名称，用于从 config 中读取指定配置
        dialect_num, input_nums, output_nums: 创建模型的参数
        prefix (str): 输出路径前缀
    """

    conf = next((c for c in config['models'] if c['name'] == name)).copy()
    model, _ = build_model(
        conf,
        dialect_num=config['dialect_num'],
        input_nums=config['input_nums'],
        output_nums=config['output_nums']
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

    output_dir = os.path.join(
        prefix,
        f'{name}_{int(manager.checkpoint.save_counter)}'
    )
    logging.info(f'exporting {name} weights to {output_dir}...')
    os.makedirs(output_dir, exist_ok=True)
    for v in model.variables:
        a = v.numpy()
        np.savetxt(
            os.path.join(output_dir, v.name.partition(':')[0]),
            np.reshape(a, (-1, a.shape[-1])) if a.ndim > 2 else a
        )
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

    if args.command in ('train', 'evaluate'):
        did, cid, character, initial, final, tone = load_dictionaries(
            config.get('dictionary_dir', '.')
        )
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

    if args.command == 'mkdict':
        mkdict(
            load_datasets(config['datasets']),
            config.get('dictionary_dir', '.'),
            config.get('min_freq')
        )

    elif args.command == 'train':
        train(config, args.model, data)

    elif args.command == 'evaluate':
        evaluate(config, args.model, data)

    elif args.command == 'benchmark':
        benchmark(config, load_datasets(config['datasets']))

    elif args.command == 'export':
        export(config, args.model, args.output)
