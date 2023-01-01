#!/usr/bin/python3 -O
# -*- coding: utf-8 -*-

'''
根据多个现代方言点读音重构祖语音节表.

使用基于决策树的自动编码器进行去噪和聚类，
使用 logistic 回归补全缺失读音。
'''

__author__ = '黄艺华 <lernanto.wong@gmail.com>'


import sys
import logging
import argparse
import json
import re
import numpy as np
import pandas as pd
import scipy.stats as st
from sklearn.pipeline import make_pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import joblib

import sinetym


def impute(data):
    '''
    根据已有读音数据填充缺失读音.

    使用基于 logistic 回归的分类器迭代填充缺失值
    '''

    # 先把读音字符串转成编码
    cat = pd.concat([data[c].astype('category') for c in data.columns], axis=1)
    codes = np.stack([cat[c].cat.codes for c in cat.columns], axis=1)

    # 使用 logistic 回归填充缺失读音
    pipeline = make_pipeline(
        OneHotEncoder(categories='auto', handle_unknown='ignore', dtype=np.int32),
        LogisticRegression(solver='saga', multi_class='multinomial', penalty='l1', C=1, max_iter=10)
    )

    imputer = IterativeImputer(
        missing_values=-1,
        initial_strategy='most_frequent',
        estimator=pipeline,
        max_iter=3
    )
    imputed = imputer.fit_transform(codes).astype(np.int32)

    # 填充的编号转回字符串
    return pd.DataFrame(
        np.stack([cat.iloc[:, i].cat.categories[imputed[:, i]] \
            for i in range(cat.shape[1])], axis=1),
        index=data.index,
        columns=data.columns
    )

def denoise(data):
    '''
    使用基于决策树的自动编码器去除数据噪音.
    '''

    # 训练基于决策树的自动编码器
    encoder = OneHotEncoder(categories='auto', dtype=np.int32)
    tree = DecisionTreeClassifier(criterion='entropy', min_impurity_decrease=0.0001)
    features = encoder.fit_transform(data)
    tree.fit(features, data)

    # 使用训练的自动编码器重建数据
    recon = pd.DataFrame(
        tree.predict(features),
        index=data.index,
        columns=data.columns
    )

    return make_pipeline(encoder, tree), recon, tree.apply(features)

def get_syllables(data, labels):
    '''
    根据去噪结果返回音节表.

    决策树每个叶节点代表祖语的一个独立音节
    '''

    def get_freq(row):
        '''
        统计祖语音节在现代方言点每个表现型的出现频率.
        '''

        pron, freq = np.unique(row, return_counts=True)
        # 按频率从高到低排序，最多取前10个
        idx = np.argsort(freq)[::-1]
        return ','.join('{}:{}'.format(pron[i], freq[i]) for i in idx[:10])

    # 根据自动编码器分类结果聚类
    cluster = data.groupby(labels).first()
    cluster.loc[:, ('freq', 'initial')] = \
        cluster.loc[:, pd.IndexSlice[:, 'initial']].apply(get_freq, axis=1)
    cluster.loc[:, ('freq', 'final')] = \
        cluster.loc[:, pd.IndexSlice[:, 'final']].apply(get_freq, axis=1)
    cluster.loc[:, ('freq', 'tone')] = \
        cluster.loc[:, pd.IndexSlice[:, 'tone']].apply(get_freq, axis=1)

    return cluster

def reconstruct(data, min_sample=3, initial_mid=0.002, final_mid=0.002, tone_mid=0.01):
    '''
    使用决策树重构祖语的声韵调类.

    分别对现代方言的声母、韵母、调类训练多输出、多分类器，配合剪枝，
    实际上相当于单独针对声母、韵母、调类训练编码器。
    训练出来的决策树每个叶节点大致相当于祖语的一个声母/韵母/调类。
    '''

    encoder = OneHotEncoder(dtype=np.int32)
    features = encoder.fit_transform(data)

    initial_tree = DecisionTreeClassifier(
        criterion='entropy',
        min_samples_leaf=min_sample,
        min_impurity_decrease=initial_mid
    ).fit(features, data.loc[:, pd.IndexSlice[:, 'initial']])
    final_tree = DecisionTreeClassifier(
        criterion='entropy',
        min_samples_leaf=min_sample,
        min_impurity_decrease=final_mid
    ).fit(features, data.loc[:, pd.IndexSlice[:, 'final']])
    tone_tree = DecisionTreeClassifier(
        criterion='entropy',
        min_samples_leaf=min_sample,
        min_impurity_decrease=tone_mid
    ).fit(features, data.loc[:, pd.IndexSlice[:, 'tone']])

    # 根据决策树的预测结果重构方言读音
    recon = pd.DataFrame(
        np.concatenate([
            initial_tree.predict(features),
            final_tree.predict(features),
            tone_tree.predict(features)
        ], axis=1),
        index=data.index,
        columns=pd.MultiIndex.from_product((
            ('initial', 'final', 'tone'),
            data.columns.levels[0]
        ))
    ).swaplevel(axis=1).sort_index(axis=1, level=0, sort_remaining=False)

    recon.loc[:, ('cluster', 'initial')] = initial_tree.apply(features)
    recon.loc[:, ('cluster', 'final')] = final_tree.apply(features)
    recon.loc[:, ('cluster', 'tone')] = tone_tree.apply(features)

    return recon, encoder, initial_tree, final_tree, tone_tree

def get_stats(data, labels):
    '''
    统计重构的声韵调的若干统计数据.

    - 每个叶节点在现代方言的读音分布
    - 每个叶节点现代方言读音的熵，及叶节点熵的均值
    '''

    def _get_stats(expanded, key, name):
        stats = expanded.groupby(key)[[c for c in expanded.columns if name in c]].mean()

        for d in data.columns.levels[0]:
            stats[d] = st.entropy(stats[[c for c in stats.columns if c.startswith(d)]].values.T)

        stats['entropy'] = stats[data.columns.levels[0]].mean(axis=1)
        return stats

    expanded = pd.get_dummies(data)
    initial_stats = _get_stats(expanded, labels['initial'], 'initial')
    final_stats = _get_stats(expanded, labels['final'], 'final')
    tone_stats = _get_stats(expanded, labels['tone'], 'tone')

    return initial_stats, final_stats, tone_stats

def main():
    parser = argparse.ArgumentParser(globals().get('__doc__', ''))
    parser.add_argument('--input', help='原始方言读音文件所在目录，CSV 格式')
    parser.add_argument('--imputed-output', help='填充缺失值的数据的输出文件')
    parser.add_argument('--recon-output', help='根据自动编码器重构的读音的输出文件')
    parser.add_argument('--encoder-output', help='输入读音编码器模型输出文件')
    parser.add_argument('--initial-tree-output', help='声母决策树输出文件')
    parser.add_argument('--final-tree-output', help='韵母决策树输出文件')
    parser.add_argument('--tone-tree-output', help='声调决策树输出文件')
    parser.add_argument('--initial-stats-output', help='声母聚类统计数据输出文件')
    parser.add_argument('--final-stats-output', help='韵母聚类统计数据输出文件')
    parser.add_argument('--tone-stats-output', help='声调聚类统计数据输出文件')
    parser.add_argument('dialects', nargs='*', help='要加载的方言读音，为空加载全部方言')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG)

    # 加载方言字音数据
    data = sinetym.datasets.transform_data(
        sinetym.datasets.load_data(args.input, *args.dialects) \
            .rename(columns={'tone_category': 'tone'}),
        index='cid',
        values=['initial', 'final', 'tone'],
        aggfunc='first'
    ).replace('', pd.NA)

    # 丢弃缺失读音太多的记录
    data.dropna(axis=1, thresh=data.shape[0] * 0.2, inplace=True)
    data.dropna(axis=0, thresh=data.shape[1] * 0.5, inplace=True)

    # 分类器不能处理缺失值，先根据已有数据填充缺失读音
    imputed = impute(data)
    imputed.to_hdf(args.imputed_output, key='imputed')

    # 使用自动编码器重构祖语声韵调类
    recon, encoder, initial_tree, final_tree, tone_tree = reconstruct(imputed)

    # 输出结果
    recon.to_hdf(args.recon_output, key='recon')
    joblib.dump(encoder, args.encoder_output)
    joblib.dump(initial_tree, args.initial_tree_output)
    joblib.dump(final_tree, args.final_tree_output)
    joblib.dump(tone_tree, args.tone_tree_output)

    initial_stats, final_stats, tone_stats = get_stats(
        data,
        recon.loc[:, 'cluster']
    )
    initial_stats.to_hdf(args.initial_stats_output, key='initial_stats')
    final_stats.to_hdf(args.final_stats_output, key='final_stats')
    tone_stats.to_hdf(args.tone_stats_output, key='tone_stats')


if __name__ == '__main__':
    main()
