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


def load(fname):
    '''
    加载方言字音数据.

    JSON 格式，每行一个字
    '''

    data = []
    for line in open(fname):
        try:
            obj = json.loads(line)

            if 'dialects' in obj:
                # 中古音
                row = {'id': obj['id'], 'char': obj['char'].strip()}
                for k in ('攝', '聲調', '韻目', '字母', '開合', '等第', '清濁'):
                    row['_'.join(('中古音', k))] = obj['middle_chinese'][k]

                # 方言读音
                for d in obj['dialects']:
                    for k in ('聲母', '韻母', '調類', '調值'):
                        key = '_'.join((d['方言點'].strip(), k))
                        val = d[k].strip()
                        row[key] = val

                data.append(row)

        except:
            logging.error('error parsing line! {}'.format(line[:50]), exc_info=True)

    output = pd.DataFrame(data)
    output.index = output.pop('id')
    return output

def clean(data):
    '''
    基于规则的数据清洗.
    '''

    def get_rows(col, max_row):
        rows = []
        for i, r in enumerate(col):
            if max_row[i] <= 1:
                # 该行不包含多音字，直接追加
                rows.append(r)

            elif isinstance(r, str) and '/' in r:
                # 包含多音字，展开
                vals = r.split('/')
                for j in range(max_row[i]):
                    # 读音数量少于最大读音数的，缺少的部分以第一个读音填充
                    rows.append(vals[j] if j < len(vals) else vals[0])

            else:
                # 该行包含多音字但该列不包含多音字，重复多次
                rows.extend([r] * max_row[i])

        return np.asarray(rows, dtype=np.object_)

    # 原始数据中有些格子有多个读音，以斜杠分隔，需要展开
    columns = [c for c in data.columns if c.partition('_')[-1] in ('聲母', '韻母', '調類', '調值')]
    # 预先计算每行最多的读音数
    max_row = np.asarray([(data[c].str.count('/') + 1).fillna(0) for c in columns], dtype=np.int32).max(axis=0)
    output = data.reset_index().apply(get_rows, axis=0, max_row=max_row)

    # 删除太长的读音
    for c in columns:
        output.loc[output[c].str.len() > 4, c] = np.NaN

    for c in output.columns:
        if c.endswith('聲母'):
            # 只允许国际音标
            output[c].replace(r'.*[^0A-Za-z()\u0080-\u03ff\u1d00-\u1d7f\u2070-\u209f].*', np.NaN, inplace=True)
            # 零声母统一用0填充
            output[c].replace('', '0', inplace=True)

        elif c.endswith('韻母'):
            output[c].replace(r'.*[^0A-Za-z()\u0080-\u03ff\u1d00-\u1d7f\u2070-\u209f].*', np.NaN, inplace=True)

        elif c.endswith('調類'):
            output[c].replace(r'.*[^上中下變陰陽平去入].*', np.NaN, inplace=True)

        elif c.endswith('調值'):
            output[c].replace(r'.*[^0-9].*', np.NaN, inplace=True)

    # 丢弃只出现一次的数据
    for c in output.columns:
        if c not in ('id', 'char'):
            count = output[c].value_counts()
            output[c].replace(count[count <= 1].index, np.NaN, inplace=True)

    return output

def impute(data):
    '''
    根据已有读音数据填充缺失读音.

    使用基于 logistic 回归的分类器迭代填充缺失值
    '''

    # 要填充的列
    columns = [c for c in data.columns if c.startswith('中古音') or c.partition('_')[-1] in ('聲母', '韻母', '調類', '調值')]

    # 先把读音字符串转成编码
    cat = pd.DataFrame()
    for c in columns:
        cat[c] = data[c].astype('category')

    codes = np.empty(cat.shape, dtype=np.int)
    for i, c in enumerate(columns):
        codes[:, i] = cat[c].cat.codes

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
    recon = pd.DataFrame()

    # 不需要填充的列保持原状
    for c in data.columns:
        if c not in columns:
            recon[c] = data[c].values

    # 要填充的列根据填充结果转回字符串
    for i, c in enumerate(columns):
        recon[c] = cat[c].cat.categories[imputed[:, i]]

    return recon

def denoise(data):
    '''
    使用基于决策树的自动编码器去除数据噪音.
    '''

    # 用于训练自动编码器的列
    columns = [c for c in data.columns if c.partition('_')[-1] in ('聲母', '韻母', '調值')]

    # 训练基于决策树的自动编码器
    encoder = OneHotEncoder(categories='auto', dtype=np.int32)
    tree = DecisionTreeClassifier(criterion='entropy', min_impurity_decrease=0.0001)
    features = encoder.fit_transform(data[columns])
    tree.fit(features, data[columns])

    # 使用训练的自动编码器重建数据
    recon = pd.DataFrame(columns=columns, data=tree.predict(features))
    recon['cluster'] = tree.apply(features)

    # 不需要去噪的列保持原状
    for c in data.columns:
        if c not in columns:
            recon[c] = data[c].values

    return make_pipeline(encoder, tree), recon

def get_syllables(data):
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
    tmp = data.groupby('cluster')
    cluster = tmp.first()
    cluster['char'] = tmp['char'].agg(lambda c: ''.join(set(c)))
    cluster['initial'] = cluster[[c for c in data.columns if '聲母' in c]].apply(get_freq, axis=1)
    cluster['rhyme'] = cluster[[c for c in data.columns if '韻母' in c]].apply(get_freq, axis=1)
    cluster['tone'] = cluster[[c for c in data.columns if '調值' in c]].apply(get_freq, axis=1)

    return cluster

def reconstruct(data, min_sample=3, initial_mid=0.002, final_mid=0.002, tone_mid=0.01):
    '''
    使用决策树重构祖语的声韵调类.

    分别对现代方言的声母、韵母、调类训练多输出、多分类器，配合剪枝，
    实际上相当于单独针对声母、韵母、调类训练编码器。
    训练出来的决策树每个叶节点大致相当于祖语的一个声母/韵母/调类。
    '''

    encoder = OneHotEncoder(dtype=np.int32)
    features = encoder.fit_transform(data[[c for c in data.columns if c.partition('_')[-1] in ('聲母', '韻母', '調類')]])
    initial_cols = [c for c in data.columns if c.endswith('聲母')]
    final_cols = [c for c in data.columns if c.endswith('韻母')]
    tone_cols = [c for c in data.columns if c.endswith('調類')]
    initials = data[initial_cols]
    finals = data[final_cols]
    tones = data[tone_cols]

    initial_tree = DecisionTreeClassifier(
        criterion='entropy',
        min_samples_leaf=min_sample,
        min_impurity_decrease=initial_mid
    ).fit(features, initials)
    final_tree = DecisionTreeClassifier(
        criterion='entropy',
        min_samples_leaf=min_sample,
        min_impurity_decrease=final_mid
    ).fit(features, finals)
    tone_tree = DecisionTreeClassifier(
        criterion='entropy',
        min_samples_leaf=min_sample,
        min_impurity_decrease=tone_mid
    ).fit(features, tones)

    # 根据决策树的预测结果重构方言读音
    recon = pd.DataFrame(
        columns=np.concatenate([initial_cols, final_cols, tone_cols]),
        data=np.concatenate([
            initial_tree.predict(features),
            final_tree.predict(features),
            tone_tree.predict(features)
        ], axis=1)
    )

    recon['char'] = data['char'].values
    recon['initial'] = initial_tree.apply(features)
    recon['final'] = final_tree.apply(features)
    recon['tone'] = tone_tree.apply(features)

    return recon, encoder, initial_tree, final_tree, tone_tree

def get_stats(data):
    '''
    统计重构的声韵调的若干统计数据.

    - 每个叶节点在现代方言的读音分布
    - 每个叶节点现代方言读音的熵，及叶节点熵的均值
    '''

    def _get_stats(expanded, key, name):
        locations = np.unique([c.partition('_')[0] for c in data.columns if c.endswith(name)])
        tmp = expanded.groupby(key)

        stats = tmp[[c for c in expanded.columns if name in c]].mean()
        stats['char'] = tmp['char'].agg(lambda s: ''.join(set(s)))

        for l in locations:
            stats[l] = st.entropy(stats[[c for c in stats.columns if c.startswith(l)]].values.T)

        stats['entropy'] = stats[locations].mean(axis=1)
        return stats

    expanded = pd.get_dummies(data, columns=[c for c in data.columns if c.partition('_')[-1] in ('聲母', '韻母', '調類')])
    initial_stats = _get_stats(expanded, 'initial', '聲母')
    final_stats = _get_stats(expanded, 'final', '韻母')
    tone_stats = _get_stats(expanded, 'tone', '調類')

    return initial_stats, final_stats, tone_stats

def main():
    parser = argparse.ArgumentParser(globals().get('__doc__', ''))
    parser.add_argument('--clean-output', help='去除噪音的数据的输出文件')
    parser.add_argument('--imputed-output', help='填充缺失值的数据的输出文件')
    parser.add_argument('--recon-output', help='根据自动编码器重构的读音的输出文件')
    parser.add_argument('--encoder-output', help='输入读音编码器模型输出文件')
    parser.add_argument('--initial-tree-output', help='声母决策树输出文件')
    parser.add_argument('--final-tree-output', help='韵母决策树输出文件')
    parser.add_argument('--tone-tree-output', help='声调决策树输出文件')
    parser.add_argument('--initial-stats-output', help='声母聚类统计数据输出文件')
    parser.add_argument('--final-stats-output', help='韵母聚类统计数据输出文件')
    parser.add_argument('--tone-stats-output', help='声调聚类统计数据输出文件')
    parser.add_argument('input', nargs='+', help='原始数据输入文件，JSON 格式')
    args = parser.parse_args()

    # 加载方言字音数据
    if (len(args.input) == 1):
        # 只有一个输入文件，直接加载
        data = load(args.input[0])

    else:
        # 多个输入文件，分别加载然后拼接
        data = [load(f) for f in args.input]
        data = pd.concat(
            [data[0][['char'] + [c for c in data[0].columns if c.startswith('中古音')]]] \
                + [df[[c for c in df.columns if c.partition('_')[-1] in ('聲母', '韻母', '調類', '調值')]] for df in data],
            axis=1
        )

    # 清洗
    data = clean(data)

    # 丢弃生僻字及缺失读音太多的记录
    data.dropna(axis=1, thresh=data.shape[0] * 0.2, inplace=True)
    data.dropna(axis=0, thresh=data.shape[1] * 0.5, inplace=True)
    data = data[data['char'].str.match(r'^[\u4e00-\u9fa5]$')].reset_index(drop=True)
    data.to_hdf(args.clean_output, key='clean')

    # 分类器不能处理缺失值，先根据已有数据填充缺失读音
    imputed = impute(data)
    imputed.to_hdf(args.imputed_output, key='imputed')

    # 使用自动编码器重构祖语声韵调类
    recon, encoder, initial_tree, final_tree, tone_tree = reconstruct(imputed)
    data[['initial', 'final', 'tone']] = recon[['initial', 'final', 'tone']]

    # 输出结果
    recon.to_hdf(args.recon_output, key='recon')
    joblib.dump(encoder, args.encoder_output)
    joblib.dump(initial_tree, args.initial_tree_output)
    joblib.dump(final_tree, args.final_tree_output)
    joblib.dump(tone_tree, args.tone_tree_output)

    initial_stats, final_stats, tone_stats = get_stats(data)
    initial_stats.to_hdf(args.initial_stats_output, key='initial_stats')
    final_stats.to_hdf(args.final_stats_output, key='final_stats')
    tone_stats.to_hdf(args.tone_stats_output, key='tone_stats')


if __name__ == '__main__':
    main()
