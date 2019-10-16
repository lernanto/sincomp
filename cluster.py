#!/usr/bin/python3 -O
# -*- coding: utf-8 -*-

'''
根据多个方言点现代读音对祖语音节聚类.

使用决策树补全缺失数据和聚类
'''

__author__ = 'Edward Wong <lernanto.wong@gmail.com>'


import sys
import logging
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


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
                tmp = {}
                max_row = 0
                for d in obj['dialects']:
                    for k in ('聲母', '韻母', '調值'):
                        key = '_'.join((d['方言點'], k))
                        # 有些字有多个读音，以斜杠分隔
                        val = d[k].split('/')
                        tmp[key] = val
                        max_row = max(max_row, len(val))

                for i in range(max_row):
                    row = {'id': obj['id'], 'char': obj['char'].strip()}
                    for key, val in tmp.items():
                        # 对于有些字有些方言点读音比其他点少的，以第一个读音填充
                        row[key] = val[i] if i < len(val) else val[0]
                    data.append(row)

        except:
            logging.error('error parsing line! {}'.format(line[:50]), exc_info=True)

    return pd.DataFrame(data, dtype='category')

def main():
    infile = sys.argv[1]
    outfile = sys.argv[2]

    # 加载方言字音数据
    data = load(infile)

    ids = data.pop('id')
    chars = data.pop('char')
    # 丢弃缺失读音太多的记录
    idx = data.notna().sum(axis=1) / data.shape[1] > 0.5
    data = data[idx]
    ids = ids[idx]
    chars = chars[idx]

    # 根据已有数据填充缺失读音
    # 先把读音字符串转成编码
    codes = np.empty(data.shape, dtype=np.int)
    for i in range(data.shape[1]):
        codes[:, i] = data.iloc[:, i].cat.codes

    # 使用决策树填充缺失读音
    pipeline = make_pipeline(
        OneHotEncoder(categories='auto', handle_unknown='ignore'),
        DecisionTreeClassifier()
    )

    imputer = IterativeImputer(missing_values=-1, initial_strategy='most_frequent', estimator=pipeline)
    imputed = imputer.fit_transform(codes).astype(np.int)

    # 使用决策树作为自动编码器去除数据噪音
    encoder = OneHotEncoder(categories='auto')
    tree = DecisionTreeClassifier(criterion='entropy', min_impurity_decrease=0.0001)
    pipeline = make_pipeline(encoder, tree)
    pipeline.fit(imputed, imputed)

    # 使用训练的自动编码器重建数据，相当于对去噪后的数据聚类
    restore = pd.DataFrame({
        'id': ids,
        'char': chars,
        'cluster': tree.apply(encoder.transform(imputed))
    })

    for i in range(data.shape[1]):
        restore[data.columns[i]] = data.iloc[:, i].cat.categories[imputed[:, i]]

    # 根据自动编码器分类结果聚类
    tmp = restore.groupby('cluster')
    result = tmp.first()
    result['char'] = tmp.agg({'char': lambda c: ''.join(set(c))}).char

    # 输出聚类结果
    result.to_csv(outfile, sep='\t')


if __name__ == '__main__':
    main()