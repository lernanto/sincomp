#!/usr/bin/python3 -O
# -*- coding: utf-8 -*-

"""
计算方言之间的预测相似度.

如果 A 方言的音类能很大程度预测 B 方言的音类，表明 A 方言到 B 方言单向的相似度高
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import argparse
import logging
import pandas

import sinetym.datasets
import sinetym.similarity


if __name__ == '__main__':
    parser = argparse.ArgumentParser(globals().get('__doc__'))
    parser.add_argument(
        '-m',
        '--method',
        choices=('chi2', 'entropy'),
        default='entropy',
        help='计算方言间相似度的方法'
    )
    parser.add_argument('dataset', nargs='?', default='zhongguoyuyan', help='输入数据集')
    parser.add_argument('-o', '--output', help='输出文件名')
    args = parser.parse_args()

    output = f'{args.dataset}_{args.method}.csv' if args.output is None \
        else args.output

    logging.getLogger().setLevel(logging.INFO)
    logging.info(
        f'compute {args.method} similarity between dialects, '
        f'dataset = {args.dataset}, output = {output}'
    )

    data = getattr(sinetym.datasets, args.dataset)
    if data is sinetym.datasets.zhongguoyuyan:
        data = data.filter(variant='mb01')

    # 对每个字取第一个声母、韵母、声调均非空的读音
    # 声韵调部分有值部分为空会导致统计数据细微偏差
    data = data.force_complete().transform(
        index='cid',
        values=['initial', 'final', 'tone'],
        aggfunc='first'
    )
    ids = data.columns.levels[0]

    sim = getattr(sinetym.similarity, args.method)(data.values, parallel=4)
    pandas.DataFrame(sim, index=ids, columns=ids) \
        .to_csv(output, lineterminator='\n')