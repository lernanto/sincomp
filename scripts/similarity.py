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

import sinetym


if __name__ == '__main__':
    parser = argparse.ArgumentParser(globals().get('__doc__'))
    parser.add_argument(
        '-m',
        '--method',
        choices=('chi2', 'entropy'),
        default='entropy',
        help='计算方言间相似度的方法'
    )
    parser.add_argument('-o', '--output', help='输出文件名')
    parser.add_argument('input', default='.', help='输入方言字音文件所在目录')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    output = f'{args.method}.csv' if args.output is None else args.output
    logging.info(f'compute {args.method} similarity between dialects, ' \
        f'input = {args.input}, output = {output}')

    # 对每个字取第一个声母、韵母、声调均非空的读音
    # 声韵调部分有值部分为空会导致统计数据细微偏差
    data = sinetym.datasets.transform_data(
        sinetym.datasets.zhongguoyuyan.force_complete(
            sinetym.datasets.load_data(args.input, suffix='mb01dz.csv')
        ),
        index='cid',
        values=['initial', 'final', 'tone'],
        aggfunc='first'
    )

    ids = data.columns.levels[0]

    sim = getattr(sinetym.similarity, args.method)(data.values, parallel=4)
    pandas.DataFrame(sim, index=ids, columns=ids) \
        .to_csv(output, line_terminator='\n')