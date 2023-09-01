#!/usr/bin/python3 -O
# -*- coding: utf-8 -*-

"""
根据方言数据构建词典.

读取环境变量 XIAOXUETANG_HOME 和 ZHONGGUOYUYAN_HOME 获取数据集的路径，
然后读取数据集全部数据生成词典。
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import logging
import argparse
import os
import pandas as pd
import csv

from sinetym.datasets import xiaoxuetang, zhongguoyuyan


if __name__ == '__main__':
    parser = argparse.ArgumentParser(globals().get('__doc__'))
    parser.add_argument(
        '-m',
        '--min-freq',
        type=int,
        default=2,
        help='出现频次不小于该值才计入词典'
    )
    parser.add_argument('output_prefix', nargs='?', default='.', help='输出词典路径前缀')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.info(f'make dictionaries to {args.output_prefix}')

    data = pd.concat([
        xiaoxuetang,
        zhongguoyuyan.assign(did=zhongguoyuyan['did'] + zhongguoyuyan['variant'])
    ], axis=0, ignore_index=True)

    for name in ('did', 'cid', 'character', 'initial', 'final', 'tone'):
        dic = data.loc[data[name] != '', name].value_counts().rename('count')
        dic.index.rename(name, inplace=True)

        if args.min_freq > 1:
            dic = dic[dic >= args.min_freq]

        fname = f'{os.path.join(args.output_prefix, name)}.csv'
        logging.info(f'save dictionary {fname}')
        dic.sort_values(ascending=False).to_csv(
            fname,
            encoding='utf-8',
            quoting=csv.QUOTE_NONE,
            lineterminator='\n'
        )