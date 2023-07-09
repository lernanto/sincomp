#!/usr/bin/python3 -O
# -*- coding: utf-8 -*-

"""
根据方言数据构建词典.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import logging
import argparse
import os
import pandas as pd
import csv
import sinetym


if __name__ == '__main__':
    parser = argparse.ArgumentParser(globals().get('__doc__'))
    parser.add_argument('-s', '--suffix', default='mb01dz.csv', help='方言数据文件后缀')
    parser.add_argument('input_prefix', default='.', help='方言数据根目录')
    parser.add_argument('output_prefix', nargs='?', default='.', help='输出词典路径前缀')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)
    logging.info(f'make dictionaries from {args.input_prefix} to {args.output_prefix}')

    dics = {}
    for id in sorted(e.name[:-len(args.suffix)] \
        for e in os.scandir(args.input_prefix) \
        if e.is_file() and e.name.endswith(args.suffix)):
        data = sinetym.datasets.load_data(
            args.input_prefix,
            id,
            suffix=args.suffix
        )

        for name in ('lid', 'cid', 'initial', 'final', 'tone'):
            col = data.loc[data[name].notna() & (data[name] != ''), name]
            if name in dics:
                dics[name] = pd.concat([dics[name], col], ignore_index=True) \
                    .drop_duplicates()
            else:
                dics[name] = col.drop_duplicates()

    for name, dic in dics.items():
        fname = f'{os.path.join(args.output_prefix, name)}.csv'
        logging.info(f'save dictionary {fname}')
        dic.sort_values().to_csv(
            fname,
            header=False,
            index=False,
            encoding='utf-8',
            quoting=csv.QUOTE_NONE,
            lineterminator='\n'
        )