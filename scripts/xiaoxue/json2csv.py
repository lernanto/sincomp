#!/usr/bin/python3 -O
# -*- coding: utf-8 -*-

"""
方言读音数据 JSON lines 格式转 CSV 格式.

搜索输入目录中所有 JSON lines 文件，把能识别出字音数据的转换为 CSV 格式，
保存文件以方言名称命名。同时把所有方言信息和字信息保存为 CSV 文件。
"""

__author__ = '黄艺华 <lernanto.wong@gmail.com>'


import os
import logging
import argparse
import jsonlines
import pandas

from sinetym.datasets import xiaoxue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(globals().get('__doc__'))
    parser.add_argument(
        '-d',
        '--debug',
        default=False,
        action='store_true',
        help='输出调试信息'
    )
    parser.add_argument('-o', '--output', default='.', help='输出文件所在目录')
    parser.add_argument('input', default='.', help='输入文件所在目录')
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.DEBUG if args.debug else logging.INFO)

    logging.info(f'convert data from JSON lines to CSV: {args.input} -> {args.output}')
    dialect_dir = os.path.join(args.output, 'dialect')
    os.makedirs(dialect_dir, exist_ok=True)

    dialect_info = []
    char_info = []

    # 扫描输入目录中所有 .jsonl 文件，尝试转换为 CSV
    for entry in os.scandir(args.input):
        if entry.is_file():
            root, ext = os.path.splitext(entry.name)
            if ext == '.jsonl':
                logging.debug(f'parsing {entry.path} ...')
                ret = xiaoxue.load_json(entry.path)

                if ret is not None:
                    dialect, data = ret
                    fname = os.path.join(dialect_dir, f'{dialect}.csv')
                    logging.info(f'{entry.path} -> {fname}')
                    data[[
                        'id',
                        '字形',
                        '方言點',
                        '聲母',
                        '韻母',
                        '調值',
                        '調類',
                        '備註'
                    ]].to_csv(
                        fname,
                        index=False,
                        encoding='utf-8',
                        lineterminator='\n'
                    )

                    # 提取方言信息
                    dialect_info.append(data.assign(**{'方言': dialect}) \
                        .reindex(columns=['方言', '區', '小區', '片', '小片', '方言點']) \
                        .drop_duplicates())

                    # 提取字的基本信息，包括中古音
                    try:
                        with jsonlines.open(entry.path) as f:
                            char_info.append(pandas.json_normalize(f) \
                                .drop(dialect, axis='columns') \
                                .drop_duplicates('id') \
                                .rename(columns=lambda x: x.replace('中古音.', '')))
                    except Exception as e:
                        logging.warning(e, exc_info=True)

    # 保存方言信息
    if len(dialect_info) > 0:
        dialect_info = pandas.concat(dialect_info, axis=0).drop_duplicates()
        dialect_info = dialect_info.sort_values(dialect_info.columns.tolist())

        fname = os.path.join(args.output, 'dialect.csv')
        logging.info(f'save dialect information to {fname}')
        dialect_info.to_csv(
            fname,
            index=False,
            encoding='utf-8',
            lineterminator='\n'
        )

    else:
        logging.warning(f'no dialect information found in {args.input}!')

    # 保存字信息
    if len(char_info) > 0:
        char_info = pandas.concat(char_info, axis=0) \
            .drop_duplicates('id') \
            .set_index('id') \
            .sort_index()

        fname = os.path.join(args.output, 'char.csv')
        logging.info(f'save character information to {fname}')
        char_info.to_csv(fname, encoding='utf-8', lineterminator='\n')

    else:
        logging.warning(f'no character information found in {args.input}!')