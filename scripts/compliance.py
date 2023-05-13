#!/usr/bin/python3 -O
# -*- coding: utf-8 -*-

"""
根据字音数据及语音规则集计算方言对于规则的符合度.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import logging
import argparse
import os
import sinetym


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(globals().get('__doc__'))
    parser.add_argument('-r', '--rule', help='语音规则文件')
    parser.add_argument(
        '-n',
        '--norm',
        choices=('no', 'l1', 'l2'),
        default='l2',
        help='计算规则符合度的正则化方法'
    )
    parser.add_argument(
        '-s',
        '--suffix',
        default='mb01dz.csv',
        help='只使用匹配该后缀的文件作为输入'
    )
    parser.add_argument('-o', '--output', help='输出文件名')
    parser.add_argument('input', default='.', help='输入的方言字音文件所在目录')
    args = parser.parse_args()

    norm = None if args.norm == 'no' else args.norm
    output = f'compliance_{args.norm}.csv' if args.output is None else args.output
    logging.info(f'compute rule compliance of {args.norm} norm, ' \
        f'input = {os.path.join(args.input, "*")}{args.suffix}, output = {output}')

    rule = sinetym.compare.load_rule(args.rule)
    logging.info(f'{rule.shape[0]} rules loaded.')

    data = sinetym.datasets.load_data(args.input, suffix=args.suffix)
    data = sinetym.datasets.transform_data(
        data.loc[
            ~data['memo'].str.contains('文'),
            ['lid', 'cid', 'initial', 'final', 'tone']
        ],
        index='cid'
    )

    comp = sinetym.compare.compliance(data, rule, norm=norm)
    comp.to_csv(output, lineterminator='\n')
