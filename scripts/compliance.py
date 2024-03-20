#!/usr/bin/env -S python3 -O
# -*- coding: utf-8 -*-

"""
根据字音数据及语音规则集计算方言对于规则的符合度.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import logging
import argparse

import sincomp.datasets
import sincomp.preprocess
import sincomp.compare


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
    parser.add_argument('-o', '--output', help='输出文件名')
    parser.add_argument('dataset', default='zhongguoyuyan', help='指定输入方言数据集')
    args = parser.parse_args()

    norm = None if args.norm == 'no' else args.norm
    output = f'{args.dataset}_compliance_{args.norm}.csv' \
        if args.output is None else args.output
    logging.info(
        f'compute rule compliance of {args.norm} norm for {args.dataset}, '
        f'output = {output}'
    )

    rule = sincomp.compare.load_rule(args.rule)
    logging.info(f'{rule.shape[0]} rules loaded.')

    data = getattr(sincomp.datasets, args.dataset).data
    data = sincomp.preprocess.transform(
        data[~data['note'].str.contains('文', na=False)],
        index='cid',
        values=['initial', 'final', 'tone'],
        aggfunc='first'
    )

    comp = sincomp.compare.compliance(data, rule, norm=norm)
    comp.to_csv(output, lineterminator='\n')
