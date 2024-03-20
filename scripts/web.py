#!/usr/bin/env -S python3 -O
# -*- coding: utf-8 -*-

"""
使用 D-Tale 创建服务展示方言数据.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import argparse
import pandas as pd
import dtale

import sincomp.datasets
import sincomp.preprocess


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('-H', '--host', default='localhost', help='监听的地址或主机名')
    parser.add_argument('-p', '--port', type=int, default=40000, help='监听的端口')
    parser.add_argument(
        'dataset',
        nargs='?',
        default='zhongguoyuyan',
        help='展示的方言数据集'
    )
    args = parser.parse_args()

    dataset = getattr(sincomp.datasets, args.dataset)
    dialect = dataset.metadata['dialect_info']
    char = dataset.metadata['char_info']
    data = dataset.data
    data = data[data['cid'].isin(char.index)]

    dialect.fillna(
        {'group': '', 'subgroup': '', 'cluster': '', 'subcluster': ''},
        inplace=True
    )
    dtale.views.startup(
        data=dialect[[
            'province',
            'city',
            'county',
            'group',
            'subgroup',
            'cluster',
            'subcluster',
            'spot',
            'latitude',
            'longitude'
        ]],
        name='dialect',
        locked=['spot']
    )

    # 拼接声韵调成为完整读音
    pronunciation = sincomp.preprocess.transform(
        pd.DataFrame({
            'did': data['did'],
            'cid': data['cid'],
            'pronunciation': data[[
                'initial',
                'final',
                'tone',
                'note'
            ]].fillna('').replace('∅', '').apply(''.join, axis=1)
        }),
        values='pronunciation',
        index='cid',
        aggfunc=' '.join
    )
    pronunciation.columns = pronunciation.columns.get_level_values(0) \
        + dialect.loc[pronunciation.columns.get_level_values(0), 'spot']
    pronunciation.set_index(char.index + char['character'], inplace=True)

    dtale.views.startup(
        data=pronunciation,
        name='pronunciation',
        inplace=True,
        show_columns=['index'] \
            + (dialect[:10].index + dialect[:10]['spot']).tolist()
    )

    # 启动 D-Tale 服务
    dtale.app.build_app().run(host=args.host, port=args.port)
