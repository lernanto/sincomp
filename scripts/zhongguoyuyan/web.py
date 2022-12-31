#!/usr/bin/python3 -O
# -*- coding: utf-8 -*-

"""
使用 D-Tale 创建服务展示方言数据.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import argparse
import os
import pandas as pd
import dtale

import sinetym


if __name__ == '__main__':
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('-H', '--host', default='localhost', help='监听的地址或主机名')
    parser.add_argument('-p', '--port', type=int, default=40000, help='监听的端口')
    parser.add_argument('path', help='方言数据根目录路径')
    args = parser.parse_args()

    dialect_path = os.path.join(args.path, 'dialect')
    location = sinetym.datasets.zhongguoyuyan.load_location(
        os.path.join(args.path, 'location.csv')
    )
    char = pd.read_csv(os.path.join(args.path, 'words.csv'), index_col=0)
    data = sinetym.datasets.zhongguoyuyan.load_data(
        dialect_path,
        suffix='mb01dz.csv'
    )
    data = data[data['cid'].isin(char.index)]

    location[['area', 'slice', 'slices']] \
        = location[['area', 'slice', 'slices']].fillna('')
    dtale.views.startup(
        data=location[[
            'province',
            'city',
            'county',
            'dialect',
            'area',
            'slice',
            'slices',
            'latitude',
            'longitude'
        ]],
        name='location',
        locked=['city', 'county']
    )

    # 字表及代表方言点的读音
    indeces = [
        '03E88',    # 北京
        '10G68',    # 南京
        '12177',    # 太原
        '10G71',    # 苏州
        '08210',    # 温州
        '21J03',    # 黄山
        '26516',    # 长沙
        '26509',    # 双峰
        '18358',    # 南昌
        '15233',    # 梅州
        '15231',    # 广州
        '01G23',    # 南宁
        '02G49',    # 厦门
        '02193'     # 建瓯
    ]
    character = sinetym.datasets.transform_data(
        data.loc[
            data['lid'].isin(indeces),
            ['lid', 'cid', 'initial', 'final', 'tone']
        ],
        index='cid',
        agg=lambda x: ' '.join(set(x))
    )
    character.set_index(
        character.index.astype(str) + char.reindex(character.index)['item'],
        inplace=True
    )
    character.columns = pd.MultiIndex.from_product((
        location.reindex(character.columns.levels[0])[['city', 'county']] \
            .apply(''.join, axis=1),
        character.columns.levels[1]
    ))

    dtale.views.startup(data=character, name='character', inplace=True)

    # 拼接声韵调成为完整读音
    pronunciation = sinetym.datasets.transform_data(
        pd.DataFrame({
            'lid': data['lid'],
            'cid': data['cid'],
            'pronunciation': data[[
                'initial',
                'final',
                'tone',
                'memo'
            ]].replace('∅', '').apply(''.join, axis=1)
        }),
        index='cid',
        agg=' '.join
    )
    pronunciation.columns = pronunciation.columns.get_level_values(0) \
        + location.loc[pronunciation.columns.get_level_values(0), 'city'] \
        + location.loc[pronunciation.columns.get_level_values(0), 'county']
    pronunciation.set_index(char.index.astype(str) + char['item'], inplace=True)

    dtale.views.startup(
        data=pronunciation,
        name='pronunciation',
        inplace=True,
        show_columns=['index'] + (indeces + location.loc[indeces, 'city'] \
            + location.loc[indeces,'county']).tolist()
    )

    # 启动 D-Tale 服务
    dtale.app.build_app().run(host=args.host, port=args.port)