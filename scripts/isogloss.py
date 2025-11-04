#!/usr/bin/env -S python3 -O
# -*- coding: utf-8 -*-

"""
根据方言点的语音规则符合度绘制关于符合度的同言线图.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import argparse
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

import sincomp.datasets
import sincomp.plot


def isogloss(
    data,
    lat,
    lon,
    val,
    name=None,
    ax=None,
    proj=ccrs.PlateCarree(),
    background=None,
    geo=None,
    fill=True,
    cmap=None,
    color=None,
    extent=None,
    levels=np.linspace(0, 1, 11),
    alpha=None,
    title=None,
    **kwargs
):
    """
    绘制带背景的同言线图.
    """

    if ax is None:
        ax = plt.axes(projection=proj)

    # 绘制背景图政区边界
    if background is not None:
        ax.imshow(
            background,
            transform=proj,
            extent=[-180, 180, -90, 90]
        )

    if geo is not None:
        geo = tuple(geo)
        ax.add_geometries(geo, proj, edgecolor='gray', facecolor='none')

    if cmap is None and color is None:
        cmap = 'coolwarm'

    # 绘制同言线图
    if alpha is None:
        alpha = 0.7 if fill else 1

    _, extent, _ = sincomp.plot.geography.isogloss(
        data.loc[:, lat],
        data.loc[:, lon],
        values=data.loc[:, val],
        ax=ax,
        fill=fill,
        cmap=cmap,
        colors=color,
        vmin=0,
        vmax=1,
        extent=extent,
        clip=geo,
        levels=levels,
        alpha=alpha,
        **kwargs
    )

    # 绘制样本点散点图
    sincomp.plot.geography.scatter(
        data.loc[:, lat],
        data.loc[:, lon],
        values=None if cmap is None else data.loc[:, val],
        ax=ax,
        extent=extent,
        clip=geo,
        vmin=0,
        vmax=1,
        marker='.',
        cmap=cmap,
        color=color
    )

    # 标注地名
    left, right, bottom, top = extent
    if name is not None:
        for _, r in data[(data[lon] > left) & (data[lon] < right) \
            & (data[lat] > bottom) & (data[lat] < top)].iterrows():
            ax.annotate(r[name], xy=(r[lon], r[lat]))

    # 添加经纬度
    gl = ax.gridlines(crs=proj, draw_labels=True)
    gl.xlines = False
    gl.ylines = False

    ax.set_extent(extent, crs=proj)

    if title is not None:
        ax.set_title(title)

    return ax, extent

def float_array(s):
    return [float(i) for i in s.split(',')]

if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    parser = argparse.ArgumentParser(globals().get('__doc__'))
    parser.add_argument(
        '-s',
        '--size',
        type=float_array,
        default=(16, 9),
        help='输出图片大小，为半角逗号分隔的2个实数，单位英寸'
    )
    parser.add_argument('-b', '--background', help='指定背景图文件')
    parser.add_argument('-g', '--geography', help='政区图文件')
    parser.add_argument(
        '-e',
        '--extent',
        type=float_array,
        help='绘制范围的经纬度，为半角逗号分隔的4个实数'
    )
    parser.add_argument('-o', '--output-prefix', default='', help='输出路径前缀')
    parser.add_argument('-f', '--format', default='png', help='保存的图片格式')
    parser.add_argument('-r', '--rule-file', help='语音规则文件')
    parser.add_argument('data', nargs='+', help='前面为数据集列表，后面为对应的规则符合度文件列表')
    args = parser.parse_args()

    output_prefix = os.path.join(os.getcwd(), args.output_prefix)
    logging.info(
        f'create isogloss, rules = {args.rule_file}, '
        f'output prefix = {output_prefix}.'
    )

    bg = None if args.background is None else plt.imread(args.background)
    geo = None if args.geography is None \
        else tuple(Reader(args.geography).geometries())

    if args.rule_file is not None:
        rules = pd.read_json(args.rule_file, orient='records', encoding='utf-8')
        if 'id' in rules.columns:
            rules.set_index('id', inplace=True)
        rules.set_index(rules.index.astype(str), inplace=True)

    n = len(args.data) // 2
    datasets = [sincomp.datasets.get(d) for d in args.data[:n]]
    names = [d.name for d in datasets]
    dialects = pd.concat([d.dialects for d in datasets], axis=0, keys=names)
    data = pd.concat(
        [pd.read_csv(f, dtype={'did': str}).set_index('did') for f in args.data[n:]],
        axis=0,
        keys=names
    )
    logging.info(f'loaded {data.shape[0]} dialects x {data.shape[1]} rules.')

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    columns = data.columns
    data[['latitude', 'longitude']] = dialects[['latitude', 'longitude']]

    for c in columns:
        if args.rule_file is None:
            path = f'{output_prefix}{c}.{args.format}'
        else:
            path = f'{output_prefix}{c}_{rules.at[c, "name"]}.{args.format}'
        logging.info(f'creating {path} ...')

        fig = plt.figure(figsize=args.size)
        isogloss(
            data,
            'latitude',
            'longitude',
            c,
            background=bg,
            geo=geo,
            extent=args.extent
        )
        fig.savefig(path, format=args.format, bbox_inches='tight')
        plt.close()

    logging.info(f'done. totally {len(columns)} isoglosses created.')