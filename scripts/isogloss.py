#!/usr/bin/python3 -O
# -*- coding: utf-8 -*-

"""
根据方言点的语音规则符合度绘制关于符合度的同言线图.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import logging
import argparse
import os
import pandas as pd
import numpy as np
import sinetym
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter


def isogloss(
    data,
    lat,
    lon,
    val,
    ax=None,
    proj=ccrs.PlateCarree(),
    background=None,
    geo=None,
    extent=None
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

    # 绘制同言线图
    _, extent, _ = sinetym.plot.geography.isogloss(
        data.loc[:, lat],
        data.loc[:, lon],
        values=data.loc[:, val],
        ax=ax,
        vmin=0,
        vmax=1,
        extent=extent,
        clip=geo,
        alpha=0.7,
        levels=np.arange(0, 1.1, 0.1)
    )

    # 绘制样本点散点图
    sinetym.plot.geography.scatter(
        data.loc[:, lat],
        data.loc[:, lon],
        values=data.loc[:, val],
        ax=ax,
        extent=extent,
        clip=geo,
        vmin=0,
        vmax=1,
        marker='.',
        cmap='coolwarm'
    )

    # 计算坐标经纬度
    left, right, bottom, top = extent
    step = max(right - left, top - bottom) / 5
    step = 1 if step <= 1 else 5 if step <= 5 else 10
    ax.set_xticks(np.arange(left // step, right // step + 1) * step, crs=proj)
    ax.set_yticks(np.arange(bottom // step, top // step + 1) * step, crs=proj)
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.set_extent(extent, crs=proj)

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
    parser.add_argument('prefix', help='样本点信息等文件的路径前缀')
    parser.add_argument('data', help='规则符合度数据文件')
    parser.add_argument('rule', nargs='?', help='语音规则文件')
    args = parser.parse_args()

    logging.info(
        f'create isogloss, input prefix = {args.prefix}, '
        f'data = {args.data}, rule = {args.rule}, '
        f'output prefix = {args.output_prefix}.'
    )

    bg = None if args.background is None else plt.imread(args.background)
    geo = None if args.geography is None \
        else tuple(Reader(args.geography).geometries())
    location = sinetym.datasets.zhongguoyuyan.load_location(
        os.path.join(args.prefix, 'location.csv'),
        predict=False
    )

    logging.info(f'loading data from {args.data} ...')
    data = pd.read_csv(args.data, index_col=0)
    logging.info(f'done. loaded {data.shape[0]} dialects x {data.shape[1]} rules.')

    if args.rule is not None:
        char = pd.read_csv(os.path.join(args.prefix, 'words.csv'), index_col='cid')
        rule = sinetym.compare.load_rule(args.rule, characters=char['item'])
        data.columns = rule.loc[data.columns.astype(int), 'name']

    columns = data.columns
    data[['latitude', 'longitude']] = location[['latitude', 'longitude']]

    for c in columns:
        fname = f'{args.output_prefix}{c}.{args.format}'
        logging.info(f'creating f{fname} ...')

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
        fig.savefig(fname, format=args.format, bbox_inches='tight')
        plt.close()

    logging.info(f'done. totally {len(columns)} isoglosses created.')
