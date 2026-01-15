#!/usr/bin/env -S python3 -O
# -*- coding: utf-8 -*-

"""
根据方言点的语音规则符合度绘制关于符合度的同言线图.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import argparse
import cartopy.crs as ccrs
import geopandas as gpd
import logging
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import rasterio

import sincomp.auxiliary
import sincomp.compare
import sincomp.datasets
import sincomp.plot


logger = logging.getLogger(__name__)


def isogloss(
    data,
    lat,
    lon,
    val,
    name=None,
    ax=None,
    proj=ccrs.PlateCarree(),
    background=None,
    background_extent=None,
    geo=None,
    fill=True,
    cmap=None,
    color=None,
    extent=None,
    resolution=100,
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

    if extent is None:
        extent = sincomp.auxiliary.extent(data.loc[:, lon], data.loc[:, lat])

    ax.set_extent(extent)

    # 绘制背景图政区边界
    pc = ccrs.PlateCarree()
    if background is not None:
        ax.imshow(background, transform=pc, extent=background_extent)

    if geo is not None:
        ax.add_geometries(
            geo,
            ccrs.CRS(geo.crs) if hasattr(geo, 'crs') else pc,
            edgecolor='gray',
            facecolor='none'
        )

    if cmap is None and color is None:
        cmap = 'coolwarm'

    # 绘制同言线图
    if alpha is None:
        alpha = 0.7 if fill else 1

    # 投影变换后绘制范围可能会超出指定经纬度，绘制同言线时预留更大的范围
    sincomp.plot.geography.isogloss(
        data.loc[:, lat],
        data.loc[:, lon],
        values=data.loc[:, val],
        ax=ax,
        fill=fill,
        cmap=cmap,
        colors=color,
        vmin=0,
        vmax=1,
        extent=(
            1.5 * extent[0] - 0.5 * extent[1],
            1.5 * extent[1] - 0.5 * extent[0],
            1.5 * extent[2] - 0.5 * extent[3],
            1.5 * extent[3] - 0.5 * extent[2]
        ),
        resolution=resolution * 2,
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
    ax.gridlines(draw_labels=True)

    if title is not None:
        ax.set_title(title)

    return ax, extent

def float_array(s):
    return [float(i) for i in s.split(',')]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(globals().get('__doc__'))
    parser.add_argument(
        '--log-level',
        default='WARNING',
        help='输出日志级别'
    )
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
    parser.add_argument(
        '-r',
        '--rule-file',
        help='语音规则文件，为 JSON 格式，为规则的列表'
    )
    parser.add_argument(
        'input_file',
        nargs='?',
        type=argparse.FileType('r', encoding='utf-8'),
        default='-',
        help='''
        输入文件，为 CSV 格式，每行为一个方言，包含如下字段：
            - datasets 方言所在数据集
            - did 方言 ID
            - value1, value2, ... 余下每列用于绘制一组同言线
        '''
    )
    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.log_level.upper()))

    output_prefix = os.path.join(os.getcwd(), args.output_prefix)
    logging.info(
        f'create isogloss, rules = {args.rule_file}, '
        f'input file = {args.input_file.name}, '
        f'output prefix = {output_prefix}.'
    )

    if args.background is None:
        background = None
        background_extent = None
    else:
        background = rasterio.open(args.background)
        lon0, lat0, lon1, lat1 = background.bounds
        background = np.transpose(background.read([1, 2, 3]), (1, 2, 0)) \
            if background.count >= 3 \
            else background.read(1)
        background_extent = (lon0, lon1, lat0, lat1)

    geo = None if args.geography is None else gpd.read_file(args.geography)

    if args.rule_file is not None:
        rules = sincomp.compare.load_rules(args.rule_file)

    data = pd.read_csv(args.input_file, dtype={'dataset': str, 'did': str})
    columns = data.columns.drop(['dataset', 'did'])
    logging.info(f'loaded {data.shape[0]} dialects x {columns.shape[0]} rules.')

    for dataset, d in data.groupby('dataset', sort=False):
        data.loc[d.index, ['latitude', 'longitude']] = \
            sincomp.datasets.get(dataset).dialects[['latitude', 'longitude']] \
            .reindex(d['did']).values

    os.makedirs(os.path.dirname(output_prefix), exist_ok=True)

    if args.extent is None:
        extent = sincomp.auxiliary.extent(
            data.loc[:, 'longitude'],
            data.loc[:, 'latitude']
        )
    else:
        extent = args.extent

    proj = ccrs.LambertConformal(
        0.5 * (extent[0] + extent[1]),
        0.5 * (extent[2] + extent[3]),
    )

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
            proj=proj,
            background=background,
            background_extent=background_extent,
            geo=geo['geometry'],
            extent=extent
        )
        fig.savefig(path, format=args.format, bbox_inches='tight')
        plt.close()

    logging.info(f'done. totally {len(columns)} isoglosses created.')