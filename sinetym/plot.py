# -*- coding: utf-8 -*-

"""
绘图功能函数.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import logging
import numpy
import scipy.sparse
import scipy.cluster.hierarchy
from sklearn.preprocessing import StandardScaler
import colorspacious
import seaborn
import plotly.express
import plotly.figure_factory
import folium


def plot_map(
    base,
    points,
    boundary=None,
    pc=None,
    ax=None,
    figsize=None,
    **kwargs
):
    '''
    绘制指定地区的方言点地图

    如果指定了方言点主成分，通过方言点的颜色来表示方言之间的相似度，
    方言点的颜色由相似度矩阵的主成分确定，可以近似表示相似度矩阵。
    如果指定了方言点分类标签，则根据标签确定方言点颜色。
    '''

    if boundary is None:
        boundary = base.unary_union

    # 根据指定的地理边界筛选边界内部的点
    mask = points.within(boundary).values

    # 根据筛选得到的点集确定绘制边界
    minx, miny, maxx, maxy = points[mask].total_bounds
    xmargin = 0.05 * (maxx - minx)
    ymargin = 0.05 * (maxy - miny)
    minx -= xmargin
    maxx += xmargin
    miny -= ymargin
    maxy += ymargin

    # 绘制地图底图
    ax = base.boundary.plot(color='gray', ax=ax, figsize=figsize)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    if pc is not None:
        # 根据相似度矩阵的主成分计算方言点颜色，降维后的相似度矩阵决定方言点的 Lab 色度
        # 缩放传入的颜色值使其接近高斯分布，标准差为0.4的高斯分布98%以上落在 [-1, 1] 区间
        pc = StandardScaler().fit_transform(pc[:, :2]) * 0.4

        # 对绘制边界内的方言点稍微改变一下颜色分布，增加对比度，但保证变换后的均值不变
        inner_pc = pc[mask]
        inner_mean = numpy.mean(inner_pc, axis=0)
        scale = (pc - inner_mean) * 2 + inner_mean
        weight = numpy.exp(-numpy.sum(numpy.square(pc - inner_mean), axis=1) * 3)
        mix = weight[:, None] * scale + (1 - weight[:, None]) * pc

        # 把变换后的主成分拉伸到 Lab 色度的表示范围 [-100, 100]，亮度固定50
        lab = numpy.empty((points.shape[0], 3), dtype=numpy.float32)
        lab[:, 0] = 50
        lab[:, 1:] = mix * 50
        if numpy.any(numpy.abs(lab[mask, 1:]) > 50):
            logging.warning(
                '{}/{} points are out of [-50, 50], whose color may not show properly'.format(
                    numpy.count_nonzero(numpy.any(
                        numpy.abs(lab[mask, 1:]) > 50,
                        axis=1
                    )),
                    inner_pc.shape[0]
                )
            )

        # Lab 色系转 RGB 色系用于显示
        rgb = numpy.clip(colorspacious.cspace_convert(lab, 'CIELab', 'sRGB1'), 0, 1)
        points.plot(ax=ax, color=rgb, **kwargs)

    else:
        # 根据标签确定方言点颜色
        points.plot(ax=ax, **kwargs)

    return ax

def plot_heatmap(dist, labels, **kwargs):
    '''绘制方言相似度矩阵热度图'''

    # 根据距离矩阵层级聚类
    linkage = scipy.cluster.hierarchy.linkage(
        dist[numpy.triu_indices_from(dist, 1)],
        method='average',
        optimal_ordering=True
    )

    # 根据聚类结果重新排列距离矩阵，距离越短的点顺序越靠近
    leaves = scipy.cluster.hierarchy.leaves_list(linkage)

    return seaborn.heatmap(
        dist[leaves][:, leaves],
        square=True,
        xticklabels=labels[leaves],
        yticklabels=labels[leaves],
        **kwargs
    )

def dendro_heat(
    dist,
    labels,
    linkagefun=scipy.cluster.hierarchy.average,
    width=1000,
    height=1000
):
    '''
    根据距离矩阵绘制带热度图的树状图
    '''

    # 绘制树状图
    # 由于 Plotly 不接受预计算的距离矩阵，需要使用自定义距离函数，这个函数的返回值是距离矩阵的上三角阵
    dendro = plotly.figure_factory.create_dendrogram(
        dist,
        distfun=lambda x: x[numpy.triu_indices_from(x, 1)],
        linkagefun=linkagefun
    )
    for d in dendro.data:
        d.yaxis = 'y2'

    # 绘制热度图
    leaves = dendro.layout.xaxis.ticktext.astype(numpy.int32)
    heat = plotly.express.imshow(
        dist[leaves][:, leaves],
        x=labels[leaves],
        y=labels[leaves],
    )
    heat.data[0].x = dendro.layout.xaxis.tickvals
    heat.data[0].y = dendro.layout.xaxis.tickvals

    # 合并树状图和热度图，并调整大小占比
    fig = dendro
    fig.add_trace(heat.data[0])
    fig.update_layout(width=width, height=height)
    fig.update_layout(xaxis={'ticktext': labels[leaves]})
    fig.update_layout(yaxis={'domain': [0, 0.8], 'ticktext': labels[leaves]})
    fig.update_layout(yaxis2={'domain': [0.8, 1]})

    return fig

def label_map(
    latitudes,
    longitudes,
    labels,
    tips,
    zoom=5,
    tiles='https://wprd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scl=2&style=7&x={x}&y={y}&z={z}',
    attr='AutoNavi'
):
    '''绘制带分类坐标点的地图'''

    # 绘制底图
    mp = folium.Map(
        location=(numpy.mean(latitudes), numpy.mean(longitudes)),
        zoom_start=zoom,
        tiles=tiles,
        attr=attr
    )

    # 添加坐标点
    for i in range(latitudes.shape[0]):
        # 根据标签序号分配一个颜色
        r = (labels[i] * 79 + 31) & 0xff
        g = (labels[i] * 37 + 43) & 0xff
        b = (labels[i] * 73 + 17) & 0xff

        folium.CircleMarker(
            (latitudes[i], longitudes[i]),
            radius=5,
            tooltip='{} {}'.format(labels[i], tips[i]),
            color='#{:02x}{:02x}{:02x}'.format(r, g, b),
            fill=True
        ).add_to(mp)

    return mp