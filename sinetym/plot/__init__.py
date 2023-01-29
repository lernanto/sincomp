# -*- coding: utf-8 -*-

"""
绘图功能函数.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import numpy
import scipy.sparse
import scipy.cluster.hierarchy
import seaborn
import plotly.express
import plotly.figure_factory

from . import geography


def heatmap(dist, labels, **kwargs):
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