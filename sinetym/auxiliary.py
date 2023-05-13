# -*- coding: utf-8 -*-

"""
辅助工具函数.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import logging
import pandas
import numpy
import scipy.sparse
from sklearn.preprocessing import normalize, OrdinalEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import CountVectorizer
import colorspacious
import matplotlib
import shapely
import cartopy


def encode(data, dtype=numpy.int32, missing_values='', unknown_value=-1):
    """
    把方言读音编码为整数.

    Parameters:
        data (`pandas.DataFrame`): M x N 字符串矩阵，每行为一个字，
            每列为一个方言点的声母/韵母/声调，空串代表空值

    Returns:
        codes (`numpy.ndarray`): M x N 整数矩阵，空值为 -1
        categories (list of `numpy.ndarray`): 长度为 N 的列表，每个元素是每一列的类别
    """

    encoder = OrdinalEncoder(
        dtype=dtype,
        handle_unknown='use_encoded_value',
        unknown_value=unknown_value
    )
    encoder.fit(
        # 为了让编码器正常工作，先补全缺失特征
        SimpleImputer(missing_values=missing_values, strategy='most_frequent') \
            .fit_transform(data)
    )

    return encoder.transform(data), encoder.categories_

def vectorize(data, sep=' ', binary=False, dtype=numpy.int32, norm=None):
    """
    对一个方言读音的数组或包含多个方言读音的矩阵进行稀疏编码.

    原始数据以字为行，以方言点的声韵调为列，允许一格包含多个音，以指定分隔符分隔。

    Parameters:
        data (array): 长度为 M 的数组 或 M x N 矩阵，当为矩阵时，每列为一个方言点的声母/韵母/声调
        sep (str): 分隔多音字的多个音的分隔符
        binary (bool): 为真时，返回的编码为 0/1 编码，否则返回读音的计数
        norm (str): 是否对返回编码归一化：
            - None: 不归一化
            - 'l1': 返回的编码除以向量的1范数
            - 'l2': 返回的编码除以向量的2范数

    Returns:
        code (`scipy.sparse.csr_matrix`): 稀疏编码得到的稀疏矩阵，行数为 M，列数为所有列读音数之和
        limits (`numpy.ndarray`): 仅当 data 为矩阵时返回，表示编码边界的数组，
            长度为 N + 1，data[:, i] 的编码为 code[:, limits[i]:limits[i + 1]]
    """

    if isinstance(data, pandas.DataFrame) or isinstance(data, pandas.Series):
        data = data.values

    if data.ndim == 1:
        # 一维数组，直接编码返回
        code = CountVectorizer(
            lowercase=False,
            tokenizer=lambda s: s.split(sep),
            token_pattern=None,
            stop_words=[''],
            binary=binary,
            dtype=dtype
        ).fit_transform(data)

        return code if norm is None else normalize(code, norm=norm)

    # 矩阵，分别编码每列然后拼接
    categories = []
    codes = []
    columns = []

    for i in range(data.shape[1]):
        c = CountVectorizer(
            lowercase=False,
            tokenizer=lambda s: s.split(sep),
            token_pattern=None,
            stop_words=[''],
            binary=binary,
            dtype=dtype
        ).fit_transform(data[:, i])

        codes.append(c if norm is None else normalize(c, norm=norm))
        columns.append(c.shape[1])

    code = scipy.sparse.hstack(codes)
    # 计算稀疏编码的边界
    limits = numpy.empty(len(columns) + 1, dtype=int)
    limits[0] = 0
    numpy.cumsum(columns, out=limits[1:])
    return code, limits

def pc2color(pc):
    """
    根据矩阵主成分分解的结果生成颜色，使样本点的颜色能反映主成分的差异.

    取主成分的前两维变换至 Lab 色度的表示范围 [-50, 50]，亮度固定50，
    然后从 Lab 色系转为 RGB 色系。

    Parameters:
        pc (`numpy.ndarray`): 矩阵分解的主成分，列数至少为2

    Returns:
        rgb (`numpy.ndarray`): RGB 颜色值，行数和 pc 相同
    """

    # 缩放传入的主成分使其接近高斯分布，标准差为0.4的高斯分布98%以上落在 [-1, 1] 区间
    pc = StandardScaler().fit_transform(pc[:, :2]) * 0.4

    # 把变换后的主成分拉伸到 Lab 色度的表示范围 [-50, 50]，亮度固定50
    lab = numpy.empty((pc.shape[0], 3), dtype=numpy.float32)
    lab[:, 0] = 50
    lab[:, 1:] = pc * 50
    if numpy.any(numpy.abs(lab[:, 1:]) > 50):
        logging.warning(
            f'{numpy.count_nonzero(numpy.any( numpy.abs(lab[:, 1:]) > 50, axis=1))}/'
            f'{pc.shape[0]} points are out of [-50, 50], whose color may not show properly'
        )

    # Lab 色系转 RGB 色系用于显示
    return numpy.clip(colorspacious.cspace_convert(lab, 'CIELab', 'sRGB1'), 0, 1)

def make_clip_path(polygons, extent=None):
    """
    根据绘制范围及指定的多边形生成图形的裁剪路径.

    Parameters:
        polygons (`shapely.geometry.multipolygon.MultiPolygon`):
            裁剪的范围，只保留该范围内的图形
        extent: 绘制的范围 (左, 右, 下, 上)

    Returns:
        path (`matplotlib.path.Path`): 生成的裁剪路径，如果传入的多边形为空返回 None
    """

    if polygons is None:
        return None

    polygons = tuple(polygons) if hasattr(polygons, '__iter__') else (polygons,)
    if len(polygons) == 0:
        return None

    if extent is not None:
        # 先对绘图区域和裁剪区域取交集
        xmin, xmax, ymin, ymax = extent
        poly = shapely.geometry.Polygon((
            (xmin, ymin),
            (xmin, ymax),
            (xmax, ymax),
            (xmax, ymin),
            (xmin, ymin)
        ))
        polygons = [poly.intersection(c) for c in polygons]

    return matplotlib.path.Path.make_compound_path(
        *cartopy.mpl.patch.geos_to_path(polygons)
    )

def clip_paths(paths, polygons, extent=None):
    """
    根据绘制范围及指定的多边形裁剪 matplotlib 绘制的图形.

    Parameters:
        paths (`matplotlib.PathCollection` or list of `matplotlib.PathCollection`):
            待裁剪的图形
        polygons (`shapely.geometry.multipolygon.MultiPolygon`):
            裁剪的范围，只保留该范围内的图形
        extent: 绘制的范围 (左, 右, 下, 上)
    """

    path = make_clip_path(polygons, extent=extent)
    if path is not None:
        # 裁剪图形
        if hasattr(paths, '__iter__'):
            for c in paths:
                c.set_clip_path(path, transform=c.axes.transData)
        else:
            paths.set_clip_path(path, transform=paths.axes.transData)
