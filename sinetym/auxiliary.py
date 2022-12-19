# -*- coding: utf-8 -*-

"""
辅助工具函数.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import pandas
import numpy
import scipy.sparse
from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize(data, sep=' ', binary=False, dtype=int, norm=None):
    """
    对方言读音进行稀疏编码.

    原始数据以字为行，以方言点的声韵调为列，允许一格包含多个音，以指定分隔符分隔。

    Parameters:
        data (array): M x N 矩阵，元素为字符串，每行为一个字，每列为一个方言点的声母/韵母/声调
        sep (str): 分隔多音字的多个音的分隔符
        binary (bool): 为真时，返回的编码为 0/1 编码，否则返回读音的计数
        norm (str): 是否对返回编码归一化：
            - None: 不归一化
            - 'l1': 返回的编码除以向量的1范数
            - 'l2': 返回的编码除以向量的2范数

    Returns:
        code (`scipy.sparse.csr_matrix`): 稀疏编码得到的稀疏矩阵，行数为 M，列数为所有列读音数之和
        limits (`numpy.ndarray`): 表示编码边界的数组，长度为 N + 1，data[:, i]
            的编码为 code[:, limits[i]:limits[i + 1]]
    """

    if isinstance(data, pandas.DataFrame):
        data = data.values

    categories = []
    codes = []

    for i in range(data.shape[1]):
        vectorizer = TfidfVectorizer(
            lowercase=False,
            tokenizer=lambda s: s.split(sep),
            stop_words=('',),
            binary=binary,
            dtype=dtype,
            norm=norm,
            use_idf=False
        )
        codes.append(vectorizer.fit_transform(data[:, i]))
        categories.append(len(vectorizer.vocabulary_))

    code = scipy.sparse.hstack(codes)
    # 计算稀疏编码的边界
    limits = numpy.empty(len(categories) + 1, dtype=int)
    limits[0] = 0
    numpy.cumsum(categories, out=limits[1:])

    return code, limits