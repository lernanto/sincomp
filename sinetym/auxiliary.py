# -*- coding: utf-8 -*-

"""
辅助工具函数.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import pandas
import numpy
import scipy.sparse
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer


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
            stop_words=[''],
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