# -*- encoding: utf-8 -*-

"""
计算方言或字的低维稠密向量表示
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import numpy
import scipy
import sklearn.compose
import sklearn.metrics.pairwise
import sklearn.pipeline
import sklearn.feature_extraction.text
import sklearn.feature_selection
import sklearn.decomposition
from typing import Optional, Sequence, Type, Union
import pandas


class PhoneSimilarity(sklearn.base.BaseEstimator):
    """
    计算方言读音相似度矩阵

    Parameters:
        dtype: 数据类型，默认为 `numpy.float64`。
    """

    def __init__(self, dtype: Type = numpy.float64):
        self.dtype = dtype

    def fit(
        self,
        X: Union[pandas.DataFrame, numpy.ndarray],
        y: Optional[numpy.ndarray] = None
    ) -> 'PhoneSimilarity':
        """
        训练模型，返回当前对象

        Parameters:
            X: 输入数据，可以是 DataFrame 或 NumPy 数组。
            y: 目标变量，可选。

        Returns:
            PhoneSimilarity: 当前对象。
        """

        self.n_features_in_ = X.shape[1]
        if isinstance(X, pandas.DataFrame):
            self.feature_names_in_ = X.columns.values

        return self

    def transform(
        self,
        X: Union[pandas.DataFrame, numpy.ndarray]
    ) -> scipy.sparse.csr_matrix:
        """
        将输入数据转换为相似度矩阵，返回稀疏矩阵

        Parameters:
            X: 输入数据，可以是 DataFrame 或 NumPy 数组。

        Returns:
            scipy.sparse.csr_matrix: 相似度矩阵。
        """

        X = numpy.asarray(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'X has {X.shape[1]} features, but PhoneSimilarity is expecting {self.n_features_in_} features as input.')

        trius = []
        for i in range(X.shape[0]):
            try:
                features = sklearn.feature_extraction.text.CountVectorizer(
                    lowercase=False,
                    tokenizer=str.split,
                    token_pattern=None,
                    min_df=2,
                    binary=True,
                    dtype=self.dtype
                ).fit_transform(X[i])
                sim = sklearn.metrics.pairwise.cosine_similarity(
                    features,
                    dense_output=False
                )

            except ValueError:
                sim = scipy.sparse.csr_matrix(
                    (X.shape[1], X.shape[1]),
                    dtype=self.dtype
                )

            trius.append(scipy.sparse.triu(sim, 1).reshape((1, -1)))

        return scipy.sparse.vstack(trius, format='csr')

    def fit_transform(
        self,
        X: Union[pandas.DataFrame, numpy.ndarray],
        y: Optional[numpy.ndarray] = None
    ) -> scipy.sparse.csr_matrix:
        """
        同时训练和转换数据，返回稀疏矩阵

        Parameters:
            X: 输入数据，可以是 DataFrame 或 NumPy 数组。
            y: 目标变量，可选。

        Returns:
            scipy.sparse.csr_matrix: 相似度矩阵。
        """

        return self.fit(X, y).transform(X)

    def get_feature_names_out(
        self,
        input_features: Optional[Sequence[str]] = None
    ) -> numpy.ndarray:
        """
        获取特征名称，返回一个数组

        Parameters:
            input_features: 输入特征名称，可选。

        Returns:
            np.ndarray: 特征名称数组。
        """

        if input_features is None:
            input_features = getattr(
                self,
                'feature_names_in_',
                [f'x{i}' for i in range(self.n_features_in_)]
            )

        return numpy.asarray(
            list(f'{i}_{j}' for i in input_features for j in input_features)
        )


class DialectEmbedding(sklearn.pipeline.Pipeline):
    """
    根据方言中字音的两两相似度，计算方言向量

    Parameters:
        features: 用于提取特征的模式列表，每个特征对应输入数据的若干列，其列名必须包含该特征的模式字符串
        embedding_size: 生成的向量维度
        min_variance: 生成读音相似度矩阵后，过滤掉方差小于 min_variance 的列

    1. 根据方言字音计算字音相似度矩阵
    2. 取相似度矩阵的上三角矩阵，打平得到稀疏的方言向量，所有方言的向量组成方言稀疏矩阵
    3. 根据方差过滤矩阵中的大部分列
    4. 使用 IDF 更新矩阵中的值
    5. 使用 SVD 把稀疏矩阵转换成稠密向量
    """

    def __init__(
        self,
        features: list[str] = ['initial', 'final', 'tone'],
        embedding_size: int = 128,
        min_variance: float = 0.05 * 0.95
    ):
        super().__init__([
            ('vectorizer', sklearn.compose.ColumnTransformer(
                [(f, PhoneSimilarity(), sklearn.compose.make_column_selector(f)) for f in features]
            )),
            ('selector', sklearn.feature_selection.VarianceThreshold(min_variance)),
            ('embedding', sklearn.decomposition.TruncatedSVD(embedding_size))
        ])

        self.features = features
        self.embedding_size = embedding_size
        self.min_variance = min_variance