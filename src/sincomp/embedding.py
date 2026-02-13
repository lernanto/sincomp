# -*- encoding: utf-8 -*-

"""
计算方言或字的低维稠密向量表示
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import numpy
import scipy
import sklearn.cluster
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
        self.n_features_out_ = X.shape[1] * X.shape[1]
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

    def inverse_transform(
        self,
        X: Union[numpy.ndarray, scipy.sparse.csr_matrix]
    ) -> numpy.ndarray:
        """
        逆变换，将变换后的数据转换回原始数据

        Parameters:
           X: 变换后的数据

        Returns:
           X_original: 原始数据
        """

        if X.shape[1] != self.n_features_out_:
            raise ValueError(
                f'X has {X.shape[1]} features, but {type(self).__name__} '
                f'is expecting {self.n_features_out_} features as input.'
            )

        outputs = []
        for i in range(X.shape[0]):
            Xi = numpy.reshape(
                numpy.asarray(X[i:i + 1]),
                (self.n_features_in_, self.n_features_in_)
            )
            dist = numpy.clip(1 - (Xi + Xi.T), 0, 1)
            ac = sklearn.cluster.AgglomerativeClustering(
                n_clusters=None,
                metric='precomputed',
                linkage='average',
                distance_threshold=0.5
            ).fit(dist)

            if hasattr(self, 'feature_names_in_'):
                # 如果指定了特征名，以该类的第一个特征为类名
                names = numpy.empty(self.n_features_in_, dtype=object)
                for j in range(ac.n_clusters_):
                    mask = ac.labels_ == j
                    names[mask] = self.feature_names_in_[mask][0]
                    
                names = names.astype(str)

            else:
                names = ac.labels_.astype(str)

            outputs.append(names)

        return numpy.stack(outputs, axis=0)


class DialectVectorizer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """
    合并方言字音各部分的相似度矩阵
    """

    def __init__(
        self,
        column_selectors: Sequence[Union[str, Sequence, slice]] = ['initial', 'final', 'tone'],
        mean_threshold: float = 0.05,
        dtype: Type = numpy.float64
    ):
        self.column_selectors = list(column_selectors)
        self.mean_threshold = mean_threshold
        self.dtype = dtype

    def _get_support_masks(self) -> list[numpy.ndarray[bool]]:
        return [((m > self.mean_threshold) & (m < 1 - self.mean_threshold)) for m in self.means_]

    def fit_transform(
        self,
        X: Union[numpy.ndarray, pandas.DataFrame],
        y: Optional[numpy.ndarray] = None
    ) -> scipy.sparse.csr_matrix:
        """
        训练模型并转换输入数据。

        Parameters:
            X: 输入数据，可以是 NumPy 数组或 Pandas DataFrame。
            y: 目标变量，可选。

        Returns:
            X_new: 转换后的相似度矩阵。
        """

        self.n_features_in_ = X.shape[1]
        if isinstance(X, pandas.DataFrame):
            self.feature_names_in_ = X.columns.values

        self.transformers_ = []
        indices = numpy.arange(X.shape[1])
        X_arr = numpy.asarray(X)
        vectors = []
        for i, s in enumerate(self.column_selectors):
            if isinstance(s, str):
                assert isinstance(X, pandas.DataFrame)
                name = s
                idx = indices[X.columns.str.contains(s, regex=False)]
                Xi = X.iloc[:, idx]
            else:
                name = f'phonesimilarity{i}'
                idx = indices[s]
                Xi = X_arr[:, idx]

            t = PhoneSimilarity(self.dtype)
            self.transformers_.append((name, t, idx))
            vectors.append(t.fit_transform(Xi))

        self.means_ = [numpy.asarray(v.mean(axis=0)).squeeze() for v in vectors]
        masks = self._get_support_masks()
        start = 0
        self.output_indices_ = []
        for m in masks:
            stop = start + numpy.count_nonzero(m)
            self.output_indices_.append(slice(start, stop))
            start = stop

        self.n_features_out_ = start

        return scipy.sparse.hstack([v[:, m] for (v, m) in zip(vectors, masks)])

    def fit(
        self,
        X: Union[numpy.ndarray, pandas.DataFrame],
        y: Optional[numpy.ndarray] = None
    ) -> 'DialectVectorizer':
        """
        训练模型，设置内部状态以准备转换数据。

        Parameters:
            X: 输入数据，可以是 NumPy 数组或 pandas DataFrame。
            y: 目标变量，可选。

        Returns:
            self: 当前对象。
        """

        self.fit_transform(X, y)
        return self

    def transform(
        self,
        X: Union[numpy.ndarray, pandas.DataFrame]
    ) -> scipy.sparse.csr_matrix:
        """
        将输入数据转换为合并的相似度矩阵。

        Parameters:
            X: 输入数据，可以是 NumPy 数组或 pandas DataFrame。

        Returns:
            X_new: 合并的相似度矩阵。
        """

        X = numpy.asarray(X)
        masks = self._get_support_masks()
        return scipy.sparse.hstack(
            [t.transform(X[:, i])[:, m] for (_, t, i), m in zip(self.transformers_, masks)],
            format='csr'
        )

    def inverse_transform(
        self,
        X: scipy.sparse.csr_matrix
    ) -> numpy.ndarray:
        """
        将转换后的相似度矩阵逆转换为原始数据。

        Parameters:
            X: 转换后的相似度矩阵。

        Returns:
            X_original: 原始数据数组。
        """

        if X.shape[1] != self.n_features_out_:
            raise ValueError(
                f'X has {X.shape[1]} features, but {type(self).__name__} '
                f'is expecting {self.n_features_out_} features as input.'
            )

        X = numpy.asarray(X)
        X_original = numpy.full(
            (X.shape[0], self.n_features_in_),
            '',
            dtype=object
        )
        masks = self._get_support_masks()
        for (_, t, ii), oi, mean, mask in zip(
            self.transformers_,
            self.output_indices_,
            self.means_,
            masks
        ):
            X_inter = numpy.repeat(mean[None, :], X.shape[0], axis=0)
            X_inter[:, mask] = X[:, oi]
            X_original[:, ii] = t.inverse_transform(X_inter)

        return X_original.astype(str)

    def get_feature_names_out(
        self,
        input_features: Optional[Sequence[str]] = None
    ) -> numpy.ndarray:
        """
        获取转换后特征的名称。

        Parameters:
            input_features: 输入特征名称列表，可选。如果提供，则用于各子转换器的字段选择；
                否则使用默认输入特征名称。

        Returns:
            feature_names_out: 输出特征名称数组，由各个子转换器的名称组成。
        """

        masks = self._get_support_masks()
        return numpy.concatenate(
            [t.get_feature_names_out(None if input_features is None else input_features[i])[m] \
                for (_, t, i), m in zip(self.transformers_, masks)]
        )


class DialectEmbedding(sklearn.pipeline.Pipeline):
    """
    根据方言中字音的两两相似度，计算方言向量

    Parameters:
        column_selectors: 用于提取特征的模式列表，每个特征对应输入数据的若干列，其列名必须包含该特征的模式字符串
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
        column_selectors: Sequence[Union[str, Sequence, slice]] = ['initial', 'final', 'tone'],
        embedding_size: int = 128,
        mean_threshold: float = 0.05
    ):
        super().__init__([
            ('vectorizer', DialectVectorizer(column_selectors, mean_threshold)),
            ('embedding', sklearn.decomposition.TruncatedSVD(embedding_size))
        ])

        self.column_selectors = list(column_selectors)
        self.embedding_size = embedding_size
        self.mean_threshold = mean_threshold