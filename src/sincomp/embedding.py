# -*- encoding: utf-8 -*-

"""
计算方言或字的低维稠密向量表示
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import numpy
import scipy
import sklearn.cluster
import sklearn.compose
import sklearn.decomposition
import sklearn.metrics.pairwise
import sklearn.pipeline
import sklearn.feature_extraction.text
import sklearn.feature_selection
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

        if self.n_features_in_ == 1:
            name = self.feature_names_in_[0] if hasattr(
                self, 'feature_names_in_'
            ) else '0'
            return numpy.full((X.shape[0], 1), name, dtype=str)

        outputs = []
        for i in range(X.shape[0]):
            Xi = X[i:i + 1].toarray() \
                if scipy.sparse.issparse(X) else numpy.asarray(X[i:i + 1])
            Xi = numpy.reshape(Xi, (self.n_features_in_, self.n_features_in_))
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
    合并方言字音各部分的相似度矩阵。

    Parameters:
        column_selectors: 字音特征的选择器列表。每个选择器可以是列名模式字符串，
            也可以是列索引序列或 slice。
        dtype: 计算相似度时使用的 NumPy 数据类型。
    """

    def __init__(
        self,
        column_selectors: Sequence[Union[str, Sequence, slice]] = ['initial', 'final', 'tone'],
        dtype: Type = numpy.float64
    ):
        self.column_selectors = list(column_selectors)
        self.dtype = dtype

    def fit(
        self,
        X: Union[numpy.ndarray, pandas.DataFrame],
        y: Optional[numpy.ndarray] = None
    ) -> 'DialectVectorizer':
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
        self.output_indices_ = []
        start = 0
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

            t = PhoneSimilarity(self.dtype).fit(Xi)
            stop = start + t.n_features_out_
            self.transformers_.append((name, t, idx))
            self.output_indices_.append(slice(start, stop))
            start = stop

        self.n_features_out_ = start
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

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'X has {X.shape[1]} features, but DialectVectorizer is expecting {self.n_features_in_} features as input.')

        X = X.toarray() if scipy.sparse.issparse(X) else numpy.asarray(X)
        return scipy.sparse.hstack(
            [t.transform(X[:, i]) for _, t, i in self.transformers_],
            format='csr'
        )

    def fit_transform(
        self,
        X: Union[numpy.ndarray, pandas.DataFrame],
        y: Optional[numpy.ndarray] = None
    ) -> scipy.sparse.csr_matrix:
       return self.fit(X, y).transform(X)

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

        X = X.toarray() if scipy.sparse.issparse(X) else numpy.asarray(X)
        X_original = numpy.full(
            (X.shape[0], self.n_features_in_),
            '',
            dtype=object
        )
        for (_, t, ii), oi in zip(self.transformers_, self.output_indices_):
            X_original[:, ii] = t.inverse_transform(X[:, oi])

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

        if input_features is None:
            input_features = getattr(
                self,
                'feature_names_in_',
                [f'x{i}' for i in range(self.n_features_in_)]
            )
        input_features = numpy.asarray(input_features)
        return numpy.concatenate(
            [t.get_feature_names_out(input_features[i])
             for _, t, i in self.transformers_]
        )


class DialectEmbedding(sklearn.base.BaseEstimator):
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
        embedding_size: int = 128,
        mean_threshold: float = 0.05
    ):
        self.embedding_size = embedding_size
        self.mean_threshold = mean_threshold

    def _get_support_mask(self):
        return (self.mean_ > self.mean_threshold) & (self.mean_ < 1 - self.mean_threshold)

    def fit_transform(
        self,
        X: scipy.sparse.csr_matrix,
        y: Optional[numpy.ndarray] = None
    ) -> numpy.ndarray:
        self.embedding_size_ = min(self.embedding_size, X.shape[0] - 1, X.shape[1] - 1)
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = self.embedding_size_

        self.mean_ = numpy.squeeze(numpy.asarray(X.mean(axis=0)))

        u, s, vt = scipy.sparse.linalg.svds(
            X[:, self._get_support_mask()],
            self.embedding_size_
        )
        u, s, vt = u[:, ::-1], s[::-1], vt[::-1]

        self.components_ = numpy.asarray(vt)
        self.singular_values_ = s
        self.explained_variance_ = numpy.var(u * s[None, :], axis=0)
        var_sum = X.multiply(X).sum() / X.shape[0] \
            - numpy.linalg.norm(self.mean_) ** 2
        self.explained_variance_ratio_ = self.explained_variance_ / var_sum

        Xt = numpy.asarray(u) * s[None, :]
        self.scale_ = numpy.exp(
            numpy.mean(numpy.log(numpy.linalg.norm(Xt, axis=1)))
        )
        return Xt / self.scale_

    def fit(
        self,
        X: scipy.sparse.csr_matrix,
        y: Optional[numpy.ndarray] = None
    ) -> 'DialectEmbedding':
        self.fit_transform(X, y)
        return self

    def transform(self, X: scipy.sparse.csr_matrix) -> numpy.ndarray:
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'X has {X.shape[1]} features, but DialectEmbedding is expecting {self.n_features_in_} features as input.')

        return X[:, self._get_support_mask()] @ self.components_.T / self.scale_

    def inverse_transform(
        self,
        X: Union[numpy.ndarray, pandas.DataFrame]
    ) -> numpy.ndarray:
        X = numpy.asarray(X)
        if X.shape[1] != self.n_features_out_:
            raise ValueError(f'X has {X.shape[1]} features, but DialectEmbedding is expecting {self.n_features_out_} features as input.')

        X = numpy.asarray(X)
        X_original = numpy.repeat(self.mean_[None, :], X.shape[0], axis=0)
        X_original[:, self._get_support_mask()] = X * self.scale_ @ self.components_
        return X_original


class CharacterVectorizer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Convert a wide dialect phone table into character multi-hot encoding.

    The input must be a wide table with one row per character and one phone
    feature per column. DataFrame columns must be unique and must not be a
    MultiIndex.

    Parameters
    ----------
    dtype : type, default=numpy.int32
        Data type of the encoded output.

    Attributes
    ----------
    feature_names_in_ : pandas.Index or list of str
        Names of the input phone feature columns seen during fitting.
    transformer_ : sklearn.compose.ColumnTransformer
        Column transformer containing one fitted CountVectorizer per input
        column.
    vocabularies_ : list of dict
        Token-to-index vocabulary for each input column.
    """

    def __init__(self, dtype=numpy.int32):
        self.dtype = dtype

    def fit(
        self,
        X: Union[pandas.DataFrame, numpy.ndarray],
        y: Optional[numpy.ndarray] = None
    ) -> 'CharacterVectorizer':
        """Fit the transformer on a wide dialect phone table.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Wide input table with one phone feature per column.
        y : numpy.ndarray, default=None
            Ignored. This parameter exists for compatibility with the
            scikit-learn transformer API.

        Returns
        -------
        CharacterVectorizer
            Fitted transformer.
        """

        if X.shape[1] == 0:
            raise ValueError('Input must contain at least one phone column.')

        self.feature_names_in_ = X.columns if isinstance(X, pandas.DataFrame) \
            else [f'x{i}' for i in range(X.shape[1])]

        vectorizer = sklearn.feature_extraction.text.CountVectorizer(
            lowercase=False,
            tokenizer=str.split,
            token_pattern=None,
            stop_words=None,
            binary=True,
            dtype=self.dtype
        )
        self.transformer_ = sklearn.compose.ColumnTransformer(
            [(n, vectorizer, n) for n in self.feature_names_in_],
            remainder='drop',
            sparse_threshold=1.0
        ).fit(X)

        self.vocabularies_ = [
            self.transformer_.named_transformers_[n].vocabulary_
            for n in self.feature_names_in_
        ]

        return self

    def transform(
        self,
        X: Union[pandas.DataFrame, numpy.ndarray]
    ) -> scipy.sparse.spmatrix:
        """Transform a wide dialect phone table into character encodings.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Wide input table with phone feature columns matching the data
            used during fitting.

        Returns
        -------
        scipy.sparse.spmatrix
            Sparse multi-hot encoding matrix.
        """

        if not hasattr(self, 'transformer_'):
            raise ValueError('CharacterVectorizer is not fitted yet.')

        return self.transformer_.transform(X)

    def fit_transform(
        self,
        X: Union[pandas.DataFrame, numpy.ndarray],
        y: Optional[numpy.ndarray] = None
    ) -> scipy.sparse.spmatrix:
        """Fit the transformer and transform the input in one step.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Wide input table with phone feature columns.
        y : numpy.ndarray, default=None
            Ignored. This parameter exists for compatibility with the
            scikit-learn transformer API.

        Returns
        -------
        scipy.sparse.spmatrix
            Sparse multi-hot encoding matrix.
        """

        return self.fit(X, y).transform(X)

    def get_feature_names_out(
        self,
        input_features: Optional[Sequence[str]] = None
    ) -> numpy.ndarray:
        """Get output feature names for the multi-hot character encoding.

        Parameters
        ----------
        input_features : Sequence[str], default=None
            Ignored. The output names are determined by the fitted
            vectorizers.

        Returns
        -------
        numpy.ndarray
            Output feature names.
        """

        if not hasattr(self, 'transformer_'):
            raise ValueError('CharacterVectorizer is not fitted yet.')
        return self.transformer_.get_feature_names_out()


class CharacterEmbedding(
    sklearn.base.BaseEstimator,
    sklearn.base.TransformerMixin
):
    """Convert character multi-hot encodings into dense embeddings.

    The embedding is computed with truncated singular value decomposition.

    Parameters
    ----------
    embedding_size : int, default=128
        Maximum number of dimensions in the output embedding. The effective
        size is limited by the number of samples and input features.
    dtype : type, default=numpy.float32
        Data type of the embedding output.

    Attributes
    ----------
    n_features_in_ : int
        Number of input features seen during fitting.
    n_features_out_ : int
        Number of output embedding dimensions.
    svd_ : sklearn.decomposition.TruncatedSVD
        Fitted truncated SVD estimator.
    character_embeddings_ : numpy.ndarray
        Training data transformed into the embedding space.
    components_ : numpy.ndarray
        Principal components of the fitted SVD model.
    singular_values_ : numpy.ndarray
        Singular values of the fitted SVD model.
    explained_variance_ : numpy.ndarray
        Explained variance for each selected component.
    explained_variance_ratio_ : numpy.ndarray
        Percentage of variance explained by each selected component.
    """

    def __init__(
        self,
        embedding_size: int = 128,
        dtype: Type = numpy.float32
    ):
        self.embedding_size = embedding_size
        self.dtype = dtype

    def fit(
        self,
        X: Union[scipy.sparse.spmatrix, numpy.ndarray],
        y: Optional[numpy.ndarray] = None
    ) -> 'CharacterEmbedding':
        """Fit a TruncatedSVD model on character encodings.

        Parameters
        ----------
        X : scipy.sparse.spmatrix or numpy.ndarray
            Character multi-hot encoding matrix.
        y : numpy.ndarray, default=None
            Ignored. This parameter exists for compatibility with the
            scikit-learn transformer API.

        Returns
        -------
        CharacterEmbedding
            Fitted transformer.
        """

        X = scipy.sparse.csr_matrix(X, dtype=self.dtype) \
            if not scipy.sparse.isspmatrix(X) else X.astype(self.dtype)
        self.n_features_in_ = X.shape[1]
        embedding_size = min(
            self.embedding_size,
            X.shape[0],
            X.shape[1]
        )
        self.n_features_out_ = embedding_size

        self.svd_ = sklearn.decomposition.TruncatedSVD(embedding_size)
        self.character_embeddings_ = self.svd_.fit_transform(X)
        self.components_ = self.svd_.components_
        self.singular_values_ = self.svd_.singular_values_
        self.explained_variance_ = self.svd_.explained_variance_
        self.explained_variance_ratio_ = self.svd_.explained_variance_ratio_
        return self

    def transform(
        self,
        X: Union[scipy.sparse.spmatrix, numpy.ndarray]
    ) -> numpy.ndarray:
        """Transform character multi-hot encoding into dense embeddings.

        Parameters
        ----------
        X : scipy.sparse.spmatrix or numpy.ndarray
            Character multi-hot encoding matrix with the same number of
            features as the data used during fitting.

        Returns
        -------
        numpy.ndarray
            Dense character embeddings.
        """

        if not hasattr(self, 'svd_'):
            raise ValueError('CharacterEmbedding is not fitted yet.')
        X = scipy.sparse.csr_matrix(X, dtype=self.dtype) \
            if not scipy.sparse.isspmatrix(X) else X.astype(self.dtype)
        return self.svd_.transform(X)

    def fit_transform(
        self,
        X: Union[scipy.sparse.spmatrix, numpy.ndarray],
        y: Optional[numpy.ndarray] = None
    ) -> numpy.ndarray:
        """Fit the transformer and transform the input in one step.

        Parameters
        ----------
        X : scipy.sparse.spmatrix or numpy.ndarray
            Character multi-hot encoding matrix.
        y : numpy.ndarray, default=None
            Ignored. This parameter exists for compatibility with the
            scikit-learn transformer API.

        Returns
        -------
        numpy.ndarray
            Dense character embeddings.
        """

        return self.fit(X, y).transform(X)

    def inverse_transform(
        self,
        X: Union[numpy.ndarray, pandas.DataFrame]
    ) -> numpy.ndarray:
        """Inverse transform dense embeddings back to feature space.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Dense character embeddings.

        Returns
        -------
        numpy.ndarray
            Reconstructed multi-hot-like feature matrix.
        """

        if not hasattr(self, 'svd_'):
            raise ValueError('CharacterEmbedding is not fitted yet.')
        X = numpy.asarray(X)
        return self.svd_.inverse_transform(X)

    def get_feature_names_out(
        self,
        input_features: Optional[Sequence[str]] = None
    ) -> numpy.ndarray:
        """Get output feature names for the dense embeddings.

        Parameters
        ----------
        input_features : Sequence[str], default=None
            Ignored. Embedding feature names are generated from the fitted
            embedding size.

        Returns
        -------
        numpy.ndarray
            Output embedding feature names.
        """

        return numpy.asarray([
            f'embed_{i}' for i in range(self.n_features_out_)
        ])
