# -*- coding: utf-8 -*-

'''
根据方言同音字对挖掘方言区别特征.
'''

__author__ = '黄艺华 <lernanto@foxmail.com>'


import os
import logging
import pandas
import numpy
import scipy.sparse
import copy
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.cluster import KMeans, SpectralClustering, spectral_clustering
from util import clean_data


class FeatureClustering:
    def __init__(self, method='kmeans', n_clusters=100, pooling_func='mean'):
        self.method = method
        self.n_clusters = n_clusters
        self.pooling_func = pooling_func

        if self.method == 'kmeans':
            self.transformer_ = KMeans(n_clusters=self.n_clusters)
        elif self.method == 'spectral_clustering':
            self.transformer_ = SpectralClustering(n_clusters=self.n_clusters)
        else:
            self.transformer_ = copy.deepcopy(self.method)

    def fit(self, X, y=None):
        labels = self.transformer_.fit_predict(X.T)
        encoder = LabelEncoder().fit(labels)
        self.classes_ = encoder.classes_
        self.labels_ = encoder.transform(labels)
        return self

    def transform(self, X):
        if scipy.sparse.issparse(X) and self.pooling_func in ('sum', 'mean'):
            # 利用稀疏矩阵运算加快速度
            T = scipy.sparse.csr_matrix((
                numpy.ones_like(self.labels_),
                (numpy.arange(self.labels_.shape[0]), self.labels_)
            ))

            if self.pooling_func == 'mean':
                counts = T.sum(axis=0).A.squeeze()
                diag = scipy.sparse.diags(numpy.where(counts == 0, 0, 1 / counts))
                T *= diag

            return X * T

        else:
            if self.pooling_func == 'sum':
                pooling_func = numpy.sum
            elif self.pooling_func == 'mean':
                pooling_func = numpy.mean
            else:
                pooling_func = self.pooling_func

            X_t = [pooling_func(
                (X.iloc if hasattr(X, 'iloc') else X)[:, self.labels_ == i],
                axis=1
            ) for i in self.classes_]

            if scipy.sparse.issparse(X_t[0]):
                return scipy.sparse.hstack(X_t, format='csr')
            else:
                return numpy.column_stack(X_t)

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        
        if self.pooling_func == numpy.mean \
            and hasattr(self.transformer_, 'cluster_centers_'):
            X_t = self.transformer_.cluster_centers_.T
            if scipy.sparse.issparse(X):
                X_t = scipy.sparse.csr_matrix(X_t)

            return X_t

        else:
            return self.transform(X)

class PhoneVectorizer(ColumnTransformer):
    def __init__(self, dtype=numpy.float64, norm='l2', n_jobs=None):
        self.dtype = dtype
        self.norm = norm
        self.n_jobs = n_jobs

    def _reset(self, X):
        vectorizer = TfidfVectorizer(
            lowercase=False,
            token_pattern=r'\S+',
            dtype=self.dtype,
            norm=self.norm,
            use_idf=False
        )
        super().__init__(
            transformers=[(
                f'{type(vectorizer).__name__.lower()}{i}',
                vectorizer,
                i
            ) for i in range(X.shape[1])],
            n_jobs=self.n_jobs
        )

    def fit(self, X, y=None):
        self._reset(X)
        return super().fit(X, y)

    def transform(self, X):
        return super().transform(X)

    def fit_transform(self, X, y=None):
        self._reset(X)
        return super().fit_transform(X, y)

class PhoneClustering(FeatureClustering):
    def __init__(
        self,
        method='kmeans',
        n_clusters=100,
        dtype=numpy.float64,
        n_jobs=None
    ):
        self.dtype = dtype
        self.n_jobs = n_jobs

        if method == 'kmeans':
            transformer = KMeans(n_clusters=n_clusters)
        elif method == 'spectral_clustering':
            transformer = SpectralClustering(n_clusters=n_clusters)
        else:
            transformer = copy.deepcopy(method)

        super().__init__(
            method=make_pipeline(
                PhoneVectorizer(dtype=self.dtype, n_jobs=self.n_jobs),
                transformer
            ),
            n_clusters=n_clusters,
            pooling_func=lambda x, axis: numpy.apply_along_axis(' '.join, axis, x)
        )

    def get_feature_names(self, input_features=None):
        if input_features is None:
            return [f'x{numpy.where(self.labels_ == i)[0][0]}' \
                for i in range(self.classes_.shape[0])]
        else:
            return list(pandas.Series(input_features) \
                .groupby(self.labels_).agg(''.join))

class SmoothSimilarity:
    def __init__(self, smooth=1):
        self.smooth = smooth

    def __call__(self, X, Y=None, **kwargs):
        X_norm = X.sum(axis=1).A.squeeze() + self.smooth
        if Y is None:
            Y = X
            Y_norm = X_norm
        else:
            Y_norm = Y.sum(axis=1).A.squeeze() + self.smooth

        if scipy.sparse.issparse(X) and scipy.sparse.issparse(Y):
            return scipy.sparse.diags(1 / X_norm) * X \
                * Y.T * scipy.sparse.diags(1 / Y_norm)
        else:
            return numpy.dot(
                numpy.asarray(X / X_norm[:, None]),
                numpy.asarray(Y / Y_norm[:, None]).T
            )

class Homophone:
    def __init__(
        self,
        affinity=SmoothSimilarity(),
        interaction_only=False,
        dtype=numpy.float64
    ):
        self.affinity = affinity
        self.interaction_only = interaction_only
        self.dtype = dtype

    def fit(self, X, y=None):
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X):
        trius = []
        for i in range(X.shape[0]):
            features = CountVectorizer(
                lowercase=False,
                token_pattern=r'\S+',
                dtype=self.dtype
            ).fit_transform(X.iloc[i] if hasattr(X, 'iloc') else X[i])
            sim = self.affinity(features, dense_output=False)

            if scipy.sparse.issparse(sim):
                # 利用稀疏矩阵的特性加速上三角阵计算
                # 根据坐标映射关系计算变换后的坐标
                if self.interaction_only:
                    # 不计入和自己的读音相似度，排除相似度矩阵的对角线元素
                    row, col = scipy.sparse.triu(sim, 1).nonzero()
                    new_col = row * (2 * sim.shape[1] - row - 3) // 2 + col
                    shape = sim.shape[1] * (sim.shape[1] - 1) // 2
                else:
                    row, col = scipy.sparse.triu(sim).nonzero()
                    new_col = row * (2 * sim.shape[1] - row - 1) // 2 + col
                    shape = sim.shape[1] * (sim.shape[1] + 1) // 2

                data = sim[row, col].A.squeeze()
                new_row = numpy.zeros_like(new_col)
                triu = scipy.sparse.csr_matrix(
                    (data, (new_row, new_col)),
                    shape=(1, shape)
                )
            else:
                if self.interaction_only:
                    indices = numpy.triu_indices_from(sim, 1)
                else:
                    indices = numpy.triu_indices_from(sim)

                triu = sim[indices]

            trius.append(triu)

        if scipy.sparse.issparse(trius[0]):
            return scipy.sparse.vstack(trius, format='csr')
        else:
            return numpy.stack(trius)

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names(self, input_features=None):
        indices = numpy.triu_indices(self.n_features_)
        if input_features is None:
            return [f'x{i}=x{j}' for i, j in zip(*indices)]
        else:
            return [f'{input_features[i]}={input_features[j]}' \
                for i, j in zip(*indices)]

class HomophoneClustering(FeatureClustering):
    def __init__(self, n_clusters=1000, **kwargs):
        super().__init__(n_clusters=n_clusters, **kwargs)

    def get_feature_names(self, input_features=None):
        if input_features is None:
            return [f'x{numpy.where(self.labels_ == i)[0][0]}' \
                for i in range(self.classes_.shape[0])]
        else:
            return list(pandas.Series(input_features) \
                .groupby(self.labels_).agg('_'.join))

def load_data(prefix, ids, suffix='mb01dz.csv'):
    '''加载方言字音数据'''

    logging.info('loading {} data files ...'.format(len(ids)))

    load_ids = []
    dialects = []
    for id in ids:
        try:
            fname = os.path.join(prefix, id + suffix)
            logging.info(f'loading {fname} ...')
            d = pandas.read_csv(
                fname,
                encoding='utf-8',
                index_col='iid',
                usecols=('iid', 'initial', 'finals', 'tone'),
                dtype={ 'iid': int, 'initial': str, 'finals': str, 'tone': str}
            )
        except Exception as e:
            logging.error('cannot load file {}: {}'.format(fname, e))
            continue

        d = clean_data(d)
        dialects.append(d)
        load_ids.append(id)

    logging.info('done. {} data file loaded'.format(len(dialects)))

    data = pandas.concat(
        [d.groupby(d.index).agg(' '.join).unstack() for d in dialects],
        axis=1,
        keys=load_ids
    ).dropna(axis=0, how='all').fillna('').transpose()

    logging.info(f'load data of {data.shape[0]} dialects x {data.columns.levels[1].shape[0]} characters')
    return data