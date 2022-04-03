# -*- coding: utf-8 -*-

'''
根据方言同音字对挖掘方言区别特征.
'''

__author__ = '黄艺华 <lernanto@foxmail.com>'


import os
import logging
from tokenize import Special
import pandas
import numpy
import scipy.sparse
import copy
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans, SpectralClustering, spectral_clustering
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from util import clean_data


class FeatureClustering:
    def __init__(self, method='kmeans', n_clusters=100, pooling_func=numpy.mean):
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
        if scipy.sparse.issparse(X) and self.pooling_func == numpy.mean:
            # 利用稀疏矩阵运算加快速度
            group = scipy.sparse.csr_matrix((
                numpy.ones_like(self.labels_),
                (numpy.arange(self.lables_.shape[0]), self.labels_)
            ))
            counts = group.sum(axis=0).A.squeeze()
            diag = scipy.sparse.diags(numpy.where(counts == 0, 0, 1 / counts))
            return X * (group * diag)

        else:
            return numpy.column_stack([self.pooling_func(
                (X.iloc if hasattr(X, 'iloc') else X)[:, self.labels_ == i],
                axis=1
            ) for i in self.classes_])

    def fit_transform(self, X, y=None):
        self.fit(X, y)
        
        if self.pooling_func == numpy.mean \
            and hasattr(self.transformer_, 'cluster_centers_'):
            return self.transformer_.cluster_centers_
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
    def __init__(self, method='kmeans', n_clusters=100, n_jobs=None):
        self.n_jobs = n_jobs

        if method == 'kmeans':
            transformer = KMeans(n_clusters=n_clusters)
        elif method == 'spectral_clustering':
            transformer = SpectralClustering(n_clusters=n_clusters)
        else:
            transformer = copy.deepcopy(method)

        super().__init__(
            method=make_pipeline(PhoneVectorizer(n_jobs=self.n_jobs), transformer),
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

class Homophone:
    def __init__(self, affinity=cosine_similarity, dtype=numpy.float64):
        self.affinity = affinity
        self.dtype = dtype

    def fit(self, X, y=None):
        self.n_features_ = X.shape[1]
        return self

    def transform(self, X):
        result = []
        for i in range(X.shape[0]):
            features = CountVectorizer(
                lowercase=False,
                token_pattern=r'\S+',
                dtype=self.dtype
            ).fit_transform( X.iloc[i] if hasattr(X, 'iloc') else X[i])
            sim = self.affinity(features)
            result.append(scipy.sparse.csr_matrix(sim[numpy.triu_indices_from(sim)]))

        return scipy.sparse.vstack(result, format='csr')

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names(self, input_features=None):
        indices = numpy.triu_indices(self.n_features_)
        if input_features is None:
            return [f'x{i}=x{j}' for i, j in zip(*indices)]
        else:
            return [f'{input_features[i]}={input_features[j]}' for i, j in zip(*indices)]

class HomophoneClustering(FeatureClustering):
    def __init__(self, method='kmeans', n_clusters=1000, **kwargs):
        super().__init__(method=method, n_clusters=n_clusters, **kwargs)

    def get_feature_names(self, input_features=None):
        if input_features is None:
            return [f'x{numpy.where(self.labels_ == i)[0][0]}' for i in range(self.classes_.shape[0])]
        else:
            return list(pandas.Series(input_features).groupby(self.labels_).agg('_'.join))

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