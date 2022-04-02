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
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import VarianceThreshold
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import linear_kernel
from util import clean_data


class Homophone:
    def __init__(self, clustering=None):
        self.clustering_ = copy.deepcopy(clustering)

    def encode(self, X):
        return CountVectorizer(lowercase=False, token_pattern=r'\S+', binary=True) \
            .fit_transform(X)

    def fit(self, X, y=None):
        self.n_features_ = X.shape[1]

        features = scipy.sparse.hstack(
            [self.encode(X.iloc[i] if hasattr(X, 'iloc') else X[i]) \
                for i in range(X.shape[0])],
            format='csr'
        )

        if self.clustering_ is not None:
            labels = self.clustering_.fit_predict(features)
            encoder = LabelEncoder().fit(labels)
            self.classes_ = encoder.classes_
            self.labels_ = encoder.transform(labels)

        return self

    def transform(self, X):
        result = []
        for i in range(X.shape[0]):
            features = self.encode(X.iloc[i] if hasattr(X, 'iloc') else X[i])
            if self.clustering_ is not None:
                features = scipy.sparse.csr_matrix(numpy.stack(
                    [features[self.labels_ == i].mean(axis=0) \
                        for i in range(self.classes_.shape[0])]
                ))

            sim = linear_kernel(features)
            result.append(scipy.sparse.csr_matrix(sim[numpy.triu_indices_from(sim)]))

        return scipy.sparse.vstack(result, format='csr')

    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

    def get_feature_names(self, input_features=None):
        if self.clustering_ is None:
            feature_names = [f'x{i}=x{j}' for i, j in zip(*numpy.triu_indices(self.n_features_))]

        else:
            if input_features is None:
                names = [f'x{numpy.where(self.labels_ == i)[0][0]}' for i in range(self.classes_.shape[0])]
            else:
                names = pandas.Series(input_features).groupby(self.labels_).agg(''.join).values
                if False:
                    names = [''.join(numpy.take(input_features, numpy.nonzero(self.labels_ == i)[0])) \
                        for i in range(self.classes_.shape[0])]

            feature_names = [f'{names[i]}={names[j]}' for i, j in zip(*numpy.triu_indices(self.classes_.shape[0]))]

        return feature_names

class HomophoneGroup:
    def __init__(self, clustering=KMeans(1000)):
        self.clustering_ = copy.deepcopy(clustering)

    def fit(self, X, y=None):
        labels = self.clustering_.fit_predict(X.T)
        encoder = LabelEncoder().fit(labels)
        self.classes_ = encoder.classes_
        self.labels_ = encoder.transform(labels)
        return self

    def transform(self, X):
        return numpy.column_stack([(X.iloc[:, self.labels_ == i] if hasattr(X, 'iloc') else X[:, self.labels_ == i]).mean(axis=1) \
            for i in range(self.classes_.shape[0])])

    def fit_transform(self, X, y=None):
        self.fit(X, y)

        if hasattr(self.clustering_, 'cluster_centers_'):
            return self.clustering_.cluster_centers_[self.classes_].T
        else:
            return self.transform(X)

    def get_feature_names(self, input_features=None):
        if input_features is None:
            return [f'x{numpy.where(self.labels_ == i)[0][0]}' for i in range(self.classes_.shape[0])]
        else:
            return list(pandas.Series(input_features).groupby(self.labels_).agg('_'.join))
            return ['_'.join(numpy.take(input_features, numpy.nonzero(self.labels_ == i)[0])) \
                for i in range(self.classes_.shape[0])]

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

    logging.info('done. {} data loaded'.format(len(dialects)))

    data = pandas.concat(
        [d.groupby(d.index).agg(' '.join).unstack() for d in dialects],
        axis=1,
        keys=load_ids
    ).dropna(axis=0, how='all').fillna('').transpose()

    logging.info(f'load data of {data.shape[0]} dialects x {data.columns.levels[1].shape[0]} characters')
    return data