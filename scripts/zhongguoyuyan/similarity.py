#!/usr/bin/python3 -O
# -*- coding: utf-8 -*-

'''
计算方言之间的预测相似度.

如果 A 方言的音类能很大程度预测 B 方言的音类，表明 A 方言到 B 方言单向的相似度高
'''

__author__ = '黄艺华 <lernanto@foxmail.com>'


import sys
import os
import logging
import pandas
import numpy
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer, MissingIndicator
import sklearn.feature_selection
import joblib


def load_data(prefix, ids, suffix='mb01dz.csv'):
    '''加载方言字音数据'''

    logging.info('loading {} data files ...'.format(len(ids)))

    columns = ['initial', 'finals', 'tone']
    load_ids = []
    dialects = []
    for id in ids:
        try:
            fname = os.path.join(prefix, id + suffix)
            d = pandas.read_csv(fname, encoding='utf-8', dtype=str)

            # 记录部分有值部分为空会导致统计数据细微偏差
            empty = d[columns].isna() | (d[columns] == '')
            empty = empty.any(axis=1) & ~empty.all(axis=1)
            empty_num = numpy.count_nonzero(empty)
            if empty_num > 0:
                logging.warning('{}/{} records from {} are partially empty, drop to avoid problems'.format(
                    empty_num,
                    d.shape[0],
                    fname
                ))
                logging.warning(d[empty])

                d = d[~empty]

            dialects.append(d)
            load_ids.append(id)
        except Exception as e:
            logging.error('cannot load file {}: {}'.format(fname, e))

    logging.info('done. {} data loaded'.format(len(dialects)))

    data = pandas.concat(
        [d.drop_duplicates('iid').set_index('iid')[columns] for d in dialects],
        axis=1,
        keys=load_ids
    ).fillna('')

    logging.info('load data of {} characters'.format(data.shape[0]))
    return load_ids, data

def encode_targets(data):
    '''编码预测目标'''

    logging.info('encoding targets ...')
    # 先记录缺失值的位置
    mask = MissingIndicator(missing_values='', features='all').fit_transform(data)

    # 把目标编码成数字，为了让编码器正常工作，先补全缺失值
    targets = OrdinalEncoder().fit_transform(
        SimpleImputer(
            missing_values='',
            strategy='most_frequent'
        ).fit_transform(data)
    )

    # 重新删除缺失值
    targets[mask] = numpy.nan
    logging.info('done. totally {} targets'.format(targets.shape[1]))
    return targets

def cross_features(data, column=3):
    '''构造交叉特征'''

    logging.info('constructing cross features ...')
    features = []
    for i in range(column):
        j = (i + 1) % column
        features.append(
            data[:, numpy.arange(i, data.shape[1], column)] \
                + data[:, numpy.arange(j, data.shape[1], column)]
        )

    logging.info('done. totally {} cross features'.format(len(features)))
    return features

def chi2(data, column=3, parallel=1):
    '''
    使用卡方检验计算方言之间的相似度

    思路为，如果 A 方言的声母加韵母能很大程度预测 B 方言的声母，说明 B 方言单向地接近 A 方言。
    这种预测能力通过卡方检验来衡量，即以 A 方言的声母加韵母为特征、B 方言的声母为目标执行卡方检验，
    卡方值越大表示越相似。遍历 A 方言及 B 方言的声母、韵母、声调，汇总后得到一个 A 方言对 B 方言预测能力的总评分。

    理论上由于自由度不同，卡方值不能直接比较，而应该比较相应的 p-value，但由于统计得出的卡方值非常大，
    导致计算出的 p-value 下溢为0，不能比较，另一方面由于自由度都比较大，卡方值近似于高斯分布，
    因此直接拿卡方值作为相似度来比较是可行的。

    这种方式计算出来的相似度是单向的，即相似度矩阵不是对称阵，如果应用于要求对称的一些聚类算法，
    可以取相似度矩阵和其转置的平均为最终的相似度矩阵。直观的解释是如果 A 方言能预测 B 方言，
    同时 B 方言也能预测 A 方言，那么两个方言是相似的。
    '''

    location = data.shape[1] // column
    logging.info('compute Chi square for {} locations {} characters {} columns'.format(
        location,
        data.shape[0],
        column
    ))

    targets = encode_targets(data)
    features = cross_features(data, column)

    # 特征 one-hot 编码
    logging.info('encoding features ...')
    limits = []
    for i, fea in enumerate(features):
        # 先记录缺失特征的位置
        mi = MissingIndicator(missing_values='')
        mask = mi.fit_transform(fea)

        # 为了让编码器正常工作，先补全缺失特征
        onehot = OneHotEncoder()
        fea = onehot.fit_transform(
            SimpleImputer(
                missing_values='',
                strategy='most_frequent'
            ).fit_transform(fea)
        )

        # 根据先前记录的缺失特征位置，把补全的特征从转换后的编码删除
        lim = numpy.cumsum([0] + [len(c) for c in onehot.categories_])
        for j, f in enumerate(mi.features_):
            fea[mask[:, j], lim[f]:lim[f + 1]] = 0

        features[i] = fea
        limits.append(lim)

    logging.info('done. totally {} features'.format(sum(f.shape[1] for f in features)))

    def compute_chi2(i):
        '''计算所有其他方言对 i 方言的相似度'''

        chi2s = []
        # 对声母、韵母、声调分别执行统计
        for j in range(column):
            # 需要剔除目标方言中的缺失样本
            target = targets[:, i * column + j]
            mask = ~numpy.isnan(target)
            target = target[mask]

            # 对其他方言的每一种交叉特征，也要分别统计
            for k, fea in enumerate(features):
                # 注意当特征包含预测目标的时候才执行预测，如声母 + 韵母预测声母
                # 否则受目标方言音系影响，即使同个方言的声母 + 韵母预测自己的声调相似度也不高
                l = (k + 1) % column
                if k == j or l == j:
                    if target.shape[0] < fea.shape[0]:
                        fea = fea[mask]

                    # 计算特征和目标的卡方统计量
                    # 由于样本缺失偶然有些组合会统计出 NaN，填充为0
                    chi2 = sklearn.feature_selection.chi2(fea, target)[0]
                    chi2 = numpy.where(numpy.isnan(chi2), 0, chi2)
                    # 某个特征的卡方统计量是该特征所有取值的卡方统计量之和
                    chi2s.append(numpy.asarray(
                        [numpy.sum(chi2[limits[k][m]:limits[k][m + 1]]) \
                             for m in range(location)]
                    ))

        # 多组特征预测同一个目标，取卡方统计量中最大的
        return numpy.mean(
            numpy.max(numpy.stack(chi2s).reshape(column, -1, location), axis=1),
            axis=0
        )

    # 计算所有方言组合的卡方统计量
    logging.info('computing Chi square ...')
    if parallel > 1:
        gen = joblib.Parallel(n_jobs=parallel)(
            joblib.delayed(compute_chi2)(i) for i in range(location)
        )
    else:
        gen = (compute_chi2(i) for i in range(location))

    sim = numpy.empty((location, location))
    for i, chi2 in enumerate(gen):
        sim[:, i] = chi2
        if (i + 1) % 10 == 0:
            logging.info('finished {} locations'.format(i + 1))

    logging.info('done. finished {} locations'.format(location))
    return sim


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)

    prefix = sys.argv[1]
    location = pandas.read_csv(
        os.path.join(prefix, 'location.csv'),
        encoding='utf-8',
        index_col=0
    )
    ids, data = load_data(prefix, location.index)

    chisq = chi2(data.values, parallel=4)
    pandas.DataFrame(chisq, index=ids, columns=ids).to_csv(sys.stdout)