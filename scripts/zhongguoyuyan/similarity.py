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
import scipy.sparse
import scipy.cluster.hierarchy
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib
import colorspacious
import seaborn
import plotly.express
import plotly.figure_factory
import folium


def clean_data(data):
    '''清洗方言字音数据中的录入错误'''

    ipa = 'A-Za-z\u00c0-\u03ff\u1d00-\u1dbf\u1e00-\u1eff\u2205\u2c60-\u2c7f' \
        + '\ua720-\ua7ff\uab30-\uab6f\ufb00-\ufb4f\ufffb' \
        + '\U00010780-\U000107ba\U0001df00-\U0001df1e'

    clean = data.copy()

    # 有些符号使用了多种写法，统一成较常用的一种
    clean['initial'] = clean['initial'].fillna('').str.lower() \
        .str.replace(f'[^{ipa}]', '') \
        .str.replace('[\u00f8\u01ff]', '\u2205') \
        .str.replace('\ufffb', ' ') \
        .str.replace('\u02a3', 'dz') \
        .str.replace('\u02a4', 'dʒ') \
        .str.replace('\u02a5', 'dʑ') \
        .str.replace('\u02a6', 'ts') \
        .str.replace('\u02a7', 'tʃ') \
        .str.replace('\u02a8', 'tɕ') \
        .str.replace('[\u02b0\u02b1]', 'h') \
        .str.replace('g', 'ɡ')

    mask = clean['initial'] != data['initial']
    if numpy.count_nonzero(mask):
        for (r, c), cnt in pandas.DataFrame({
            'raw': data.loc[mask, 'initial'],
            'clean': clean.loc[mask, 'initial']
        }).value_counts().iteritems():
            logging.warning(f'replace {r} -> {c} {cnt}')

    clean['finals'] = clean['finals'].fillna('').str.lower() \
        .str.replace(f'[^{ipa}]', '')

    mask = clean['finals'] != data['finals']
    if numpy.count_nonzero(mask):
        for (r, c), cnt in pandas.DataFrame({
            'raw': data.loc[mask, 'finals'],
            'clean': clean.loc[mask, 'finals']
        }).value_counts().iteritems():
            logging.warning(f'replace {r} -> {c} {cnt}')

    # 部分声调被错误转为日期格式，还原成数字
    clean['tone'].fillna('', inplace=True)
    mask = clean['tone'].str.match(r'^\d+年\d+月\d+日$')
    clean.loc[mask, 'tone'] = pandas.to_datetime(
        clean.loc[mask, 'tone'],
        format=r'%Y年%m月%d日'
    ).dt.dayofyear.astype(str)
    clean.loc[~mask, 'tone'] = clean.loc[~mask, 'tone'].str.lower() \
        .str.replace(r'[^1-5]', '')

    mask = clean['tone'] != data['tone']
    if numpy.count_nonzero(mask):
        for (r, c), cnt in pandas.DataFrame({
            'raw': data.loc[mask, 'tone'],
            'clean': clean.loc[mask, 'tone']
        }).value_counts().iteritems():
            logging.warning(f'replace {r} -> {c} {cnt}')

    return clean

def load_data(prefix, ids, suffix='mb01dz.csv'):
    '''加载方言字音数据'''

    logging.info('loading {} data files ...'.format(len(ids)))

    columns = ['initial', 'finals', 'tone']
    load_ids = []
    dialects = []
    for id in ids:
        try:
            fname = os.path.join(prefix, id + suffix)
            logging.info(f'loading {fname} ...')
            d = pandas.read_csv(fname, encoding='utf-8', dtype=str)
        except Exception as e:
            logging.error('cannot load file {}: {}'.format(fname, e))
            continue

        d = clean_data(d)

        # 记录部分有值部分为空会导致统计数据细微偏差
        invalid = (d[columns].isna() | (d[columns] == '')).any(axis=1)
        invalid_num = numpy.count_nonzero(invalid)
        if invalid_num > 0:
            logging.warning('drop {}/{} invalid records from {}'.format(
                invalid_num,
                d.shape[0],
                fname
            ))
            logging.warning(d[invalid])

            d = d[~invalid]

        dialects.append(d)
        load_ids.append(id)

    logging.info('done. {} data loaded'.format(len(dialects)))

    data = pandas.concat(
        [d.drop_duplicates('iid').set_index('iid')[columns] for d in dialects],
        axis=1,
        keys=load_ids
    ).fillna('')

    logging.info(f'load data of {data.shape[0]} characters x {len(dialects)} dialects, ' \
        + f'{sum(d.shape[0] for d in dialects)} valid records')
    return load_ids, data

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

    features = numpy.stack(features, axis=2)
    logging.info('done. totally {} cross features'.format(features.shape[2]))
    return features

def encode_features(features):
    '''特征 one-hot 编码'''

    logging.info('encoding features ...')

    # 为了让编码器正常工作，先补全缺失特征
    encoder = OneHotEncoder(
        sparse=True,
        dtype=numpy.float32,
        handle_unknown='ignore'
    )
    features = encoder.fit(
        SimpleImputer(
            missing_values='',
            strategy='most_frequent'
        ).fit_transform(features)
    ).transform(features)

    categories = numpy.asarray([len(c) for c in encoder.categories_])

    logging.info('done. totally {} features'.format(features.shape[1]))
    return features, categories

def freq2prob(freqs, limits):
    '''根据出现频次估算概率'''

    return numpy.concatenate(
        [freqs[limits[i]:limits[i + 1]] \
            / numpy.sum(freqs[limits[i]:limits[i + 1]]) \
            for i in range(limits.shape[0] - 1)]
    )

def chi2_block(
    features,
    feature_categories,
    feature_probs,
    targets,
    target_categories,
    target_probs
):
    '''
    分块计算卡方.

    x2 = sum((o - e)^2 / e) = sum(o^2 / e) + n
    其中 e = n * pi * pj，因此 sum(o^2 / e) = sum(o^2 / n / pi / pj)
    = 1/n sum(1/pi * sum(o^2 / pj))
    只需要计算求和中观察数 o 非零的项，因此也只需处理对应的概率 pj
    '''

    logging.debug('features = {}, feature categories = {} feature probs = {}, targets = {}, target categories = {}, target probs = {}'.format(
        features,
        feature_categories,
        feature_probs,
        targets,
        target_categories,
        target_probs
    ))

    freq = features.T * targets
    chisq = scipy.sparse.csr_matrix(freq)
    # 利用 CSR 矩阵的内部结构，只计算非0项的期望数
    chisq.data = numpy.square(chisq.data) / target_probs[chisq.indices]

    # 按列归并目标取值
    target_limits = numpy.concatenate([[0], numpy.cumsum(target_categories)])
    freq = numpy.column_stack([numpy.sum(
        freq[:, target_limits[i]:target_limits[i + 1]],
        axis=1
    ) for i in range(target_limits.shape[0] - 1)]).A
    chisq = numpy.column_stack([numpy.sum(
        chisq[:, target_limits[i]:target_limits[i + 1]],
        axis=1
    ) for i in range(target_limits.shape[0] - 1)]).A

    chisq /= feature_probs[:, None]

    # 按行归并特征取值
    feature_limits = numpy.concatenate([[0], numpy.cumsum(feature_categories)])
    freq = numpy.stack([numpy.sum(
        freq[feature_limits[i]:feature_limits[i + 1]],
        axis=0
    ) for i in range(feature_limits.shape[0] - 1)])
    chisq = numpy.stack([numpy.sum(
        chisq[feature_limits[i]:feature_limits[i + 1]],
        axis=0
    ) for i in range(feature_limits.shape[0] - 1)])

    # 得到真正的卡方值
    chisq = chisq / freq - freq

    # 计算自由度
    dof = numpy.outer(feature_categories - 1, target_categories - 1) \
        .astype(numpy.float32)
    # 标准化卡方值使之接近标准正态分布
    return (chisq - dof) / numpy.sqrt(2 * dof)

def chi2(
    src,
    dest=None,
    column=3,
    blocksize=(100, 100),
    parallel=1
):
    '''
    使用卡方检验计算方言之间的相似度

    思路为，如果 A 方言的声母 + 韵母能很大程度预测 B 方言的声母 + 韵母，说明 B 方言接近 A 方言。
    这种预测能力通过卡方检验来衡量，卡方值越大表示越相似。
    遍历 A 方言及 B 方言的声母、韵母、声调组合，取平均得到 A、B 方言之间的相似度评分。

    理论上由于自由度不同，卡方值不能直接比较，而应该比较相应的 p-value，但由于统计得出的卡方值非常大，
    导致计算出的 p-value 下溢为0，不能比较，另一方面由于自由度都比较大，卡方值近似于正态分布，
    可以把卡方值正则化到标准正态分布后进行比较。

    卡方检验对于 A、B 方言是对称的，即理论上相似度矩阵是对称阵，但由于计算精度的原因可能出现极小的误差导致不对称。
    可以取相似度矩阵和其转置的平均来强制保证对称性。
    '''

    src_num = src.shape[1] // column
    dest_num = src_num if dest is None else dest.shape[1] // column

    logging.info(('compute chi square for {} x {} dialects, ' \
        + 'characters = {}, columns = {}, block size = {}, parallel = {}').format(
        src_num,
        dest_num,
        src.shape[0],
        column,
        blocksize,
        parallel
    ))

    # 特征交叉
    if column > 1:
        features = cross_features(src, column)
        feature_column = features.shape[2]
        features = numpy.swapaxes(features, 1, 2).reshape(features.shape[0], -1)
    else:
        features = src
        feature_column = 1

    # 特征 one-hot 编码
    features, feature_categories = encode_features(features)
    feature_limits = numpy.concatenate([[0], numpy.cumsum(feature_categories)])
    # 根据特征的出现频率估算概率
    feature_probs = freq2prob(features.sum(axis=0).A.flatten(), feature_limits)

    if dest is None:
        targets = features
        target_categories = feature_categories
        target_limits = feature_limits
        target_probs = feature_probs
    else:
        # 由于卡方统计的对称性，对预测目标做和特征相同处理
        if column > 1:
            targets = cross_features(dest, column)
            targets = numpy.swapaxes(targets, 1, 2).reshape(targets.shape[0], -1)
        else:
            targets = dest

        targets, target_categories = encode_features(targets)
        target_limits = numpy.concatenate([[0], numpy.cumsum(target_categories)])
        target_probs = freq2prob(targets.sum(axis=0).A.flatten(), target_limits)

    # 分块并行计算特征到目标的卡方
    row_block = (src_num + blocksize[0] - 1) // blocksize[0]
    col_block = (dest_num + blocksize[1] - 1) // blocksize[1]

    logging.info('computing chi square for {} x {} x {} blocks, block size = {} ...'.format(
        feature_column,
        row_block,
        col_block,
        blocksize
    ))

    chisq = numpy.zeros((src_num, dest_num), dtype=numpy.float32)

    count = 0
    for i in range(feature_column):
        feature_base = i * src_num
        target_base = i * dest_num
        fc = feature_categories[feature_base:feature_base + src_num]
        fl = feature_limits[feature_base:feature_base + src_num + 1]
        tc = target_categories[target_base:target_base + dest_num]
        tl = target_limits[target_base:target_base + dest_num + 1]

        gen = joblib.Parallel(n_jobs=parallel)(
            joblib.delayed(chi2_block)(
                features[:, fl[j]:fl[min(j + blocksize[0], fl.shape[0] - 1)]],
                fc[j:j + blocksize[0]],
                feature_probs[fl[j]:fl[min(j + blocksize[0], fl.shape[0] - 1)]],
                targets[:, tl[k]:tl[min(k + blocksize[1], tl.shape[0] - 1)]],
                tc[k:k + blocksize[1]],
                target_probs[tl[k]:tl[min(k + blocksize[1], tl.shape[0] - 1)]]
            ) for j in range(0, src_num, blocksize[0]) \
                for k in range(0, dest_num, blocksize[1])
        )

        for j, ch in enumerate(gen):
            row = j // col_block * blocksize[0]
            col = j % col_block * blocksize[1]
            chisq[row:row + blocksize[0], col:col + blocksize[1]] += ch

            count += 1
            if count % 10 == 0:
                logging.info('finished {} blocks'.format(count))

        logging.info('done. finished {} blocks'.format(count))

    # 取多组特征卡方的均值
    if feature_column > 1:
        chisq /= feature_column

    return chisq

def joint_entropy(features, feature_limits, targets, target_limits):
    '''分块计算方言间联合熵'''

    # 计算共现频次及未归一化的信息量
    freq = features.T * targets
    entropy = scipy.sparse.csr_matrix(freq)
    entropy.data = numpy.where(
        entropy.data == 0,
        0,
        entropy.data * numpy.log(entropy.data)
    )

    # 对共现矩阵的行分组求和，把特征不同取值的频次归并在一起
    freq = numpy.stack([numpy.sum(
        freq[feature_limits[i]:feature_limits[i + 1]],
        axis=0
    ) for i in range(feature_limits.shape[0] - 1)]).A
    entropy = numpy.stack([numpy.sum(
        entropy[feature_limits[i]:feature_limits[i + 1]],
        axis=0
    ) for i in range(feature_limits.shape[0] - 1)]).A

    # 对共现矩阵的列分组求和，把目标不同取值的频次归并在一起
    freq = numpy.column_stack([numpy.sum(
        freq[:, target_limits[i]:target_limits[i + 1]],
        axis=1
    ) for i in range(target_limits.shape[0] - 1)])
    entropy = numpy.column_stack([numpy.sum(
        entropy[:, target_limits[i]:target_limits[i + 1]],
        axis=1
    ) for i in range(target_limits.shape[0] - 1)])

    # 根据总频次归一化信息量，得到真正的联合熵
    return numpy.log(freq) - entropy / freq

def marginal_entropy(features, feature_limits):
    '''计算特征边缘熵'''

    freq = features.sum(axis=0).A[0]
    entropy = numpy.where(freq == 0, 0, freq * numpy.log(freq))

    freq = numpy.stack([numpy.sum(
        freq[feature_limits[i]:feature_limits[i + 1]]
    ) for i in range(feature_limits.shape[0] - 1)])
    entropy = numpy.stack([numpy.sum(
        entropy[feature_limits[i]:feature_limits[i + 1]]
    ) for i in range(feature_limits.shape[0] - 1)])
    return numpy.log(freq) - entropy / freq

def entropy(
    src,
    dest=None,
    column=3,
    blocksize=(100, 100),
    parallel=1
):
    '''
    计算方言之间的条件熵.
    '''

    if dest is None:
        dest = src

    src_num = src.shape[1] // column
    dest_num = dest.shape[1] // column

    logging.info(('compute joint entropy for {} sources {} destinations, ' \
        + 'characters = {}, columns = {}, block size = {}, parallel = {}').format(
        src_num,
        dest_num,
        src.shape[0],
        column,
        blocksize,
        parallel
    ))

    # 特征交叉
    if column > 1:
        features = cross_features(src, column)
        feature_column = features.shape[2]
        features = features.reshape(features.shape[0], -1)
    else:
        features = src
        feature_column = 1

    # 特征 one-hot 编码
    features, feature_categories = encode_features(features)
    feature_limits = numpy.concatenate([[0], numpy.cumsum(feature_categories)])

    # H(Y|X) = H(X, Y) - H(X)，分别计算联合熵和特征的边缘熵，然后相减
    # 计算特征边缘熵
    logging.info('computing marginal entropy for {} features ...'.format(
        feature_limits.shape[0] - 1
    ))
    feature_entropy = marginal_entropy(features, feature_limits)
    logging.info('done.')

    # 预测目标编码
    targets, target_categories = encode_features(dest)
    target_limits = numpy.concatenate([[0], numpy.cumsum(target_categories)])

    # 计算特征到目标的条件熵
    row_block = (src_num + blocksize[0] - 1) // blocksize[0]
    col_block = (dest_num + blocksize[1] - 1) // blocksize[1]

    logging.info('computing joint entropy for {} x {} blocks, block size = {} ...'.format(
        row_block,
        col_block,
        blocksize
    ))

    # 分块并行计算联合熵
    feature_block = blocksize[0] * feature_column
    target_block = blocksize[1] * column
    gen = joblib.Parallel(n_jobs=parallel)(
        joblib.delayed(joint_entropy)(
            features[:, feature_limits[i]:feature_limits[min(
                i + feature_block,
                feature_limits.shape[0] - 1
            )]],
            feature_limits[i:i + feature_block + 1] - feature_limits[i],
            targets[:, target_limits[j]:target_limits[min(
                j + target_block,
                target_limits.shape[0] - 1
            )]],
            target_limits[j:j + target_block + 1] - target_limits[j]
        ) for i in range(0, src.shape[1], feature_block) \
            for j in range(0, dest.shape[1], target_block)
    )

    ent = numpy.empty((src_num, dest_num), dtype=numpy.float32)
    for i, e in enumerate(gen):
        row = i // col_block * blocksize[0]
        col = i % col_block * blocksize[1]

        # H(y|x) = H(x, y) - H(x)
        e -= feature_entropy[
            row * feature_column:row * feature_column + feature_block,
            None
        ]
        minus = numpy.count_nonzero(e < 0)
        if minus > 0:
            # 正常情况下计算出来的条件熵都大等于0，小于0可能是由于计算精度造成的
            logging.warning('{}/{} conditional entropy < 0'.format(
                minus,
                e.size
            ))

        # 归并同一组方言对多个特征和目标的条件熵
        ent[row:row + blocksize[0], col:col + blocksize[1]] = numpy.sum(
            numpy.min(
                e.reshape(
                    min(blocksize[0], ent.shape[0] - row),
                    feature_column,
                    min(blocksize[1], ent.shape[1] - col),
                    column
                ),
                axis=1
            ),
            axis=-1
        )

        if (i + 1) % 10 == 0:
            logging.info('finished {} blocks'.format(i + 1))

    logging.info('done. finished {} blocks'.format(i + 1))
    if numpy.any(numpy.isnan(ent)):
        logging.warning('result conditional entropy contains NaN')

    return ent

def normalize_sim(sim):
    '''正则化相似度矩阵到取值 [-1, 1] 区间的对称阵'''

    # 和转置取平均构造对称阵
    sim = (sim + sim.T) / 2
    # 除以对角线元素的平方根缩放到 [-1, 1] 区间，前提是对角线元素的相似度最大
    d = numpy.sqrt(numpy.diagonal(sim))
    sim /= d[:, None]
    sim /= d[None, :]

    # 如果有元素异常超出 [-1, 1] 区间，强制裁剪到 [-1, 1]
    overflow = numpy.count_nonzero((sim < -1) | (sim > 1))
    if overflow > 1:
        logging.warning('{}/{} similarity out of [-1, 1], clip'.format(
            overflow,
            sim.size
        ))
        sim = numpy.clip(sim, -1, 1)

    return sim

def sim2dist(sim):
    '''
    相似度矩阵转换成距离矩阵.

    假设相似阵的元素是欧氏空间向量的内积 sij = xi * xj，
    因此 dij^2 = (xi - xj)^2 = xi^2 + xj^2 - 2xi * xj = sii + sjj - 2sij
    '''

    d2 = numpy.diagonal(sim)
    d2 = -2 * sim + d2[:, None] + d2[None, :]

    # 有少量元素 < 0 是由于计算相似度的时候取近似导致的，强制为0
    minus = numpy.count_nonzero(d2 < 0)
    if minus > 0:
        logging.warning('{}/{} distance square < 0, clip to 0'.format(
            minus,
            d2.size
        ))
        d2 = numpy.maximum(d2, 0)

    return numpy.sqrt(d2)

def dist2sim(dist):
    '''距离矩阵转换成相似度矩阵'''

    max_sqrt = numpy.sqrt(numpy.max(dist, axis=0))
    return 1 - dist / max_sqrt[:, None] / max_sqrt[None, :]

def plot_map(
    base,
    points,
    boundary=None,
    pc=None,
    ax=None,
    figsize=None,
    **kwargs
):
    '''
    绘制指定地区的方言点地图

    如果指定了方言点主成分，通过方言点的颜色来表示方言之间的相似度，
    方言点的颜色由相似度矩阵的主成分确定，可以近似表示相似度矩阵。
    如果指定了方言点分类标签，则根据标签确定方言点颜色。
    '''

    if boundary is None:
        boundary = base.unary_union

    # 根据指定的地理边界筛选边界内部的点
    mask = points.within(boundary).values

    # 根据筛选得到的点集确定绘制边界
    minx, miny, maxx, maxy = points[mask].total_bounds
    xmargin = 0.05 * (maxx - minx)
    ymargin = 0.05 * (maxy - miny)
    minx -= xmargin
    maxx += xmargin
    miny -= ymargin
    maxy += ymargin

    # 绘制地图底图
    ax = base.boundary.plot(color='gray', ax=ax, figsize=figsize)
    ax.set_xlim(minx, maxx)
    ax.set_ylim(miny, maxy)

    if pc is not None:
        # 根据相似度矩阵的主成分计算方言点颜色，降维后的相似度矩阵决定方言点的 Lab 色度
        # 缩放传入的颜色值使其接近高斯分布，标准差为0.4的高斯分布98%以上落在 [-1, 1] 区间
        pc = StandardScaler().fit_transform(pc[:, :2]) * 0.4

        # 对绘制边界内的方言点稍微改变一下颜色分布，增加对比度，但保证变换后的均值不变
        inner_pc = pc[mask]
        inner_mean = numpy.mean(inner_pc, axis=0)
        scale = (pc - inner_mean) * 2 + inner_mean
        weight = numpy.exp(-numpy.sum(numpy.square(pc - inner_mean), axis=1) * 3)
        mix = weight[:, None] * scale + (1 - weight[:, None]) * pc

        # 把变换后的主成分拉伸到 Lab 色度的表示范围 [-100, 100]，亮度固定50
        lab = numpy.empty((points.shape[0], 3), dtype=numpy.float32)
        lab[:, 0] = 50
        lab[:, 1:] = mix * 50
        if numpy.any(numpy.abs(lab[mask, 1:]) > 50):
            logging.warning(
                '{}/{} points are out of [-50, 50], whose color may not show properly'.format(
                    numpy.count_nonzero(numpy.any(
                        numpy.abs(lab[mask, 1:]) > 50,
                        axis=1
                    )),
                    inner_pc.shape[0]
                )
            )

        # Lab 色系转 RGB 色系用于显示
        rgb = numpy.clip(colorspacious.cspace_convert(lab, 'CIELab', 'sRGB1'), 0, 1)
        points.plot(ax=ax, color=rgb, **kwargs)

    else:
        # 根据标签确定方言点颜色
        points.plot(ax=ax, **kwargs)

    return ax

def plot_heatmap(dist, labels, **kwargs):
    '''绘制方言相似度矩阵热度图'''

    # 根据距离矩阵层级聚类
    linkage = scipy.cluster.hierarchy.linkage(
        dist[numpy.triu_indices_from(dist, 1)],
        method='average',
        optimal_ordering=True
    )

    # 根据聚类结果重新排列距离矩阵，距离越短的点顺序越靠近
    leaves = scipy.cluster.hierarchy.leaves_list(linkage)

    return seaborn.heatmap(
        dist[leaves][:, leaves],
        square=True,
        xticklabels=labels[leaves],
        yticklabels=labels[leaves],
        **kwargs
    )

def dendro_heat(
    dist,
    labels,
    linkagefun=scipy.cluster.hierarchy.average,
    width=1000,
    height=1000
):
    '''
    根据距离矩阵绘制带热度图的树状图
    '''

    # 绘制树状图
    # 由于 Plotly 不接受预计算的距离矩阵，需要使用自定义距离函数，这个函数的返回值是距离矩阵的上三角阵
    dendro = plotly.figure_factory.create_dendrogram(
        dist,
        distfun=lambda x: x[numpy.triu_indices_from(x, 1)],
        linkagefun=linkagefun
    )
    for d in dendro.data:
        d.yaxis = 'y2'

    # 绘制热度图
    leaves = dendro.layout.xaxis.ticktext.astype(numpy.int32)
    heat = plotly.express.imshow(
        dist[leaves][:, leaves],
        x=labels[leaves],
        y=labels[leaves],
    )
    heat.data[0].x = dendro.layout.xaxis.tickvals
    heat.data[0].y = dendro.layout.xaxis.tickvals

    # 合并树状图和热度图，并调整大小占比
    fig = dendro
    fig.add_trace(heat.data[0])
    fig.update_layout(width=width, height=height)
    fig.update_layout(xaxis={'ticktext': labels[leaves]})
    fig.update_layout(yaxis={'domain': [0, 0.8], 'ticktext': labels[leaves]})
    fig.update_layout(yaxis2={'domain': [0.8, 1]})

    return fig

def label_map(
    latitudes,
    longitudes,
    labels,
    tips,
    zoom=5,
    tiles='https://wprd01.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scl=2&style=7&x={x}&y={y}&z={z}',
    attr='AutoNavi'
):
    '''绘制带分类坐标点的地图'''

    # 绘制底图
    mp = folium.Map(
        location=(numpy.mean(latitudes), numpy.mean(longitudes)),
        zoom_start=zoom,
        tiles=tiles,
        attr=attr
    )

    # 添加坐标点
    for i in range(latitudes.shape[0]):
        # 根据标签序号分配一个颜色
        r = (labels[i] * 79 + 31) & 0xff
        g = (labels[i] * 37 + 43) & 0xff
        b = (labels[i] * 73 + 17) & 0xff

        folium.CircleMarker(
            (latitudes[i], longitudes[i]),
            radius=5,
            tooltip='{} {}'.format(labels[i], tips[i]),
            color='#{:02x}{:02x}{:02x}'.format(r, g, b),
            fill=True
        ).add_to(mp)

    return mp


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
    pandas.DataFrame(chisq, index=ids, columns=ids) \
        .to_csv('chi2.csv', line_terminator='\n')

    ent = entropy(data.values, parallel=4)
    pandas.DataFrame(ent, index=ids, columns=ids) \
        .to_csv('entropy.csv', line_terminator='\n')
