# -*- coding: utf-8 -*-

"""
计算方言之间的预测相似度.

如果 A 方言的音类能很大程度预测 B 方言的音类，表明 A 方言到 B 方言单向的相似度高。
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import logging
import pandas
import numpy
import scipy.sparse
import scipy.cluster.hierarchy
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib

from . import datasets
from . import preprocess


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
        sparse_output=True,
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
    src: datasets.Dataset | pandas.DataFrame | numpy.ndarray[str],
    dest: datasets.Dataset | pandas.DataFrame | numpy.ndarray[str] | None = None,
    feature_num: int = 3,
    blocksize: tuple[int, int] = (100, 100),
    parallel: int = 1
) -> pandas.DataFrame | numpy.ndarray[float]:
    """
    使用卡方检验计算方言之间的相似度

    Parameters:
        src: 源方言数据表
        dest: 目标方言数据表，为 None 时和 `src` 相同
        feature_num: `src` 和 `dest` 中特征数量，当 `src` 为 pandas.DataFrame 时，
            从 `src` 自动推导
        blocksize: 指定并行计算时每块数据的大小
        parallel: 并行计算的并行数

    Returns:
        chi2: 源方言和目标方言两两之间的条件熵矩阵，当 `src` 和 `dest` 为
            pandas.DataFrame 包含方言名称时，指定 `chi2` 的行列为相应名称

    思路为，如果 A 方言的声母 + 韵母能很大程度预测 B 方言的声母 + 韵母，说明 B 方言接近 A 方言。
    这种预测能力通过卡方检验来衡量，卡方值越大表示越相似。
    遍历 A 方言及 B 方言的声母、韵母、声调组合，取平均得到 A、B 方言之间的相似度评分。

    理论上由于自由度不同，卡方值不能直接比较，而应该比较相应的 p-value，但由于统计得出的卡方值非常大，
    导致计算出的 p-value 下溢为0，不能比较，另一方面由于自由度都比较大，卡方值近似于正态分布，
    可以把卡方值正则化到标准正态分布后进行比较。

    卡方检验对于 A、B 方言是对称的，即理论上相似度矩阵是对称阵，但由于计算精度的原因可能出现极小的误差导致不对称。
    可以取相似度矩阵和其转置的平均来强制保证对称性。
    """

    if dest is None:
        dest = src

    if isinstance(src, datasets.Dataset | pandas.DataFrame):
        index = src.columns.levels[0]
        src_num = index.shape[0]
        feature_num = src.columns.levels[1].shape[0]
        src = src.values
    else:
        index = None
        src_num = src.shape[1] // feature_num

    if isinstance(dest, datasets.Dataset | pandas.DataFrame):
        columns = dest.columns.levels[0]
        dest_num = columns.shape[0]
        dest = dest.values
    else:
        columns = None
        dest_num = dest.shape[1] // feature_num

    logging.info(
        f'compute X2 for {src_num} x {dest_num} dialects, '
        f'characters = {src.shape[0]}, features = {feature_num}, '
        f'block size = {blocksize}, parallel = {parallel}'
    )

    # 特征交叉
    if feature_num > 1:
        features = cross_features(src, feature_num)
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
        if feature_num > 1:
            targets = cross_features(dest, feature_num)
            targets = numpy.swapaxes(targets, 1, 2).reshape(targets.shape[0], -1)
        else:
            targets = dest

        targets, target_categories = encode_features(targets)
        target_limits = numpy.concatenate([[0], numpy.cumsum(target_categories)])
        target_probs = freq2prob(targets.sum(axis=0).A.flatten(), target_limits)

    # 分块并行计算特征到目标的卡方
    row_block = (src_num + blocksize[0] - 1) // blocksize[0]
    col_block = (dest_num + blocksize[1] - 1) // blocksize[1]

    logging.info(
        f'computing X2 for {feature_num} x {row_block} x {col_block} blocks, '
        f'block size = {blocksize} ...'
    )

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
                logging.info(f'finished {count} blocks')

        logging.info(f'done. finished {count} blocks')

    # 取多组特征卡方的均值
    if feature_column > 1:
        chisq /= feature_column

    return chisq if index is None and columns is None \
        else pandas.DataFrame(chisq, index=index, columns=columns)

def _entropy(
    features: scipy.sparse.csr_matrix,
    feature_limits: numpy.ndarray[int],
    targets: scipy.sparse.csr_matrix,
    target_limits: numpy.ndarray[int]
) -> numpy.ndarray[float]:
    """
    计算稀疏特征矩阵到目标矩阵的条件熵

    Parameters:
        features: 源方言编码的特征稀疏矩阵
        feature_limits: 表明特征矩阵中特征边界的数组
        targets: 目标方言编码的目标稀疏矩阵
        target_limits: 表明目标矩阵中目标边界的数组

    Returns:
        entropy: 每个特征到每个目标的条件熵矩阵

    X 到 Y 的条件熵是条件概率 P(Y|X) 的期望 H(Y|X) = E[-log P(Y|X)]。
    本函数通过构造 X 和 Y 的共现矩阵来计算 H(Y|X)，因为
    P(Y|X) * -log P(y|x) = f(x, y) / f(x) * -log f(x, y) / f(x)
    = (f(x, y) * log f(x) - f(x, y) * log f(x, y)) / f(x)
    不直接计算 P(Y|X)，而是使用稀疏矩阵计算一些中间值来减少计算量。
    """

    # 计算共现频次及其对数
    freq = features.T @ targets
    entropy = scipy.sparse.csc_matrix(freq)
    entropy.data = numpy.where(
        entropy.data == 0,
        0,
        entropy.data * numpy.log(entropy.data)
    )

    # 对共现矩阵的列分组求和，把目标不同取值的频次归并在一起
    freq = numpy.asarray(numpy.column_stack([numpy.sum(
        freq[:, target_limits[i]:target_limits[i + 1]],
        axis=1
    ) for i in range(target_limits.shape[0] - 1)]))
    entropy = numpy.asarray(numpy.column_stack([numpy.sum(
        entropy[:, target_limits[i]:target_limits[i + 1]],
        axis=1
    ) for i in range(target_limits.shape[0] - 1)]))

    # 计算特征频次的对数，然后减去共现频次的对数
    feature_entropy = numpy.where(freq == 0, 0, freq * numpy.log(freq))
    entropy = feature_entropy - entropy

    # 对共现矩阵的行分组求和，把特征不同取值的频次归并在一起
    freq = numpy.stack([numpy.sum(
        freq[feature_limits[i]:feature_limits[i + 1]],
        axis=0
    ) for i in range(feature_limits.shape[0] - 1)])
    entropy = numpy.stack([numpy.sum(
        entropy[feature_limits[i]:feature_limits[i + 1]],
        axis=0
    ) for i in range(feature_limits.shape[0] - 1)])

    # 频次对数除以样本总频次，得到真正的条件熵
    entropy /= freq
    invalid = numpy.count_nonzero(~(numpy.isfinite(entropy) & (entropy >= 0)))
    if invalid > 0:
        logging.warning(f'{invalid}/{entropy.size} invalid conditional entropy')

    return entropy

def entropy(
    src: datasets.Dataset | pandas.DataFrame | numpy.ndarray[str],
    dest: datasets.Dataset | pandas.DataFrame | numpy.ndarray[str] | None = None,
    feature_num: int = 3,
    blocksize: tuple[int, int] = (100, 100),
    parallel: int = 1
) -> pandas.DataFrame | numpy.ndarray[float]:
    """
    计算方言之间的条件熵

    Parameters:
        src: 源方言数据表
        dest: 目标方言数据表，为 None 时和 `src` 相同
        feature_num: `src` 和 `dest` 中特征数量，当 `src` 为 pandas.DataFrame 时，
            从 `src` 自动推导
        blocksize: 指定并行计算时每块数据的大小
        parallel: 并行计算的并行数

    Returns:
        entropy: 源方言和目标方言两两之间的条件熵矩阵，当 `src` 和 `dest` 为
            pandas.DataFrame 包含方言名称时，指定 `entropy` 的行列为相应名称
    """

    if dest is None:
        dest = src

    if isinstance(src, datasets.Dataset | pandas.DataFrame):
        index = src.columns.levels[0]
        src_num = index.shape[0]
        feature_num = src.columns.levels[1].shape[0]
        src = src.values
    else:
        index = None
        src_num = src.shape[1] // feature_num

    if isinstance(dest, datasets.Dataset | pandas.DataFrame):
        columns = dest.columns.levels[0]
        dest_num = columns.shape[0]
        dest = dest.values
    else:
        columns = None
        dest_num = dest.shape[1] // feature_num

    logging.info(
        f'compute conditional entropy for {src_num} sources '
        f'{dest_num} destinations, characters = {src.shape[0]}, '
        f'features = {feature_num}, block size = {blocksize}, parallel = {parallel}'
    )

    # 特征交叉
    if feature_num > 1:
        features = cross_features(src, feature_num)
        feature_column = features.shape[2]
        features = features.reshape(features.shape[0], -1)
    else:
        features = src
        feature_column = 1

    # 特征 one-hot 编码
    features, feature_categories = encode_features(features)
    feature_limits = numpy.concatenate([[0], numpy.cumsum(feature_categories)])

    # 预测目标编码
    targets, target_categories = encode_features(dest)
    target_limits = numpy.concatenate([[0], numpy.cumsum(target_categories)])

    # 计算特征到目标的条件熵
    row_block = (src_num + blocksize[0] - 1) // blocksize[0]
    col_block = (dest_num + blocksize[1] - 1) // blocksize[1]

    logging.info(
        f'computing conditional entropy for {row_block} x {col_block} blocks, '
        f'block size = {blocksize} ...'
    )

    # 分块并行计算联合熵
    feature_block = blocksize[0] * feature_column
    target_block = blocksize[1] * feature_num
    gen = joblib.Parallel(n_jobs=parallel)(
        joblib.delayed(_entropy)(
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

        # 归并同一组方言对多个特征和目标的条件熵
        ent[row:row + blocksize[0], col:col + blocksize[1]] = numpy.sum(
            numpy.min(
                e.reshape(
                    min(blocksize[0], ent.shape[0] - row),
                    feature_column,
                    min(blocksize[1], ent.shape[1] - col),
                    feature_num
                ),
                axis=1
            ),
            axis=-1
        )

        if (i + 1) % 10 == 0:
            logging.info(f'finished {i + 1} blocks')

    logging.info(f'done. finished {i + 1} blocks')
    return ent if index is None and columns is None \
        else pandas.DataFrame(ent, index=index, columns=columns)

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


if __name__ == '__main__':
    import os
    import argparse

    parser = argparse.ArgumentParser('根据指定方言数据集计算方言之间的预测相似度')
    parser.add_argument(
        '-m',
        '--method',
        choices=('chi2', 'entropy'),
        help='计算方言间相似度的方法，如果不指定，计算所有方法的结果'
    )
    parser.add_argument(
        '-o',
        '--output',
        help='输出路径，如果只有一个数据集及一个方法，为输出文件名，否则为输出目录'
    )
    parser.add_argument(
        'dataset',
        nargs='*',
        default=('ccr',),
        help='方言数据集名称或数据文件或目录路径'
    )
    args = parser.parse_args()

    if args.method is None:
        methods = 'chi2', 'entropy'
    else:
        methods = (args.method,)

    for dts in args.dataset:
        try:
            data = getattr(datasets, dts)
        except AttributeError:
            # 如果在数据集不在支持的列表中，视为数据文件或目录路径，文件为 CSV 格式
            if os.path.isdir(dts):
                # 目录，递归检索目录下的所有文件，视每个文件为一个方言数据，文件名为方言 ID
                data = datasets.FileDataset(path=dts)
                dts = os.path.basename(dts)
            else:
                data = pandas.read_csv(dts, dtype=str)
                dts = os.path.splitext(os.path.basename(dts))[0]

        data = preprocess.transform(
            data,
            index='cid',
            columns='did',
            values=['initial', 'final', 'tone'],
            aggfunc='first'
        ).fillna('')

        for method in methods:
            # 如果输出文件只有一个，使用指定的路径作为文件名，否则作为输出目录
            if len(args.dataset) > 1 or len(methods) > 1:
                output = os.path.join(
                    os.getcwd() if args.output is None else args.output,
                    f'{dts}_{method}.csv'
                )
            else:
                output = os.path.join(os.getcwd(), f'{dts}_{method}.csv') \
                    if args.output is None else args.output

            print(f'compute {method} between {dts} dialects -> {output}')

            os.makedirs(os.path.dirname(output), exist_ok=True)
            sim = globals()[method](data, parallel=4)
            sim.to_csv(output, lineterminator='\n')
