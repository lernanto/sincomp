# -*- coding: utf-8 -*-

"""
根据读音对齐不同语料集中的多音字

多音字是指一个数据集中字 ID 不同，但字形相同的字，由于不同数据集编码的 ID 不同，
需要判断哪些 ID 对应同一个字。

1. 取所有数据集中单音字，按字拼接为一个大特征矩阵。
2. 对上述矩阵执行矩阵分解，得到变换矩阵。
3. 分别对每个数据集中的多音字，使用上一步得到的变换矩阵执行矩阵分解降维，得到字音向量。
4. 计算不同数据集的多音字的向量之间的距离。
5. 对每个多音字，根据上述距离聚类，不同数据集的字 ID 属于同一个类的即为同一个字。
"""


import typing
import logging
import pandas
import numpy
import scipy.sparse
import scipy.sparse.linalg
import sklearn.cluster
import sklearn.compose
import sklearn.feature_extraction.text
import sklearn.metrics
import sklearn.preprocessing
import sklearn.pipeline
import itertools

from .preprocess import transform


def prepare(
    dataset: pandas.DataFrame,
    chars: pandas.Series | None = None
) -> tuple[pandas.DataFrame, pandas.Series]:
    """
    把数据集长表预处理成宽表

    Parameters:
        dataset: 数据集长表，必须包含字段 cid, initial, final, tone，
            如果 `chars` 为空，还必须包含 character 字段用于生成字表
        chars: cid 到字形的映射表，索引为 cid

    Returns:
        matrix: 数据集变换并编码后的特征矩阵列表，每行代表一个字
        transformed_chars: transformed 中包含的字表，顺序和 `matrix` 相同
    """

    transformed = transform(
        dataset.fillna({'initial': '', 'final': '', 'tone': ''}),
        index='cid',
        columns='did',
        values=['initial', 'final', 'tone'],
        aggfunc=' '.join,
        fill_value=''
    )

    if chars is None:
        # 从 dataset 生成字表
        transformed_chars = dataset[['cid', 'character']] \
            .dropna() \
            .sort_values(['cid', 'character']) \
            .drop_duplicates('cid') \
            .set_index('cid')['character']

    else:
        # dataset 和 chars 的字取交集
        index = transformed.index.intersection(
            chars.dropna().index.drop_duplicates()
        ).sort_values()
        transformed_chars = chars.loc[index]

    transformed = transformed.loc[transformed_chars.index]
    matrix = sklearn.compose.make_column_transformer(
        *[(sklearn.pipeline.make_pipeline(
            sklearn.feature_extraction.text.CountVectorizer(
                lowercase=False,
                tokenizer=str.split,
                token_pattern=None,
                stop_words=None
            ),
            sklearn.preprocessing.Normalizer('l2')
        ), i) for i in range(transformed.shape[1])]
    ).fit_transform(transformed)
    
    return matrix, transformed_chars

def polyphone_distance(
    *datasets: list[tuple[scipy.sparse.csr_matrix, numpy.ndarray[str]]],
    emb_size: int = 10,
    metric: typing.Callable = sklearn.metrics.pairwise.paired_cosine_distances
) -> tuple[
    numpy.ndarray[int],
    numpy.ndarray[int],
    numpy.ndarray[int],
    numpy.ndarray[int],
    numpy.ndarray[str],
    numpy.ndarray[float]
]:
    """
    计算不同语料集的多音字之间的距离

    Parameters:
        datasets: 数据集的列表，每个数据集为如下的二元组：
            - matrix: 数据集的编码矩阵，每行代表一个字
            - chars: 字表，字的数量和顺序和 matrix 相同，在多音字的情况下，会包含重复的字
        emb_size: 矩阵分解使用的字音向量长度
        metric: 计算字向量距离的函数，接受2个参数，各为 char_num * emb_size 的矩阵，
            返回长度为 char_num 的距离数组

    Returns:
        dataset_index1, dataset_index2: 指明后面的距离为哪两个数据集的字
        char_index1, char_index2: 指明后面的距离为数据集中各自哪个位置的字
        chars: 上述位置的字
        distances: 上述两个字的距离。以上只包含多音字，即字表中重复出现的字，
            且只包含不同数据集之间字形相同的字

    1. 取所有数据集中单音字，按字拼接为一个大特征矩阵。
    2. 对上述矩阵执行矩阵分解，得到变换矩阵。
    3. 分别对每个数据集中的多音字，使用上一步得到的变换矩阵执行矩阵分解降维，得到字音向量。
    4. 计算不同数据集的多音字的向量之间的距离。
    """

    datasets = [(m, pandas.Series(range(len(c)), index=c)) for m, c in datasets]

    # 构造单音字矩阵
    monophones = pandas.concat(
        [c.loc[c.index.drop_duplicates(keep=False)] for _, c in datasets],
        axis=1,
        join='inner'
    )
    if monophones.shape[0] == 0:
        raise ValueError('Common monophones between datasets must not be empty.')

    mat = scipy.sparse.hstack(
        [m[monophones.iloc[:, i]] for i, (m, _) in enumerate(datasets)]
    )

    # 对单音字矩阵 SVD 矩阵分解
    _, _, vt = scipy.sparse.linalg.svds(mat, emb_size)
    # 为每个数据集计算读音编码到字音向量的变换，是 VT 的子矩阵的伪逆
    limits = numpy.cumsum([0] + [m.shape[1] for m, _ in datasets])
    trans = []
    for i in range(len(datasets)):
        trans.append(numpy.linalg.pinv(vt[:, limits[i]:limits[i + 1]].T).T)

    dataset_index1 = [numpy.empty(0, dtype=int)]
    dataset_index2 = [numpy.empty(0, dtype=int)]
    char_index1 = [numpy.empty(0, dtype=int)]
    char_index2 = [numpy.empty(0, dtype=int)]
    chars = [numpy.empty(0, dtype=str)]
    distances = [numpy.empty(0, dtype=float)]

    for (i, (m1, c1)), (j, (m2, c2)) in itertools.combinations(enumerate(datasets), 2):
        # 数据集两两之间，为每种可能的多音字对应关系计算距离
        # 只计算在任一数据集为多音字的字
        polyphones = pandas.merge(
            c1.to_frame(),
            c2.to_frame(),
            left_index=True,
            right_index=True
        )
        polyphones = polyphones[
            polyphones.index.to_series().groupby(level=0).transform('count') > 1
        ]

        if polyphones.shape[0] > 0:
            idx1 = polyphones.iloc[:, 0]
            idx2 = polyphones.iloc[:, 1]
            dist = metric(m1[idx1] * trans[i], m2[idx2] * trans[j])

            dataset_index1.append(numpy.full(idx1.shape[0], i))
            dataset_index2.append(numpy.full(idx2.shape[0], j))
            char_index1.append(idx1)
            char_index2.append(idx2)
            chars.append(polyphones.index)
            distances.append(dist)

    return (
        numpy.concatenate(dataset_index1),
        numpy.concatenate(dataset_index2),
        numpy.concatenate(char_index1),
        numpy.concatenate(char_index2),
        numpy.concatenate(chars),
        numpy.concatenate(distances)
    )

def cluster(
    dataset_index1: numpy.ndarray[int],
    dataset_index2: numpy.ndarray[int],
    char_index1: numpy.ndarray[int],
    char_index2: numpy.ndarray[int],
    chars: numpy.ndarray[str],
    distances: numpy.ndarray[float],
    max_distance: float = 10,
    distance_threshold: float = 0.3,
) -> tuple[
    numpy.ndarray[int],
    numpy.ndarray[int],
    numpy.ndarray[str],
    numpy.ndarray[int]
]:
    """
    根据各方言多音字之间的距离进行聚类

    Parameters:
        dataset_index1, dataset_index2: 指明后面的距离为哪两个数据集的字
        char_index1, char_index2: 指明后面的距离为数据集中各自哪个位置的字
        chars: 上述位置的字
        distances: 上述位置的字在对应方言之间的距离
        max_distance: 代表非常大的距离，使两个字不可能合为一类，应设置成远大于距离函数的最大值
        distance_threshold: 距离均值小于该值的两组字会合为一类

    Returns:
        dataset_indeces: 数据集的位置列表
        char_indeces: 字的位置列表
        chars: 上述对应位置的数据集的字
        clusters: 上述对应位置的字的聚类 ID
    """

    polyphones = pandas.DataFrame({
        'dataset_index1': dataset_index1,
        'dataset_index2': dataset_index2,
        'char_index1': char_index1,
        'char_index2': char_index2,
        'character': chars,
        'distance': distances
    })
    polyphones['index1'] = polyphones[['dataset_index1', 'char_index1']] \
        .apply(tuple, axis=1)
    polyphones['index2'] = polyphones[['dataset_index2', 'char_index2']] \
        .apply(tuple, axis=1)

    dataset_indeces = [numpy.empty(0, dtype=int)]
    char_indeces = [numpy.empty(0, dtype=int)]
    chars = [numpy.empty(0, dtype=str)]
    clusters = [numpy.empty(0, dtype=int)]

    for char in polyphones['character'].drop_duplicates():
        # 为单个多音字构造距离矩阵
        dist = pandas.pivot_table(
            polyphones[polyphones['character'] == char],
            values='distance',
            index=('dataset_index1', 'char_index1'),
            columns=('dataset_index2', 'char_index2'),
            aggfunc='first'
        )
        idx = dist.index.union(dist.columns)
        dist = dist.reindex(idx).reindex(idx, axis=1)
        # 同个数据集的不同读音之间应设置非常大的距离，使算法不会把它们合并为一类
        dist = dist.where(dist.notna(), dist.transpose()).fillna(max_distance)

        # 根据距离聚类
        cls = sklearn.cluster.AgglomerativeClustering(
            n_clusters=None,
            metric='precomputed',
            linkage='average',
            distance_threshold=distance_threshold
        ).fit_predict(dist)

        dataset_indeces.append(dist.index.get_level_values(0).values)
        char_indeces.append(dist.index.get_level_values(1).values)
        chars.append(numpy.full(cls.shape[0], char))
        clusters.append(cls)

    return (
        numpy.concatenate(dataset_indeces),
        numpy.concatenate(char_indeces),
        numpy.concatenate(chars),
        numpy.concatenate(clusters)
    )

def align(
    *datasets: list[tuple[pandas.DataFrame, numpy.ndarray[str] | None]],
    emb_size: int = 10
) -> list[pandas.DataFrame]:
    """
    根据读音对其多个方言数据集中的多音字

    Parameters:
        datasets: 数据集的列表，每个数据集为如下的二元组：
            - dataset: 方言读音数据集长表，必须包含字段 cid, initial, final, tone，
                如果 `chars` 为空，还必须包含 character 字段用于生成字表
            - chars: cid 到字形的映射表，索引为 cid，在多音字的情况下，会包含重复的字
        emb_size: 矩阵分解使用的字音向量长度

    Returns:
        char_lists: 字映射表的列表，顺序和 datasets 一致。每个字表包含了对应 `chars` 所有 cid 非空的字，
            索引为 cid，label 列为该字新的 ID
    """

    datasets = [prepare(d, c) for d, c in datasets]
    data = polyphone_distance(*datasets, emb_size=emb_size)
    dataset_indeces, char_indeces, chars, clusters = cluster(*data)

    # 对每个数据集的字表，建立原始字 ID 到对齐后的字 ID 的映射
    char_lists = []
    for i, (_, c) in enumerate(datasets):
        # 单音字的聚类均为 0
        labels = pandas.Series(0, index=c.index)
        mask = (dataset_indeces == i)
        labels.iloc[char_indeces[mask]] = clusters[mask]
        char_lists.append(
            c.to_frame('character').assign(label=c + '-' + labels.astype(str))
        )

    # 对字 ID 重新编码，使每个字的 ID 均不同
    encoder = sklearn.preprocessing.LabelEncoder().fit(
        pandas.concat(char_lists, axis=0, ignore_index=True)['label'].sort_values()
    )

    for chars in char_lists:
        chars['label'] = encoder.transform(chars['label'])

    return char_lists

if __name__ == '__main__':
    import argparse
    import opencc

    from . import datasets


    parser = argparse.ArgumentParser('对齐指定的数据集生成新的汇总数据集')
    parser.add_argument(
        '-e',
        '--embedding-size',
        type=int,
        default=10,
        help='用于对齐多音字的字向量长度'
    )
    parser.add_argument(
        '-o',
        '--output',
        default='dataset.csv',
        help='对齐后的数据集输出文件'
    )
    parser.add_argument(
        '-c',
        '--char-output',
        default='char.csv',
        help='对齐后的新字 ID 到各数据集的原字 ID 的映射文件'
    )
    parser.add_argument(
        'dataset',
        nargs='*',
        default=('ccr', 'mcpdict'),
        help='要对齐的数据集列表'
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    t2s = opencc.OpenCC('t2s')

    dss = []
    for name in args.dataset:
        try:
            data = getattr(datasets, name)
        except AttributeError:
            # 当前只接受预定义的数据集
            logging.warning(f'{name} is not a predefined dataset, ignored.')
            continue

        if data is datasets.ccr:
            # 从数据集生成字表，下同
            # TODO: 为对齐多音字，当前把字形统一转换为简体字，应分别处理繁体和简体
            chars = data[['cid', 'character']].sort_values(['cid', 'character']) \
                .drop_duplicates('cid') \
                .dropna() \
                .set_index('cid')['character'] \
                .map(t2s.convert)

        elif data is datasets.mcpdict:
            chars = data['character'].drop_duplicates().dropna()
            chars = pandas.Series(
                chars.map(t2s.convert).values,
                index=chars.values
            )
            data = data.rename(columns={'character': 'cid'})

        elif data is datasets.zhongguoyuyan:
            chars = data.metadata['char_info']['character']

        dss.append((name, data, chars))

    logging.info(
        f'align datasets {", ".join([n for n, _, _ in dss])}, '
        f'embedding size = {args.embedding_size} ...'
    )

    # 对齐多音字，生成旧字 ID 到新字 ID 的映射表
    char_lists = align(
        *[(d, c) for _, d, c in dss],
        emb_size=args.embedding_size
    )

    # 把所有数据集中的所有有效数据映射到新字 ID
    dataset = pandas.concat(
        [d.assign(cid=d['cid'].map(c['label']))[[
            'did',
            'cid',
            'initial',
            'final',
            'tone',
            'note'
        ]] for (_, d, _), c in zip(dss, char_lists)],
        axis=0,
        ignore_index=True
    )
    dataset.dropna(subset='cid', inplace=True)
    dataset['cid'] = dataset['cid'].astype(int)

    # 整合成一个总的新旧字 ID 映射表
    names = [n for n, _, _ in dss]
    chars = pandas.pivot_table(
        pandas.concat(char_lists, axis=0, keys=names) \
            .rename(columns={'label': 'cid'}) \
            .reset_index(names=['dataset', 'old_cid']),
        values='old_cid',
        index='cid',
        columns='dataset',
        aggfunc='first'
    ).reindex(names, axis=1)

    chars.insert(0, 'character', pandas.concat(
        [i.map(c['character']) for (_, i), c in zip(chars.items(), char_lists)],
        axis=1
    ).bfill(axis=1).iloc[:, 0])

    logging.info(f'save {dataset.shape[0]} aligned data to {args.output} ...')
    dataset.to_csv(
        args.output,
        index=False,
        encoding='utf-8',
        lineterminator='\n'
    )
    logging.info(f'save {chars.shape[0]} character infomation to {args.char_output} ...')
    chars.to_csv(args.char_output, encoding='utf-8', lineterminator='\n')
