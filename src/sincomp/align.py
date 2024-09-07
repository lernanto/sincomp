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


import itertools
import logging
import numpy
import opencc
import os
import pandas
import scipy.sparse
import scipy.sparse.linalg
import sklearn.cluster
import sklearn.compose
import sklearn.decomposition
import sklearn.feature_extraction.text
import sklearn.linear_model
import sklearn.metrics
import sklearn.preprocessing
import sklearn.pipeline
import typing

from .preprocess import transform


def prepare(
    dataset: pandas.DataFrame,
    chars: pandas.Series | numpy.ndarray[str] | None = None
) -> tuple[
    scipy.sparse.csr_matrix,
    numpy.ndarray[str],
    numpy.ndarray[str],
    numpy.ndarray[str] | None
]:
    """
    把数据集长表预处理成宽表

    Parameters:
        dataset: 数据集长表，必须包含字段 cid, initial, final, tone，
            如果 `chars` 为空，还必须包含 character 字段用于生成字表
        chars: cid 到字形的映射表，以 cid 为索引，在不包含索引的情况下，默认索引为从 0 开始编号

    Returns:
        matrix: 数据集变换并编码后的特征矩阵列表，每行代表一个字
        cids: transformed 中包含的字的 ID，顺序和 `matrix` 相同
        simplified: transformed 中包含的字表的简体
        traditional: transformed 中包含的字表的繁体，如原始字表未提供繁体，则为空
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
        # 从 dataset 生成字表，默认为繁体字
        chars = dataset[['cid', 'character']] \
            .dropna() \
            .sort_values(['cid', 'character']) \
            .drop_duplicates('cid') \
            .set_index('cid')['character']
        cids = chars.index

    else:
        # dataset 和 chars 的字取交集
        chars = pandas.Series(chars)
        cids = transformed.index.intersection(
            chars.dropna().index.drop_duplicates()
        ).sort_values()
        chars = chars.loc[cids]

    transformed = transformed.loc[cids]
    matrix = sklearn.compose.make_column_transformer(
        *[(sklearn.pipeline.make_pipeline(
            sklearn.feature_extraction.text.CountVectorizer(
                lowercase=False,
                tokenizer=str.split,
                token_pattern=None,
                stop_words=None,
                binary=True
            ),
            sklearn.preprocessing.Normalizer('l2')
        ), i) for i in range(transformed.shape[1])]
    ).fit_transform(transformed)

     # 假设字表为繁体，尝试转成简体
    simplified = chars.map(opencc.OpenCC('t2s').convert, na_action='ignore')
    if (chars != simplified).sum() < 3:
        # 绝大多数字转化后相同，认为原始字表为简体
        simplified = chars.values
        traditional = None
    else:
        simplified = simplified.values
        traditional = chars.values

    return matrix, cids.values, simplified, traditional

def align_chars(
    *char_lists: list[tuple[numpy.ndarray[str] | None, numpy.ndarray[str] | None]]
) -> tuple[numpy.ndarray[str], numpy.ndarray[int]]:
    """
    根据繁简字形构造数据集间的字对应表

    Parameters:
        char_lists: 方言数据集字形列表，每个数据集包含简体字表和繁体字表，
            简体和繁体顺序相同，且不能同时为空

    Returns:
        chars: 对齐后的字表，优先用简体，如果所有输入字表均为繁体，则为繁体
        indices: `chars` 中的每个字在每个数据集字表中的位置，列数等于 char_lists 长度，
            如某数据集缺某字，相应位置为 -1

    如果某数据集的字表具备简体和繁体，根据两种字形和其他数据集严格对齐；
    如果只提供了简体或繁体，只根据提供的字体对齐。
    """

    chars = pandas.DataFrame(columns=['simplified', 'traditional'], dtype=str)
    for i, (s, t) in enumerate(char_lists):
        if s is not None and t is not None:
            # 字表具备简体和繁体，根据两者严格对齐
            chars = chars.merge(
                pandas.DataFrame({
                    'simplified': s,
                    'traditional': t,
                    i: range(len(s))
                }),
                how='outer',
                on=['simplified', 'traditional']
            )
        elif s is not None:
            # 只有简体，根据简体对齐
            chars = chars.merge(
                pandas.DataFrame({'simplified': s, i: range(len(s))}),
                how='outer',
                on='simplified'
            )
        elif t is not None:
            # 只有繁体，根据繁体对齐
            chars = chars.merge(
                pandas.DataFrame({'traditional': t, i: range(len(t))}),
                how='outer',
                on='traditional'
            )
        else:
            raise ValueError('Either of implified and titional characters must be valid.')

    return (
        # 为尽可能把同形字归到一组，以简体字为准，除非简体为空
        chars['simplified'].where(
            chars['simplified'].notna(),
            chars['traditional']
        ).fillna('').values,
        chars.loc[:, range(len(char_lists))].fillna(-1).astype(int).values
    )

def polyphone_distance(
    chars: numpy.ndarray[str],
    indices: numpy.ndarray[int],
    *matrices,
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
        chars: 全部数据集对齐后的字表
        indices: `chars` 中的每个字在 `matrices` 中的行索引，列数等于 matrices 长度，
            -1 表示对应数据集无该字
        matrices: 数据集矩阵列表，每个元素为一个数据集字音的稀疏编码矩阵，每行代表一个字
        emb_size: 矩阵分解使用的字音向量长度
        metric: 计算字向量距离的函数，接受2个参数，各为 char_num * emb_size 的矩阵，
            返回长度为 char_num 的距离数组

    Returns:
        dataset_index1, dataset_index2: 指明后面的距离为哪两个数据集的字
        char_index1, char_index2: 指明后面的距离为数据集中各自哪个位置的字
        polyphone_chars: 上述位置的字形
        distances: 上述两个字的距离。以上只包含多音字，即字表中重复出现的字，
            且只包含不同数据集之间字形相同的字

    1. 取所有数据集中单音字，按字拼接为一个大特征矩阵。
    2. 对上述矩阵执行矩阵分解，得到变换矩阵。
    3. 分别对每个数据集中的多音字，使用上一步得到的变换矩阵执行矩阵分解降维，得到字音向量。
    4. 计算不同数据集的多音字的向量之间的距离。
    """

    chars = numpy.asarray(chars)
    indices = numpy.asarray(indices)

    # 检索字表中在所有数据集均为单音字的字，作为训练集
    monophone_mask = pandas.Series(chars).groupby(chars).transform('count') == 1

    monophones = indices[monophone_mask & numpy.all(indices >= 0, axis=1)]
    if monophones.shape[0] == 0:
        raise ValueError('Common monophones between datasets is empty.')

    mat = scipy.sparse.hstack(
        [m[monophones[:, i]] for i, m in enumerate(matrices)]
    )

    # 对单音字矩阵 SVD 矩阵分解
    _, _, vt = scipy.sparse.linalg.svds(mat, emb_size)
    # 为每个数据集计算读音编码到字音向量的变换，是 VT 的子矩阵的伪逆
    limits = numpy.cumsum([0] + [m.shape[1] for m in matrices])
    trans = []
    for i in range(len(matrices)):
        trans.append(numpy.linalg.pinv(vt[:, limits[i]:limits[i + 1]].T).T)

    dataset_index1 = [numpy.empty(0, dtype=int)]
    dataset_index2 = [numpy.empty(0, dtype=int)]
    char_index1 = [numpy.empty(0, dtype=int)]
    char_index2 = [numpy.empty(0, dtype=int)]
    polyphone_chars = [numpy.empty(0, dtype=str)]
    distances = [numpy.empty(0, dtype=float)]

    for (i, m1), (j, m2) in itertools.combinations(enumerate(matrices), 2):
        # 数据集两两之间，为每种可能的多音字对应关系计算距离
        # 只计算在任一数据集为多音字的字
        mask = ~monophone_mask & (indices[:, i] >= 0) & (indices[:, j] >= 0)
        idx = indices[mask]
        if idx.shape[0] > 0:
            dist = metric(
                m1[idx[:, i]] * trans[i],
                m2[idx[:, j]] * trans[j]
            )

            dataset_index1.append(numpy.full(idx.shape[0], i))
            dataset_index2.append(numpy.full(idx.shape[0], j))
            char_index1.append(idx[:, i])
            char_index2.append(idx[:, j])
            polyphone_chars.append(chars[mask])
            distances.append(dist)

    return (
        numpy.concatenate(dataset_index1),
        numpy.concatenate(dataset_index2),
        numpy.concatenate(char_index1),
        numpy.concatenate(char_index2),
        numpy.concatenate(polyphone_chars),
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
    numpy.ndarray[int]
]:
    """
    根据各方言多音字之间的距离进行聚类

    Parameters:
        dataset_index1, dataset_index2: 指明后面的距离为哪两个数据集的字
        char_index1, char_index2: 指明后面的距离为数据集中各自哪个位置的字
        chars: 上述位置的字形
        distances: 上述位置的字在对应方言之间的距离
        max_distance: 代表非常大的距离，使两个字不可能合为一类，应设置成远大于距离函数的最大值
        distance_threshold: 距离均值小于该值的两组字会合为一类

    Returns:
        dataset_indeces: 数据集的位置列表
        char_indeces: 字的位置列表
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

    dataset_indeces = [numpy.empty(0, dtype=int)]
    char_indeces = [numpy.empty(0, dtype=int)]
    clusters = [numpy.empty(0, dtype=int)]

    # 为尽可能把同形字归到一组，以简体字为准，除非简体为空
    for _, g in polyphones.groupby('character'):
        # 为单个多音字构造距离矩阵
        dist = pandas.pivot_table(
            g,
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
        clusters.append(cls)

    return (
        numpy.concatenate(dataset_indeces),
        numpy.concatenate(char_indeces),
        numpy.concatenate(clusters)
    )

def align(
    *datasets: list[tuple[
        pandas.DataFrame,
        pandas.Series | numpy.ndarray[str] | None
    ]],
    emb_size: int = 10
) -> list[pandas.DataFrame]:
    """
    根据读音对齐多个方言数据集中的多音字

    Parameters:
        datasets: 数据集的列表，每个数据集为如下的二元组：
            - dataset: 方言读音数据集长表，必须包含字段 cid, initial, final, tone，
                如果 `chars` 为空，还必须包含 character 字段用于生成字表
            - chars: cid 到字形的映射表，以 cid 为索引，在不包含索引的情况下，默认索引为从 0 开始编号。
                在多音字的情况下，会包含重复的字
        emb_size: 矩阵分解使用的字音向量长度

    Returns:
        char_lists: 字映射表的列表，顺序和 datasets 一致。每个字表包含了对应 `chars` 所有 cid 非空的字，
            索引为 cid，包含如下列：
            - simplified: 该字的简体，一般不为空
            - traditional: 该字的繁体，所有数据集都未提供该字的繁体时为空
            - label: 该字的新 ID
    """

    datasets = [prepare(d, c) for d, c in datasets]
    chars, indices = align_chars(*[(s, t) for _, _, s, t in datasets])

    data = polyphone_distance(
        chars,
        indices,
        *[m for m, _, _, _ in datasets],
        emb_size=emb_size
    )
    dataset_indeces, char_indeces, clusters = cluster(*data)

    # 对每个数据集的字表，建立原始字 ID 到对齐后的字 ID 的映射
    char_lists = []
    for i, (_, cids, s, t) in enumerate(datasets):
        # 单音字的聚类均为 0
        chr = pandas.DataFrame({
            'simplified': s,
            'traditional': t,
            'character': '',
            'label': 0
        }, index=cids)

        # 只在一个数据集中出现的多音字保持该数据集的划分
        valid = (indices >= 0).astype(int)
        idx = numpy.unique(indices[valid[:, i] == numpy.sum(valid, axis=1), i])
        chr.iloc[idx, chr.columns.get_loc('label')] = idx

        mask = (dataset_indeces == i)
        chr.iloc[char_indeces[mask], chr.columns.get_loc('label')] = clusters[mask]

        mask = indices[:, i] >= 0
        chr.iloc[indices[mask, i], chr.columns.get_loc('character')] = chars[mask]
        chr['label'] = chr['character'] + '\0' + chr['label'].astype(str)
        char_lists.append(chr)

    # 对字 ID 重新编码，使每个字的 ID 均不同
    encoder = sklearn.preprocessing.LabelEncoder().fit(
        pandas.concat(char_lists, axis=0, ignore_index=True)['label'].sort_values()
    )

    for chr in char_lists:
        chr['label'] = encoder.transform(chr['label'])
        chr.drop('character', axis=1, inplace=True)

    return char_lists

def annotate(
    embeddings: numpy.ndarray[float],
    chars: numpy.ndarray[str],
    data: numpy.ndarray[str],
    data_chars: numpy.ndarray[str]
) -> numpy.ndarray[int]:
    """
    为未区分的多音字标注相应的读音标记

    Parameters:
        embeddings: chars 对应的字向量，行数和 chars 相同
        chars: 基础数据集的字形列表，多音字会出现多次
        data: 待标注多音字的一个方言的字音数据，每行为一个待标注的字音，缺失值用空字符串表示
        data_chars: data 中每行对应的字形，缺失值用空字符串表示

    Returns:
        labels: 标注 data 每行对应 chars 的位置，如 data_chars 中的字在 chars 中不存在，
            则相应标注为 -1

    取 data 中字对齐 chars 中的对应单音字，使用线性回归拟合 embeddings 中对应的字向量，
    由此得到一个把 data 中的字音映射到字向量的线性模型。使用该模型把 data 中的多音字也映射为字向量，
    然后计算和 embeddings 中哪个字最接近，即标注为该字。
    """

    embeddings = numpy.asarray(embeddings)
    chars = pandas.Series(chars).reset_index(drop=True)
    data = numpy.asarray(data)
    data_chars = numpy.asarray(data_chars)

    # 待标注数据编码为稀疏矩阵
    matrix = sklearn.compose.make_column_transformer(
        *[(sklearn.pipeline.make_pipeline(
            sklearn.feature_extraction.text.CountVectorizer(
                lowercase=False,
                tokenizer=str.split,
                token_pattern=None,
                stop_words=None,
                binary=True
            ),
            sklearn.preprocessing.Normalizer('l2')
        ), i) for i in range(data.shape[1])]
    ).fit_transform(data)

    # 标注基础数据集中的单音字及其位置
    idx = chars[chars.groupby(chars).transform('count') == 1]
    idx = pandas.Series(idx.index, index=idx).reindex(data_chars)

    # 单音字只有一个 ID，直接标注
    labels = numpy.full(data.shape[0], -1, dtype=int)
    mask = idx.notna()
    labels[mask] = chars.index[idx[mask].astype(int)]

    if numpy.any(~mask):
        # 把待标注数据集中的多音字编码成字向量
        emb = sklearn.linear_model.LinearRegression() \
            .fit(matrix[mask], embeddings[idx[mask].astype(int)]) \
            .predict(matrix[~mask])

        # 为多音字计算最相似的字向量
        for i, j in enumerate(numpy.nonzero(~mask)[0]):
            idx2 = numpy.nonzero(chars == data_chars[j])[0]
            if len(data_chars[j]) > 0 and idx2.shape[0] > 0:
                dist = sklearn.metrics.pairwise.cosine_distances(
                    emb[i][None, :],
                    embeddings[idx2]
                )
                labels[j] = chars.index[idx2[numpy.argmin(dist[0])]]

    return labels

def align_no_cid(
    base: scipy.sparse.csr_matrix | pandas.DataFrame,
    simplified: numpy.ndarray[str] | pandas.Series,
    traditional: numpy.ndarray[str] | pandas.Series | None,
    *datasets: list,
    na_threshold: float = 0.5,
    emb_size: int = 10
) -> list[list[tuple[numpy.ndarray, numpy.ndarray[str], numpy.ndarray[str]]]]:
    """
    把没有字 ID 的数据集中的多音字对齐到基础数据集的字 ID

    Parameters:
        base: 基础数据集，每行代表一个字，多音字占多行
        simplified: base 中对应字的简体，多音字会出现多次。如类型为 pandas.Series，则索引为字 ID
        traditional: base 中对应字的繁体，为空的情况下只支持根据简体对齐
        datasets: 待对齐的无字 ID 数据集列表，每个元素为一个数据集，
            必须包含 did, initial, final, tone 字段
        na_threshold: base 中有读音的比例超过该值的字才会用于训练向量编码器。
            仅当 base 类型为 pandas.DataFrame 时有效，且无论是否参与训练编码器，最后所有字都会编码成字向量
        emb_size: 指定编码字向量的大小

    Returns:
        result: 标注结果列表，长度和 dataset 相同，每个元素又是一个列表，长度为对应数据集的方言数。
            每个元素为如下三元组：
            - labels: 标注的字 ID 列表，长度等于该方言的记录数。内容为该数据集的每条记录对应 chars 中字的位置，
                如 chars 为 pandas.Series，则为字 ID，如在 chars 中不存在该字形，则为 -1 或 None
            - simplified: labels 对应位置的字的简体，如原始数据集没有提供简体，由原始字形转换得出
            - traditional: labels 对应位置的字的繁体，如原始数据集没有提供繁体，由原始字形转换得出

    先把基础数据集降维编码成字向量，在对每个数据集中每个方言应用 annotate 标注多音字。
    """

    # 把基础数据集编码成字向量
    mask = base.isna().mean(axis=1) < na_threshold
    matrix = sklearn.compose.make_column_transformer(*[(
        sklearn.pipeline.make_pipeline(
            sklearn.feature_extraction.text.CountVectorizer(
                lowercase=False,
                tokenizer=str.split,
                token_pattern=None,
                stop_words=None,
                binary=True
            ),
            sklearn.preprocessing.Normalizer('l2')
        ), i) for i in range(base.shape[1])]).fit_transform(base.fillna(''))
    logging.debug(f'fit base dialect embeddings with {matrix[mask].shape} data.')
    emb = sklearn.decomposition.TruncatedSVD(emb_size).fit(matrix[mask]) \
        .transform(matrix)

    # 针对每个数据集中的每个方言标注多音字
    t2s = opencc.OpenCC('t2s')
    s2t = opencc.OpenCC('s2t')
    result = []
    for dataset in datasets:
        labels = []
        for data in dataset:
            sim = data['character'].map(t2s.convert, na_action='ignore')
            if (sim != data['character']).sum() < 3:
                # 认为待标注数据集为简体
                sim = data['character']
                trad = data['character'].map(s2t.convert, na_action='ignore')
                chars = simplified
                data_chars = sim
            else:
                trad = data['character']
                if traditional is None:
                    # 待标注数据集为繁体，但基础数据集未提供繁体，根据简体对齐
                    chars = simplified
                    data_chars = sim
                else:
                    # 使用繁体对齐
                    chars = traditional
                    data_chars = trad

            l = annotate(
                emb,
                chars,
                data[['initial', 'final', 'tone']].fillna(''),
                data_chars.fillna('')
            )
            # 输入字表含有字 ID，把位置转成字 ID
            if isinstance(chars, pandas.Series):
                l = numpy.where(l >= 0, chars.index[l].values, None)
            labels.append((l, sim.values, trad.values))

        result.append(labels)

    return result


if __name__ == '__main__':
    import argparse

    from . import datasets, preprocess


    parser = argparse.ArgumentParser('对齐指定的数据集生成新的汇总数据集')
    parser.add_argument(
        '-e',
        '--embedding-size',
        type=int,
        default=10,
        help='用于对齐多音字的字向量长度'
    )
    parser.add_argument(
        '--prefix',
        default='aligned',
        help='对齐后的数据集输出路径前缀'
    )
    parser.add_argument(
        '--charmap-output',
        default='charmap.csv',
        help='新旧字 ID 映射表输出文件'
    )
    parser.add_argument(
        '--char-output',
        default=os.path.join('aligned', '.characters'),
        help='对齐后的新字 ID 到各数据集的原字 ID 的映射文件'
    )
    parser.add_argument(
        '--dialect-output',
        default=os.path.join('aligned', '.dialects'),
        help='合并各数据集的方言信息文件'
    )
    parser.add_argument(
        'datasets',
        nargs='*',
        default=('CCR', 'MCPDict'),
        help='要对齐的数据集列表'
    )
    args = parser.parse_args()

    logging.getLogger().setLevel(logging.INFO)

    names = []
    dialects = []
    withcid = []
    nocid = []
    for name in args.datasets:
        data = datasets.get(name)
        if data is None:
            continue

        names.append(data.name)
        dialects.append(data.dialects)

        # 区分含有字 ID 的数据集和不含字 ID 的数据集
        if (chars := data.characters).shape[0] > 0:
            withcid.append((data, chars['character']))
        else:
            nocid.append(data)

    logging.info(
        f'align datasets {", ".join(names)}, '
        f'embedding size = {args.embedding_size}...'
    )

    dialects = pandas.concat(dialects, axis=0, keys=names) \
        .reset_index(names=['dataset', 'original_did']) \
        .rename_axis('did')

    # 对齐多音字，生成旧字 ID 到新字 ID 的映射表
    names = [d.name for d, _ in withcid]
    logging.info(f'align datasets {", ".join(names)}...')
    char_lists = align(*withcid, emb_size=args.embedding_size)

    # 整合成一个总的新旧字 ID 映射表
    charmap = pandas.concat(char_lists, axis=0, keys=names) \
        .rename_axis(['dataset', 'cid'])

    chars = pandas.pivot_table(
        charmap.reset_index(names=['dataset', 'old_cid']),
        values='old_cid',
        index='label',
        columns='dataset',
        aggfunc='first'
    ).reindex(names, axis=1)
    chars.loc[:, ['simplified', 'traditional']] = charmap.sort_index() \
        .groupby('label')[['simplified', 'traditional']].first()
    chars.rename_axis('cid', inplace=True)

    logging.info(f'annotate datasets without characer ID {", ".join([d.name for d in nocid])}...')
    base = pandas.concat(
        [preprocess.transform(
            d.dropna(subset=['cid', 'did']).replace({'cid': c['label']}),
            index='cid',
            columns='did',
            values=['initial', 'final', 'tone'],
            aggfunc=lambda x: ' '.join(x.dropna()) if x.count() > 0 else pandas.NA
        ) for (d, _), c in zip(withcid, char_lists)],
        axis=1
    ).reindex(chars.index)
    label_list = align_no_cid(
        base,
        chars['simplified'],
        chars['traditional'],
        *nocid
    )

    # 额外对不含字 ID 数据集特有的字编码
    unknown= []
    for labels in label_list:
        for l, s, t in labels:
            unknown.append(
                pandas.DataFrame({'label': l, 'simplified': s, 'traditional': t})
            )

    if unknown:
        unknown = pandas.concat(unknown, axis=0, ignore_index=True)
        unknown = unknown.loc[
            unknown['label'].isna(),
            ['simplified', 'traditional']
        ] \
            .dropna() \
            .sort_values(['simplified', 'traditional']) \
            .drop_duplicates() \
            .reset_index(drop=True) \
            .rename_axis('cid')
        unknown.index += chars.shape[0]
        chars = pandas.concat([chars, unknown], axis=0)

        # 根据生成的字 ID 对原来缺失的字赋值
        unknown_map = unknown.reset_index() \
            .set_index(['simplified', 'traditional'])['cid']
        for labels in label_list:
            for l, s, t in labels:
                l[l == None] = unknown_map.reindex(
                    pandas.MultiIndex.from_arrays([s[l == None], t[l == None]])
                )

    path = os.path.abspath(args.charmap_output)
    logging.info(f'save {charmap.shape[0]} character mapping to {path}...')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    charmap.to_csv(path, encoding='utf-8', lineterminator='\n')

    path = os.path.abspath(args.char_output)
    logging.info(f'save {chars.shape[0]} character information to {path}...')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    chars.to_csv(path, encoding='utf-8', lineterminator='\n')

    path = os.path.abspath(args.dialect_output)
    logging.info(f'save {dialects.shape[0]} dialect information to {path}...')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dialects.to_csv(path, encoding='utf-8', lineterminator='\n')

    # 把所有数据集中的所有有效数据映射到新方言 ID 和字 ID
    dialect_map = dialects[['dataset', 'original_did']].reset_index() \
        .set_index(['dataset', 'original_did'])['did']

    for (dataset, _), c in zip(withcid, char_lists):
        for did, data in dataset.items():
            did = dialect_map[(dataset.name, did)]
            data = data.assign(did=did, cid=data['cid'].map(c['label'])) \
                .reindex([
                    'did',
                    'cid',
                    'initial',
                    'final',
                    'tone',
                    'tone_category',
                    'note'
                ], axis=1) \
                .dropna(subset=['did', 'cid'])

            if data.shape[0] > 0:
                data['cid'] = data['cid'].astype(int)

                dirname = os.path.abspath(os.path.join(args.prefix, dataset.name))
                path = os.path.join(dirname, str(did))
                os.makedirs(dirname, exist_ok=True)
                logging.info(f'save {data.shape[0]} aligned data to {path}...')
                data.to_csv(path, index=False, encoding='utf-8', lineterminator='\n')

    # 为不含字 ID 的数据集加上字 ID 并保存
    for dataset, labels in zip(nocid, label_list):
        for (did, data), (cid, _, _) in zip(dataset.items(), labels):
            did = dialect_map[(dataset.name, did)]
            data = data.assign(did=did, cid=cid).reindex([
                'did',
                'cid',
                'initial',
                'final',
                'tone',
                'tone_category',
                'note'
            ], axis=1).dropna(subset=['did', 'cid'])

            if data.shape[0] > 0:
                data['cid'] = data['cid'].astype(int)

                dirname = os.path.abspath(os.path.join(args.prefix, dataset.name))
                path = os.path.join(dirname, str(did))
                os.makedirs(dirname, exist_ok=True)
                logging.info(f'save {data.shape[0]} aligned data to {path}...')
                data.to_csv(path, index=False, encoding='utf-8', lineterminator='\n')
