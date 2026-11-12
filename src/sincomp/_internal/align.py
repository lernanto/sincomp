# -*- coding: utf-8 -*-

"""
根据读音对齐不同语料集中的多音字

多音字是指一个数据集中字 ID 不同，但字形相同的字，由于不同数据集编码的 ID 不同，
需要判断哪些 ID 对应同一个字。

1. 初始先对齐所有数据集中的单音字，多音字不对齐，即每个数据集的每个多音字都看作一个单独的字
2. 对上述拼接的数据集执行矩阵分解，得到每个字的向量
3. 使用多音字的向量执行聚类，不同数据集的字 ID 属于同一个类的即为同一个字
"""


import logging
import numpy
import opencc
import pandas
import scipy.sparse
import scipy.sparse.linalg
import sklearn.cluster
import sklearn.compose
import sklearn.decomposition
import sklearn.feature_extraction.text
import sklearn.linear_model
import sklearn.metrics
import sklearn.pipeline
import typing

from . import factorize, preprocessing


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler())


def preprocess_chars(
    dataset: pandas.DataFrame,
    chars: pandas.Series | numpy.ndarray[str] | None = None
) -> tuple[
    numpy.ndarray[str | int],
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

    if chars is None:
        # 从 dataset 生成字表，默认为繁体字
        chars = dataset.loc[:, ['cid', 'character']] \
            .dropna() \
            .sort_values(['cid', 'character']) \
            .drop_duplicates('cid') \
            .set_index('cid')['character']
    elif isinstance(chars, numpy.ndarray):
        chars = pandas.Series(chars)

    # 假设字表为繁体，尝试转成简体
    simplified = chars.map(opencc.OpenCC('t2s').convert, na_action='ignore')
    if (chars != simplified).sum() < 3:
        # 绝大多数字转化后相同，认为原始字表为简体
        simplified = chars.values
        traditional = None
    else:
        simplified = simplified.values
        traditional = chars.values

    return chars.index.values, simplified, traditional

def align_chars(
    *char_lists: list[tuple[numpy.ndarray[str], numpy.ndarray[str] | None, numpy.ndarray[str] | None]]
) -> tuple[pandas.DataFrame, list[pandas.DataFrame]]:
    """
    根据繁简字形构造数据集间的字对应表

    Parameters:
        char_lists: 方言数据集字形列表，每个数据集包含字 ID 表、简体字表、繁体字表，
            简体和繁体顺序相同，且不能同时为空

    Returns:
        monophones: 对齐后的单音字表，包含以下字段：
            - character: 字形，优先使用简体，如果所有输入字表均为繁体，则为繁体
            - simplified: 简体字形
            - traditional: 繁体字形
            - 0, 1, ...: 每个数据集的字 ID，顺序和 `char_lists` 相同
        polyphones: 每个数据集的多音字表，长度和 char_lists 相同，每个元素包含以下字段：
            - character, simplified, traditional: 字形，和 `monophones` 相同
            - cid: 在对应数据集的字 ID

    如果某数据集的字表具备简体和繁体，根据两种字形和其他数据集严格对齐；
    如果只提供了简体或繁体，只根据提供的字体对齐。
    """

    chars = pandas.DataFrame(columns=['simplified', 'traditional'], dtype=str)
    for i, (cids, s, t) in enumerate(char_lists):
        if s is not None and t is not None:
            # 字表具备简体和繁体，根据两者严格对齐
            chars = chars.merge(
                pandas.DataFrame({
                    i: cids,
                    'simplified': s,
                    'traditional': t
                }),
                how='outer',
                on=['simplified', 'traditional']
            )
        elif s is not None:
            # 只有简体，根据简体对齐
            chars = chars.merge(
                pandas.DataFrame({i: cids, 'simplified': s}),
                how='outer',
                on='simplified'
            )
        elif t is not None:
            # 只有繁体，根据繁体对齐
            chars = chars.merge(
                pandas.DataFrame({i: cids, 'traditional': t}),
                how='outer',
                on='traditional'
            )
        else:
            raise ValueError('Either of implified and titional characters must be valid.')

    # 为尽可能把同形字归到一组，以简体字为准，除非简体为空
    chars['character'] = chars['simplified'].where(
        chars['simplified'].notna(),
        chars['traditional']
    )

    is_poly = chars['character'].duplicated(keep=False)
    monophones = chars[~is_poly]

    polyphones = []
    for i in range(len(char_lists)):
        polyphones.append(
            chars.loc[is_poly, ['character', 'simplified', 'traditional', i]] \
                .dropna(subset=i) \
                .drop_duplicates(i)
        )

    return monophones, polyphones

def polyphone_embs_svd(
    data: list[pandas.DataFrame],
    monophones: pandas.DataFrame,
    polyphones: list[pandas.DataFrame],
    embedding_size: int = 32
) -> list[numpy.ndarray[float]]:
    """
    使用 SVD 矩阵分解求解多音字的字向量

    Parameters:
        data: 方言数据集列表，每个元素为数据集长表，必须包含字段 did, cid, initial, final, tone
        monophones, polyphones: `align_chars` 的返回值
        embedding_size: 矩阵分解使用的字音向量长度

    Returns:
        polyphone_embs: 多音字向量列表，每个元素为 `polyphones` 对应位置的多音字向量

    1. 取所有数据集中单音字，按字拼接为一个大特征矩阵
    2. 对上述矩阵执行矩阵分解，得到变换矩阵
    3. 分别对每个数据集中的多音字，使用上一步得到的变换矩阵执行矩阵分解降维，得到字音向量
    """

    assert len(data) == len(polyphones)

    # 把每个数据集长表转换成宽表
    data = [preprocessing.transform(
        d,
        index='cid',
        columns='did',
        values=['initial', 'final', 'tone'],
        aggfunc=lambda x: ' '.join(x.dropna())
    ) for d in data]

    # 先用对齐的单音字构造包含所有数据集的读音编码矩阵
    monophones = monophones.dropna(subset=range(len(data)))
    transformers = []
    matrices = []
    for i, d in enumerate(data):
        t = sklearn.compose.make_column_transformer(
            *[sklearn.pipeline.make_pipeline(
            sklearn.feature_extraction.text.CountVectorizer(
                lowercase=False,
                tokenizer=str.split,
                token_pattern=None,
                stop_words=None,
                binary=True,
                dtype=numpy.float32
            ), i) for i in range(d.shape[1])]
        )
        m = t.fit_transform(d.loc[monophones.loc[:, i]].fillna(''))
        transformers.append(t)
        matrices.append(m)

    # 对单音字矩阵实施 SVD 分解
    _, _, vt = scipy.sparse.linalg.svds(
        scipy.sparse.hstack(matrices),
        embedding_size
    )

    # 计算每个数据集的多音字向量，求对应的稀疏线性方程的最小二乘解
    limits = numpy.cumsum([0] + [m.shape[1] for m in matrices])
    embs = []
    for i, (p, d, t) in enumerate(zip(polyphones, data, transformers)):
        vti = vt[:, limits[i]:limits[i + 1]]
        embs.append(numpy.linalg.solve(
            vti @ vti.T,
            vti @ t.transform(d.loc[p.loc[:, i]].fillna('')).T
        ).T)

    return embs

def align(
    *datasets: list[tuple[
        pandas.DataFrame,
        pandas.Series | numpy.ndarray[str] | None
    ]],
    encoder: str = 'svd',
    embedding_size: int = 32,
    metric: typing.Callable = sklearn.metrics.pairwise.euclidean_distances,
    distance_threshold: float = 10
) -> tuple[pandas.DataFrame, list[pandas.Series]]:
    """
    根据读音对齐多个方言数据集中的多音字

    Parameters:
        datasets: 数据集的列表，每个数据集为如下的二元组：
            - dataset: 方言读音数据集长表，必须包含字段 did, cid, initial, final, tone，
                如果 `chars` 为空，还必须包含 character 字段用于生成字表
            - chars: cid 到字形的映射表，以 cid 为索引，在不包含索引的情况下，默认索引为从 0 开始编号。
                在多音字的情况下，会包含重复的字
        encoder: 求字向量算法：
            - svd: SVD 矩阵分解
            - mf: 带缺失值的矩阵分解
        embedding_size: 矩阵分解使用的字音向量长度
        metric: 计算字向量距离的函数，接受1个参数，为 char_num * embedding_size 的矩阵，
            返回 char_num * char_num 的距离矩阵
        distance_threshold: 距离均值小于该值的两组字会合为一类

    Returns:
        chars: 对齐后的字表，每行为一个字，以新的字 ID 为索引，包含如下列：
            - simplified: 该字的简体，一般不为空
            - traditional: 该字的繁体，所有数据集都未提供该字的繁体时为空
            - 0, 1, ...: 该字在每个数据集的字 ID，为空表示数据集不包含该字
        charmaps: 字映射表的列表，顺序和 datasets 一致。索引为 cid，值为新字 ID

    1. 初始先对其所有数据集中的单音字，多音字不对齐，即每个数据集的每个多音字都看作一个单独的字
    2. 对上述拼接的数据集执行矩阵分解，得到每个字的向量
    3. 使用多音字的向量执行聚类，不同数据集的字 ID 属于同一个类的即为同一个字
    """

    max_dist = numpy.finfo(numpy.float32).max
    char_lists = [preprocess_chars(d, c) for d, c in datasets]
    monophones, polyphones = align_chars(*char_lists)

    if encoder == 'svd':
        # SVD 分解计算多音字向量
        embs = polyphone_embs_svd(
            [d for d, _ in datasets],
            monophones,
            polyphones,
            embedding_size=embedding_size
        )
        embs = numpy.concatenate(embs, axis=0)
        distances = metric(embs)

    elif encoder == 'fm':
        # 先根据初始的对齐结果对所有数据集中的字重新编码，以便执行矩阵分解
        charmap = pandas.concat(
            [monophones] + polyphones,
            axis=0,
            ignore_index=True
        )
        charmap.set_index(charmap.index.astype(str), inplace=True)
        data = []
        for i, (d, _) in enumerate(datasets):
            cids = charmap.loc[:, i].dropna()
            m = pandas.Series(cids.index, index=cids)

            newd = d.loc[:, ['initial', 'final', 'tone']].copy()
            # 方言 ID 增加数据集前缀，避免重复
            newd.insert(0, 'did', str(i) + d.loc[:, 'did'])
            newd.insert(1, 'cid', d.loc[:, 'cid'].map(m))
            data.append(newd)

        data = pandas.concat(data, axis=0, ignore_index=True).dropna()

        # 矩阵分解计算多音字向量
        wide = preprocessing.transform(
            data,
            index="cid",
            columns="did",
            values=["initial", "final", "tone"],
            aggfunc=lambda x: " ".join(x.dropna())
        )
        vectors, _ = preprocessing.features_to_vectors(wide.fillna(""))
        embs, _, _ = factorize.als(
            vectors,
            n_components=embedding_size
        )
        cids = numpy.asarray(wide.index)
        # 极少数多音字可能因为数据残缺而算不出向量
        idx = pandas.Series(range(len(cids)), index=cids) \
            .reindex(charmap.index[monophones.shape[0]:], fill_value=-1)
        embs = embs[idx]
        distances = metric(embs)

        # 对于无法算出字向量的多音字，保留每个数据集为一个独立的字不合并
        distances[idx < 0] = max_dist
        distances[:, idx < 0] = max_dist

    else:
        raise ValueError(f'invalid encoder {repr(encoder)}')

    # 只有相同字形的字可聚类，且相同数据集的字相互不可聚类，把聚类设成极大值避免聚类
    limits = numpy.cumsum([0] + [p.shape[0] for p in polyphones])
    for i in range(limits.shape[0] - 1):
        distances[limits[i]:limits[i + 1]][:, limits[i]:limits[i + 1]] = max_dist

    polyphones = pandas.concat(polyphones, axis=0, ignore_index=True)
    distances[
        polyphones[['character']].values != polyphones[['character']].values.T
    ] = max_dist

    # 根据距离聚类
    labels = sklearn.cluster.AgglomerativeClustering(
        n_clusters=None,
        metric='precomputed',
        linkage='average',
        distance_threshold=distance_threshold
    ).fit_predict(distances)

    # 根据聚类结果构建新的字 ID 映射表
    chars = pandas.concat(
        [monophones, polyphones.groupby(labels).first()],
        axis=0,
        ignore_index=True
    ).drop('character', axis=1)

    charmaps = []
    for i in range(len(datasets)):
        cids = chars.loc[:, i].dropna()
        charmaps.append(pandas.Series(cids.index, index=cids))

    return chars, charmaps

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
        *[(sklearn.feature_extraction.text.CountVectorizer(
            lowercase=False,
            tokenizer=str.split,
            token_pattern=None,
            stop_words=None,
            binary=True
        ), i) for i in range(data.shape[1])]
    ).fit_transform(data)

    # 标注基础数据集中的单音字及其位置
    idx = chars[~chars.duplicated(False)]
    idx = pandas.Series(idx.index, index=idx).reindex(data_chars)

    # 单音字只有一个 ID，直接标注
    labels = numpy.full(data.shape[0], -1, dtype=int)
    mask = idx.notna().values
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
    base: pandas.DataFrame,
    simplified: numpy.ndarray[str] | pandas.Series,
    traditional: numpy.ndarray[str] | pandas.Series | None,
    *datasets: list,
    encoder: str = 'svd',
    embedding_size: int = 32
) -> list[list[tuple[numpy.ndarray, numpy.ndarray[str], numpy.ndarray[str]]]]:
    """
    把没有字 ID 的数据集中的多音字对齐到基础数据集的字 ID

    Parameters:
        base: 基础数据集，为方言读音数据集长表，必须包含字段 did, cid, initial, final, tone
        simplified: base 中对应字的简体，多音字会出现多次。如类型为 pandas.Series，则索引为字 ID
        traditional: base 中对应字的繁体，为空的情况下只支持根据简体对齐
        datasets: 待对齐的无字 ID 数据集列表，每个元素为一个数据集，
            必须包含 did, initial, final, tone 字段
        encoder: 求字向量算法：
            - svd: SVD 矩阵分解
            - mf: 带缺失值的矩阵分解
        embedding_size: 指定编码字向量的大小

    Returns:
        results: 标注结果列表，长度和 datasets 相同，每个元素又是一个列表，长度为对应数据集的方言数。
            每个元素为如下三元组：
            - labels: 标注的字 ID 列表，长度等于该方言的记录数。内容为该数据集的每条记录对应 chars 中字的位置，
                如 chars 为 pandas.Series，则为字 ID，如在 chars 中不存在该字形，则为 -1 或 None
            - simplified: labels 对应位置的字的简体，如原始数据集没有提供简体，由原始字形转换得出
            - traditional: labels 对应位置的字的繁体，如原始数据集没有提供繁体，由原始字形转换得出

    先把基础数据集降维编码成字向量，在对每个数据集中每个方言应用 annotate 标注多音字。
    """

    assert simplified is not None

    # Pivot the base dataset into a wide table holding one cell per
    # dialect component; polyphones hold multiple values in one cell
    wide = preprocessing.transform(
        base,
        index="cid",
        columns="did",
        values=["initial", "final", "tone"],
        aggfunc=lambda x: " ".join(x.dropna())
    )

    # Encode the base dataset into character vectors
    if encoder == "svd":
        # Exclude characters whose coverage is below 10% from encoding
        mask = wide.count(axis=1) >= wide.shape[1] * 0.1
        wide = wide.loc[:, wide[mask].count(axis=0) > 0].fillna("")

        pipeline = sklearn.pipeline.make_pipeline(
            sklearn.compose.ColumnTransformer(
                [
                    (
                        str(c),
                        sklearn.feature_extraction.text.CountVectorizer(
                            lowercase=False,
                            tokenizer=str.split,
                            token_pattern=None,
                            stop_words=None,
                            binary=True,
                            dtype=numpy.int32
                        ),
                        c
                    )
                    for c in range(wide.shape[1])
                ],
                remainder="drop",
                sparse_threshold=1.0
            ),
            sklearn.decomposition.TruncatedSVD(embedding_size)
        )
        # Fit on the characters above the coverage threshold, then
        # project the remaining ones; char_embs keeps the row order of
        # wide
        embs = pipeline.fit_transform(wide[mask])
        char_embs = numpy.empty((wide.shape[0], embs.shape[1]))
        char_embs[mask.to_numpy()] = embs
        remainder = wide[~mask]
        if remainder.shape[0] > 0:
            char_embs[~mask.to_numpy()] = pipeline.transform(remainder)

        cids = numpy.asarray(wide.index)
    else:
        vectors, _ = preprocessing.features_to_vectors(wide.fillna(""))
        char_embs, _, _ = factorize.als(
            vectors,
            n_components=embedding_size
        )
        cids = numpy.asarray(wide.index)

    simplified = simplified.reindex(cids)
    if traditional is not None:
        traditional = traditional.reindex(cids)

    # 针对每个数据集中的每个方言标注多音字
    t2s = opencc.OpenCC('t2s')
    s2t = opencc.OpenCC('s2t')
    results = []
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
                char_embs,
                chars,
                data[['initial', 'final', 'tone']].dropna(axis=1, how='all') \
                    .fillna(''),
                data_chars.fillna('')
            )
            # 输入字表含有字 ID，把位置转成字 ID
            if isinstance(chars, pandas.Series):
                l = numpy.where(l >= 0, chars.index[l].values, None)
            labels.append((l, sim.values, trad.values))

        results.append(labels)

    return results
