# -*- coding: utf-8 -*

"""
对方言字音矩阵实施矩阵分解，得到字向量和读音向量
"""

__author__ = '黄艺华 <lernanto@foxmial.com>'


import logging
import numpy
import pandas
import sklearn.preprocessing


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler())


def _solve_char_embs(
    cooc: numpy.ndarray[int],
    phone_embs: numpy.ndarray[float],
    limits: numpy.ndarray[int],
    phone_indeces,
    l2: float = 0.0
) -> numpy.ndarray[float]:
    """
    根据字和方言共现矩阵及读音向量求解字向量

    Parameters:
        cooc: 字和方言共现矩阵，形状为 (字数, 方言数)，值为 0 或 1
        phone_embs: 读音向量列表，可以包含多于需要的方言，此时只使用其中部分方言的读音向量，
            使用的向量由 limits 指定
        limits: 方言读音向量的边界，长度为方言数 + 1，记录了每个方言的读音向量在
            phone_embs 中的起始位置和结束位置
        phone_indeces: 每个字对应的读音向量索引，长度等于字数，
            每个元素记录了该字对应 phone_embs 中的索引
        l2: L2 正则化系数，0 表示不使用正则化

    Returns:
        char_embs: 字向量列表，形状为 (字数, 向量维度)

    某个字的向量只与它在每个方言中的读音有关，使用最小二乘法分别求解每个字向量。
    """

    assert cooc.shape[1] == limits.shape[0] - 1, \
        f'{cooc.shape[1]} != {limits.shape[0] - 1}'
    assert cooc.shape[0] == len(phone_indeces), \
        f'{cooc.shape[0]} != {len(phone_indeces)}'

    embedding_size = phone_embs.shape[1]

    prods = numpy.empty(
        (limits.shape[0] - 1, embedding_size, embedding_size),
        dtype=numpy.float32
    )
    for j in range(prods.shape[0]):
        emb = phone_embs[limits[j]:limits[j + 1]]
        numpy.matmul(emb.T, emb, out=prods[j])

    a = numpy.tensordot(cooc, prods, [-1, 0])
    if l2 > 0:
        a += (numpy.eye(a.shape[1], dtype=a.dtype) * l2)[None, :, :]

    b = numpy.empty((cooc.shape[0], embedding_size, 1), dtype=numpy.float32)
    for i, indeces in enumerate(phone_indeces):
        numpy.sum(phone_embs[indeces], axis=0, out=b[i, :, 0])

    return numpy.linalg.solve(a, b)[..., 0]

def _solve_phone_embs(
    cooc: numpy.ndarray[int],
    char_embs: numpy.ndarray[float],
    char_indeces,
    l2: float = 0.0
) -> numpy.ndarray[float]:
    """
    根据字和方言共现矩阵及字向量求解读音向量

    Parameters:
        cooc: 字和方言共现矩阵，形状为 (字数, 方言数)，值为 0 或 1
        char_embs: 字向量列表
        char_indeces: 字索引列表，为方言、读音的二维数组，每个元素记录了该方言该读音
            对应的字在 char_embs 中的索引
        l2: L2 正则化系数，0 表示不使用正则化

    Returns:
        phone_embs: 读音向量列表，形状为 (读音数, 向量维度)

    某个读音向量只与它和每个字的共现关系相关，使用最小二乘法分别求解每个方言的每个读音向量。
    """

    assert cooc.shape[0] == len(char_embs), \
        f'{cooc.shape[0]} != {len(char_embs)}'
    assert cooc.shape[1] == len(char_indeces), \
        f'{cooc.shape[1]} != {len(char_indeces)}'

    embedding_size = char_embs.shape[1]

    a = numpy.tensordot(
        cooc,
        char_embs[:, :, None] @ char_embs[:, None, :],
        [0, 0]
    )
    if l2 > 0:
        a += (numpy.eye(a.shape[1], dtype=a.dtype) * l2)[None, :, :]

    limits = numpy.cumsum([0] + [len(i) for i in char_indeces], dtype=numpy.int32)
    phone_embs = numpy.empty((limits[-1], embedding_size), dtype=char_embs.dtype)
    for j, indeces in enumerate(char_indeces):
        b = numpy.empty((len(indeces), embedding_size), dtype=numpy.float32)
        for k, idx in enumerate(indeces):
            numpy.sum(char_embs[idx], axis=0, out=b[k])

        phone_embs[limits[j]:limits[j + 1]] = numpy.linalg.solve(a[j], b.T).T

    return phone_embs

def factorize(
    data: pandas.DataFrame,
    embedding_size: int = 128,
    max_iter: int = 10,
    tol: float = 0.0001,
    min_dialect_coverage: float | int = 0.2,
    min_character_coverage: float | int = 0.2,
    min_dialects: float| int = 10,
    min_characters: float | int = 100,
    l2: float = 0.0001
) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """
    对方言字音矩阵实施矩阵分解，得到字向量和读音向量

    Parameters:
        data: 方言字音数据长表，必须包含 did 列作为方言 ID、cid 列作为字 ID，其余列作为读音
        embedding_size: 字向量和读音向量的维数
        max_iter: 最大迭代轮数
        tol: 停止阈值，误差下降小于该值时停止训练
        min_dialect_coverage: 方言最小覆盖率，方言中至少覆盖该比例或数量的字才参与训练
        min_character_coverage: 字最小覆盖率，字中至少覆盖该比例或数量的方言才参与训练
        min_dialects: 最少方言数，参与训练的方言数必须大于该值
        min_characters: 最少字数，参与训练的字数必须大于该值
        l2: L2 正则化系数，0 表示不使用正则化

    Returns:
        character_embeddings: 字向量，索引为字 ID
        phone_embeddings: 读音向量，每个方言的声韵调的每个取值均占一行，
            索引为方言 ID、读音名（如声母）、取值的多级索引

    把方言读音独热编码的稀疏矩阵看成字向量和读音向量的乘积。交替固定读音向量或字向量，
    对另一向量实施线性回归求最小二乘解，迭代直到误差不再下降。求解时只考虑读音矩阵中有值的元素，
    忽略缺失值。
    为提高训练速度，第一阶段只训练满足最小覆盖率的字和方言，第二阶段再更新剩余的字和方言。
    """

    phone_names = data.columns.drop(['did', 'cid'])

    logger.debug('prepairing data for factorization ...')

    dialects = data.groupby('did')['cid'].nunique()
    chars = data.groupby('cid')['did'].nunique()
    dialect_num = dialects.shape[0]
    char_num = chars.shape[0]

    if isinstance(min_dialects, float):
        min_dialects = int(dialect_num * min_dialects)
    if isinstance(min_characters, float):
        min_characters = int(char_num * min_characters)

    if min_dialects < dialect_num and min_dialect_coverage > 0:
        # 把方言 ID 按出现频次排序，并找到满足最小覆盖率的边界
        if isinstance(min_dialect_coverage, float):
            min_dialect_coverage = int(char_num * min_dialect_coverage)

        aug = -dialects.values
        idx = numpy.argsort(aug)
        dialects = dialects.index[idx]
        dialect_pos1 = numpy.clip(
            numpy.searchsorted(aug[idx], -min_dialect_coverage, side='right'),
            min_dialects,
            dialect_num
        )
    else:
        dialects = dialects.index
        dialect_pos1 = dialect_num

    if min_characters < char_num and min_character_coverage > 0:
        # 把字 ID 按出现频次排序，并找到满足最小覆盖率的边界
        if isinstance(min_character_coverage, float):
            min_character_coverage = int(dialect_num * min_character_coverage)

        aug = -chars.values
        idx = numpy.argsort(aug)
        chars = chars.index[idx]
        char_pos1 = numpy.clip(
            numpy.searchsorted(aug[idx], -min_character_coverage, side='right'),
            min_characters,
            char_num
        )
    else:
        chars = chars.index
        char_pos1 = char_num

    # 对字 ID 和方言 ID 编码
    encoder = sklearn.preprocessing.OrdinalEncoder(
        categories=[chars, dialects],
        dtype=numpy.int32
    ).fit(data.iloc[:1][['cid', 'did']])

    # 生成临时变量，加快训练速度
    categories = [None] * dialect_num
    codes = [None] * dialect_num
    char_indeces = [None] * dialect_num
    limits = numpy.zeros(dialect_num + 1, dtype=numpy.int32)

    for _, d in data.groupby('did'):
        phone_encoder = sklearn.preprocessing.OrdinalEncoder(dtype=numpy.int32)
        c = numpy.concatenate(
            [
                encoder.transform(d[['cid', 'did']]),
                phone_encoder.fit_transform(d[phone_names])
            ],
            axis=1
        )
        j = c[0, 1]
        categories[j] = phone_encoder.categories_
        codes[j] = c

        indeces = []
        for k, cat in enumerate(phone_encoder.categories_):
            for l in range(cat.shape[0]):
                indeces.append(c[c[:, 2 + k] == l, 0])
        char_indeces[j] = indeces

        bases = numpy.cumsum(
            numpy.asarray(
                [0] + [c.shape[0] for c in phone_encoder.categories_[:-1]],
                dtype=numpy.int32
            )
        )
        c[:, 2:] += bases[None, :]
        limits[j + 1] = sum([c.shape[0] for c in phone_encoder.categories_])

    numpy.cumsum(limits, out=limits)
    phone_num = limits[-1]

    dialect_codes = []
    for j in range(dialect_pos1):
        c = codes[j]
        dialect_codes.append(c[c[:, 0] < char_pos1])

    for j, c in enumerate(codes):
        c[:, 2:] += limits[j]
    codes = numpy.concatenate(codes, axis=0)

    # 字和方言点共现矩阵
    cooc = numpy.zeros((char_num, dialect_num), dtype=numpy.int8)
    cooc[codes[:, 0], codes[:, 1]] = 1

    phone_indeces = []
    for i in range(char_num):
        idx = numpy.ravel(codes[codes[:, 0] == i, 2:])
        if dialect_pos1 < dialect_num:
            idx = idx[idx < limits[dialect_pos1]]
        phone_indeces.append(idx)

    logger.debug(f'done prepairing {char_num} characters and {dialect_num} dialects.')

    # 轮流更新字向量和方言读音向量
    char_embs = numpy.random.randn(char_num, embedding_size) \
        .astype(numpy.float32)
    phone_embs = numpy.random.randn(phone_num, embedding_size) \
        .astype(numpy.float32)

    logger.debug(
        f'training with {char_pos1} characters and {dialect_pos1} dialects, '
        f'embedding size = {embedding_size}, maximum iteration = {max_iter}, '
        f'tol = {tol}, L2 = {l2} ...'
    )

    cooc1 = cooc[:char_pos1, :dialect_pos1]
    limits1 = limits[:dialect_pos1 + 1]
    phone_indeces1 = phone_indeces[:char_pos1]

    if char_pos1 < char_num:
        # 第一阶段只训练部分字和方言，因此字索引必须限制在这个子集
        char_indeces1 = []
        for j in range(dialect_pos1):
            indeces = []
            for idx in char_indeces[j]:
                idx.sort()
                pos = numpy.searchsorted(idx, char_pos1)
                indeces.append(idx[:pos])
            char_indeces1.append(indeces)

    else:
        char_indeces1 = char_indeces[:dialect_pos1]

    prev_rmse = numpy.inf
    for it in range(max_iter):
        new_char_embs = _solve_char_embs(
            cooc1,
            phone_embs,
            limits1,
            phone_indeces1,
            l2=l2
        )

        new_phone_embs = _solve_phone_embs(
            cooc1,
            new_char_embs,
            char_indeces1,
            l2=l2
        )

        # 计算 RMSE
        square_errors = []
        counts = []
        for j, c in enumerate(dialect_codes):
            error = char_embs[c[:, 0]] @ phone_embs[limits[j]:limits[j + 1]].T
            numpy.put_along_axis(
                error,
                c[:, 2:],
                numpy.take_along_axis(error, c[:, 2:], axis=1) - 1,
                axis=1
            )
            square_errors.append(numpy.linalg.norm(error) ** 2)
            counts.append(error.size)

        rmse = numpy.sqrt(numpy.sum(square_errors) / numpy.sum(counts))
        logger.debug(f'iteration {it + 1}: RMSE = {rmse}')

        improve = prev_rmse - rmse
        if improve > 0:
            # 仅当 RMSE 下降时更新向量
            char_embs[:char_pos1] = new_char_embs
            phone_embs[:limits[dialect_pos1]] = new_phone_embs

        if improve < tol:
            # 本轮迭代 RMSE 比上轮下降小于指定阈值，认为已收敛，退出训练
            logger.debug(f'{improve} < tol = {tol}, stop training.')
            break
        else:
            prev_rmse = rmse

    else:
        logger.warning(f'maximum iteration {max_iter} reached, training not converged.')

    logger.debug('done.')

    # 计算剩余的字向量和读音向量
    if char_pos1 < char_num:
        logger.debug(f'updating rest {char_num - char_pos1} characters ...')
        char_embs[char_pos1:] = _solve_char_embs(
            cooc[char_pos1:, :dialect_pos1],
            phone_embs,
            limits1,
            phone_indeces[char_pos1:],
            l2=l2
        )
        logger.debug('done.')

    if dialect_pos1 < dialect_num:
        logger.debug(f'updating rest {dialect_num - dialect_pos1} dialects ...')
        phone_embs[limits[dialect_pos1]:] = _solve_phone_embs(
            cooc[:, dialect_pos1:],
            char_embs,
            char_indeces[dialect_pos1:],
            l2=l2
        )
        logger.debug('done.')

    # 根据方言 ID、方言读音生成读音向量索引
    aug = pandas.concat(
        [pandas.concat(
            [pandas.DataFrame(index=c) for c in cat],
            axis=0,
            keys=phone_names
        ) for cat in categories],
        axis=0,
        keys=encoder.categories_[1]
    ).rename_axis(['cid', 'phone', 'value'], axis=0)
    return (
        pandas.DataFrame(char_embs, index=encoder.categories_[0]),
        pandas.DataFrame(phone_embs, index=aug.index)
    )