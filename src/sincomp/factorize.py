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


def _update_char_embs(
    char_embs: numpy.ndarray[float],
    phone_embs: numpy.ndarray[float],
    cooc: numpy.ndarray[int],
    limits: numpy.ndarray[int],
    phone_indeces,
    l2: float = 0.0
) -> None:
    """
    更新字向量

    Parameters:
        char_embs: 字向量列表，原地更新此参数的内容
        phone_embs: 读音向量列表
        limits: 方言读音向量的边界，长度为方言数 + 1，记录了每个方言的读音向量在
            phone_embs 中的起始位置和结束位置
        cooc: 字和方言共现矩阵，形状为 (字数, 方言数)，值为 0 或 1
        phone_indeces: 每个字对应的读音向量索引，长度等于字数，
            每个元素记录了该字对应 phone_embs 中的索引
        l2: L2 正则化系数，0 表示不使用正则化

    某个字的向量只与它在每个方言中的读音有关，使用最小二乘法分别求解每个字向量。
    """

    assert char_embs.shape[0] == cooc.shape[0], \
        f'{char_embs.shape[0]} != {cooc.shape[0]}'
    assert char_embs.shape[1] == phone_embs.shape[1], \
        f'{char_embs.shape[1]} != {phone_embs.shape[1]}'
    assert limits.shape[0] - 1 == cooc.shape[1], \
        f'{limits.shape[0] - 1} != {cooc.shape[1]}'
    assert char_embs.shape[0] == len(phone_indeces), \
        f'{char_embs.shape[0]} != {len(phone_indeces)}'

    embedding_size = char_embs.shape[1]

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

    b = numpy.empty((char_embs.shape[0], embedding_size, 1), dtype=numpy.float32)
    for i, indeces in enumerate(phone_indeces):
        numpy.sum(phone_embs[indeces], axis=0, out=b[i, :, 0])

    char_embs[:] = numpy.linalg.solve(a, b)[..., 0]

def _update_phone_embs(
    char_embs: numpy.ndarray[float],
    phone_embs: numpy.ndarray[float],
    cooc: numpy.ndarray[int],
    limits: numpy.ndarray[float],
    char_indeces,
    l2: float = 0.0
) -> None:
    """
    更新读音向量

    Parameters:
        char_embs: 字向量列表
        phone_embs: 读音向量列表，原地更新此参数的内容
        limits: 方言读音向量的边界，长度为方言数 + 1，记录了每个方言的读音向量在
            phone_embs 中的起始位置和结束位置
        cooc: 字和方言共现矩阵，形状为 (字数, 方言数)，值为 0 或 1
        char_indeces: 字索引列表，为方言、特征的二维数组，每个元素记录了该方言该特征
            对应的字在 char_embs 中的索引
        l2: L2 正则化系数，0 表示不使用正则化

    某个读音向量只与它和每个字的共现关系相关，使用最小二乘法分别求解每个方言的每个读音向量。
    """

    assert char_embs.shape[0] == cooc.shape[0], \
        f'{char_embs.shape[0]} != {cooc.shape[0]}'
    assert char_embs.shape[1] == phone_embs.shape[1], \
        f'{char_embs.shape[1]} != {phone_embs.shape[1]}'
    assert limits.shape[0] -1 == cooc.shape[1], \
        f'{limits.shape[0] - 1} != {cooc.shape[1]}'
    assert cooc.shape[1] == len(char_indeces), \
        f'{cooc.shape[1]} != {len(char_indeces)}'

    a = numpy.tensordot(
        cooc,
        char_embs[:, :, None] @ char_embs[:, None, :],
        [0, 0]
    )
    if l2 > 0:
        a += (numpy.eye(a.shape[1], dtype=a.dtype) * l2)[None, :, :]

    for j, indeces in enumerate(char_indeces):
        b = numpy.empty(
            (len(indeces), phone_embs.shape[1]),
            dtype=numpy.float32
        )
        for k, idx in enumerate(indeces):
            numpy.sum(char_embs[idx], axis=0, out=b[k])

        phone_embs[limits[j]:limits[j + 1]] = numpy.linalg.solve(a[j], b.T).T

def factorize(
    data: pandas.DataFrame,
    embedding_size: int = 128,
    max_iter: int = 10,
    tol: float = 0.0001,
    l2: float = 0.0001
) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """
    对方言字音矩阵实施矩阵分解，得到字向量和读音向量

    Parameters:
        data: 方言字音数据长表，必须包含 did 列作为方言 ID、cid 列作为字 ID，其余列作为读音
        embedding_size: 字向量和读音向量的维数
        max_iter: 最大迭代轮数
        tol: 停止阈值，误差下降小于该值时停止训练
        l2: L2 正则化系数，0 表示不使用正则化

    Returns:
        character_embeddings: 字向量，索引为字 ID
        phone_embeddings: 读音向量，每个方言的声韵调的每个取值均占一行，
            索引为方言 ID、特征名（如声母）、取值的多级索引

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

    # 对字 ID 和方言 ID 编码
    encoder = sklearn.preprocessing.OrdinalEncoder(dtype=numpy.int32) \
        .fit(data[['cid', 'did']])

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
    for j, c in enumerate(codes):
        dialect_codes.append(c.copy())
        c[:, 2:] += limits[j]

    codes = numpy.concatenate(codes, axis=0)

    # 字和方言点共现矩阵
    cooc = numpy.zeros((char_num, dialect_num), dtype=numpy.int8)
    cooc[codes[:, 0], codes[:, 1]] = 1

    phone_indeces = []
    for i in range(char_num):
        phone_indeces.append(numpy.ravel(codes[codes[:, 0] == i, 2:]))

    logger.debug(f'done prepairing {char_num} characters and {dialect_num} dialects.')

    # 轮流更新字向量和方言读音向量
    char_embs = numpy.random.randn(char_num, embedding_size) \
        .astype(numpy.float32)
    phone_embs = numpy.random.randn(phone_num, embedding_size) \
        .astype(numpy.float32)

    logger.debug(
        f'training with {char_num} characters and {dialect_num} dialects, '
        f'embedding size = {embedding_size}, maximum iteration = {max_iter}, '
        f'tol = {tol}, L2 = {l2} ...'
    )

    prev_rmse = numpy.inf
    for it in range(max_iter):
        _update_char_embs(
            char_embs,
            phone_embs,
            cooc,
            limits,
            phone_indeces,
            l2=l2
        )

        _update_phone_embs(
            char_embs,
            phone_embs,
            cooc,
            limits,
            char_indeces,
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
        if (diff := prev_rmse - rmse) < tol:
            # 本轮迭代 RMSE 比上轮下降小于指定阈值，认为已收敛，退出训练
            logger.debug(f'{diff} < tol = {tol}, stop training.')
            break
        else:
            prev_rmse = rmse

    else:
        logger.warning(f'maximum iteration {max_iter} reached, training not converged.')

    logger.debug('done.')

    # 根据方言 ID、方言特征生成读音向量索引
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