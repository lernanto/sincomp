# -*- coding: utf-8 -*-

"""用于方言比较的工具函数."""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import pandas
import numpy
import sklearn.compose
import sklearn.feature_extraction.text
import sklearn.preprocessing


def load_rule(fname, characters=None):
    """
    加载语音规则.

    Parameters:
        fname (str): 语音规则文件路径
        characters (`pandas.Series`): 字 ID 到字的映射表，用于显示

    Returns:
        rules (`pandas.DataFrame`): 语音规则表，每行对应一对同音字集
    """

    rules = pandas.read_csv(
        fname,
        converters={'cid1': str.split, 'cid2': str.split},
        comment='#'
    )

    if characters is None:
        rules['name'] = rules['feature'] + ':' \
            + rules['cid1'].str[0].astype(str) + '=' \
            + rules['cid2'].str[0].astype(str)
    else:
        rules['name'] = characters[rules['cid1'].str[0]].values \
            + '=' + characters[rules['cid2'].str[0]].values
        rules['char1'] = rules['cid1'].apply(lambda x: ''.join(characters[x]))
        rules['char2'] = rules['cid2'].apply(lambda x: ''.join(characters[x]))

    return rules

def compliance(
    data: pandas.DataFrame,
    rules: pandas.DataFrame,
    dtype: numpy.dtype = numpy.float32,
    norm: int | None = 2
) -> pandas.DataFrame:
    """
    计算方言字音对语音规则的符合度
    
    针对若干条读音规则，每条规则由2个字集组成，字集中每个字在一个方言中的读音为字集的读音分布，
    2个字集的读音分布归一化后的内积为字集的读音相似度，即方言对该规则的符合度，取值为 [0, 1]。
    当取 L2 归一化时，即为余弦相似度。

    Parameters:
        data: 方言字音数据表
        rules: 语音规则数据表
        norm: 计算相似度时归一化的范数，None 表示不归一化

    Returns:
        similarities: 读音相似度数据表，每行为一个方言，每列为一条规则
    """

    comp = []
    for feature, rule in rules.groupby('feature'):
        feature_data = data.loc[:, pandas.IndexSlice[:, feature]]

        # 先对方言读音 one-hot 编码
        transformer = sklearn.compose.make_column_transformer(
            *[(sklearn.feature_extraction.text.CountVectorizer(
                lowercase=False,
                tokenizer=str.split,
                stop_words=None,
                dtype=dtype
            ), i) for i in range(feature_data.shape[1])]
        )
        code = transformer.fit_transform(feature_data.fillna(''))

        lim = numpy.empty(len(transformer.transformers_) + 1, dtype=int)
        lim[0] = 0
        numpy.cumsum(
            [len(t[1].vocabulary_) for t in transformer.transformers_],
            out=lim[1:]
        )

        # 计算字集的读音向量
        code1 = numpy.empty((rule.shape[0], code.shape[1]), dtype=dtype)
        code2 = numpy.empty((rule.shape[0], code.shape[1]), dtype=dtype)
        for i, (_, r) in enumerate(rule.iterrows()):
            code1[i] = code[data.index.get_indexer(r['cid1'])].sum(axis=0).A[0]
            code2[i] = code[data.index.get_indexer(r['cid2'])].sum(axis=0).A[0]

        # 计算读音分布相似度，对读音向量分别归一化后内积
        sim = numpy.empty((feature_data.shape[1], rule.shape[0]), dtype=dtype)
        for i in range(feature_data.shape[1]):
            x1 = code1[:, lim[i]:lim[i + 1]]
            x2 = code2[:, lim[i]:lim[i + 1]]
            if norm is not None:
                x1 /= numpy.linalg.norm(x1, norm, axis=1, keepdims=True)
                x2 /= numpy.linalg.norm(x2, norm, axis=1, keepdims=True)

            numpy.sum(x1 * x2, axis=1, out=sim[i])

        comp.append(pandas.DataFrame(
            sim,
            index=feature_data.columns.get_level_values(0),
            columns=rule.index
        ))

    # 结果数据按输入规则的顺序重新排序
    return pandas.concat(comp, axis=1).reindex(rules.index, axis=1)
