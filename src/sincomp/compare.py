# -*- coding: utf-8 -*-

"""用于方言比较的工具函数."""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import argparse
import logging
import numpy
import pandas
import sklearn.compose
import sklearn.feature_extraction.text
import sklearn.preprocessing


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler())


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


if __name__ == '__main__':
    from . import datasets, preprocess


    parser = argparse.ArgumentParser(globals().get('__doc__'))
    parser.add_argument(
        '-l',
        '--log-level',
        default='WARNING',
        help='日志级别'
    )
    parser.add_argument('-r', '--rule-file', default='rules.json', help='语音规则文件')
    parser.add_argument(
        '-m',
        '--min-coverage',
        type=float,
        default=0,
        help='覆盖方言比例达到该值的字才纳入计算'
    )
    parser.add_argument(
        '-e',
        '--embedding-size',
        type=int,
        default=128,
        help='用于矩阵分解的字向量大小'
    )
    parser.add_argument(
        '-n',
        '--norm',
        type=int,
        default=2,
        help='把规则符合度归一化到 [0, 1]，基于矩阵分解的算法只适用 L2 归一化'
    )
    parser.add_argument('-o', '--output', help='输出文件名')
    parser.add_argument('dataset', help='指定输入方言数据集')
    args = parser.parse_args()

    logger.setLevel(getattr(logging, args.log_level.upper()))

    dataset = datasets.get(args.dataset)
    output = f'{dataset.name}_compliance_l{args.norm}.csv' \
        if args.output is None else args.output

    logger.info(
        f'compute rule compliance for {dataset.name}, '
        f'min coverage = {args.min_coverage}, '
        f'norm = {args.norm}, output = {output}'
    )

    rules = pandas.read_json(args.rule_file, orient='records', encoding='utf-8')
    logger.info(f'loaded {len(rules)} rules from {args.rule_file} .')

    if 'id' in rules.columns:
        rules.set_index('id', inplace=True)

    encoder = sklearn.preprocessing.LabelEncoder()
    rules['feature_id'] = encoder.fit_transform(rules['feature'])

    data = dataset.data
    if 'cid' not in data.columns:
        # 没有字 ID 的数据集使用字形作为 ID
        data = data.rename(columns={'character': 'cid'}).dropna(subset='cid')

    if args.min_coverage > 0:
        # 删除方言覆盖率小于阈值的字
        data = data[data.groupby('cid')['did'].transform('nunique') \
            > dataset.dialects.shape[0] * args.min_coverage]

    data = preprocess.transform(
        data,
        index='cid',
        columns='did',
        values=encoder.classes_,
        aggfunc=lambda x: ' '.join(x.dropna())
    )

    comp = compliance(data, rules, norm=args.norm if args.norm > 0 else None)
    comp.to_csv(output, encoding='utf-8', lineterminator='\n')