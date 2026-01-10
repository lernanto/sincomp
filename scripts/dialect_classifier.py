#!/usr/bin/env -S python3 -O
# -*- coding: utf-8 -*-

"""
根据指定的规则训练模型对方言分类
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import argparse
import joblib
import logging
import numpy as np
import pandas as pd
import sincomp.compare
import sincomp.datasets
import sincomp.preprocess
from sklearn.compose import make_column_transformer
from sklearn.impute import KNNImputer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline, make_pipeline


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler())


def train_classifier(
        rules: pd.DataFrame,
        annotations: pd.DataFrame,
        resample: int = 0,
        samples: int = 500,
        min_rate: float = 0.5,
        max_rate: float = 0.8
) -> Pipeline:
    """
    训练方言分类器

    对方言数据重采样来作样本增强，同时使用交叉验证来寻找最优模型超参。

    Parameters:
        rules: 根据方言字音计算规则符合度的规则集
        annotations: 标注数据，每行为一个方言样本，包含如下字段：
            - dataset 样本所属数据集
            - did 方言 ID
            - stratefy 可选，用于交叉验证的分组，属于同一 stratefy 的样本大致均匀地落在各组
            - group 可选，用于交叉验证的分组，属于同一 group 的样本保证落在同一组
            - label 方言所属分类
        resample: 对每个方言重采样的次数，为0不重采样
        samples: 重采样时每个方言采样的字数
        min_rate: 重采样的字数占方言收录字数的占比不小于该值
        max_rate: 重采样的字数占方言收录字数的占比不大于该值

    Returns:
        classifier: 训练的分类器
    """

    feature_names = rules['feature'].unique().tolist()

    compliances = []
    indeces = []
    for d, a in annotations.groupby('dataset', sort=False):
        data = sincomp.datasets.get(d).select(a['did']).data

        if 'cid' not in data.columns:
            data = data.rename(columns={'character': 'cid'}).dropna(subset='cid')

        indeces.extend(a.index)
        new_data = [data]

        if resample > 0:
            # 对方言字音重采样，作为样本增强，增加模型稳定性
            indeces.extend(a.index.repeat(resample))
            for did, d in data.groupby('did', sort=False):
                n = int(np.clip(
                    samples,
                    min_rate * d.shape[0],
                    max_rate * d.shape[0]
                ))
                for i in range(resample):
                    new_data.append(d.sample(n).assign(did=f'{did}_{i}'))

        new_data = sincomp.preprocess.transform(
            pd.concat(new_data, axis=0, ignore_index=True),
            index='cid',
            columns='did',
            values=feature_names,
            aggfunc=lambda x: ' '.join(x.dropna())
        )
        compliances.append(sincomp.compare.compliance(new_data, rules))

    compliances = pd.concat(compliances, axis=0)
    annotations = annotations.loc[indeces]

    clf = make_pipeline(
        make_column_transformer(('passthrough', compliances.columns)),
        # 使用 KNN 算法填充特征中的缺失值
        KNNImputer(),
        # 训练线性分类模型，使用交叉验证来搜索最优超参
        LogisticRegressionCV(
            Cs=np.power(2.0, np.arange(-4, 5, 0.5)),
            fit_intercept=False,
            cv=list(StratifiedGroupKFold().split(
                compliances,
                annotations['stratefy'],
                annotations['group']
            )),
            penalty='l1',
            solver='saga'
        )
    ).fit(compliances, annotations['label'])

    logger.debug(f'best C = {clf.steps[-1][1].C_[0]:.4f}')
    return clf

def train(args: argparse.Namespace) -> None:
    """
    训练方言分类器
    """

    logger.debug(
        f'train dialect classifier, rule file = {args.rule_file}, '
        f'annotation file = {args.annotation_file}, '
        f'output file = {args.output_file}, resample = {args.resample}'
    )

    rules = sincomp.compare.load_rules(args.rule_file)
    annotations = pd.read_csv(
        args.annotation_file,
        dtype=str,
        comment='#',
        encoding='utf-8'
    )
    if 'stratefy' not in annotations.columns:
        annotations['stratefy'] = annotations['label']
    if 'group' not in annotations.columns:
        annotations['group'] = annotations['dataset'] + '_' + annotations['did']

    clf = train_classifier(rules, annotations, args.resample)

    fi = pd.DataFrame({
        'name': rules['name'],
        'importance': np.linalg.norm(clf.steps[-1][1].coef_, axis=0)
    })
    fi = '\n'.join(f'{r["name"]}: {r["importance"]:.4f}' \
        for _, r in fi.sort_values('importance', ascending=False).iterrows())
    logger.info(f'feature importance:\n{fi}')

    joblib.dump(clf, args.output_file)

def validate(args: argparse.Namespace) -> None:
    """
    交叉验证方言分类器准确率
    """

    logger.debug(
        f'cross validate dialect classifier, rule file = {args.rule_file}, '
        f'annotation file = {args.annotation_file}, '
        f'resample = {args.resample}'
    )

    rules = sincomp.compare.load_rules(args.rule_file)
    annotations = pd.read_csv(
        args.annotation_file,
        dtype=str,
        comment='#',
        encoding='utf-8'
    )
    if 'stratefy' not in annotations.columns:
        annotations['stratefy'] = annotations['label']
    if 'group' not in annotations.columns:
        annotations['group'] = annotations['dataset'] + '_' + annotations['did']

    feature_names = rules['feature'].unique().tolist()
    compliances = []
    indeces = []
    for d, a in annotations.groupby('dataset', sort=False):
        data = sincomp.datasets.get(d).select(a['did']).data
        if 'cid' not in data.columns:
            data = data.rename(columns={'character': 'cid'}).dropna(subset='cid')

        data = sincomp.preprocess.transform(
            data,
            index='cid',
            columns='did',
            values=feature_names,
            aggfunc=lambda x: ' '.join(x.dropna())
        )
        compliances.append(sincomp.compare.compliance(data, rules))
        indeces.extend(a.index)

    compliances = pd.concat(compliances, axis=0)
    annotations = annotations.loc[indeces]

    acc = []
    for train_idx, test_idx in StratifiedGroupKFold() \
        .split(compliances, annotations['stratefy'], annotations['group']):
        clf = train_classifier(rules, annotations.iloc[train_idx], args.resample)
        acc.append(accuracy_score(
            annotations.iloc[test_idx]['label'],
            clf.predict(compliances.iloc[test_idx])
        ))

    print(
        f'cross validation with {annotations.shape[0]} samples, '
        f'accuracy = {np.mean(acc):.4f}±{np.std(acc, ddof=1):.4f} '
    )

def predict(args: argparse.Namespace) -> None:
    """
    使用已训练的方言分类器预测方言分类
    """

    logger.debug(
        f'predict dialect class, model file = {args.model_file}, '
        f'rule file = {args.rule_file}, input file = {args.input.name}, '
        f'output file = {args.output.name}'
    )

    clf = joblib.load(args.model_file)
    if args.precomputed:
        # 从预计算的文件加载方言对规则的符合度
        compliances = pd.read_csv(args.input, dtype={'dataset': str, 'did': str})
        samples = compliances.pop('did').to_frame()
        if 'dataset' in compliances.columns:
            samples.insert(0, 'dataset', compliances.pop('dataset'))

    else:
        # 根据规则计算方言对规则符合度
        rules = sincomp.compare.load_rules(args.rule_file)
        samples = pd.read_csv(
            args.input,
            dtype=str,
            comment='#',
            encoding='utf-8'
        )

        feature_names = rules['feature'].unique().tolist()
        compliances = []
        for d, i in samples.groupby('dataset', sort=False):
            data = sincomp.datasets.get(d).select(i['did']).data
            if 'cid' not in data.columns:
                data = data.rename(columns={'character': 'cid'}).dropna(subset='cid')

            data = sincomp.preprocess.transform(
                data,
                index='cid',
                columns='did',
                values=feature_names,
                aggfunc=lambda x: ' '.join(x.dropna())
            )
            compliances.append(
                sincomp.compare.compliance(data, rules).set_index(i.index)
            )

        compliances = pd.concat(compliances, axis=0).loc[samples.index]

    probs = pd.DataFrame(
        clf.predict_proba(compliances),
        columns=clf.steps[-1][1].classes_
    )
    pd.concat([samples, probs], axis=1).to_csv(args.output, index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--log-level',
        default='WARNING',
        help='输出日志级别'
    )
    subparsers = parser.add_subparsers()

    train_parser = subparsers.add_parser(
        'train', help=train.__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )
    train_parser.add_argument(
        '--resample',
        type=int,
        default=5,
        help='对每个方言样本重采样的次数，以增加模型稳定性，为0不重采样'
    )
    train_parser.add_argument(
        'rule_file',
        help='''
        用于训练的规则文件，为 JSON 格式，为规则的数组，每条规则包含如下字段：
            - id 可选，规则 ID，缺失时以序号为 ID
            - name 可选，规则名
            - cid1 用于对比的字集1，为字 ID 的数组
            - cid2 用于对比的字集2，为字 ID 的数组
        '''
    )
    train_parser.add_argument(
        'annotation_file',
        help='''
        标注的训练样本文件，为 CSV 格式，每行为一个方言样本，包含如下字段：
            - dataset 样本所属数据集
            - did 方言 ID
            - stratefy 可选，用于交叉验证的分组，属于同一 stratefy 的样本大致均匀地落在各组
            - group 可选，用于交叉验证的分组，属于同一 group 的样本保证落在同一组
            - label 方言所属分类
        '''
    )
    train_parser.add_argument(
        'output_file',
        nargs='?',
        default='dialect_classifier.bz2',
        help='模型输出文件'
    )
    train_parser.set_defaults(func=train)

    validate_parser = subparsers.add_parser(
        'validate', help=validate.__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )
    validate_parser.add_argument(
        '--resample',
        type=int,
        default=5,
        help='对每个方言样本重采样的次数，以增加模型稳定性，为0不重采样'
    )
    validate_parser.add_argument(
        'rule_file',
        help='''
        用于训练的规则文件，为 JSON 格式，为规则的数组，每条规则包含如下字段：
            - id 可选，规则 ID，缺失时以序号为 ID
            - name 可选，规则名
            - cid1 用于对比的字集1，为字 ID 的数组
            - cid2 用于对比的字集2，为字 ID 的数组
        '''
    )
    validate_parser.add_argument(
        'annotation_file',
        help='''
        标注的训练样本文件，为 CSV 格式，每行为一个方言样本，包含如下字段：
            - dataset 样本所属数据集
            - did 方言 ID
            - stratefy 可选，用于交叉验证的分组，属于同一 stratefy 的样本大致均匀地落在各组
            - group 可选，用于交叉验证的分组，属于同一 group 的样本保证落在同一组
            - label 方言所属分类
        '''
    )
    validate_parser.set_defaults(func=validate)

    predict_parser = subparsers.add_parser(
        'predict', help=predict.__doc__,
        formatter_class=argparse.RawTextHelpFormatter
    )
    predict_parser.add_argument(
        '-p',
        '--precomputed',
        action='store_true',
        default=False,
        help='输入文件为预先计算的规则符合度，CSV 格式，每行为一个方言，每列为一条规则'
    )
    predict_parser.add_argument(
        '-m',
        '--model-file',
        default='dialect_classifier.bz2',
        help='模型文件'
    )
    predict_parser.add_argument(
        '-r',
        '--rule-file',
        help='''
        用于训练的规则文件，为 JSON 格式，为规则的数组，每条规则包含如下字段：
            - id 可选，规则 ID，缺失时以序号为 ID
            - name 可选，规则名
            - cid1 用于对比的字集1，为字 ID 的数组
            - cid2 用于对比的字集2，为字 ID 的数组
        '''
    )
    predict_parser.add_argument(
        'input',
        nargs='?',
        default='-',
        type=argparse.FileType('r'),
        help='''
        待分类方言输入文件，为 CSV 格式，每行为一个方言样本，包含如下字段：
            - dataset 样本所属数据集
            - did 方言 ID
        '''
    )
    predict_parser.add_argument(
        'output',
        nargs='?',
        default='-',
        type=argparse.FileType('w'),
        help='''
        输出文件，为 CSV 格式，每行对应输入文件的一行，除包含输入文件的所有字段外，
        还新增若干字段，每个字段对应一个方言分类，为方言属于该分类的概率
        '''
    )
    predict_parser.set_defaults(func=predict)

    args = parser.parse_args()
    logger.setLevel(getattr(logging, args.log_level.upper()))
    args.func(args)