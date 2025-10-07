#!/usr/bin/env -S python3 -O
# -*- coding: utf-8 -*-

"""
训练一个分类器，用以区分不同拉丁化方案转写的读音

当前包含如下分类：
    - IPA 国际音标
    - pinyin 汉语拼音转写的汉语
    - jyutping 粤拼转写的粤语
    - beh-oe-ji 白话字转写的闽南语
    - romaji 日语罗马字转写的日语
    - romaja 韩语罗马字转写的韩语
    - quoc ngu 国语字转写的越南语
    - other 其他
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import argparse
import numpy as np
import os
import pandas as pd
import scipy.stats as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC
import sys


# 从 MCPDict 数据集选择部分样本手工标注
annotations = {
    '1935南昌': 'IPA',
    '中世朝鮮': 'IPA',
    '中原音韻': 'IPA',
    '中唐': 'IPA',
    '党項': 'IPA',
    '劍川金華白語': 'IPA',
    '北宋': 'IPA',
    '巴陵戲': 'IPA',
    '榕江侗上古借詞': 'IPA',
    '沅陵丑溪口': 'IPA',
    '河源': 'IPA',
    '沿河': 'IPA',
    '洪雅': 'IPA',
    '淸末寧波': 'IPA',
    '湘音檢字': 'IPA',
    '溧水': 'IPA',
    '煙臺': 'IPA',
    '猗氏': 'IPA',
    '盛唐': 'IPA',
    '知乎新派': 'IPA',
    '老國音': 'IPA',
    '臨髙話': 'IPA',
    '蒙古字韻': 'IPA',
    '蔡家話': 'IPA',
    '西儒耳目資': 'IPA',
    '訓詁諧音': 'IPA',
    '資興南鄕': 'IPA',
    '遂寧觀音': 'IPA',
    '鳳翔糜杆橋': 'IPA',

    '國語': 'pinyin',
    '普通話': 'pinyin',

    '香港': 'jyutping',

    '臺灣': 'beh-oe-ji',

    '日語其他': 'romaji',
    '日語吳音': 'romaji',
    '日語漢音': 'romaji',

    '朝鮮': 'romaja',

    '越南': 'quoc ngu',

    '廣韻': 'other',
    '東干甘肅話': 'other',
    '白-沙': 'other',
    '鄭張': 'other',
}


def main(args: argparse.Namespace) -> None:
    np.random.seed(919)

    # 加载 MCPDict 数据集方言信息
    data_dir = os.path.join(args.mcpdict_dir, 'tools', 'tables', 'output')
    dialects = pd.read_json(
        os.path.join(data_dir, '_詳情.json'),
        orient='index',
        encoding='utf-8'
    )
    dialects = dialects.loc[dialects.index.isin(annotations.keys())]
    dialects['label'] = dialects.index.map(annotations)
    dialects['count'] = dialects.groupby('label', as_index=False)['label'] \
        .transform('count')

    # 读取方言读音数据
    data = []
    for i, r in dialects.iterrows():
        pron = pd.read_csv(
            os.path.join(data_dir, i + '.tsv'),
            sep='\t',
            usecols=[1],
            dtype=str,
            encoding='utf-8'
        )

        # 大部分方言点的读音以数字形式标注了声调，先去除
        pron = pron.iloc[:, 0].str.extract(
            r'([^0-9]*)(?:[0-9][0-9a-z]*)?',
            expand=False
        ).dropna()

        # 如果某一分类的样本只有一个，把读音数据随机分成两份，一份作为训练集，一份作为测试集
        if r['count'] > 1:
            data.append((pron.values, r['label']))
        else:
            pron = pron.sample(frac=1)
            n = pron.shape[0] // 2
            data.append((pron.iloc[:n].values, r['label']))
            data.append((pron.iloc[n:].values, r['label']))

    data = pd.DataFrame(data, columns=['data', 'label'])

    train_data, test_data = train_test_split(data, test_size=0.5, stratify=data['label'])
    test_data['data'] = test_data['data'].str.join('')

    # 从读音数据中随机抽样生成不小于指定数量的训练样本
    train_data['count'] = train_data.groupby('label', as_index=False)['label'] \
        .transform('count')
    data = []
    for _, r in train_data.iterrows():
        bernoulli = st.bernoulli(min(200 / r['data'].shape[0], 0.5))
        for _ in range(int(np.ceil(50 / r['count']))):
            data.append((
                ''.join(r['data'][bernoulli.rvs(r['data'].shape[0]).astype(bool)]),
                r['label']
            ))

    train_data = pd.DataFrame(data, columns=['data', 'label'])

    # 使用交叉验证训练 SVM 分类器
    cls = make_pipeline(
        CountVectorizer(
            lowercase=False,
            analyzer='char',
            vocabulary=map(chr, range(0x0021, 0x0370))
        ),
        SelectKBest(chi2, k=100),
        GridSearchCV(
            LinearSVC(fit_intercept=False),
            {'C': [0.01, 0.1, 1, 10]},
            cv=5
        )
    ).fit(train_data['data'], train_data['label'])

    # 在测试集上评估分类准确率并输出
    print(
        classification_report(
            test_data['label'], 
            cls.predict(test_data['data'])
        ),
        file=sys.stderr
    )

    # 输出分类器词汇表、类型列表、权重
    svm = cls.steps[2][1].best_estimator_
    pd.DataFrame(
        svm.coef_.T,
        index=cls.steps[1][1].get_feature_names_out(
            cls.steps[0][1].get_feature_names_out()
        ),
        columns=svm.classes_
    ).to_csv(args.output, lineterminator='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--output',
         '-o',
         type=argparse.FileType('w', encoding='utf-8'),
         default='-',
         help='输出文件，默认为标准输出'
    )
    parser.add_argument('mcpdict_dir', default='.', help='MCPDict 项目根目录')
    args = parser.parse_args()
    main(args)