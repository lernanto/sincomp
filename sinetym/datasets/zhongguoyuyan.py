# -*- encoding: utf-8 -*-

"""
一些处理数据函数.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import logging
import os
import pandas
import numpy
from sklearn.neighbors import KNeighborsClassifier


def clean_location(location):
    """
    清洗方言点地名格式.

    Parameters:
        location (`pandas.DataFrame`): 方言点信息数据表

    Returns:
        clean (`pandas.DataFrame`): 归一化市县名称的方言点数据
    """

    def norm(raw):
        clean =  raw.fillna('').str.strip() \
            .str.replace('^[(（]无[)）]$', '', regex=True) \
            .str.replace('(.)（.*）$', r'\1', regex=True) \
            .str.replace('[（）]', '', regex=True) \
            .str.replace('(?:(?:土家|布衣|蒙古|朝鲜|哈尼|.)族|哈萨克)*自治[州县]$', '', regex=True) \
            .str.replace('^(?:.*市区.*|市[内辖].+区)$', '市区', regex=True) \
            .str.replace('^(.{2,})(?:地|新|特|林|综合实验)区$', r'\1', regex=True) \
            .str.replace('^(.{2,6})[市州县区]$', r'\1', regex=True)

        mask = clean != raw
        if numpy.count_nonzero(mask):
            for (r, c), cnt in pandas.DataFrame({
                'raw': raw[mask],
                'clean': clean[mask]
            }).value_counts().items():
                logging.warning(f'replace {r} -> {c} {cnt}')

        return clean

    clean = location.assign(
        city=norm(location['city']),
        county=norm(location['country'])
    )

    # 直辖市的地区统一置空
    mask = clean['province'].isin(['北京', '天津', '上海', '重庆'])
    clean.loc[mask & clean['county'] == '', 'county'] = clean['city']
    clean.loc[mask, 'city'] = ''

    return clean

def force_complete(data):
    """
    保证一个读音的声母、韵母、声调均有效，否则删除该读音.

    Parameters:
        data (`pandas.DataFrame`): 原始方言字音数据表

    Returns:
        output (`pandas.DataFrame`): 删除不完整读音后的方言字音数据表
    """

    columns = ['initial', 'final', 'tone']
    invalid = (data[columns].isna() | (data[columns] == '')).any(axis=1)
    invalid_num = numpy.count_nonzero(invalid)
    if invalid_num > 0:
        logging.warning(f'drop {invalid_num}/{data.shape[0]} invalid records')
        logging.warning(data[invalid])

    return data.drop(data.index[invalid])

def get_dialect(location):
    """
    从方言点信息中获取所属方言区.

    Parameters:
        location (`pandas.DataFrame`): 原始方言点信息数据表

    Returns:
        dialect (`pandas.Series`): 方言点对应的方言区列表
    """

    def clean(tag):
        """清洗原始的方言区标记."""

        tag = tag.fillna('')

        return numpy.where(
            tag.str.contains('客'),
            '客家话',
            numpy.where(
                tag.str.contains('平'),
                '平话',
                numpy.where(
                    tag.str.contains('湘南|韶州'),
                    tag.str.replace('.*(湘南|韶州).*', r'\1土话', regex=True),
                    numpy.where(
                        tag.str.contains('[吴闽赣粤湘晋徽]'),
                        tag.str.replace('.*([吴闽赣粤湘晋徽]).*', r'\1语', regex=True),
                        numpy.where(
                            tag.str.contains('北京|东北|冀鲁|胶辽|中原|兰银|江淮|西南'),
                            tag.str.replace(
                                '.*(北京|东北|冀鲁|胶辽|中原|兰银|江淮|西南).*',
                                r'\1官话',
                                regex=True
                            ),
                            ''
                        )
                    )
                )
            )
        ).astype(str)

    # 有些方言区，主要是官话的大区被标在不同的字段，尽力尝试获取
    dialect = clean(location['area'])
    dialect = numpy.where(dialect != '', dialect, clean(location['slice']))

    return pandas.Series(dialect, index=location.index)

def get_cluster(location):
    """
    从方言点信息中提取所属方言片.

    Parameters:
        location (`pandas.DataFrame`): 方言信息数据表

    Returns:
        cluster (`pandas.Series`): 方言片列表
    """

    def try_get_cluster(tag):
        """尝试从标记字符串中匹配方言片."""

        return tag[tag.str.match('^.+[^小]片.*$') == True].str.replace(
            '^(?:.*[语话]区?)?([^语话片]*[^小片]片).*$',
            r'\1',
            regex=True
        ).reindex(tag.index)

    cluster = try_get_cluster(location['slice'])
    cluster = cluster.where(cluster.notna(), try_get_cluster(location['slices']))
    cluster = cluster.where(cluster.notna(), try_get_cluster(location['area']))

    slice = location.loc[
        location['slice'].str.contains('[不未]明|[语话片]$', regex=True) == False,
        'slice'
    ]
    slice = slice.where(slice.str.len() != 2, slice + '片')
    cluster = cluster.where(
        cluster.notna(),
        slice
    )

    return cluster.fillna('')

def get_subcluster(location):
    """
    从方言点信息中提取所属方言小片.

    Parameters:
        location (`pandas.DataFrame`): 方言信息数据表

    Returns:
        subcluster (`pandas.Series`): 方言小片列表
    """

    def try_get_subcluster(tag):
        """尝试从标记字符串中匹配方言小片."""

        return tag[tag.str.match('^.+小片.*$') == True].str.replace(
            '^(?:.*[语话]区?)?(?:[^语话片]*[^小片]片)?([^语话片]+小片).*$',
            r'\1',
            regex=True
        ).reindex(tag.index)

    subcluster = try_get_subcluster(location['slices'])
    subcluster = subcluster.where(
        subcluster.notna(),
        try_get_subcluster(location['slice'])
    )
    subcluster = subcluster.where(
        subcluster.notna(),
        location.loc[
            location['slices'].str.contains('[不未]明|[语话片]$', regex=True) == False,
            'slices'
        ]
    )

    return subcluster.fillna('')


def predict_dialect(location, dialect):
    """
    使用 KNN 算法根据经纬度信息预测方言区.

    Parameters:
        location (`pandas.DataFrame`): 原始方言点信息
        dialect (`pandas.Series`): 从原始信息中提取的方言区信息，无法获取方言区的为空字符串

    Returns:
        predict (`pandas.Series`): 带预测的方言区信息，已知的方言区保持不变，其余使用 KNN 预测
    """

    mask = location['latitude'].notna() & location['longitude'].notna()
    knn = KNeighborsClassifier().fit(
        location.loc[mask & (dialect != ''), ['latitude', 'longitude']],
        dialect[mask & (dialect != '')]
    )
    predict = dialect.copy()
    predict[mask & (dialect == '')] = knn.predict(
        location.loc[mask & (dialect == ''), ['latitude', 'longitude']]
    )

    return predict

def load_location(fname, predict=True):
    """
    读取方言点信息.

    对其中的市县名称使用规则归一化，并规范化方言区名称。对缺失方言区的，使用模型预测填充。

    Parameters:
        fname (str): 包含方言点数据的文件路径
        predict (bool): 对缺失的方言区信息用机器学习模型预测填充，默认开启

    Returns:
        location (`pandas.DataFrame`): 方言点信息数据表
    """

    location = clean_location(pandas.read_csv(fname, index_col=0))
    dialect = get_dialect(location)
    location['dialect'] = predict_dialect(location, dialect) if predict \
        else dialect
    location['cluster'] = get_cluster(location)
    location['subcluster'] = get_subcluster(location)

    return location