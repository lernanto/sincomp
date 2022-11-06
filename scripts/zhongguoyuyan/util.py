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
            .str.replace('((土家|布衣|蒙古|朝鲜|哈尼|.)族|哈萨克)*自治[州县]$', '', regex=True) \
            .str.replace('^(.{2,})(地|新|综合实验)区$', r'\1', regex=True) \
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

def clean_data(data, minfreq=1):
    """
    清洗方言字音数据中的录入错误.

    Parameters:
        data (`pandas.DataFrame`): 方言字音数据表
        minfreq (int): 在一种方言中出现次数少于该值的声韵调会从该方言中删除

    Returns:
        clean (`pandas.DataFram`): 清洗后的方言字音数据表
    """

    ipa = ':A-Za-z\u00c0-\u03ff\u1d00-\u1dbf\u1e00-\u1eff\u2205\u2c60-\u2c7f' \
        + '\ua720-\ua7ff\uab30-\uab6f\ufb00-\ufb4f\uff1a\ufffb' \
        + '\U00010780-\U000107ba\U0001df00-\U0001df1e'

    clean = data.copy()

    # 有些符号使用了多种写法，统一成较常用的一种
    clean['initial'] = clean['initial'].str.lower() \
        .str.replace(f'[^{ipa}]', '', regex=True) \
        .str.replace('[\u00f8\u01ff]', '\u2205', regex=True) \
        .str.replace('\ufffb', '_') \
        .str.replace('\u02a3', 'dz') \
        .str.replace('\u02a4', 'dʒ') \
        .str.replace('\u02a5', 'dʑ') \
        .str.replace('\u02a6', 'ts') \
        .str.replace('\u02a7', 'tʃ') \
        .str.replace('\u02a8', 'tɕ') \
        .str.replace('g', 'ɡ') \
        .str.replace('(.)h', r'\1ʰ', regex=True)

    # 删除出现次数少的读音
    if minfreq > 1:
        clean.loc[
            clean['initial'].groupby(clean['initial']).transform('count') <= minfreq,
            'initial'
        ] = ''

    mask = clean['initial'] != data['initial']
    if numpy.count_nonzero(mask):
        for (r, c), cnt in pandas.DataFrame({
            'raw': data.loc[mask, 'initial'],
            'clean': clean.loc[mask, 'initial']
        }).value_counts().items():
            logging.warning(f'replace {r} -> {c} {cnt}')

    clean['finals'] = clean['finals'].str.lower() \
        .str.replace(f'[^{ipa}]', '', regex=True) \
        .str.replace(':', 'ː') \
        .str.replace('：', 'ː')

    if minfreq > 1:
        clean.loc[
            clean['finals'].groupby(clean['finals']).transform('count') <= minfreq,
            'finals'
        ] = ''

    mask = clean['finals'] != data['finals']
    if numpy.count_nonzero(mask):
        for (r, c), cnt in pandas.DataFrame({
            'raw': data.loc[mask, 'finals'],
            'clean': clean.loc[mask, 'finals']
        }).value_counts().items():
            logging.warning(f'replace {r} -> {c} {cnt}')

    # 部分声调被错误转为日期格式，还原成数字
    mask = clean['tone'].str.match(r'^\d+年\d+月\d+日$')
    clean.loc[mask, 'tone'] = pandas.to_datetime(
        clean.loc[mask, 'tone'],
        format=r'%Y年%m月%d日'
    ).dt.dayofyear.astype(str)
    clean.loc[~mask, 'tone'] = clean.loc[~mask, 'tone'].str.lower() \
        .str.replace(r'[^1-5↗]', '', regex=True)

    if minfreq > 1:
        clean.loc[
            clean['tone'].groupby(clean['tone']).transform('count') <= minfreq,
            'tone'
        ] = ''

    mask = clean['tone'] != data['tone']
    if numpy.count_nonzero(mask):
        for (r, c), cnt in pandas.DataFrame({
            'raw': data.loc[mask, 'tone'],
            'clean': clean.loc[mask, 'tone']
        }).value_counts().items():
            logging.warning(f'replace {r} -> {c} {cnt}')

    return clean

def load_data(
    prefix,
    ids,
    suffix='mb01dz.csv',
    force_complete=False,
    sep=' ',
    transpose=False
):
    """
    加载方言字音数据.

    Parameters:
        prefix (str): 方言字音数据文件路径的前缀
        ids (iterable): 要加载的方言代码列表，完整的方言字音文件路径由代码加上前后缀构成
        suffix (str): 方言字音数据文件路径的后缀
        force_complete (bool): 保证一个读音的声母、韵母、声调均有效，否则舍弃该读音
        sep (str): 在返回结果中，当一个字有多个读音时，用来分隔多个读音的分隔符
        transpose (bool): 返回结果默认每行代表一个方言，当 transpose 为真时每行代表一个字

    Returns:
        data (`pandas.DataFrame`): 方言字音表
    """

    logging.info('loading {} data files ...'.format(len(ids)))

    columns = ['initial', 'finals', 'tone']
    dtypes = {
        'iid': int,
        'initial': str,
        'finals': str,
        'tone': str,
        'memo': str
    }

    load_ids = []
    dialects = []
    for id in ids:
        try:
            fname = os.path.join(prefix, id + suffix)
            logging.info(f'loading {fname} ...')

            d = pandas.read_csv(
                fname,
                encoding='utf-8',
                index_col='iid',
                usecols=dtypes.keys(),
                dtype=dtypes
            ).fillna('')

        except Exception as e:
            logging.error('cannot load file {}: {}'.format(fname, e), exc_info=True)
            continue

        d = clean_data(d)

        if force_complete:
            # 保证一个读音的声母、韵母、声调均有效，否则删除该读音
            invalid = (d[columns].isna() | (d[columns] == '')).any(axis=1)
            invalid_num = numpy.count_nonzero(invalid)
            if invalid_num > 0:
                logging.warning('drop {}/{} invalid records from {}'.format(
                    invalid_num,
                    d.shape[0],
                    fname
                ))
                logging.warning(d[invalid])

                d = d[~invalid]

        dialects.append(d)
        load_ids.append(id)

    logging.info('done. {} data file loaded'.format(len(dialects)))

    data = pandas.concat(
        [d.groupby(d.index).agg(sep.join) for d in dialects],
        axis=1,
        keys=load_ids,
        sort=True
    ).fillna('')
    logging.info(f'load data of {data.columns.levels[0].shape[0]} dialects x {data.shape[0]} characters')

    if not transpose:
        data = data.stack().transpose()

    return data

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