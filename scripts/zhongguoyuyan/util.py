# -*- encoding: utf-8 -*-

"""
一些处理数据函数.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import logging
import os
import pandas
import numpy


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
    '''从方言信息中获取所属方言区'''

    def clean(tag):
        '''清洗原始的方言区标记'''

        tag = tag.fillna('')

        return numpy.where(
            tag.str.contains('客'),
            '客家方言',
            numpy.where(
                tag.str.contains('平'),
                '平话',
                numpy.where(
                    tag.str.contains('[吴闽赣粤湘晋徽]'),
                    tag.str.replace('.*([吴闽赣粤湘晋徽]).*', r'\1方言', regex=True),
                    numpy.where(
                        tag.str.contains('北京|东北|冀鲁|胶辽|中原|兰银|江淮|西南'),
                        tag.str.replace(
                            '.*(北京|东北|冀鲁|胶辽|中原|兰银|江淮|西南).*',
                            r'\1官话',
                            regex=True
                        ),
                        numpy.where(
                            tag.str.contains('湖南|韶州'),
                            tag.str.replace('.*(湖南|韶州).*', r'\1土话', regex=True),
                            ''
                        )
                    )
                )
            )
        ).astype(str)

    # 有些方言区，主要是官话的大区被标在不同的字段，尽力尝试获取
    dialect = clean(location['area'])
    return numpy.where(dialect != '', dialect, clean(location['slice']))