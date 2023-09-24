# -*- coding: utf-8 -*-

"""
处理汉语方言读音数据的工具集.

当前支持读取：
    - 小学堂汉字古今音资料库的现代方言数据，见：https://xiaoxue.iis.sinica.edu.tw/ccr
    - 中国语言资源保护工程采录展示平台的方言数据，见：https://zhongguoyuyan.cn/
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import os
import logging
import pandas
import numpy

from . import xiaoxuetang, zhongguoyuyan


ipa = 'a-z\u00c0-\u03ff\u1d00-\u1dbf\u1e00-\u1eff\u2205\u2c60-\u2c7f' \
    + '\ua720-\ua7ff\uab30-\uab6f\ufb00-\ufb4f\uff1a\ufffb' \
    + '\U00010780-\U000107ba\U0001df00-\U0001df1e'


def clean_initial(raw):
    """
    清洗方言字音数据中的声母.

    Parameters:
        raw (`pandas.Series`): 方言字音声母列表

    Returns:
        clean (`pandas.Series`): 清洗后的方言字音声母列表
    """

    # 有些符号使用了多种写法，统一成较常用的一种
    return raw.str.replace('[0\u00d8\u00f8\u01fe\u01ff]', '∅', regex=True) \
        .str.replace('(.)∅|∅(.)', r'\1\2', regex=True) \
        .str.replace('\ufffb', '_') \
        .str.replace('\u02a3', 'dz') \
        .str.replace('\u02a4', 'dʒ') \
        .str.replace('\u02a5', 'dʑ') \
        .str.replace('\u02a6', 'ts') \
        .str.replace('\u02a7', 'tʃ') \
        .str.replace('\u02a8', 'tɕ') \
        .str.replace('g', 'ɡ') \
        .str.replace('(.)h', r'\1ʰ', regex=True) \
        .str.replace('(.)ɦ', r'\1ʱ', regex=True) \
        .str.replace('([bdɡvzʐʑʒɾ])ʰ', r'\1ʱ', regex=True) \
        .str.replace(f'[^{ipa}]', '', regex=True)

def clean_final(raw):
    """
    清洗方言字音数据中的韵母.

    Parameters:
        raw (`pandas.Series`): 方言字音韵母列表

    Returns:
        clean (`pandas.Series`): 清洗后的方言字音韵母列表
    """

    return raw.str.replace('A', 'ᴀ') \
        .str.replace('E', 'ᴇ') \
        .str.replace('I', 'ɪ') \
        .str.replace(':', 'ː') \
        .str.replace('：', 'ː') \
        .str.replace(f'[^{ipa}]', '', regex=True)

def clean_tone(raw):
    """
    清洗方言字音数据中的声调.

    Parameters:
        raw (`pandas.Series`): 方言字音声调列表

    Returns:
        clean (`pandas.Series`): 清洗后的方言字音声调列表
    """

    return raw.str.replace(r'[^1-5↗]', '', regex=True)

def clean_data(raw, minfreq=2):
    """
    清洗方言字音数据中的录入错误.

    Parameters:
        raw (`pandas.DataFrame`): 方言字音数据表
        minfreq (int): 在一种方言中出现次数少于该值的声韵调会从该方言中删除

    Returns:
        clean (`pandas.DataFram`): 清洗后的方言字音数据表
    """

    clean = pandas.DataFrame()

    for col in raw.columns:
        if col == 'initial':
            # 部分音节切分错误，重新切分
            clean.loc[raw[col] == 'Ǿŋ', col] = '∅'
            clean.loc[raw[col] == 'ku', col] = 'k'
            clean[col] = clean_initial(raw[col])

        elif col == 'finals':
            # 部分音节切分错误，重新切分
            clean.loc[raw['initial'] == 'Ǿŋ', col] = 'ŋ'
            mask = raw['initial'] == 'ku'
            clean.loc[mask, col] = 'u' + raw.loc[mask, col]
            clean[col] = clean_final(raw[col])

        elif col == 'tone':
            # 部分声调被错误转为日期格式，还原成数字
            mask = raw[col].str.match(r'^\d+年\d+月\d+日$', na='')
            clean.loc[mask, col] = pandas.to_datetime(
                raw.loc[mask, col],
                format=r'%Y年%m月%d日'
            ).dt.dayofyear.astype(str)

            clean.loc[~mask, col] = raw.loc[~mask, col]
            clean[col] = clean_tone(clean[col])

        elif col == '聲母':
            clean[col] = clean_initial(raw[col])

        elif col == '韻母':
            clean[col] = clean_final(raw[col])

        elif col == '調值':
            clean[col] = clean_tone(raw[col])

        elif col == '調類':
            clean[col] = raw[col].replace(
                r'[^上中下變陰陽平去入輕聲]',
                '',
                regex=True
            )

        else:
            clean[col] = raw[col]

        if col in ('initial', 'finals', 'tone', '聲母', '韻母', '調值', '調類') \
            and minfreq > 1:
            # 删除出现次数少的读音
            clean.loc[
                clean.groupby(col)[col].transform('count') < minfreq,
                col
            ] = ''

        mask = clean[col] != raw[col]
        if numpy.count_nonzero(mask):
            for (r, c), cnt in pandas.DataFrame({
                'raw': raw.loc[mask, col],
                'clean': clean.loc[mask, col]
            }).value_counts().items():
                logging.warning(f'replace {r} -> {c} {cnt}')

    return clean

def load_data(prefix, *ids, suffix='.csv', general_names=True):
    """
    加载方言字音数据.

    Parameters:
        prefix (str): 方言字音数据文件路径的前缀
        ids (iterable): 要加载的方言代码列表，完整的方言字音文件路径由代码加上前后缀构成，
            当为空时，加载路径中所有匹配指定后缀的文件
        suffix (str): 方言字音数据文件路径的后缀
        general_names (bool): 为真时，把数据列名转为通用名称

    Returns:
        data (`pandas.DataFrame`): 方言字音表
    """

    if len(ids) == 0:
        ids = sorted(e.name[:-len(suffix)] for e in os.scandir(prefix) \
            if e.is_file() and e.name.endswith(suffix))

    if len(ids) == 0:
        logging.error(f'no data file matching suffix {suffix} in {prefix}!')
        return

    logging.info(f'loading data from {prefix} ...')

    dialects = []
    for id in ids:
        try:
            fname = os.path.join(prefix, id + suffix)
            logging.info(f'loading {fname} ...')
            d = pandas.read_csv(
                fname,
                encoding='utf-8',
                dtype=str,
                na_filter=False
            )

        except Exception as e:
            logging.error(f'cannot load file {fname}: {e}', exc_info=True)
            continue

        d = clean_data(d)
        d.insert(0, 'lid', id)

        if general_names:
            # 替换列名为统一的名称
            d.rename(columns={
                'iid': 'cid',
                'name': 'character',
                'finals': 'final',
                '字號': 'cid',
                '字': 'character',
                '聲母': 'initial',
                '韻母': 'final',
                '調值': 'tone',
                '調類': 'tone_category',
                '備註': 'memo'
            }, inplace=True)
            d['cid'] = d['cid'].astype(int)

        dialects.append(d)

    logging.info(f'done, {len(dialects)} data file loaded')
    return pandas.concat(dialects, axis=0, ignore_index=True)

def transform_data(data, index='lid', values=None, aggfunc=' '.join):
    """
    把方言读音数据长表转换为宽表.

    当 index 为 lid 时，以地点为行，字为列，声韵调为子列。
    当 index 为 cid 时，以字为行，地点为列，声韵调为子列。

    Parameters:
        data (`pandas.DataFrame`): 原始方言读音数据表
        index (str): 指明以原始表的哪一列为行，lid 一个地点为一行，cid 一个字为一行
        aggfunc (str or callable): 相同的 lid 和 cid 有多个记录的，使用 aggfunc 函数合并

    Returns:
        output (`pandas.DataFrame`): 变换格式后的数据表
    """

    output = data.pivot_table(
        values,
        index=index,
        columns='cid' if index == 'lid' else 'lid',
        aggfunc=aggfunc,
        fill_value='',
        sort=False
    )

    if output.columns.nlevels <= 1:
        return output
    else:
        # 如果列名为多层级，把指定的列名上移到最高层级
        return output.swaplevel(axis=1).reindex(pandas.MultiIndex.from_product((
            output.columns.levels[1],
            output.columns.levels[0]
        )), axis=1)