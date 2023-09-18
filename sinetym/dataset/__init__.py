# -*- coding: utf-8 -*-

"""
处理汉语方言读音数据的工具集.

当前支持读取：
    - 小学堂汉字古今音资料库的现代方言数据，见：https://xiaoxue.iis.sinica.edu.tw/ccr
    - 中国语言资源保护工程采录展示平台的方言数据，见：https://zhongguoyuyan.cn/
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import logging
import pandas
import numpy
import functools


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


class Dataset:
    """
    数据集基类，支持延迟加载.

    子类必须实现 load() 和 load_metadata() 函数。
    """

    def __init__(self, name):
        """
        Parameters:
            name (str): 数据集名称
        """

        self.name = name

    @functools.cached_property
    def data(self):
        return self.load()

    @functools.lru_cache
    def transform(self, index='did', values=None, aggfunc=' '.join):
        """
        把方言读音数据长表转换为宽表.

        当 index 为 did 时，以地点为行，字为列，声韵调为子列。
        当 index 为 cid 时，以字为行，地点为列，声韵调为子列。

        Parameters:
            index (str): 指明以原始表的哪一列为行，did 一个地点为一行，cid 一个字为一行
            aggfunc (str or callable): 相同的 did 和 cid 有多个记录的，使用 aggfunc 函数合并

        Returns:
            output (`pandas.DataFrame`): 变换格式后的数据表
        """

        if self.data is None:
            return None

        output = self.data.pivot_table(
            values,
            index=index,
            columns='cid' if index == 'did' else 'did',
            aggfunc=aggfunc,
            fill_value='',
            sort=False
        )

        # 如果列名为多层级，把指定的列名上移到最高层级
        if transorm_data.columns.nlevels > 1:
            transorm_data = output.swaplevel(axis=1).reindex(
                pandas.MultiIndex.from_product((
                    transorm_data.columns.levels[1],
                    transorm_data.columns.levels[0]
                )),
                axis=1
            )

        return output

    def __iter__(self):
        return iter(()) if self.data is None else iter(self.data)

    def __getitem__(self, key):
        return None if self.data is None else self.data.__getitem__(key)

    def __getattr__(self, name):
        return None if self.data is None else self.data.__getattr__(name)

    def reset(self):
        """
        清除已经加载的数据和所有缓存.
        """

        self.__dict__.pop('data', None)
        self.__dict__.pop('metadata', None)
