# -*- coding: utf-8 -*-

"""
处理汉语方言读音数据的工具集.

当前支持读取：
    - 小学堂汉字古今音资料库的现代方言数据，见：https://xiaoxue.iis.sinica.edu.tw/ccr
    - 中国语言资源保护工程采录展示平台的方言数据，见：https://zhongguoyuyan.cn/
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import pandas
import functools


_ipa = 'a-z\u00c0-\u03ff\u1d00-\u1dbf\u1e00-\u1eff\u207f\u2205\u2c60-\u2c7f' \
    '\ua720-\ua7ff\uab30-\uab6f\ufb00-\ufb4f\uff1a\ufffb' \
    '\U00010780-\U000107ba\U0001df00-\U0001df1e'

# 清洗声母的字符映射表
_initial_table = {
    0x0030: 0x2205,     # ∅
    0x0067: 0x0261,     # ɡ
    0x00d8: 0x2205,     # ∅
    0x00f8: 0x2205,     # ∅
    0x1d50: 0x006d,     # m
    0x1d51: 0x014b,     # ŋ
    0x1d5b: 0x1db9,     # superscript ʋ
    0x01b2: 0x028b,     # ʋ
    0x01fe: 0x2205,     # ∅
    0x01ff: 0x2205,     # ∅
    0x0241: 0x0294,     # ʔ
    0x207f: 0x006e,     # n
    0xff4b: 0x006b,     # k
    0xfffb: 0x005f      # _
}

# 清洗韵母的字符映射表
_final_table = {
    0x003a: 0x02d0,     # ː
    0x0041: 0x1d00,     # ᴀ
    0x0045: 0x1d07,     # ᴇ
    0x0049: 0x026a,     # ɪ
    0x0059: 0x028f,     # ʏ
    0x0067: 0x0261,     # ɡ
    0x0241: 0x0294,     # ʔ
    0x02dc: 0x0303,     # nasalize
    0x0307: 0x0329,     # syllabic
    0x034a: 0x0303,     # nasalize
    0x1d02: 0x00e6,     # æ
    0x1d5b: 0x1db9,     # superscript ʋ
    0xff1a: 0x02d0,     # ː
}

def clean_initial(raw):
    """
    清洗方言字音数据中的声母.

    Parameters:
        raw (`pandas.Series`): 方言字音声母列表

    Returns:
        clean (`pandas.Series`): 清洗后的方言字音声母列表
    """

    # 有些符号使用了多种写法，统一成较常用的一种
    return raw.str.translate(_initial_table) \
        .str.lower() \
        .str.replace('\u02a3', 'dz') \
        .str.replace('\u02a4', 'dʒ') \
        .str.replace('\u02a5', 'dʑ') \
        .str.replace('\u02a6', 'ts') \
        .str.replace('\u02a7', 'tʃ') \
        .str.replace('\u02a8', 'tɕ') \
        .str.replace(f'[^{_ipa}]', '', regex=True) \
        .str.replace('([kɡŋhɦ].?)w', r'\1ʷ', regex=True) \
        .str.replace('([kɡŋhɦ].?)[vʋ]', r'\1ᶹ', regex=True) \
        .str.replace('([^ʔ∅])h', r'\1ʰ', regex=True) \
        .str.replace('([^ʔ∅])ɦ', r'\1ʱ', regex=True) \
        .str.replace('([bdɡvzʐʑʒɾ])ʱ', r'\1ʰ', regex=True) \
        .str.replace('([ʰʱ])([ʷᶹ])', r'\2\1', regex=True) \
        .str.replace('(.)∅|∅(.)', r'\1\2', regex=True)

def clean_final(raw):
    """
    清洗方言字音数据中的韵母.

    Parameters:
        raw (`pandas.Series`): 方言字音韵母列表

    Returns:
        clean (`pandas.Series`): 清洗后的方言字音韵母列表
    """

    return raw.str.translate(_final_table) \
        .str.lower() \
        .str.replace('\u00e3', 'ã') \
        .str.replace('\u00f5', 'õ') \
        .str.replace('\u0129|\u0131\u0303', 'ĩ', regex=True) \
        .str.replace('\u0169', 'ũ') \
        .str.replace('\u1ebd', 'ẽ') \
        .str.replace('\u1ef9', 'ỹ') \
        .str.replace('\u01eb', 'o̜') \
        .str.replace('\u1e47', 'n̩') \
        .str.replace(f'[^{_ipa}]', '', regex=True)

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
