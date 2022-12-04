# -*- encoding: utf-8 -*-

"""
公共的数据处理函数.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


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
    return raw.fillna('') \
        .str.replace('[0\u00f8\u01fe\u01ff]', '∅', regex=True) \
        .str.replace('\ufffb', '_') \
        .str.replace('\u02a3', 'dz') \
        .str.replace('\u02a4', 'dʒ') \
        .str.replace('\u02a5', 'dʑ') \
        .str.replace('\u02a6', 'ts') \
        .str.replace('\u02a7', 'tʃ') \
        .str.replace('\u02a8', 'tɕ') \
        .str.replace('g', 'ɡ') \
        .str.replace('(.)h', r'\1ʰ', regex=True) \
        .str.replace(f'[^{ipa}]', '', regex=True)

def clean_final(raw):
    """
    清洗方言字音数据中的韵母.

    Parameters:
        raw (`pandas.Series`): 方言字音韵母列表

    Returns:
        clean (`pandas.Series`): 清洗后的方言字音韵母列表
    """

    return raw.fillna('') \
        .str.replace('A', 'ᴀ') \
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

    return raw.fillna('').str.replace(r'[^1-5↗]', '', regex=True)