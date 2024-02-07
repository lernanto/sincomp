# -*- coding: utf-8 -*-

"""
处理汉语方言读音数据的工具集.

当前支持读取：
    - 小学堂汉字古今音资料库的现代方言数据，见：https://xiaoxue.iis.sinica.edu.tw/ccr
    - 中国语言资源保护工程采录展示平台的方言数据，见：https://zhongguoyuyan.cn/
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import collections
import pandas
import numpy
import functools
from sklearn.neighbors import KNeighborsClassifier


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

def predict_group(
    features: pandas.DataFrame | numpy.ndarray,
    labels: pandas.Series | numpy.ndarray[str]
) -> pandas.Series | numpy.ndarray[str]:
    """
    使用 KNN 算法根据经纬度信息预测方言区

    Parameters:
        features: 作为预测特征的方言点信息
        labels: 从原始信息中提取的方言区信息，无法获取方言区的为空字符串

    Returns:
        predict: 带预测的方言区信息，已知的方言区保持不变，其余使用 KNN 预测
    """

    predict = labels.copy()

    mask = numpy.all(numpy.isfinite(features), axis=1)
    predict[mask & (labels == '')] = KNeighborsClassifier().fit(
        features[mask & (labels != '')],
        labels[mask & (labels != '')]
    ).predict(features[mask & (labels == '')])

    return predict


class Dataset:
    """数据集基类."""

    def __init__(self, name, data=None, metadata=dict()):
        """
        Parameters:
            name (str): 数据集名称
            data (`pandas.DataFrame`): 方言字音数据表
            metadata (dict): 数据集元信息
        """

        self.name = name
        self._data = data
        self.metadata = metadata

    @property
    def data(self):
        return self._data

    @functools.lru_cache
    def _force_complete(self, columns=None):
        """
        保证一行数据指定的列都有值，否则删除该读音.

        Parameters:
            columns (array-like): 需要保证有值的列，空表示所有列

        Returns:
            output (`pandas.DataFrame`): 删除不完整读音后的方言字音数据表
        """

        if columns is None:
            columns = self.data.columns
        else:
            columns = list(columns)

        invalid = (self.data[columns].isna() | (self.data[columns] == '')) \
            .any(axis=1)
        return self.data.drop(self.data.index[invalid])

    def force_complete(self, columns=['initial', 'final', 'tone']):
        """
        保证一行数据指定的列都有值，否则删除该读音.

        Parameters:
            columns (array-like): 需要保证有值的列

        Returns:
            other (`sinetym.dataset.Dataset`): 删除不完整读音后的方言字音数据集
        """

        if self.data is None:
            return Dataset(self.name, metadata=self.metadata)

        try:
            columns = tuple(columns)
        except ValueError:
            ...

        return Dataset(
            self.name,
            data=self._force_complete(columns=columns),
            metadata=self.metadata
        )

    @functools.lru_cache
    def _transform(self, index='did', values=None, aggfunc=' '.join):
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

        output = self.data.pivot_table(
            values,
            index=index,
            columns='cid' if index == 'did' else 'did',
            aggfunc=aggfunc,
            fill_value='',
            sort=False
        )

        # 如果列名为多层级，把指定的列名上移到最高层级
        if output.columns.nlevels > 1:
            output = output.swaplevel(axis=1).reindex(
                pandas.MultiIndex.from_product((
                    output.columns.levels[1],
                    output.columns.levels[0]
                )),
                axis=1
            )

        return output

    def transform(
        self,
        index: str = 'did',
        values: list[str] = ['initial', 'final', 'tone'],
        aggfunc: collections.abc.Callable | str = ' '.join
    ):
        """
        把方言读音数据长表转换为宽表

        当 index 为 did 时，以地点为行，字为列，声韵调为子列。
        当 index 为 cid 时，以字为行，地点为列，声韵调为子列。

        Parameters:
            index: 指明以原始表的哪一列为行，did 一个地点为一行，cid 一个字为一行
            values: 用于变换的列，变换后成为二级列
            aggfunc: 相同的 did 和 cid 有多个记录的，使用 aggfunc 函数合并

        Returns:
            other: 变换格式后的数据集
        """

        if self.data is None:
            return Dataset(self.name, metadata=self.metadata)

        try:
            values = tuple(values)
        except (TypeError, ValueError):
            ...

        return Dataset(
            self.name,
            data=self._transform(index, values, aggfunc),
            metadata=self.metadata
        )

    def __iter__(self):
        return iter(()) if self.data is None else iter(self.data)

    def __getitem__(self, key):
        return None if self.data is None else self.data.__getitem__(key)

    def __getattr__(self, name):
        if name.startswith('_'):
            raise(AttributeError(
                f'{repr(type(self).__name__)} object has no attribute {repr(name)}',
                name=name,
                obj=self
            ))

        return None if self.data is None else self.data.__getattr__(name)

    def __repr__(self):
        return f'{type(self).__name__}({repr(self.name)})'

    def __str__(self):
        return self.name

def concat(*datasets):
    """
    把多个数据集按顺序拼接为一个.

    Parameters:
        datasets (iterable of Dataset or `pandas.DataFrame`): 待拼接的数据集

    Returns:
        output (Dataset): 拼接后的数据集
    """

    return Dataset(
        'dataset',
        data=pandas.concat(
            [d.data if isinstance(d, Dataset) else d for d in datasets],
            axis=0,
            ignore_index=True
        )
    )
