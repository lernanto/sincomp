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
