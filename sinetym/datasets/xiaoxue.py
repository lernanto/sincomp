# -*- coding: utf-8 -*-

"""
处理小学堂字音数据的功能函数.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import re
import numpy
import pandas


def load_data(*files):
    """
    加载方言字音数据.

    Parameters:
        files (iterable): 文件或字符串的列表，每个是一个 CSV 格式的字音文件

    Returns:
        data (`pandas.DataFrame`): 方言字音数据表
    """

    if (len(files) == 1):
        # 只有一个输入文件，直接加载
        return pandas.read_csv(files[0], comment='#', index_col='id', dtype=str)

    elif files > 1:
        # 多个输入文件，分别加载然后拼接
        data = []
        for f in files:
            if isinstance(f, str):
                f = open(f)

            # CSV 文件的第一行是大方言名称
            dialect = re.sub(r'^\s*#', '', next(f)).strip()
            if dialect:
                # 每个列名都加上大方言作为前缀
                data.append(pandas.read_csv(f, index_col='id', dtype=str) \
                        .add_prefix('{}_'.format(dialect)))

        return pandas.concat(data, axis=1)

def clean_data(data):
    """
    清洗方言字音数据中的错误.

    Parameters:
        data (`pandas.DataFrame`): 原始方言字音数据表

    Returns:
        output (`pandas.DataFrame`): 清洗后的方言字音数据表
    """

    def get_rows(col, max_row):
        rows = []
        for i, r in enumerate(col):
            if max_row[i] <= 1:
                # 该行不包含多音字，直接追加
                rows.append(r)

            elif isinstance(r, str) and '/' in r:
                # 包含多音字，展开
                vals = r.split('/')
                for j in range(max_row[i]):
                    # 读音数量少于最大读音数的，缺少的部分以第一个读音填充
                    rows.append(vals[j] if j < len(vals) else vals[0])

            else:
                # 该行包含多音字但该列不包含多音字，重复多次
                rows.extend([r] * max_row[i])

        return numpy.asarray(rows, dtype=numpy.object_)

    # 原始数据中有些格子有多个读音，以斜杠分隔，需要展开
    columns = [c for c in data.columns if c.split('_')[-1] in ('聲母', '韻母', '調類', '調值')]
    # 预先计算每行最多的读音数
    max_row = numpy.asarray(
        [(data[c].str.count('/') + 1).fillna(0) for c in columns],
        dtype=numpy.int32
    ).max(axis=0)
    output = data.reset_index().apply(get_rows, axis=0, max_row=max_row)

    # 删除太长的读音
    for c in columns:
        output.loc[output[c].str.len() > 4, c] = numpy.NaN

    for c in output.columns:
        if c.endswith('聲母'):
            # 只允许国际音标
            output[c].replace(
                r'.*[^0A-Za-z()\u0080-\u03ff\u1d00-\u1d7f\u2070-\u209f].*',
                numpy.NaN,
                inplace=True
            )
            # 零声母统一用0填充
            output[c].replace('', '0', inplace=True)

        elif c.endswith('韻母'):
            output[c].replace(
                r'.*[^0A-Za-z()\u0080-\u03ff\u1d00-\u1d7f\u2070-\u209f].*',
                numpy.NaN,
                inplace=True
            )

        elif c.endswith('調類'):
            output[c].replace(r'.*[^上中下變陰陽平去入].*', numpy.NaN, inplace=True)

        elif c.endswith('調值'):
            output[c].replace(r'.*[^0-9].*', numpy.NaN, inplace=True)

    # 丢弃只出现一次的数据
    for c in output.columns:
        if c.split('_')[-1] in ('聲母', '韻母', '調類', '調值'):
            count = output[c].value_counts()
            output[c].replace(count[count <= 1].index, numpy.NaN, inplace=True)

    return output