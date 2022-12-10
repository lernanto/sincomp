# -*- coding: utf-8 -*-

"""
处理小学堂字音数据的功能函数.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import os
import logging
import numpy
import pandas
import jsonlines

from . import common


def load_json(fname):
    """
    载入 JSON lines 格式的方言字音数据.

    Parameters:
        fname (str): 方言字音数据文件，每行为一个 JSON 字符串

    Returns:
        dialect (str): 方言名称
        data (`pandas.DataFrame`): 载入的方言读音数据表
    """

    # 尝试根据第一条记录确定是什么方言，查找记录中第一个类型为数组的元素
    with jsonlines.open(fname) as f:
        try:
            dialect = next(k for r in f for k, v in r.items() if isinstance(v, list))
        except StopIteration:
            logging.error(f'{fname} format error!')
            return

    with jsonlines.open(fname) as f:
        data = pandas.json_normalize(
            (r for r in jsonlines.open(fname) if dialect in r),
            record_path=dialect,
            meta=['id', '字形']
        )

    return dialect, data

def load_data(prefix, *dialects, suffix='.csv', sep=' '):
    """
    加载方言字音数据.

    Parameters:
        prefix (str): 方言字音数据文件路径前缀
        dialects (iterable): 方言列表，每个方言对应一个 CSV 文件，
            当为空时，加载路径下所有匹配指定后缀的文件
        suffix (str): 文件后缀
        sep (str): 分隔多音字的多个音的分隔符

    Returns:
        data (`pandas.DataFrame`): 方言字音数据表
    """

    if len(dialects) == 0:
        dialects = sorted(e.name[:-len(suffix)] for e in os.scandir(prefix) \
            if e.is_file() and e.name.endswith(suffix))

    if len(dialects) == 0:
        logging.error(f'no data file matching suffix {suffix} in {prefix}!')
        return

    logging.info(f'loading data from {prefix} ...')

    data_list = []
    for d in dialects:
        fname = os.path.join(prefix, d + suffix)
        logging.info(f'loading {fname} ...')
        data = clean_data(pandas.read_csv(
            fname,
            encoding='utf-8',
            dtype={'id': int, '聲母': str, '韻母': str, '調值': str, '調類': str}
        ), sep=sep)

        # 不同方言文件中可能存在相同的方言点名称，加上方言名称以区别
        data.insert(0, 'lid', f'{d}_' + data.pop('方言點'))
        # 替换列名为统一的名称
        data.rename(
            columns={
                'id': 'cid',
                '字形': 'character',
                '聲母': 'initial',
                '韻母': 'final',
                '調值': 'tone',
                '備註': 'memo'
            },
            inplace=True
        )
        data_list.append(data.sort_values(['lid', 'cid']))

    logging.info(f'done, {len(data_list)} data files loaded.')
    return pandas.concat(data_list, axis=0, ignore_index=True)

def clean_data(raw, minfreq=2, sep=' '):
    """
    清洗方言字音数据中的录入错误.

    Parameters:
        raw (`pandas.DataFrame`): 原始方言字音数据表
        minfreq (int): 在一种方言中出现次数少于该值的声韵调会从该方言中删除
        sep (str): 分隔多音字的多个音的分隔符

    Returns:
        clean (`pandas.DataFrame`): 清洗后的方言字音数据表
    """

    clean = pandas.DataFrame()
    for name in raw.columns:
        if name in ('聲母', '韻母', '調值', '調類'):
            raw_col = raw[name].fillna('').str.split('/').explode()

            if name == '聲母':
                col = common.clean_initial(raw_col)
            elif name == '韻母':
                col = common.clean_final(raw_col)
            elif name == '調值':
                col = common.clean_tone(raw_col)
            elif name == '調類':
                col = raw_col.replace(r'[^上中下變陰陽平去入輕聲]', '', regex=True)

            # 删除出现次数少的读音
            if minfreq > 1:
                col[pandas.DataFrame({'方言點': raw['方言點'], name: col}) \
                    .groupby(['方言點', name])[name].transform('count') < minfreq] = ''

            mask = col != raw_col
            if numpy.count_nonzero(mask):
                for (r, c), cnt in pandas.DataFrame({
                    'raw': raw_col[mask],
                    'clean': col[mask]
                }).value_counts().items():
                    logging.warning(f'{name} {r} -> {c} {cnt}')

            clean[name] = col.groupby(col.index).agg(sep.join)

        else:
            clean[name] = raw[name]

    return clean

def expand_polyphone(data, sep=' '):
    """
    把数据中的多音字展开成多行.

    数据中每个方言的声母、韵母、声调分别单独为一列，清洗后每列的多个音以空格分隔。
    每个字在不同方言的读音数量不一样，有些方言有多个音，有些只有一个音，都是多音的读音数量也可能不一样。
    在少数情况下，同一个字在同一个方言中声韵调的数量也不一样，这可能是录入错误导致的，
    也有些多音字的多个音只有声韵调其中部分不同，其他部分是相同的，相同的部分就只录入了一个。
    以上复杂的情况导致数据中每一格包含的元素数量不同，为对齐带来了困难。
    当前只采取简单的方式，即同一行所有列的元素按顺序对齐，元素数量少的列就用第一个元素补足。

    Parameters:
        data (`pandas.DataFrame`): 方言字音数据表
        sep (str): 分隔多音字的多个音的分隔符

    Returns:
        output (`pandas.DataFrame`): 展开多音字后的方言字音数据表，行数为每个字的最大读音数之和
    """

    def get_rows(col, max_row):
        rows = []
        for i, r in enumerate(col):
            if max_row[i] <= 1:
                # 该行不包含多音字，直接追加
                rows.append(r)

            elif isinstance(r, str) and sep in r:
                # 包含多音字，展开
                vals = r.split(sep)
                for j in range(max_row[i]):
                    # 读音数量少于最大读音数的，缺少的部分以第一个读音填充
                    rows.append(vals[j] if j < len(vals) else vals[0])

            else:
                # 该行包含多音字但该列不包含多音字，重复多次
                rows.extend([r] * max_row[i])

        return numpy.asarray(rows, dtype=numpy.object_)

    # 原始数据中有些格子有多个读音需要展开
    # 预先计算每行最多的读音数
    max_row = numpy.asarray(
        [(data.iloc[:, i].str.count(sep) + 1).fillna(0) \
            for (i, n) in enumerate(data.columns.get_level_values(-1)) \
            if n in ('initial', 'final', 'tone', '調類')],
        dtype=numpy.int32
    ).max(axis=0)

    return data.reset_index() \
        .apply(get_rows, axis=0, max_row=max_row) \
        .set_index('cid') \
        .set_axis(data.columns, axis=1)