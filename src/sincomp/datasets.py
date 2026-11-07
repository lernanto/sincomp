# -*- coding: utf-8 -*-

"""
Dialect pronunciation datasets.

This module is the public entry point of :mod:`sincomp._internal.dataset`.
It loads the long-form pronunciation records of Chinese dialects -- one
row per dialect and character, with the character ID, the character
itself and the initial, final and tone readings -- and normalizes them
into a common schema, whatever the source format is.

Supported sources
-----------------

- Beida zihui: the dialect data of the 漢語方音字彙 (Peking
  University), https://github.com/lexibank/beidazihui
- CCR: the modern dialects of the Xiaoxue archival database,
  https://xiaoxue.iis.sinica.edu.tw/ccrdata/
- MCPDict: the modern dialects of the 汉字音典,
  https://mcpdict.sourceforge.io/
- zhongguoyuyan: the dialect data of the 中国语言资源保护工程采录
  展示平台, https://zhongguoyuyan.cn/

Usage
-----

The predefined datasets are exposed as ready-to-use module attributes
(:data:`beidazihui`, :data:`ccr`, :data:`mcpdict` and
:data:`zhongguoyuyan`); :func:`get` looks up a predefined dataset by
name, or loads a custom dataset from a CSV file or a directory of
per-dialect CSV files, and :func:`predefined` maps every registered
dataset. The dataset classes are not exposed here: the base types
``Dataset`` and ``FileDataset`` are importable from the package root
(``sincomp.Dataset``), and the concrete dataset classes remain
internal, so always create datasets through :func:`get`. The module
attributes :data:`cache_dir` and :data:`dataset_dir` locate the local
data cache, evaluated at import time from the ``SINCOMP_CACHE``
environment variable.

Examples
--------
>>> import sincomp.datasets
>>> dataset = sincomp.datasets.get("MCPDict")             # doctest: +SKIP
>>> dataset.get_data("1800長沙").head()                   # doctest: +SKIP
>>> custom = sincomp.datasets.get("path/to/dialects")     # doctest: +SKIP
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import logging
import os
import sys

import pandas

from ._internal import dataset


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler())


cache_dir = os.getenv(
    'SINCOMP_CACHE',
    os.path.join(
        os.getenv(
            'LOCALAPPDATA',
            os.path.expanduser('~')
        ) if sys.platform.startswith('win')
        else os.getenv(
            'XDG_DATA_HOME',
            os.path.join(os.path.expanduser('~'), '.local', 'share')
        ),
        'sincomp'
    )
)
dataset_dir = os.path.join(cache_dir, 'datasets')

ccr = dataset.CCRDataset(os.path.join(dataset_dir, 'ccr'))
mcpdict = dataset.MCPDictDataset(os.path.join(dataset_dir, 'mcpdict'))
beidazihui = dataset.BeidazihuiDataset(os.path.join(dataset_dir, 'beidazihui'))
zhongguoyuyan = dataset.ZhongguoyuyanDataset(
    os.path.join(dataset_dir, 'zhongguoyuyan')
)
_datasets = [beidazihui, ccr, mcpdict, zhongguoyuyan]
_dataset_mapping = {
    'beidazihui': beidazihui,
    'CCR': ccr,
    'ccr': ccr,
    'xiaoxue': ccr,
    'MCPDict': mcpdict,
    'mcpdict': mcpdict,
    'zhongguoyuyan': zhongguoyuyan,
    'yubao': zhongguoyuyan
}


def predefined() -> list[dataset.Dataset]:
    """
    List all predefined datasets.

    Returns
    -------
    datasets : list of Dataset
        List of all predefined dataset objects.
    """

    return _datasets.copy()


def get(name: str) -> dataset.Dataset | None:
    """
    获取或加载数据集

    Parameters:
        name: 预定义数据集的名字，或本地数据集的路径

    Returns:
        dataset: 预定义或创建的数据集对象

    优先把 `name` 当作预定义数据集的名字查询，如果不成功，当作本地路径创建数据集。
    """

    try:
        return _dataset_mapping[name]

    except KeyError:
        # 不是预定义数据集，尝试把入参作为路径从本地加载
        logger.info(
            f'{name} is not a predefined dataset, try loading data from files.'
        )

        if os.path.isdir(name):
            # name 是目录，使用目录下的数据创建数据集
            return dataset.FileDataset(data_dir=name)

        elif os.path.isfile(name):
            # name 是文件，直接加载数据后包装成数据集
            return dataset.Dataset(
                pandas.read_csv(name, dtype=str, encoding='utf-8'),
                name=os.path.splitext(os.path.basename(name))[0]
            )

        else:
            logger.warning(f'{name} not found!')
