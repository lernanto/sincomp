# -*- coding: utf-8 -*-

"""
Dialect pronunciation datasets.

This module is the public entry point of :mod:`sincomp._internal.datasets`.
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
internal, so always create datasets through :func:`get`. Remote
sources are downloaded lazily and cached under the directory given by
the ``SINCOMP_CACHE`` environment variable.

Examples
--------
>>> import sincomp.datasets
>>> dataset = sincomp.datasets.get("MCPDict")             # doctest: +SKIP
>>> dataset.get_data("1800長沙").head()                   # doctest: +SKIP
>>> custom = sincomp.datasets.get("path/to/dialects")     # doctest: +SKIP
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'

from ._internal.datasets import (
    beidazihui,
    ccr,
    get,
    mcpdict,
    predefined,
    zhongguoyuyan,
)

__all__ = [
    "beidazihui",
    "ccr",
    "get",
    "mcpdict",
    "predefined",
    "zhongguoyuyan",
]
