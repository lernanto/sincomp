# -*- coding: utf-8 -*-

"""
测试用例的公用函数
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import unittest
import unittest.mock
import os
import tempfile
import urllib.parse
import urllib.request
import numpy
import pandas


data_dir = os.path.join(os.path.dirname(__file__), 'data')
tmp_dir = tempfile.TemporaryDirectory().name


def make_wide():
    """Build a small (dialect, feature) wide pronunciation table."""
    cells = [
        ["b", "ei", "51", "p", "ei", "24"],
        ["p", "ɔ", "213", "p", "ɔ", "53"],
        ["b", "ɔ", "24", "b", "ei", "51"],
        ["m", "ei", "51", "", "", ""],
    ]
    columns = pandas.MultiIndex.from_tuples([
        ("beijing", "initial"),
        ("beijing", "final"),
        ("beijing", "tone"),
        ("guangzhou", "initial"),
        ("guangzhou", "final"),
        ("guangzhou", "tone"),
    ])
    return pandas.DataFrame(
        cells,
        columns=columns,
        index=["char1", "char2", "char3", "char4"],
    )


def make_rules():
    """Build a small sound-change rule table."""
    rules = pandas.DataFrame([
        {
            "id": "r1",
            "feature": "initial",
            "cid1": ["char1", "char3"],
            "cid2": ["char2"],
        },
        {
            "id": "r2",
            "feature": "tone",
            "cid1": ["char1"],
            "cid2": ["char2", "char3"],
        },
    ])
    return rules.set_index("id")




def mock_urlopen(url, *args, **kwargs):
    if isinstance(url, urllib.request.Request):
        url = url.full_url

    return open(
        os.path.join(
            data_dir,
            urllib.parse.urlparse(url).path.split('/')[-1]
        ),
        'rb'
    )


def setUpModule():
    """为测试产生的文件创建临时目录，为数据集设置环境变量，使用模拟函数取代真正的网络请求"""

    global env_patcher
    global urlopen_patcher

    env_patcher = unittest.mock.patch.dict(os.environ, {
        'SINCOMP_CACHE': os.path.join(tmp_dir, 'cache'),
        'ZHONGGUOYUYAN_HOME': os.path.join(data_dir, 'zhongguoyuyan')
    })
    urlopen_patcher = unittest.mock.patch.object(
        urllib.request,
        'urlopen',
        mock_urlopen
    )

    env_patcher.start()
    urlopen_patcher.start()

    import importlib
    import sincomp.datasets
    importlib.reload(sincomp.datasets)


def tearDownModule():
    urlopen_patcher.stop()
    env_patcher.stop()

    import importlib
    import sincomp.datasets
    importlib.reload(sincomp.datasets)