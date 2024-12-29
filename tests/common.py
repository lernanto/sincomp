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


data_dir = os.path.join(os.path.dirname(__file__), 'data')
tmp_dir = tempfile.TemporaryDirectory().name


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
    import sincomp.datasets


def tearDownModule():
    urlopen_patcher.stop()
    env_patcher.stop()
