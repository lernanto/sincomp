# -*- coding: utf-8 -*-

"""
处理小学堂字音数据的功能函数.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import pandas


def load_location(path):
    """
    加载方言点信息.

    Parameters:
        path (str): 方言点文件路径，CSV 格式

    Returns:
        data (`pandas.DataFrame`): 方言点信息数据表，只包含文件中编号非空的方言点
    """

    location = pandas.read_csv(path, dtype={'編號': str})
    return location[location['編號'].notna()].set_index('編號')