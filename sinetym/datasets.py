# -*- coding: utf-8 -*-

"""
汉语方言读音数据集.

当前包含如下数据集：
    - 小学堂汉字古今音资料库的现代方言数据，见：https://xiaoxue.iis.sinica.edu.tw/ccr
    - 中国语言资源保护工程采录展示平台的方言数据，见：https://zhongguoyuyan.cn/
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import os

from .dataset.xiaoxuetang import XiaoxuetangDataset
from .dataset.zhongguoyuyan import ZhongguoyuyanDataset


xiaoxuetang = XiaoxuetangDataset(os.environ.get('XIAOXUETANG_HOME'))
zhongguoyuyan = ZhongguoyuyanDataset(os.environ.get('ZHONGGUOYUYAN_HOME'))
