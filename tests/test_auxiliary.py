# -*- coding: utf-8 -*-

"""
测试辅助函数
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import unittest
import numpy
import sincomp.auxiliary


class TestAuxiliary(unittest.TestCase):
    def test_split_data(self):
        ori = numpy.random.choice(20, size=100)
        a1, b1 = sincomp.auxiliary.split_data(ori, random_state=123)
        a2, b2 = sincomp.auxiliary.split_data(ori, random_state=123)
        self.assertTrue(numpy.all(a1 == a2) and numpy.all(b1 == b2))
