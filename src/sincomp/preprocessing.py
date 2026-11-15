# -*- coding: utf-8 -*-

"""Preprocess dialect pronunciation tables.

:func:`transform` converts the long-form pronunciation records into
the wide feature matrices (one row per character, one column per
dialect feature) consumed by the estimators of the package.
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"


from ._internal.preprocessing import transform
