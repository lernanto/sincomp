# -*- coding: utf-8 -*-

"""
Neural models for dialect pronunciation data.

This module is the public entry point of :mod:`sincomp._internal.models`.
It provides scikit-learn compatible neural building blocks on top of
TensorFlow for encoding dialect pronunciation tables and reconstructing
readings between dialects.

Classes
-------

- :class:`Processor`: preprocesses long-form dialect tables into
  integer-encoded tensors, mapping symbols through pre-built vocabularies
  with dedicated codes for missing and unknown values.
- :class:`EncoderBase`: base class of the encoder models, wrapping a
  TensorFlow Keras model with fit/predict helpers following the
  scikit-learn estimator interface.
- :class:`BilinearEncoder`: an encoder whose dialect embedding is the
  product of learned character and feature embedding matrices.

TensorFlow is an optional dependency: the names below are importable
only when it is installed.

Examples
--------
>>> import sincomp.nn
>>> processor = sincomp.nn.Processor(
...     dialect_vocabs, char_vocabs, target_vocabs)  # doctest: +SKIP
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'

try:
    from ._internal.models import (
        BilinearEncoder,
        EncoderBase,
        Processor,
    )
except ImportError:  # pragma: no cover
    pass

__all__ = [
    "BilinearEncoder",
    "EncoderBase",
    "Processor",
]
