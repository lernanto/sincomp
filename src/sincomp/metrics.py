# -*- coding: utf-8 -*-

"""Compute pairwise similarities between dialects.

This module provides :func:`dialect_similarities`, a dispatcher over
the pairwise dialect similarity measures implemented in the private
``sincomp._internal.similarity`` module.
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"


import numpy

from ._internal.similarity import chi2, entropy


def dialect_similarities(
    src: numpy.ndarray,
    dest: numpy.ndarray | None = None,
    metric: str = "chi2",
    n_jobs: int | None = None,
    **kwargs
):
    """Compute the pairwise similarities between dialects.

    Two dialects are similar when the pronunciations of one dialect predict
    the pronunciations of the other one well. The predictive power is measured
    either by the normalized chi-square statistic (``metric="chi2"``,
    larger is more similar) or by the conditional entropy
    (``metric="entropy"``, smaller is more similar) over the
    co-occurrences of the phone segments of every dialect pair.

    Parameters
    ----------
    src : numpy.ndarray
        Source dialect table with one row per character and one block
        of ``feature_num`` columns per dialect holding the whitespace
        separated phone transcriptions; empty strings mark missing
        values.
    dest : numpy.ndarray, default=None
        Target dialect table holding the same number of features per
        dialect as `src`; when ``None``, the similarities of `src`
        with itself are computed.
    metric : str, default="chi2"
        The similarity measure, either ``"chi2"`` or ``"entropy"``.
    n_jobs : int, default=None
        Number of parallel jobs; ``None`` runs sequentially.
    **kwargs
        Additional keyword arguments passed to the underlying measure,
        e.g. ``feature_num`` and ``blocksize``.

    Returns
    -------
    numpy.ndarray
        Similarity matrix of shape ``(n_src_dialects, n_dest_dialects)``.

    Raises
    ------
    ValueError
        If `metric` is not one of ``"chi2"`` and ``"entropy"``.

    Examples
    --------
    Compare two dialects, each represented by three phone features:

    >>> import numpy as np
    >>> from sincomp.metrics import dialect_similarities
    >>> features = np.array([
    ...     ["p", "a", "1", "p", "a", "1"],
    ...     ["t", "a", "2", "t", "a", "2"],
    ...     ["k", "i", "1", "k", "i", "1"],
    ...     ["p", "u", "2", "b", "u", "2"],
    ...     ["t", "o", "1", "t", "o", "1"],
    ...     ["k", "e", "2", "g", "e", "2"],
    ... ], dtype=str)
    >>> similarities = dialect_similarities(features, feature_num=3)
    >>> similarities.shape
    (2, 2)
    """

    if metric == "chi2":
        measure = chi2
    elif metric == "entropy":
        measure = entropy
    else:
        raise ValueError(
            f"invalid metric {metric!r}, expected 'chi2' or 'entropy'."
        )

    return measure(
        src,
        dest,
        parallel=1 if n_jobs is None else n_jobs,
        **kwargs
    )
