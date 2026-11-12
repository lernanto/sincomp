# -*- coding: utf-8 -*-

"""Preprocess dialect pronunciation tables.

This module exposes the preprocessing utilities of SinComp as
scikit-learn compatible estimators and functions. The public interface
consists of:

- :class:`DialectImputer`: a stateless transformer that fills the
  missing readings of a wide pronunciation table (one row per
  character, one column per dialect feature) with the weighted votes
  of the nearest observed characters;
- :func:`transform`: converts long-form pronunciation records into the
  wide feature matrices consumed by the estimators of this module and
  the rest of the package.

The estimators in this module operate on the wide feature tables
produced by :func:`transform`. Like every scikit-learn transformer,
they support the ``fit``/``transform``/``fit_transform`` lifecycle,
which makes them composable into
:class:`sklearn.pipeline.Pipeline` objects.

Examples
--------
>>> import pandas
>>> from sincomp.preprocessing import DialectImputer
>>> X = pandas.DataFrame(
...     [["b ei 51", "p ɔ 213"], ["p ɔ 213", ""]],
...     columns=["d1_initial", "d1_final"]
... )
>>> DialectImputer().fit_transform(X).shape
(2, 2)
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"


import numpy
import pandas
import sklearn.base
from typing import Optional, Sequence, Union

from ._internal import preprocessing
from ._internal.preprocessing import transform


class DialectImputer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Fill the missing readings of a wide pronunciation table.

    The input is a wide table with one row per character and one column
    per dialect feature, e.g. the initial, final or tone of one dialect.
    Missing values are marked with empty strings or ``pandas.NA``. For
    every missing value, the readings of the ``n_neighbors`` nearest
    characters with an observed value vote for the replacement, weighted
    inversely by their distance. The distances are computed over the
    observed feature values of the characters with the KNN algorithm.

    The transformer is stateless: fitting only records the shape of the
    input, all the work happens in :meth:`transform`.

    Parameters
    ----------
    n_neighbors : int, default=3
        Number of nearest characters whose readings vote for every
        missing value.
    n_jobs : int, default=1
        Number of parallel jobs used to impute the dialect features.

    Attributes
    ----------
    n_features_in_ : int
        Number of input columns seen during fitting.
    feature_names_in_ : pandas.Index
        Labels of the input columns seen during fitting.

    Examples
    --------
    >>> import pandas
    >>> from sincomp.preprocessing import DialectImputer
    >>> X = pandas.DataFrame(
    ...     [["b ei 51", "p ɔ 213"], ["p ɔ 213", ""]],
    ...     columns=["d1_initial", "d1_final"]
    ... )
    >>> DialectImputer().fit_transform(X).shape
    (2, 2)
    """

    def __init__(self, n_neighbors: int = 3, n_jobs: int = 1):
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

    def fit(
        self,
        X: Union[pandas.DataFrame, numpy.ndarray],
        y: Optional[numpy.ndarray] = None
    ) -> "DialectImputer":
        """Record the shape of the input table.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Wide input table with one row per character and one column
            per dialect feature.
        y : numpy.ndarray, default=None
            Ignored. This parameter exists for compatibility with the
            scikit-learn transformer API.

        Returns
        -------
        DialectImputer
            Fitted transformer.
        """

        self.n_features_in_ = X.shape[1]
        if isinstance(X, pandas.DataFrame):
            self.feature_names_in_ = X.columns

        return self

    def transform(
        self,
        X: Union[pandas.DataFrame, numpy.ndarray]
    ) -> Union[pandas.DataFrame, numpy.ndarray]:
        """Fill the missing readings of the input table.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Wide input table in the same column structure as the data
            used during fitting, missing values marked with empty
            strings or ``pandas.NA``.

        Returns
        -------
        pandas.DataFrame or numpy.ndarray
            Imputed table of the same type and shape as the input.
        """

        if not hasattr(self, "n_features_in_"):
            raise ValueError(
                "This DialectImputer instance is not fitted yet."
            )

        return preprocessing.impute(
            X,
            n_neighbors=self.n_neighbors,
            n_jobs=self.n_jobs
        )

    def get_feature_names_out(
        self,
        input_features: Optional[Sequence[str]] = None
    ) -> numpy.ndarray:
        """Get the labels of the output columns.

        Parameters
        ----------
        input_features : Sequence[str], default=None
            Ignored. The output columns match the input columns.

        Returns
        -------
        numpy.ndarray
            Labels of the output columns.
        """

        if not hasattr(self, "n_features_in_"):
            raise ValueError(
                "This DialectImputer instance is not fitted yet."
            )

        if input_features is None:
            input_features = getattr(
                self,
                "feature_names_in_",
                [f"x{i}" for i in range(self.n_features_in_)]
            )

        return numpy.asarray(input_features)
