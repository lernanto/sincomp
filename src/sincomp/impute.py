# -*- coding: utf-8 -*-

"""Imputation of missing values in wide dialect pronunciation tables.

This module exposes the imputation algorithms of SinComp as
scikit-learn compatible transformers. Both fill the missing values
of a wide pronunciation table (one row per character, one column per
dialect feature) and are stateless: fitting only records the shape of
the input, all the work happens in :meth:`transform`.

- :class:`KNeighborsImputer`: fills every missing cell with the
  distance-weighted vote of the nearest characters with an observed
  value of the same feature;
- :class:`FactorizedImputer`: embeds the characters with a
  factorization of the binary multi-hot encoding of their observed
  values (masked ALS or truncated SVD), projects the embedding into
  one space per feature and fills every missing cell with the
  weighted vote of the nearest observed characters in that space.

Examples
--------
>>> import pandas
>>> from sincomp.impute import KNeighborsImputer
>>> X = pandas.DataFrame(
...     [["b", "p"], ["p", pandas.NA]],
...     columns=["d1_initial", "d2_initial"]
... )
>>> imputed = KNeighborsImputer().fit_transform(X)
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"


import joblib
import numpy
import pandas
import scipy.sparse
import sklearn.base
import sklearn.decomposition
import sklearn.feature_extraction.text
import sklearn.metrics
from typing import Optional, Sequence, Union

from ._internal import factorize, preprocessing


def _impute_column(
    values: numpy.ndarray,
    distances: numpy.ndarray[float],
    n_neighbors: int = 1
) -> numpy.ndarray:
    """Fill the missing values of one column with the weighted vote of
    the nearest characters.

    Every missing value is replaced with the value that receives the
    highest weighted vote among the ``n_neighbors`` nearest characters
    with an observed value, weighted inversely by their distance. If
    several values tie at the highest vote, the first one in the
    vocabulary order wins.

    Parameters
    ----------
    values : numpy.ndarray[str]
        Values of one column with one element per character; missing
        values are marked with empty strings.
    distances : numpy.ndarray[float]
        Precomputed distances between the characters, of shape
        ``(len(values), len(values))``.
    n_neighbors : int, default=1
        Number of nearest characters with an observed value that vote
        for every missing value.

    Returns
    -------
    numpy.ndarray[str]
        Values with the missing ones filled, in the same order as
        ``values``.
    """

    vectorizer = sklearn.feature_extraction.text.CountVectorizer(
        lowercase=False,
        tokenizer=str.split,
        token_pattern=None,
        stop_words=None,
        binary=True,
        dtype=distances.dtype
    )

    missing = values == ''
    observed = ~missing
    output = values.copy()
    if not missing.any() or not observed.any():
        return output

    dist = distances[missing][:, observed]
    idx = numpy.argsort(dist, axis=1)[:, :n_neighbors]
    vec = vectorizer.fit_transform(values)
    # the closer the neighbors are, the higher their weight
    weights = vec[observed].toarray()[idx]
    weights /= numpy.take_along_axis(dist, idx, axis=1)[..., None] + 1e-4
    scores = numpy.sum(weights, axis=1)
    names = vectorizer.get_feature_names_out()

    # the highest voted value wins, ties are resolved by the first one
    # in the vocabulary order
    output[numpy.flatnonzero(missing)] = names[
        numpy.argmax(scores, axis=1)
    ]
    return output


def _knn_impute(
    vectors,
    values: numpy.ndarray,
    n_neighbors: int = 1,
    n_jobs: int = 1
) -> numpy.ndarray:
    """Fill the missing values of one feature's columns with the
    weighted vote of the nearest observed characters.

    The distance between the characters is the cosine distance of
    their rows in ``vectors``. Every missing value is replaced with
    the value that receives the highest weighted vote among the
    ``n_neighbors`` nearest characters with an observed value,
    weighted inversely by their distance.

    Parameters
    ----------
    vectors : numpy.ndarray or scipy.sparse matrix
        Character vectors in the space of one feature, of shape
        ``(n_characters, n_dims)``; they may be sparse multi-hot codes
        or dense embeddings, the imputation only needs their cosine
        distances.
    values : numpy.ndarray
        Values of the feature's columns, of shape
        ``(n_characters, n_columns)``; missing values are marked with
        empty strings.
    n_neighbors : int, default=1
        Number of nearest characters with an observed value that vote
        for every missing value.
    n_jobs : int, default=1
        Number of parallel jobs used to impute the columns.

    Returns
    -------
    numpy.ndarray
        Values with the missing ones filled, of the same shape as
        ``values``.
    """

    dist = sklearn.metrics.pairwise.cosine_distances(vectors)
    output = numpy.empty(values.shape, dtype=object)
    output[:] = values[:]
    for i, filled in enumerate(joblib.Parallel(n_jobs)(
        joblib.delayed(_impute_column)(
            values[:, i],
            dist,
            n_neighbors=n_neighbors
        ) for i in range(values.shape[1])
    )):
        output[:, i] = filled

    return output


class KNeighborsImputer(
    sklearn.base.BaseEstimator, sklearn.base.TransformerMixin
):
    """Fill the missing values of a wide pronunciation table with
    the nearest characters.

    The input is a wide table with one row per character and one column
    per dialect feature, e.g. the initial, final or tone of one dialect.
    Missing values are marked with empty strings or ``pandas.NA``. For
    every missing value, the values of the ``n_neighbors`` nearest
    characters with an observed value vote for the replacement, weighted
    inversely by their distance. The distances between the characters
    are computed over the observed values of the same feature only:
    initials are imputed from initials, finals from finals, and so on.
    A DataFrame with a MultiIndex column label is split into one group
    of columns per feature, the last label level, and the groups are
    imputed one after another; every other input is imputed as a
    whole.

    The transformer is stateless: fitting only records the shape of the
    input, all the work happens in :meth:`transform`.

    Parameters
    ----------
    n_neighbors : int, default=3
        Number of nearest characters whose values vote for every
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
    >>> from sincomp.impute import KNeighborsImputer
    >>> X = pandas.DataFrame(
    ...     [["b", "ei", "51"], ["p", "ɔ", pandas.NA]],
    ...     columns=["d1_initial", "d1_final", "d1_tone"]
    ... )
    >>> KNeighborsImputer().fit_transform(X)
      d1_initial d1_final d1_tone
    0          b       ei      51
    1          p        ɔ      51
    """

    def __init__(self, n_neighbors: int = 3, n_jobs: int = 1):
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

    def fit(
        self,
        X: Union[pandas.DataFrame, numpy.ndarray],
        y: Optional[numpy.ndarray] = None
    ) -> "KNeighborsImputer":
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
        KNeighborsImputer
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
        """Fill the missing values of the input table.

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
                "This KNeighborsImputer instance is not fitted yet."
            )

        is_dataframe = isinstance(X, pandas.DataFrame)
        if is_dataframe:
            data = X.fillna("").to_numpy(dtype=str)
            output = numpy.empty(data.shape, dtype=object)
            output[:] = data
            if isinstance(X.columns, pandas.MultiIndex):
                # characters imputing one feature are compared by that
                # feature only, so the columns are grouped by their
                # feature label, the last level of the column labels
                features = X.columns.get_level_values(-1)
                groups = [
                    numpy.flatnonzero(features == f)
                    for f in features.unique()
                ]
            else:
                groups = [numpy.arange(data.shape[1])]
        else:
            data = X
            output = X.copy()
            groups = [numpy.arange(data.shape[1])]

        for cols in groups:
            # the multi-hot codes of the columns of one feature form
            # the character vectors of that feature
            blocks, _ = preprocessing.features_to_vectors(data[:, cols])
            vectors = scipy.sparse.hstack(blocks, format="csr")
            output[:, cols] = _knn_impute(
                vectors,
                data[:, cols],
                n_neighbors=self.n_neighbors,
                n_jobs=self.n_jobs,
            )

        if is_dataframe:
            return pandas.DataFrame(
                output, index=X.index, columns=X.columns
            )
        return output

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
                "This KNeighborsImputer instance is not fitted yet."
            )

        if input_features is None:
            input_features = getattr(
                self,
                "feature_names_in_",
                [f"x{i}" for i in range(self.n_features_in_)]
            )

        return numpy.asarray(input_features)


class FactorizedImputer(
    sklearn.base.BaseEstimator, sklearn.base.TransformerMixin
):
    """Fill the missing values of a wide pronunciation table with
    factorized character embeddings.

    Every column of the input is binary multi-hot encoded into one
    sparse block, one block per column; the blocks are factorized into
    one embedding per character and one embedding per distinct value
    with the requested algorithm:

    - ``"als"``: masked alternating least squares, where missing cells
      never influence the learned embeddings;
    - ``"svd"``: truncated SVD of the concatenated blocks, where
      missing cells are treated as zeros.

    The character embedding is then projected into one space per
    feature with the singular value decomposition of the value
    embeddings of that feature, and every missing cell is filled with
    the value that receives the highest weighted vote among the
    ``n_neighbors`` nearest observed characters in the space of its
    feature (cosine distance). A DataFrame with a MultiIndex column
    label groups its columns by the last label level; every other
    column forms its own feature.

    Columns with fewer than two observed values or a single distinct
    value cannot predict and are kept unchanged. The transformer
    is stateless: fitting only records the shape of the input, all the
    work happens in :meth:`transform`. Missing values are marked with
    ``pandas.NA``.

    Parameters
    ----------
    algorithm : {"als", "svd"}, default="als"
        The factorization algorithm used to embed the characters.
    n_components : int, default=128
        Dimension of the character and value embeddings.
    max_iter : int, default=10
        Maximum number of ALS iterations, only used with
        ``algorithm="als"``.
    tol : float, default=1e-4
        ALS convergence tolerance, only used with
        ``algorithm="als"``.
    alpha : float, default=1.0
        ALS L2 regularization coefficient, only used with
        ``algorithm="als"``. Regularization strong enough to keep the
        embeddings generalizable is essential, weakly regularized
        embeddings overfit the observed cells of each character.
    n_neighbors : int, default=1
        Number of nearest observed characters whose values vote for
        every missing value.
    random_state : int, numpy.random.RandomState or None, default=None
        Seed of the random initialization. ``None`` draws from the
        global NumPy random state.

    Attributes
    ----------
    n_features_in_ : int
        Number of input columns seen during fitting.
    feature_names_in_ : pandas.Index
        Labels of the input columns seen during fitting.

    Examples
    --------
    >>> import pandas
    >>> from sincomp.impute import FactorizedImputer
    >>> X = pandas.DataFrame(
    ...     [["b", "p"], ["p", pandas.NA]],
    ...     columns=["d1_initial", "d2_initial"]
    ... )
    >>> imputed = FactorizedImputer().fit_transform(X)  # doctest: +SKIP
    """

    def __init__(
        self,
        algorithm: str = "als",
        n_components: int = 128,
        max_iter: int = 10,
        tol: float = 1e-4,
        alpha: float = 1.0,
        n_neighbors: int = 1,
        random_state=None
    ):
        self.algorithm = algorithm
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.n_neighbors = n_neighbors
        self.random_state = random_state

    def fit(
        self,
        X: pandas.DataFrame,
        y: Optional[numpy.ndarray] = None
    ) -> "FactorizedImputer":
        """Record the shape of the input table.

        Parameters
        ----------
        X : pandas.DataFrame
            Wide input table with one row per character and one column
            per dialect feature.
        y : numpy.ndarray, default=None
            Ignored. This parameter exists for compatibility with the
            scikit-learn transformer API.

        Returns
        -------
        FactorizedImputer
            Fitted transformer.
        """

        self.n_features_in_ = X.shape[1]
        if isinstance(X, pandas.DataFrame):
            self.feature_names_in_ = X.columns

        return self

    def transform(self, X: pandas.DataFrame) -> pandas.DataFrame:
        """Fill the missing values of the input table.

        Parameters
        ----------
        X : pandas.DataFrame
            Wide input table in the same column structure as the data
            used during fitting, cells holding whitespace-separated
            values and missing values marked with ``pandas.NA``.

        Returns
        -------
        pandas.DataFrame
            Table of the same shape as ``X`` with the missing values
            filled with predicted values.

        Raises
        ------
        ValueError
            If ``algorithm`` is not ``"als"`` or ``"svd"``.
        """

        if not hasattr(self, "n_features_in_"):
            raise ValueError(
                "This FactorizedImputer instance is not fitted yet."
            )
        if self.algorithm not in ("als", "svd"):
            raise ValueError(
                f"Unknown algorithm {self.algorithm!r}, "
                "expected 'als' or 'svd'."
            )

        output = X.copy()
        data = X.fillna("").to_numpy(dtype=str)
        observed = X.notna().to_numpy()

        # columns with too little observed data carry no usable
        # information for the factorization and are kept unchanged
        valid = [
            j for j in range(X.shape[1])
            if observed[:, j].sum() >= 2
            and len(set(data[observed[:, j], j])) >= 2
        ]
        if not valid:
            return output

        blocks, vocabularies = preprocessing.features_to_vectors(
            data[:, valid]
        )
        if self.algorithm == "als":
            # masked ALS, missing cells never influence the embeddings
            embeddings, components, _ = factorize.als(
                blocks,
                n_components=self.n_components,
                max_iter=self.max_iter,
                tol=self.tol,
                alpha=self.alpha,
                random_state=self.random_state,
            )
        else:
            # truncated SVD treats missing cells as zero rows of their
            # block
            mat = scipy.sparse.hstack(blocks, format="csr")
            size = max(min(self.n_components, mat.shape[1] - 1), 1)
            svd = sklearn.decomposition.TruncatedSVD(
                size, random_state=self.random_state
            )
            embeddings = svd.fit_transform(mat.astype(numpy.float32))
            components = svd.components_

        if isinstance(X.columns, pandas.MultiIndex):
            features = [name[-1] for name in X.columns[valid]]
        else:
            features = list(range(len(valid)))
        limits = numpy.concatenate(
            [[0], numpy.cumsum([len(v) for v in vocabularies])]
        )

        # project the character embedding into one space per feature,
        # the left singular vectors of the value embeddings of that
        # feature, and fill every missing cell with the weighted vote
        # of the nearest observed characters in that space
        for feature in dict.fromkeys(features):
            idx = [i for i, f in enumerate(features) if f == feature]
            component = numpy.concatenate(
                [components[:, limits[i]:limits[i + 1]] for i in idx],
                axis=1,
            )
            u, s, _ = numpy.linalg.svd(component, full_matrices=False)
            feat_emb = embeddings @ (u * s)

            cols = [valid[i] for i in idx]
            filled = _knn_impute(
                feat_emb,
                data[:, cols],
                n_neighbors=self.n_neighbors,
            )
            for i, j in enumerate(cols):
                output.iloc[:, j] = filled[:, i]

        return output

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
                "This FactorizedImputer instance is not fitted yet."
            )

        if input_features is None:
            input_features = getattr(
                self,
                "feature_names_in_",
                [f"x{i}" for i in range(self.n_features_in_)]
            )

        return numpy.asarray(input_features)
