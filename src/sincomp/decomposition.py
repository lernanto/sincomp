# -*- encoding: utf-8 -*-

"""
Dense vector representations of dialects, characters and phones

The estimators of this module embed dialect phone data into low
dimensional continuous spaces:

- :class:`DialectEmbedding` projects the phone-similarity features of
  dialects onto their truncated SVD subspace;
- :class:`CharacterEmbedding` projects character multi-hot encodings
  onto their truncated SVD subspace;
- :class:`Factorizer` factorizes a wide dialect-by-feature table into
  character and feature-value embeddings with masked alternating
  least squares, ignoring missing cells.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import numpy
import scipy
import sklearn.cluster
import sklearn.decomposition
import sklearn.metrics.pairwise
import sklearn.feature_extraction.text
from typing import Optional, Sequence, Type, Union
import pandas

from ._internal.factorize import als, solve_embeddings
from ._internal.preprocessing import features_to_vectors


class DialectEmbedding(sklearn.base.BaseEstimator):
    """Embed dialects into a dense space with truncated SVD.

    The input is a sparse matrix with one row per dialect holding the
    flattened phone-similarity features of the dialect, e.g. the
    output of :class:`PhoneSimilarity`. Fitting computes the mean of
    every feature, keeps only the features whose mean falls inside
    ``(mean_threshold, 1 - mean_threshold)``, projects the filtered
    matrix onto its truncated SVD subspace and rescales the embeddings
    so their norms average to one in the geometric sense.

    Parameters
    ----------
    embedding_size : int, default=128
        Maximum dimension of the dialect embeddings, further limited
        by the numbers of dialects and features.
    mean_threshold : float, default=0.05
        Features with a mean inside ``(mean_threshold,
        1 - mean_threshold)`` are kept, all others are dropped.

    Attributes
    ----------
    embedding_size_ : int
        Effective dimension of the embeddings.
    n_features_in_ : int
        Number of input features seen during fitting.
    n_features_out_ : int
        Dimension of the embeddings, equal to `embedding_size_`.
    mean_ : numpy.ndarray
        Mean of every input feature.
    components_ : numpy.ndarray
        Right singular vectors of the filtered matrix, of shape
        (embedding_size_, n_kept_features).
    singular_values_ : numpy.ndarray
        Singular values of the filtered matrix.
    explained_variance_ : numpy.ndarray
        Variance explained by each embedding dimension.
    explained_variance_ratio_ : numpy.ndarray
        Fraction of the total variance explained by each embedding
        dimension.
    scale_ : float
        Geometric mean norm of the training embeddings used to
        rescale them and their projections.
    """

    def __init__(
        self,
        embedding_size: int = 128,
        mean_threshold: float = 0.05
    ):
        self.embedding_size = embedding_size
        self.mean_threshold = mean_threshold

    def _get_support_mask(self):
        """Return the boolean mask of the kept input features."""

        return (self.mean_ > self.mean_threshold) \
            & (self.mean_ < 1 - self.mean_threshold)

    def fit_transform(
        self,
        X: scipy.sparse.csr_matrix,
        y: Optional[numpy.ndarray] = None
    ) -> numpy.ndarray:
        """Fit the embedding and return the training embeddings.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix
            Sparse dialect feature matrix with one row per dialect.
        y : numpy.ndarray, default=None
            Ignored. This parameter exists for compatibility with the
            scikit-learn estimator API.

        Returns
        -------
        numpy.ndarray
            Dense dialect embeddings of shape
            (n_dialects, embedding_size_).
        """

        self.embedding_size_ = min(
            self.embedding_size, X.shape[0] - 1, X.shape[1] - 1
        )
        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = self.embedding_size_

        self.mean_ = numpy.squeeze(numpy.asarray(X.mean(axis=0)))

        u, s, vt = scipy.sparse.linalg.svds(
            X[:, self._get_support_mask()],
            self.embedding_size_
        )
        u, s, vt = u[:, ::-1], s[::-1], vt[::-1]

        self.components_ = numpy.asarray(vt)
        self.singular_values_ = s
        self.explained_variance_ = numpy.var(u * s[None, :], axis=0)
        var_sum = X.multiply(X).sum() / X.shape[0] \
            - numpy.linalg.norm(self.mean_) ** 2
        self.explained_variance_ratio_ = self.explained_variance_ / var_sum

        Xt = numpy.asarray(u) * s[None, :]
        self.scale_ = numpy.exp(
            numpy.mean(numpy.log(numpy.linalg.norm(Xt, axis=1)))
        )
        return Xt / self.scale_

    def fit(
        self,
        X: scipy.sparse.csr_matrix,
        y: Optional[numpy.ndarray] = None
    ) -> 'DialectEmbedding':
        """Fit the embedding and return the estimator itself.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix
            Sparse dialect feature matrix with one row per dialect.
        y : numpy.ndarray, default=None
            Ignored. This parameter exists for compatibility with the
            scikit-learn estimator API.

        Returns
        -------
        DialectEmbedding
            Fitted estimator.
        """

        self.fit_transform(X, y)
        return self

    def transform(self, X: scipy.sparse.csr_matrix) -> numpy.ndarray:
        """Project dialects onto the fitted embedding space.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix
            Sparse dialect feature matrix in the same column structure
            as the data used during fitting.

        Returns
        -------
        numpy.ndarray
            Dense dialect embeddings of shape
            (n_dialects, embedding_size_).

        Raises
        ------
        ValueError
            If the input feature count differs from the count seen during
            fitting.
        """

        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f'X has {X.shape[1]} features, but DialectEmbedding is '
                f'expecting {self.n_features_in_} features as input.'
            )

        return X[:, self._get_support_mask()] \
            @ self.components_.T / self.scale_

    def inverse_transform(
        self,
        X: Union[numpy.ndarray, pandas.DataFrame]
    ) -> numpy.ndarray:
        """Reconstruct the sparse feature vectors of the embeddings.

        The embedding of every dialect is scaled back and multiplied
        with the fitted components; the filtered-out features fall
        back to their training means.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Dense dialect embeddings.

        Returns
        -------
        numpy.ndarray
            Reconstructed feature vectors of shape
            (n_dialects, n_features_in_).

        Raises
        ------
        ValueError
            If the input embedding dimension differs from the fitted
            embedding dimension.
        """

        X = numpy.asarray(X)
        if X.shape[1] != self.n_features_out_:
            raise ValueError(
                f'X has {X.shape[1]} features, but DialectEmbedding is '
                f'expecting {self.n_features_out_} features as input.'
            )

        X_original = numpy.repeat(self.mean_[None, :], X.shape[0], axis=0)
        X_original[:, self._get_support_mask()] = \
            X * self.scale_ @ self.components_
        return X_original


class CharacterEmbedding(
    sklearn.base.BaseEstimator,
    sklearn.base.TransformerMixin
):
    """Convert character multi-hot encodings into dense embeddings.

    The embedding is computed with truncated singular value decomposition.

    Parameters
    ----------
    embedding_size : int, default=128
        Maximum number of dimensions in the output embedding. The effective
        size is limited by the number of samples and input features.
    dtype : type, default=numpy.float32
        Data type of the embedding output.

    Attributes
    ----------
    n_features_in_ : int
        Number of input features seen during fitting.
    n_features_out_ : int
        Number of output embedding dimensions.
    svd_ : sklearn.decomposition.TruncatedSVD
        Fitted truncated SVD estimator.
    character_embeddings_ : numpy.ndarray
        Training data transformed into the embedding space.
    components_ : numpy.ndarray
        Principal components of the fitted SVD model.
    singular_values_ : numpy.ndarray
        Singular values of the fitted SVD model.
    explained_variance_ : numpy.ndarray
        Explained variance for each selected component.
    explained_variance_ratio_ : numpy.ndarray
        Percentage of variance explained by each selected component.
    """

    def __init__(
        self,
        embedding_size: int = 128,
        dtype: Type = numpy.float32
    ):
        self.embedding_size = embedding_size
        self.dtype = dtype

    def fit(
        self,
        X: Union[scipy.sparse.spmatrix, numpy.ndarray],
        y: Optional[numpy.ndarray] = None
    ) -> 'CharacterEmbedding':
        """Fit a TruncatedSVD model on character encodings.

        Parameters
        ----------
        X : scipy.sparse.spmatrix or numpy.ndarray
            Character multi-hot encoding matrix.
        y : numpy.ndarray, default=None
            Ignored. This parameter exists for compatibility with the
            scikit-learn transformer API.

        Returns
        -------
        CharacterEmbedding
            Fitted transformer.
        """

        X = scipy.sparse.csr_matrix(X, dtype=self.dtype) \
            if not scipy.sparse.isspmatrix(X) else X.astype(self.dtype)
        self.n_features_in_ = X.shape[1]
        embedding_size = min(
            self.embedding_size,
            X.shape[0],
            X.shape[1]
        )
        self.n_features_out_ = embedding_size

        self.svd_ = sklearn.decomposition.TruncatedSVD(embedding_size)
        self.character_embeddings_ = self.svd_.fit_transform(X)
        self.components_ = self.svd_.components_
        self.singular_values_ = self.svd_.singular_values_
        self.explained_variance_ = self.svd_.explained_variance_
        self.explained_variance_ratio_ = self.svd_.explained_variance_ratio_
        return self

    def transform(
        self,
        X: Union[scipy.sparse.spmatrix, numpy.ndarray]
    ) -> numpy.ndarray:
        """Transform character multi-hot encoding into dense embeddings.

        Parameters
        ----------
        X : scipy.sparse.spmatrix or numpy.ndarray
            Character multi-hot encoding matrix with the same number of
            features as the data used during fitting.

        Returns
        -------
        numpy.ndarray
            Dense character embeddings.
        """

        if not hasattr(self, 'svd_'):
            raise ValueError('CharacterEmbedding is not fitted yet.')
        X = scipy.sparse.csr_matrix(X, dtype=self.dtype) \
            if not scipy.sparse.isspmatrix(X) else X.astype(self.dtype)
        return self.svd_.transform(X)

    def fit_transform(
        self,
        X: Union[scipy.sparse.spmatrix, numpy.ndarray],
        y: Optional[numpy.ndarray] = None
    ) -> numpy.ndarray:
        """Fit the transformer and transform the input in one step.

        Parameters
        ----------
        X : scipy.sparse.spmatrix or numpy.ndarray
            Character multi-hot encoding matrix.
        y : numpy.ndarray, default=None
            Ignored. This parameter exists for compatibility with the
            scikit-learn transformer API.

        Returns
        -------
        numpy.ndarray
            Dense character embeddings.
        """

        return self.fit(X, y).transform(X)

    def inverse_transform(
        self,
        X: Union[numpy.ndarray, pandas.DataFrame]
    ) -> numpy.ndarray:
        """Inverse transform dense embeddings back to feature space.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Dense character embeddings.

        Returns
        -------
        numpy.ndarray
            Reconstructed multi-hot-like feature matrix.
        """

        if not hasattr(self, 'svd_'):
            raise ValueError('CharacterEmbedding is not fitted yet.')
        X = numpy.asarray(X)
        return self.svd_.inverse_transform(X)

    def get_feature_names_out(
        self,
        input_features: Optional[Sequence[str]] = None
    ) -> numpy.ndarray:
        """Get output feature names for the dense embeddings.

        Parameters
        ----------
        input_features : Sequence[str], default=None
            Ignored. Embedding feature names are generated from the fitted
            embedding size.

        Returns
        -------
        numpy.ndarray
            Output embedding feature names.
        """

        return numpy.asarray([
            f'embed_{i}' for i in range(self.n_features_out_)
        ])


class Factorizer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Factorize a wide dialect-by-feature table with masked ALS.

    The input is a wide table with one row per character and one column
    per dialect feature, e.g. the initial, final or tone of one dialect;
    any feature segmentation is supported. Each cell holds the
    whitespace-separated feature values observed for the character in
    that dialect feature, so a character with multiple pronunciations in one
    dialect contributes several values in the same cell. Empty cells and
    NaN mark missing values.

    The table is factorized into one embedding per character and one
    embedding per distinct feature value with alternating least squares
    (ALS). Only observed cells contribute to the loss, so missing cells
    never influence the learned embeddings. To save computation the
    characters and columns are ordered by coverage and the ALS
    iterations run on the densest prefix only; the remaining ones are
    updated once in a second stage, followed by a final pass re-solving
    all character embeddings when the second stage updates any
    feature-value embedding. The training
    loop itself lives in
    :func:`sincomp._internal.factorize.als`; this estimator only
    normalizes the input, delegates to it and exposes the results under
    the scikit-learn fitted-attribute conventions.

    Parameters
    ----------
    n_components : int, default=128
        Dimension of the character and feature-value embeddings.
    max_iter : int, default=10
        Maximum number of ALS iterations.
    tol : float, default=1e-4
        Stop when the RMSE improvement between iterations drops below
        this value.
    alpha : float, default=1e-4
        L2 regularization coefficient, 0 disables regularization.
    min_sample_coverage : float or int, default=0.2
        Characters observing fewer columns than this fraction (or
        absolute number) are only updated in the second training stage.
    min_feature_coverage : float or int, default=0.2
        Columns observed in fewer characters than this fraction (or
        absolute number) are only updated in the second training stage.
    min_samples : float or int, default=100
        The first training stage uses at least this fraction (or
        absolute number) of the characters.
    min_features : float or int, default=10
        The first training stage uses at least this fraction (or
        absolute number) of the columns.
    random_state : int, numpy.random.RandomState or None, default=None
        Seed of the random initialization. ``None`` draws from the
        global NumPy random state.

    Attributes
    ----------
    components_ : numpy.ndarray
        Feature-value embeddings of shape
        ``(n_components, n_feature_values)``, one column per distinct
        feature value of each input column, ordered by input column.
        Following the scikit-learn convention, the rows are the
        ``n_components`` basis vectors and the columns are the feature
        values.
    feature_bounds_ : numpy.ndarray
        Cumulative boundaries of the feature-value blocks of the
        input columns within the columns of ``components_``, of shape
        ``(n_features_in_ + 1,)``.
    vocabularies_ : list of dict
        Feature-value-to-index vocabulary of each input column, with
        global indices aligned with the columns of ``components_``.
    feature_names_in_ : pandas.Index
        Labels of the input columns.
    n_features_in_ : int
        Number of input columns seen during fitting.
    n_features_out_ : int
        Dimension of the embeddings, equal to `n_components`.
    n_iter_ : int
        Number of ALS iterations performed.

    Notes
    -----
    :meth:`fit_transform` returns one embedding per input row, in
    input order. For a DataFrame input with a
    ``(dialect, feature)`` MultiIndex over the columns, every level-0
    label forms one block of columns; with a flat column index every
    column forms its own block.

    :meth:`transform` projects new characters by solving their masked
    least-squares systems against the fitted components with
    :func:`sincomp._internal.factorize.solve_embeddings`, the same
    projection used to obtain the training embeddings, so transforming
    the training data reproduces the output of :meth:`fit_transform`.
    Unknown feature values are ignored; a character whose feature
    values are all unknown cannot be projected and raises an error.

    Examples
    --------
    >>> import pandas
    >>> from sincomp.decomposition import Factorizer
    >>> X = pandas.DataFrame(
    ...     [["b ei 51", "p ei 24"], ["p ɔ 213", "p ɔ 53"]],
    ...     columns=pandas.MultiIndex.from_tuples(
    ...         [("beijing", "initial"), ("beijing", "final")]
    ...     )
    ... )
    >>> factorizer = Factorizer(n_components=2).fit(X)
    >>> factorizer.transform(X).shape
    (2, 2)
    """

    def __init__(
        self,
        n_components: int = 128,
        max_iter: int = 10,
        tol: float = 1e-4,
        alpha: float = 1e-4,
        min_sample_coverage: float | int = 0.2,
        min_feature_coverage: float | int = 0.2,
        min_samples: float | int = 100,
        min_features: float | int = 10,
        random_state=None
    ):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.min_sample_coverage = min_sample_coverage
        self.min_feature_coverage = min_feature_coverage
        self.min_samples = min_samples
        self.min_features = min_features
        self.random_state = random_state

    def _check_X(self, X) -> tuple[numpy.ndarray, pandas.Index]:
        """Normalize the input into a string object array with labels.

        Parameters
        ----------
        X : pandas.DataFrame or array-like
            Wide table with whitespace-separated feature values per cell.

        Returns
        -------
        cells : numpy.ndarray
            String object array of shape (n_characters, n_features),
            missing values normalized to empty strings.
        columns : pandas.Index
            Labels of the input columns.
        """

        if isinstance(X, pandas.DataFrame):
            columns = X.columns
            cells = X.to_numpy(dtype=object)
        else:
            cells = numpy.asarray(X, dtype=object)
            if cells.ndim != 2:
                raise ValueError(
                    f"Expected a 2-dimensional table, got array with "
                    f"shape {cells.shape}."
                )
            columns = pandas.Index(
                [f"x{i}" for i in range(cells.shape[1])]
            )

        cells = numpy.where(pandas.isna(cells), "", cells).astype(str)
        return cells, columns

    def fit(
        self,
        X: pandas.DataFrame | numpy.ndarray,
        y: Optional[numpy.ndarray] = None
    ) -> "Factorizer":
        """Fit the factorization on a wide dialect-by-feature table.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Wide input table with one row per character and one column
            per dialect feature. Cells hold whitespace-separated feature
            values; empty cells and NaN mark missing values.
        y : numpy.ndarray, default=None
            Ignored. This parameter exists for compatibility with the
            scikit-learn estimator API.

        Returns
        -------
        Factorizer
            Fitted factorizer.

        Raises
        ------
        ValueError
            If the input is empty, has no feature column, has duplicated
            column labels or contains no observed feature value.
        """

        self.fit_transform(X, y)
        return self

    def fit_transform(
        self,
        X: pandas.DataFrame | numpy.ndarray,
        y: Optional[numpy.ndarray] = None
    ) -> numpy.ndarray:
        """Fit the factorization and return the training embeddings.

        The returned embeddings agree with projecting the training data
        through :meth:`transform`, because every training iteration
        ends with a character solve against the latest components.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Wide input table with one row per character and one column
            per dialect feature. Cells hold whitespace-separated feature
            values; empty cells and NaN mark missing values.
        y : numpy.ndarray, default=None
            Ignored. This parameter exists for compatibility with the
            scikit-learn estimator API.

        Returns
        -------
        numpy.ndarray
            Character embeddings of shape
            ``(n_samples, n_components)`` in input row order.

        Raises
        ------
        ValueError
            If the input is empty, has no feature column, has duplicated
            column labels or contains no observed feature value.
        """

        cells, columns = self._check_X(X)
        if columns.duplicated().any():
            raise ValueError("Input columns must be unique.")
        if cells.shape[1] == 0:
            raise ValueError(
                "Input must contain at least one feature column."
            )

        vectors, vocabularies = features_to_vectors(cells)

        # the vocabulary indices are global, aligned with the
        # concatenation of the per-column code blocks
        offset = 0
        for vocab in vocabularies:
            for token in vocab:
                vocab[token] += offset
            offset += len(vocab)

        embeddings, components, n_iter = als(
            vectors,
            n_components=self.n_components,
            max_iter=self.max_iter,
            tol=self.tol,
            alpha=self.alpha,
            min_sample_coverage=self.min_sample_coverage,
            min_feature_coverage=self.min_feature_coverage,
            min_samples=self.min_samples,
            min_features=self.min_features,
            random_state=self.random_state
        )

        self.components_ = components
        # the boundaries are derived from the per-column vocabularies, so
        # they stay aligned with both n_features_in_ and the global
        # vocabulary indices
        self.feature_bounds_ = numpy.concatenate(
            ([0], numpy.cumsum([len(v) for v in vocabularies]))
        ).astype(numpy.int32)
        self.vocabularies_ = vocabularies
        self.feature_names_in_ = columns
        self.n_features_in_ = cells.shape[1]
        self.n_features_out_ = self.n_components
        self.n_iter_ = n_iter

        return embeddings

    def transform(
        self,
        X: pandas.DataFrame | numpy.ndarray
    ) -> numpy.ndarray:
        """Project new characters onto the fitted embedding space.

        Every character is embedded by solving its masked
        least-squares system against the fitted components, the same
        projection used to obtain the training embeddings. Unknown
        feature values are ignored and completely missing cells do not
        contribute to the projection; a character whose feature values
        are all unknown cannot be projected and raises an error.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Wide input table in the same column structure as the data
            used during fitting.

        Returns
        -------
        numpy.ndarray
            Character embeddings of shape
            ``(n_samples, n_components)`` in input row order.

        Raises
        ------
        ValueError
            If the estimator is not fitted, the column structure does
            not match the fitted one, or a character holds no known
            feature value.
        """

        if not hasattr(self, "components_"):
            raise ValueError("This Factorizer instance is not fitted yet.")

        cells, columns = self._check_X(X)
        if cells.shape[1] != self.n_features_in_:
            raise ValueError(
                f"X has {cells.shape[1]} features, but Factorizer is "
                f"expecting {self.n_features_in_} features as input."
            )
        if isinstance(X, pandas.DataFrame) \
                and not columns.equals(self.feature_names_in_):
            raise ValueError(
                "X columns do not match the columns seen during fitting."
            )

        # encode the cells against the fitted vocabularies, one sparse
        # block per input column; the vocabularies hold global indices,
        # so shift them back into each block's local coordinate system;
        # unknown feature values are ignored
        n_samples = cells.shape[0]
        offsets = self.feature_bounds_[:-1]
        blocks = []
        for j, vocab in enumerate(self.vocabularies_):
            offset = int(offsets[j])
            indptr = [0]
            indices = []
            for cell in cells[:, j]:
                # duplicate values in one cell count once, matching the
                # binary encoding used during fitting
                indices.extend(
                    vocab[t] - offset
                    for t in set(cell.split()) if t in vocab
                )
                indptr.append(len(indices))
            blocks.append(scipy.sparse.csr_matrix(
                (numpy.ones(len(indices), dtype=numpy.float32),
                 indices, indptr),
                shape=(n_samples, len(vocab))
            ))

        mask = numpy.stack([b.getnnz(axis=1) > 0 for b in blocks], axis=1)
        known = mask.any(axis=1)
        if not known.all():
            raise ValueError(
                f"Row {int(numpy.nonzero(~known)[0][0])} contains no "
                "known feature values, cannot project it onto the "
                "embedding space."
            )

        bounds = self.feature_bounds_
        components = [
            self.components_[:, bounds[j]:bounds[j + 1]]
            for j in range(self.n_features_in_)
        ]
        return solve_embeddings(
            blocks, mask, components, self.alpha
        )

    def get_feature_names_out(
        self,
        input_features: Optional[Sequence[str]] = None
    ) -> numpy.ndarray:
        """Get the feature-value labels of the fitted components.

        Parameters
        ----------
        input_features : Sequence[str], default=None
            Ignored. The output names are determined by the fitted
            vocabularies.

        Returns
        -------
        numpy.ndarray
            Feature-value labels, aligned with the columns of
            ``components_``.
        """

        if not hasattr(self, "components_"):
            raise ValueError("This Factorizer instance is not fitted yet.")

        names = []
        for vocab in self.vocabularies_:
            names.extend(sorted(vocab, key=vocab.get))

        return numpy.asarray(names)
