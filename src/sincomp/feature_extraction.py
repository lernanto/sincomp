# -*- coding: utf-8 -*-

"""Extract feature representations from dialect pronunciation tables.

The estimators in this module turn raw pronunciation tables into
feature matrices. The public interface consists of:

- :class:`PhoneSimilarity`: derives, for every dialect, the pairwise
  similarity matrix of its phones from their distributional behavior
  over the characters; :meth:`transform` outputs one row per dialect,
  holding the flattened upper triangle of the similarity matrix;
- :class:`DialectVectorizer`: aggregates the phone similarity matrices
  of the individual dialect features into one vector per dialect;
- :class:`CharacterVectorizer`: multi-hot encodes the phones of every
  character over the columns of a wide dialect phone table;
- :class:`DialectCompliance`: scores how well every dialect complies
  with each sound-change rule by comparing the feature distributions
  of the two character sets of the rule.

All the estimators are scikit-learn compatible and consume the wide
feature tables produced by :func:`sincomp.transform`.
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"


import numpy
import pandas
import scipy.sparse
import sklearn.base
import sklearn.compose
import sklearn.feature_extraction.text
from typing import Optional, Sequence, Type, Union

from ._internal.compare import compliance


class PhoneSimilarity(sklearn.base.BaseEstimator):
    """Compute the phone similarity matrix of every dialect.

    For every dialect, two phones are similar when they occur with the
    same characters: the similarity matrix is the cosine similarity of
    the character-distribution vectors of the phones. The output of
    :meth:`transform` holds one row per dialect, each row being the
    flattened upper triangle of the phone-by-phone similarity matrix.

    Parameters
    ----------
    dtype : type, default=numpy.float64
        Data type of the similarity matrices.

    Attributes
    ----------
    n_features_in_ : int
        Number of input columns seen during fitting.
    n_features_out_ : int
        Number of flattened similarity features, equal to the square of
        ``n_features_in_``.
    feature_names_in_ : numpy.ndarray of str
        Names of the input columns. Only defined when fitting on a
        pandas DataFrame.
    """

    def __init__(self, dtype: Type = numpy.float64):
        self.dtype = dtype

    def fit(
        self,
        X: Union[pandas.DataFrame, numpy.ndarray],
        y: Optional[numpy.ndarray] = None
    ) -> 'PhoneSimilarity':
        """Record the input shape and fit nothing.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Pronunciation data with one row per phone and one column
            per dialect; cells hold whitespace-separated phone
            transcriptions, empty strings mark missing values.
        y : numpy.ndarray, default=None
            Ignored. This parameter exists for compatibility with the
            scikit-learn estimator API.

        Returns
        -------
        PhoneSimilarity
            Fitted estimator.
        """

        self.n_features_in_ = X.shape[1]
        self.n_features_out_ = X.shape[1] * X.shape[1]
        if isinstance(X, pandas.DataFrame):
            self.feature_names_in_ = X.columns.values

        return self

    def transform(
        self,
        X: Union[pandas.DataFrame, numpy.ndarray]
    ) -> scipy.sparse.csr_matrix:
        """Compute the phone similarity matrix of every dialect.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Pronunciation data in the same column structure as the
            data used during fitting.

        Returns
        -------
        scipy.sparse.csr_matrix
            One flattened upper-triangular similarity matrix per
            dialect, of shape (n_dialects, n_features_out_).
        """

        X = numpy.asarray(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                f'X has {X.shape[1]} features, but PhoneSimilarity is '
                f'expecting {self.n_features_in_} features as input.'
            )

        trius = []
        for i in range(X.shape[0]):
            try:
                features = sklearn.feature_extraction.text.CountVectorizer(
                    lowercase=False,
                    tokenizer=str.split,
                    token_pattern=None,
                    min_df=2,
                    binary=True,
                    dtype=self.dtype
                ).fit_transform(X[i])
                sim = sklearn.metrics.pairwise.cosine_similarity(
                    features,
                    dense_output=False
                )

            except ValueError:
                sim = scipy.sparse.csr_matrix(
                    (X.shape[1], X.shape[1]),
                    dtype=self.dtype
                )

            trius.append(scipy.sparse.triu(sim, 1).reshape((1, -1)))

        return scipy.sparse.vstack(trius, format='csr')

    def fit_transform(
        self,
        X: Union[pandas.DataFrame, numpy.ndarray],
        y: Optional[numpy.ndarray] = None
    ) -> scipy.sparse.csr_matrix:
        """Fit the estimator and compute the similarity matrices.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Pronunciation data with one row per phone and one column
            per dialect.
        y : numpy.ndarray, default=None
            Ignored. This parameter exists for compatibility with the
            scikit-learn estimator API.

        Returns
        -------
        scipy.sparse.csr_matrix
            One flattened upper-triangular similarity matrix per
            dialect, of shape (n_dialects, n_features_out_).
        """

        return self.fit(X, y).transform(X)

    def get_feature_names_out(
        self,
        input_features: Optional[Sequence[str]] = None
    ) -> numpy.ndarray:
        """Get the output feature names of the similarity vectors.

        Parameters
        ----------
        input_features : Sequence[str], default=None
            Names of the input dialects; defaults to the column labels
            seen during fitting or generated ``x{i}`` placeholders.

        Returns
        -------
        numpy.ndarray
            Pairs of input feature names joined with an underscore,
            aligned with the flattened similarity vector of one
            dialect.
        """

        if input_features is None:
            input_features = getattr(
                self,
                'feature_names_in_',
                [f'x{i}' for i in range(self.n_features_in_)]
            )

        return numpy.asarray(
            list(f'{i}_{j}' for i in input_features for j in input_features)
        )

    def inverse_transform(
        self,
        X: Union[numpy.ndarray, scipy.sparse.csr_matrix]
    ) -> numpy.ndarray:
        """Group phones into classes from their similarity vectors.

        The flattened similarity vector of every dialect is reshaped
        into a phone-by-phone matrix, turned into distances and
        clustered hierarchically; every cluster receives the name of
        its first phone when the input feature names are known.

        Parameters
        ----------
        X : numpy.ndarray or scipy.sparse.csr_matrix
            Flattened similarity vectors, one row per dialect.

        Returns
        -------
        numpy.ndarray
            One phone label per input feature, of shape
            (n_dialects, n_features_in_).
        """

        if X.shape[1] != self.n_features_out_:
            raise ValueError(
                f'X has {X.shape[1]} features, but {type(self).__name__} '
                f'is expecting {self.n_features_out_} features as input.'
            )

        if self.n_features_in_ == 1:
            name = self.feature_names_in_[0] if hasattr(
                self, 'feature_names_in_'
            ) else '0'
            return numpy.full((X.shape[0], 1), name, dtype=str)

        outputs = []
        for i in range(X.shape[0]):
            Xi = X[i:i + 1].toarray() \
                if scipy.sparse.issparse(X) else numpy.asarray(X[i:i + 1])
            Xi = numpy.reshape(Xi, (self.n_features_in_, self.n_features_in_))
            dist = numpy.clip(1 - (Xi + Xi.T), 0, 1)
            ac = sklearn.cluster.AgglomerativeClustering(
                n_clusters=None,
                metric='precomputed',
                linkage='average',
                distance_threshold=0.5
            ).fit(dist)

            if hasattr(self, 'feature_names_in_'):
                # label every cluster with its first phone
                names = numpy.empty(self.n_features_in_, dtype=object)
                for j in range(ac.n_clusters_):
                    mask = ac.labels_ == j
                    names[mask] = self.feature_names_in_[mask][0]
                names = names.astype(str)

            else:
                names = ac.labels_.astype(str)

            outputs.append(names)

        return numpy.stack(outputs, axis=0)


class DialectVectorizer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Combine the phone similarity vectors of dialect feature groups.

    Parameters
    ----------
    column_selectors : sequence, default=['initial', 'final', 'tone']
        Selectors for the input columns of each dialect feature group.
        Each selector is a string that matches column names by literal
        substring, a sequence of column indices, or a slice.
    dtype : type, default=numpy.float64
        Data type used to compute the phone similarity vectors.
    """

    def __init__(
        self,
        column_selectors: Sequence[Union[str, Sequence, slice]] = ['initial', 'final', 'tone'],
        dtype: Type = numpy.float64
    ):
        self.column_selectors = list(column_selectors)
        self.dtype = dtype

    def fit(
        self,
        X: Union[numpy.ndarray, pandas.DataFrame],
        y: Optional[numpy.ndarray] = None
    ) -> 'DialectVectorizer':
        """Fit the vectorizer to the dialect feature table.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Input table with one row per dialect and one column per
            dialect phone feature.
        y : numpy.ndarray, default=None
            Ignored. This parameter exists for compatibility with the
            scikit-learn estimator API.

        Returns
        -------
        DialectVectorizer
            Fitted vectorizer.
        """

        self.n_features_in_ = X.shape[1]
        if isinstance(X, pandas.DataFrame):
            self.feature_names_in_ = X.columns.values

        self.transformers_ = []
        indices = numpy.arange(X.shape[1])
        X_arr = numpy.asarray(X)
        self.output_indices_ = []
        start = 0
        for i, s in enumerate(self.column_selectors):
            if isinstance(s, str):
                assert isinstance(X, pandas.DataFrame)
                name = s
                idx = indices[X.columns.str.contains(s, regex=False)]
                Xi = X.iloc[:, idx]
            else:
                name = f'phonesimilarity{i}'
                idx = indices[s]
                Xi = X_arr[:, idx]

            t = PhoneSimilarity(self.dtype).fit(Xi)
            stop = start + t.n_features_out_
            self.transformers_.append((name, t, idx))
            self.output_indices_.append(slice(start, stop))
            start = stop

        self.n_features_out_ = start
        return self

    def transform(
        self,
        X: Union[numpy.ndarray, pandas.DataFrame]
    ) -> scipy.sparse.csr_matrix:
        """Transform the dialect feature table into similarity vectors.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Input table with one row per dialect and the same columns as
            the data used during fitting.

        Returns
        -------
        scipy.sparse.csr_matrix
            One concatenated similarity vector per dialect, of shape
            ``(n_dialects, n_features_out_)``.
        """

        if X.shape[1] != self.n_features_in_:
            raise ValueError(f'X has {X.shape[1]} features, but DialectVectorizer is expecting {self.n_features_in_} features as input.')

        X = X.toarray() if scipy.sparse.issparse(X) else numpy.asarray(X)
        return scipy.sparse.hstack(
            [t.transform(X[:, i]) for _, t, i in self.transformers_],
            format='csr'
        )

    def fit_transform(
        self,
        X: Union[numpy.ndarray, pandas.DataFrame],
        y: Optional[numpy.ndarray] = None
    ) -> scipy.sparse.csr_matrix:
        """Fit the vectorizer and transform the dialect feature table.

        Parameters
        ----------
        X : numpy.ndarray or pandas.DataFrame
            Input table with one row per dialect and one column per
            dialect phone feature.
        y : numpy.ndarray, default=None
            Ignored. This parameter exists for compatibility with the
            scikit-learn estimator API.

        Returns
        -------
        scipy.sparse.csr_matrix
            One concatenated similarity vector per dialect, of shape
            ``(n_dialects, n_features_out_)``.
        """

        return self.fit(X, y).transform(X)

    def inverse_transform(
        self,
        X: scipy.sparse.csr_matrix
    ) -> numpy.ndarray:
        """Reconstruct phone labels from the similarity vectors.

        Parameters
        ----------
        X : scipy.sparse.csr_matrix
            Concatenated similarity vectors, one row per dialect.

        Returns
        -------
        numpy.ndarray
            Reconstructed phone labels, with shape
            ``(n_dialects, n_features_in_)``. Unselected input columns
            contain empty strings.
        """

        if X.shape[1] != self.n_features_out_:
            raise ValueError(
                f'X has {X.shape[1]} features, but {type(self).__name__} '
                f'is expecting {self.n_features_out_} features as input.'
            )

        X = X.toarray() if scipy.sparse.issparse(X) else numpy.asarray(X)
        X_original = numpy.full(
            (X.shape[0], self.n_features_in_),
            '',
            dtype=object
        )
        for (_, t, ii), oi in zip(self.transformers_, self.output_indices_):
            X_original[:, ii] = t.inverse_transform(X[:, oi])

        return X_original.astype(str)

    def get_feature_names_out(
        self,
        input_features: Optional[Sequence[str]] = None
    ) -> numpy.ndarray:
        """Get output feature names for the concatenated similarity vectors.

        Parameters
        ----------
        input_features : Sequence[str], default=None
            Input feature names. If ``None``, use the names seen during
            fitting or generated ``x{i}`` placeholders.

        Returns
        -------
        numpy.ndarray
            Output names for the flattened pairwise similarity features
            of each selected dialect feature group.
        """

        if input_features is None:
            input_features = getattr(
                self,
                'feature_names_in_',
                [f'x{i}' for i in range(self.n_features_in_)]
            )
        input_features = numpy.asarray(input_features)
        return numpy.concatenate(
            [t.get_feature_names_out(input_features[i])
             for _, t, i in self.transformers_]
        )


class CharacterVectorizer(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Convert a wide dialect phone table into character multi-hot encoding.

    The input must be a wide table with one row per character and one phone
    feature per column. DataFrame columns must be unique and must not be a
    MultiIndex.

    Parameters
    ----------
    dtype : type, default=numpy.int32
        Data type of the encoded output.

    Attributes
    ----------
    feature_names_in_ : pandas.Index or list of str
        Names of the input phone feature columns seen during fitting.
    transformer_ : sklearn.compose.ColumnTransformer
        Column transformer containing one fitted CountVectorizer per input
        column.
    vocabularies_ : list of dict
        Token-to-index vocabulary for each input column.
    """

    def __init__(self, dtype=numpy.int32):
        self.dtype = dtype

    def fit(
        self,
        X: Union[pandas.DataFrame, numpy.ndarray],
        y: Optional[numpy.ndarray] = None
    ) -> 'CharacterVectorizer':
        """Fit the transformer on a wide dialect phone table.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Wide input table with one phone feature per column.
        y : numpy.ndarray, default=None
            Ignored. This parameter exists for compatibility with the
            scikit-learn transformer API.

        Returns
        -------
        CharacterVectorizer
            Fitted transformer.
        """

        if X.shape[1] == 0:
            raise ValueError('Input must contain at least one phone column.')

        self.feature_names_in_ = X.columns if isinstance(X, pandas.DataFrame) \
            else [f'x{i}' for i in range(X.shape[1])]

        vectorizer = sklearn.feature_extraction.text.CountVectorizer(
            lowercase=False,
            tokenizer=str.split,
            token_pattern=None,
            stop_words=None,
            binary=True,
            dtype=self.dtype
        )
        self.transformer_ = sklearn.compose.ColumnTransformer(
            [(str(c), vectorizer, c) for c in self.feature_names_in_],
            remainder="drop",
            sparse_threshold=1.0
        ).fit(X)

        self.vocabularies_ = [
            self.transformer_.named_transformers_[str(c)].vocabulary_
            for c in self.feature_names_in_
        ]

        return self

    def transform(
        self,
        X: Union[pandas.DataFrame, numpy.ndarray]
    ) -> scipy.sparse.spmatrix:
        """Transform a wide dialect phone table into character encodings.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Wide input table with phone feature columns matching the data
            used during fitting.

        Returns
        -------
        scipy.sparse.spmatrix
            Sparse multi-hot encoding matrix.
        """

        if not hasattr(self, 'transformer_'):
            raise ValueError('CharacterVectorizer is not fitted yet.')

        return self.transformer_.transform(X)

    def fit_transform(
        self,
        X: Union[pandas.DataFrame, numpy.ndarray],
        y: Optional[numpy.ndarray] = None
    ) -> scipy.sparse.spmatrix:
        """Fit the transformer and transform the input in one step.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Wide input table with phone feature columns.
        y : numpy.ndarray, default=None
            Ignored. This parameter exists for compatibility with the
            scikit-learn transformer API.

        Returns
        -------
        scipy.sparse.spmatrix
            Sparse multi-hot encoding matrix.
        """

        return self.fit(X, y).transform(X)

    def get_feature_names_out(
        self,
        input_features: Optional[Sequence[str]] = None
    ) -> numpy.ndarray:
        """Get output feature names for the multi-hot character encoding.

        Parameters
        ----------
        input_features : Sequence[str], default=None
            Ignored. The output names are determined by the fitted
            vectorizers.

        Returns
        -------
        numpy.ndarray
            Output feature names.
        """

        if not hasattr(self, 'transformer_'):
            raise ValueError('CharacterVectorizer is not fitted yet.')
        return self.transformer_.get_feature_names_out()


class DialectCompliance(sklearn.base.BaseEstimator, sklearn.base.TransformerMixin):
    """Score how well every dialect complies with a set of sound rules.

    For each rule, which is made up of two character sets, the
    distribution of the features of each character set in a dialect is
    summarized as a feature-value vector. The similarity between the
    normalized vectors of the two character sets is the compliance score
    of the dialect for the rule, in the range ``[0, 1]``. With L2
    normalization the score is the cosine similarity.

    The input is a wide table with one row per character, a
    ``(dialect, feature)`` MultiIndex over the columns and character IDs
    as the row index, as produced by :func:`sincomp.transform`. The
    output is a DataFrame with one row per dialect and one column per
    rule.

    Parameters
    ----------
    rules : pandas.DataFrame
        Sound-change rules, one rule per row.
        Required columns are ``feature``, ``cid1`` and ``cid2``, where
        ``cid1`` and ``cid2`` hold the character IDs of the two character
        sets of each rule.
    norm : int or None, default=2
        The norm used to normalize the feature-value vectors before the
        inner product. ``None`` skips the normalization.
    dtype : type, default=numpy.float32
        Data type of the compliance scores.

    Attributes
    ----------
    n_features_in_ : int
        Number of input columns seen during fitting.
    feature_names_in_ : pandas.Index
        Labels of the input columns seen during fitting.

    Notes
    -----
    Fitting only validates the rules and records the shape of the input,
    the actual scoring happens in :meth:`transform`.
    """

    def __init__(
        self,
        rules: pandas.DataFrame,
        norm: Optional[int] = 2,
        dtype: Type = numpy.float32
    ):
        self.rules = rules
        self.norm = norm
        self.dtype = dtype

    def fit(
        self,
        X: pandas.DataFrame,
        y: Optional[numpy.ndarray] = None
    ) -> 'DialectCompliance':
        """Validate the rules and record the shape of the input.

        Parameters
        ----------
        X : pandas.DataFrame
            Wide input table with a ``(dialect, feature)`` MultiIndex
            over the columns and character IDs as the row index.
        y : numpy.ndarray, default=None
            Ignored. This parameter exists for compatibility with the
            scikit-learn transformer API.

        Returns
        -------
        DialectCompliance
            Fitted estimator.

        Raises
        ------
        ValueError
            If the rules are not a DataFrame or lack one of the required
            columns ``feature``, ``cid1`` and ``cid2``.
        """

        if not isinstance(self.rules, pandas.DataFrame):
            raise ValueError("rules must be a pandas.DataFrame.")
        missing = {"feature", "cid1", "cid2"} - set(self.rules.columns)
        if missing:
            raise ValueError(
                f"rules is missing the required columns {sorted(missing)}."
            )

        self.n_features_in_ = X.shape[1]
        if isinstance(X, pandas.DataFrame):
            self.feature_names_in_ = X.columns

        return self

    def transform(self, X: pandas.DataFrame) -> pandas.DataFrame:
        """Compute the rule compliance of every dialect in the table.

        Parameters
        ----------
        X : pandas.DataFrame
            Wide input table with a ``(dialect, feature)`` MultiIndex
            over the columns and character IDs as the row index, matching
            the data used during fitting.

        Returns
        -------
        pandas.DataFrame
            Compliance scores, one row per dialect and one column per
            rule.
        """

        if not hasattr(self, "n_features_in_"):
            raise ValueError(
                "This DialectCompliance instance is not fitted yet."
            )

        return compliance(X, self.rules, dtype=self.dtype, norm=self.norm)
