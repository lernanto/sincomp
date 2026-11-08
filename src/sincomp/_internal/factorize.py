# -*- coding: utf-8 -*-

"""
Masked alternating least squares (ALS) factorization of multi-hot
vector matrices.

The input is a list of sparse multi-hot matrices, one per feature
column of a wide observation table (e.g. one pronunciation component
of one dialect). Every non-zero entry marks a feature value observed
for one sample, so a sample holding several values in one column
contributes several entries. The matrices are factorized into one
dense embedding per sample and one dense embedding per distinct
feature value by alternating least squares, where only observed cells
contribute to the loss: missing cells never influence the learned
embeddings.

To save computation the samples and columns are ordered by coverage
and the ALS iterations run on the densest prefix only; the remaining
samples and columns are updated once in a second stage. The public
entry point is :func:`als`, and the two half-steps of one
iteration live in :func:`solve_embeddings` and
:func:`solve_components`.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import logging
import numpy
import scipy.sparse
import sklearn.utils


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler())


def solve_embeddings(blocks, mask, components, alpha):
    """Least-squares solve of the sample embeddings.

    Parameters
    ----------
    blocks : list of scipy.sparse.csr_matrix
        One multi-hot vector block of shape (n_samples, vocab_size)
        per column, all restricted to the samples being solved.
    mask : numpy.ndarray
        Boolean matrix of shape (n_samples, len(blocks)); ``True``
        marks an observed cell whose block contributes to the loss.
    components : list of numpy.ndarray
        Current feature-value embeddings, one array of shape
        (n_components, vocab_size) per column, aligned with `blocks`.
    alpha : float
        L2 regularization coefficient, 0 disables regularization.

    Returns
    -------
    numpy.ndarray
        Solved sample embeddings of shape (n_samples, n_components).

    Notes
    -----
    Every sample :math:`i` solves an independent linear system against
    the right-hand side :math:`b_i`, the sum of the embeddings of the
    feature values observed for the sample:

    .. math::

        A_i = \\sum_{j \\, : \\, M_{ij} = 1} V_j V_j^\\top + \\alpha I,
        \\qquad
        b_i = \\sum_{j} X_{j\\, i, :} \\, V_j^\\top,
        \\qquad
        u_i = A_i^{-1} b_i

    where :math:`V_j` is the feature-value embedding matrix of block
    :math:`j`, :math:`X_j` its multi-hot vector block and :math:`M`
    the observation mask. The system is positive definite whenever
    :math:`\\alpha > 0`.
    """

    k = components[0].shape[0]
    prods = numpy.stack([c @ c.T for c in components])
    a = numpy.tensordot(mask.astype(prods.dtype), prods, [-1, 0])
    if alpha > 0:
        a += (numpy.eye(k, dtype=a.dtype) * alpha)[None, :, :]

    b = numpy.zeros((mask.shape[0], k), dtype=a.dtype)
    for block, c in zip(blocks, components):
        b += numpy.asarray(block @ c.T)

    return numpy.linalg.solve(a, b[:, :, None])[..., 0]


def solve_components(blocks, mask, embeddings, alpha):
    """Least-squares solve of the feature-value embeddings block by block.

    Parameters
    ----------
    blocks : list of scipy.sparse.csr_matrix
        One multi-hot vector block of shape (n_samples, vocab_size)
        per column.
    mask : numpy.ndarray
        Boolean matrix of shape (n_samples, len(blocks)); ``True``
        marks an observed cell contributing its sample to the block
        system.
    embeddings : numpy.ndarray
        Current sample embeddings of shape (n_samples, n_components).
    alpha : float
        L2 regularization coefficient, 0 disables regularization.

    Returns
    -------
    list of numpy.ndarray
        Solved feature-value embeddings, one array of shape
        (n_components, vocab_size) per column.

    Notes
    -----
    Block :math:`j` solves an independent linear system against the
    right-hand side :math:`B_j = U^\\top X_j`, where :math:`X_j` is
    the multi-hot vector block of the column: the column of every
    feature value is the sum of the embeddings of the samples
    observing it.

    .. math::

        A_j = \\sum_{i \\, : \\, M_{ij} = 1} u_i u_i^\\top + \\alpha I,
        \\qquad
        B_j = U^\\top X_j,
        \\qquad
        V_j = A_j^{-1} B_j

    where :math:`U` collects the sample embeddings and :math:`M` the
    observation mask. Each system is positive definite whenever
    :math:`\\alpha > 0`.
    """

    k = embeddings.shape[1]
    components = []
    for j, block in enumerate(blocks):
        observers = embeddings[mask[:, j]]
        a = observers.T @ observers
        if alpha > 0:
            a = a + numpy.eye(k, dtype=a.dtype) * alpha

        targets = numpy.asarray(embeddings.T @ block)
        components.append(numpy.linalg.solve(a, targets))

    return components


def als(
    vectors: list[scipy.sparse.csr_matrix],
    n_components: int = 128,
    max_iter: int = 10,
    tol: float = 1e-4,
    alpha: float = 1e-4,
    min_sample_coverage: float | int = 0.2,
    min_feature_coverage: float | int = 0.2,
    min_samples: float | int = 100,
    min_features: float | int = 10,
    random_state=None
) -> tuple[numpy.ndarray, numpy.ndarray, int]:
    """Factorize masked multi-hot code blocks with ALS.

    The input is a list of sparse multi-hot vector matrices, one per
    block (e.g. one dialect feature column of a wide table). Each cell
    of a block holds the multi-hot encoding of one observed value
    combination, so a sample observed several values in one block
    contributes several 1s. The blocks are factorized into one
    embedding per sample and one embedding per distinct feature value
    with alternating least squares (ALS). Only observed cells
    contribute to the loss: a cell is observed iff its block row is
    non-zero, so missing cells never influence the learned embeddings.

    To save computation the samples and blocks are ordered by coverage
    and the ALS iterations run on the densest prefix only; the
    remaining samples and blocks are updated once in a second stage,
    and a final pass re-solves all sample embeddings against the final
    components. The results are returned in the input order.

    Parameters
    ----------
    vectors : list of scipy.sparse.csr_matrix
        One multi-hot code matrix of shape (n_samples, vocab_size) per
        block, all with the same number of rows. The blocks are used
        in place without conversion, so they must already be CSR
        matrices. A block must not be empty.
    n_components : int, default=128
        Dimension of the sample and feature-value embeddings.
    max_iter : int, default=10
        Maximum number of ALS iterations.
    tol : float, default=1e-4
        Stop when the RMSE improvement between iterations drops below
        this value.
    alpha : float, default=1e-4
        L2 regularization coefficient, 0 disables regularization. With
        ``alpha = 0`` a completely empty sample or block makes its
        least-squares system singular.
    min_sample_coverage : float or int, default=0.2
        Samples observing fewer blocks than this fraction (or absolute
        number) are only updated in the second training stage.
    min_feature_coverage : float or int, default=0.2
        Blocks observed in fewer samples than this fraction (or
        absolute number) are only updated in the second training stage.
    min_samples : float or int, default=100
        The first training stage uses at least this fraction (or
        absolute number) of the samples.
    min_features : float or int, default=10
        The first training stage uses at least this fraction (or
        absolute number) of the blocks.
    random_state : int, numpy.random.RandomState or None, default=None
        Seed of the random initialization. ``None`` draws from the
        global NumPy random state.

    Returns
    -------
    embeddings : numpy.ndarray
        Sample embeddings of shape (n_samples, n_components) in the
        input row order.
    components : numpy.ndarray
        Feature-value embeddings of shape (n_components, n_features),
        one column per feature value, grouped by block in the input
        block order. Following the scikit-learn convention, the rows
        of this matrix are the ``n_components`` basis vectors and its
        columns are the feature values.
    n_iter : int
        Number of first-stage ALS iterations performed.

    Raises
    ------
    ValueError
        If `vectors` is empty, holds an empty block, or contains no
        observed cell.

    Notes
    -----
    Let :math:`U \\in \\mathbb{R}^{n \\times k}` denote the sample
    embeddings, :math:`V_j \\in \\mathbb{R}^{k \\times m_j}` the
    feature-value embeddings of block :math:`j`, :math:`X_j` the
    multi-hot vectors of block :math:`j` and :math:`M \\in \\{0, 1\\}^{n
    \\times m}` the observation mask derived from the non-zero pattern
    of the blocks. ALS minimizes the masked least-squares objective

    .. math::

        \\min_{U, V} \\;
        \\sum_{(i, j) \\, : \\, M_{ij} = 1}
        \\left\\| u_i^\\top V_j - x_{ij} \\right\\|^2
        + \\alpha \\left( \\| U \\|_F^2 + \\| V \\|_F^2 \\right)

    where :math:`x_{ij}` is the multi-hot target of cell
    :math:`(i, j)` over the vocabulary of block :math:`j`. Because
    :math:`x_{ij}` is multi-hot, the squared error of a cell expands
    to the prediction squares over its whole block vocabulary minus
    twice the predictions of the observed values plus the number of
    observed values, which allows a closed-form RMSE:

    .. math::

        \\mathrm{RMSE}^2 = \\frac{1}{\\sum_j o_j m_j}
        \\sum_{(i, j) \\, : \\, M_{ij} = 1} \\Big(
        \\sum_{t} (u_i^\\top v_{jt})^2
        - 2 \\sum_{t \\in x_{ij}} u_i^\\top v_{jt}
        + |x_{ij}| \\Big)

    with :math:`o_j` the number of cells observing block :math:`j` and
    :math:`m_j` its vocabulary size. Each ALS half-step solves the
    objective exactly in one variable group with the other fixed:

    - sample step, an independent :math:`k \\times k` system per
      sample:

      .. math::

          A_i = \\sum_{j \\, : \\, M_{ij} = 1} V_j V_j^\\top
              + \\alpha I, \\qquad
          b_i = \\sum_{j} X_{j\\, i, :} \\, V_j^\\top, \\qquad
          u_i = A_i^{-1} b_i

    - feature step, an independent :math:`k \\times k` system per
      block:

      .. math::

          A_j = \\sum_{i \\, : \\, M_{ij} = 1} u_i u_i^\\top
              + \\alpha I, \\qquad
          B_j = X_j^\\top U, \\qquad
          V_j = A_j^{-1} B_j^\\top

    Both systems are positive definite whenever
    :math:`\\alpha > 0`. The training loop stops as soon as the RMSE
    improvement between two iterations drops below `tol`, keeping an
    update only when the RMSE actually improves.

    The two-stage schedule concentrates the iterations on the samples
    and blocks with the highest coverage: the first stage alternates
    the sample and feature steps on the densest prefix of the
    reordered data, and the second stage solves the remaining samples
    against the first-stage feature embeddings and the remaining
    blocks against all sample embeddings once each. A final pass
    re-solves all sample embeddings against the final components.
    """

    if len(vectors) == 0:
        raise ValueError("vectors must contain at least one block.")

    n_samples = vectors[0].shape[0]
    sizes = numpy.asarray([v.shape[1] for v in vectors], dtype=numpy.int32)
    if (sizes == 0).any():
        raise ValueError("vectors must not contain empty blocks.")
    if n_samples == 0:
        raise ValueError("vectors must contain at least one sample.")

    n_blocks = len(vectors)

    # derive the sample and feature coverages from the non-zero pattern
    # of the vector blocks; the dense observation mask is only built for
    # the reordered data, where the training uses it
    sample_cov = numpy.zeros(n_samples, dtype=numpy.int32)
    feature_cov = numpy.zeros(n_blocks, dtype=numpy.int32)
    for j, v in enumerate(vectors):
        observed = v.getnnz(axis=1) > 0
        sample_cov += observed
        feature_cov[j] = numpy.count_nonzero(observed)

    if not sample_cov.any():
        raise ValueError("vectors contain no observed cell.")

    if isinstance(min_features, float):
        min_features = int(n_blocks * min_features)
    if min_features < n_blocks and min_feature_coverage > 0:
        if isinstance(min_feature_coverage, float):
            thr = int(n_samples * min_feature_coverage)
        else:
            thr = int(min_feature_coverage)

        aug = -feature_cov
        feature_order = numpy.argsort(aug, kind="stable")
        feature_pos = int(numpy.clip(
            numpy.searchsorted(aug[feature_order], -thr, side='right'),
            max(min_features, 1),
            n_blocks
        ))
    else:
        feature_order = numpy.arange(n_blocks)
        feature_pos = n_blocks

    if isinstance(min_samples, float):
        min_samples = int(n_samples * min_samples)
    if min_samples < n_samples and min_sample_coverage > 0:
        if isinstance(min_sample_coverage, float):
            thr = int(n_blocks * min_sample_coverage)
        else:
            thr = int(min_sample_coverage)

        aug = -sample_cov
        sample_order = numpy.argsort(aug, kind="stable")
        sample_pos = int(numpy.clip(
            numpy.searchsorted(aug[sample_order], -thr, side='right'),
            max(min_samples, 1),
            n_samples
        ))
    else:
        sample_order = numpy.arange(n_samples)
        sample_pos = n_samples

    logger.debug(
        f"training with {sample_pos} of {n_samples} samples and "
        f"{feature_pos} of {n_blocks} blocks, "
        f"n_components = {n_components}, maximum iteration = {max_iter}, "
        f"tol = {tol}, alpha = {alpha} ..."
    )

    # reorder the blocks and the samples so the first-stage subset forms
    # a dense prefix; reordering blocks is a cheap list permutation
    vectors = [vectors[c] for c in feature_order]
    vectors = [v[sample_order].astype(numpy.float32) for v in vectors]
    for v in vectors:
        v.sort_indices()

    mask = numpy.empty((n_samples, n_blocks), dtype=bool)
    for j, v in enumerate(vectors):
        mask[:, j] = v.getnnz(axis=1) > 0

    rng = sklearn.utils.check_random_state(random_state)
    k = n_components
    embeddings = rng.randn(n_samples, k).astype(numpy.float32)
    components = [
        rng.randn(k, v.shape[1]).astype(numpy.float32)
        for v in vectors
    ]

    # first-stage structures, precomputed once and reused every
    # iteration
    blocks_ul = [
        v[:sample_pos] for v in vectors[:feature_pos]
    ]
    mask_ul = mask[:sample_pos, :feature_pos]

    prev_rmse = numpy.inf
    n_iter = max_iter
    for it in range(max_iter):
        new_embeddings = solve_embeddings(
            blocks_ul, mask_ul, components[:feature_pos], alpha
        )
        new_components = solve_components(
            blocks_ul, mask_ul, new_embeddings[:sample_pos], alpha
        )

        # closed-form RMSE over the observed first-stage cells, computed
        # with the current embeddings
        sse = 0.0
        count = 0
        for j, (block, c) in enumerate(
            zip(blocks_ul, components[:feature_pos])
        ):
            pred = embeddings[:sample_pos] @ c
            sse -= 2.0 * float(block.multiply(pred).sum())
            seg = numpy.sum(pred ** 2, axis=1)
            sse += float(seg[mask_ul[:, j]].sum())
            count += int(mask_ul[:, j].sum()) * c.shape[1]

        if count == 0:
            logger.warning("no observed cell in the first stage, stop.")
            n_iter = 0
            break

        rmse = numpy.sqrt(
            max(sse + sum(b.nnz for b in blocks_ul), 0.0) / count
        )
        logger.debug(f"iteration {it + 1}: RMSE = {rmse}")

        improve = prev_rmse - rmse
        if improve > 0:
            # keep the update only when the RMSE improves
            embeddings[:sample_pos] = new_embeddings
            components[:feature_pos] = new_components

        if improve < tol:
            # the RMSE improvement dropped below the threshold, stop
            logger.debug(f"{improve} < tol = {tol}, stop training.")
            n_iter = it + 1
            break

        prev_rmse = rmse
    else:
        logger.warning(
            f"maximum iteration {max_iter} reached, training not converged."
        )

    logger.debug("done.")

    # second stage: update the remaining samples and blocks with the
    # first-stage results
    if sample_pos < n_samples:
        logger.debug(f"updating rest {n_samples - sample_pos} samples ...")
        blocks_r = [v[sample_pos:] for v in vectors[:feature_pos]]
        embeddings[sample_pos:] = solve_embeddings(
            blocks_r,
            mask[sample_pos:, :feature_pos],
            components[:feature_pos],
            alpha
        )
        logger.debug("done.")

    if feature_pos < n_blocks:
        logger.debug(f"updating rest {n_blocks - feature_pos} blocks ...")
        components[feature_pos:] = solve_components(
            vectors[feature_pos:],
            mask[:, feature_pos:],
            embeddings,
            alpha
        )
        logger.debug("done.")

    # final pass: re-solve all sample embeddings against the final
    # components over the reordered data, so the returned training
    # embeddings agree with projecting the training data through
    # solve_embeddings
    logger.debug("solving the final sample embeddings ...")
    embeddings = solve_embeddings(
        vectors, mask, components, alpha
    )
    logger.debug("done.")

    # restore the input order of the samples; the feature-value
    # embeddings are restored to the input block order and concatenated
    # along their feature axis
    embs = numpy.empty_like(embeddings)
    embs[sample_order] = embeddings

    comps = [None] * n_blocks
    for j, c in enumerate(feature_order):
        comps[c] = components[j]

    return embs, numpy.concatenate(comps, axis=1), n_iter
