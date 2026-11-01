# -*- coding: utf-8 -*-
"""Simulate the formation of dialects from a common ancestor.

The simulation starts from one ancestor dialect of ``n_characters``
characters, each carrying one value of every feature — a segment
type such as initial, final or tone. The ancestor samples its value
inventories from Dirichlet distributions: for each feature, a
1-of-m distribution over its ``m`` values is drawn from
``Dirichlet(alpha)`` and every character independently samples its
value from that distribution.

The simulation then runs ``n_iter`` generations. In each generation,
every live dialect splits into a left and a right child. The left child
inherits the parent unchanged and the right child applies
one rule sound change: one value of one randomly selected feature is
replaced with another value of the feature vocabulary. The feature
is drawn with equal probability among all features, the replaced
source value among the values the dialect actually holds, and the
target either uniformly among the other vocabulary values or, when a
transition matrix is given for the feature, with the probabilities
of the matrix row of the source as given — the target never equals
the source — and it may or may not merge two of the values the
dialect holds into one. After the
split, every character of every child mutates every feature
independently with probability ``mutation_rate``, changing it to one
of the other values the child holds for that feature — uniformly, or
with the weights of the transition matrix row of the current value.
The population is capped at ``max_dialects``: every live dialect
conceives a left and a right child, and a random selection cuts the
conceived children down to the cap when the population exceeds it.
The lineages are settled first, and the values are generated
afterwards in lineage order — only for the dialects with descendants
surviving to the deepest level, the modern ones and their ancestors,
so the cut children and the extinct lineages never carry values.
Every generated right child draws its own rule change, e.g.
``i5 > i2`` — the change it carries from its parent — while the root
and the left children inherit unchanged; the tree records a node's
change on its own row.

:func:`gen_dialects` returns the global binary-tree numbers of every
dialect conceived in every generation — cut children included — as one
ordered array, plus the numbers of the dialects whose lineages survive
to the deepest level, their values, and their incoming sound changes;
a node's parent number and generation follow from its own number as
``(number - 1) // 2`` and ``(number + 1).bit_length() - 1``.
:func:`gen_dataset`
rebuilds the phylogenetic tree from this metadata, then samples
character coverage for the modern dialects, mimicking real dialect
data whose recorded counts differ across dialects and whose
occurrences concentrate on a few characters.

The values are simulated as integer codes, one vocabulary of the
codes ``0 ... size - 1`` per feature; translating the codes to and
from actual value names is the caller's responsibility.
:func:`gen_dataset` wraps this translation around :func:`gen_dialects`.
Dataset dialect IDs
keep the legacy ``d<level>_<n>`` format, where ``n`` is the node's
zero-based position within its level, derived from the global
binary-tree index.

Usage::

    from sincomp.simulation import gen_dataset, gen_dialects

    dataset, tree = gen_dataset(seed=42)
    dataset.get_data("d4_3")
    tree[tree["changed_feature"].notna()]

    (
        dialect_numbers, survived_numbers,
        survived_values, survived_changes,
    ) = gen_dialects(100, n_categories=[4, 3], seed=42)
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"


import logging

import numpy
import pandas

from sincomp import datasets


logger = logging.getLogger(__name__)

# the features of the initial-final-tone wrapper
FEATURES = ("initial", "final", "tone")


def _as_rng(seed: int | numpy.random.Generator | None) -> numpy.random.Generator:
    """Return a random generator from a seed or a generator itself.

    Parameters
    ----------
    seed : int | numpy.random.Generator | None
        The seed of a new generator, a generator to reuse, or ``None``
        for fresh entropy.

    Returns
    -------
    numpy.random.Generator
        The random generator.
    """

    if isinstance(seed, numpy.random.Generator):
        return seed
    return numpy.random.default_rng(seed)


def _init_ancestor(
    n_characters: int,
    n_categories: list[int],
    alpha: float,
    rng: numpy.random.Generator,
    values: numpy.ndarray | None = None,
) -> dict:
    """Initialize the value codes of the ancestor dialect.

    The codes are either given outright through ``values`` or, when
    it is omitted, sampled from a Dirichlet-drawn value distribution
    of each feature.

    Parameters
    ----------
    n_characters : int
        Number of characters of the dialect.
    n_categories : list[int]
        Number of value categories of each feature.
    alpha : float
        Concentration of the Dirichlet distribution the 1-of-m value
        distribution of each feature is drawn from; unused when
        ``values`` is given.
    rng : numpy.random.Generator
        The random generator.
    values : numpy.ndarray | None
        The value code of every character for every feature as a 2-D
        integer array of shape ``(n_characters, n_features)`` that
        seeds the ancestor dialect directly; the codes need not cover
        the whole vocabulary, so a value that is never observed
        simply never merges.

    Returns
    -------
    dict
        The value code of every character per feature, keyed by the
        feature position.
    """

    if values is None:
        return {
            f: rng.choice(
                size, size=n_characters,
                p=rng.dirichlet(numpy.full(size, alpha)),
            )
            for f, size in enumerate(n_categories)
        }
    return {f: values[:, f].astype(int) for f in range(len(n_categories))}


def _mutate(
    indices: numpy.ndarray,
    rate: float,
    rng: numpy.random.Generator,
    matrix: numpy.ndarray | None = None,
) -> numpy.ndarray:
    """Mutate the value of each character in place.

    Each character keeps its value with probability ``1 - rate`` and
    changes it with probability ``rate`` to one of the other values
    the dialect holds, so a mutation never introduces a value that
    the dialect does not use. The new value is drawn uniformly from
    the other present values, or, when a transition matrix is given,
    with the weights of the matrix row of the current value
    restricted to the other present values; the rows without any
    weight on another present value cannot mutate.

    Parameters
    ----------
    indices : numpy.ndarray
        The value indices of the characters, mutated in place.
    rate : float
        Mutation probability of each character.
    rng : numpy.random.Generator
        The random generator.
    matrix : numpy.ndarray | None
        The transition probability matrix of the feature over its
        vocabulary; ``None`` redraws uniformly.

    Returns
    -------
    numpy.ndarray
        The mutated value indices.

    Raises
    ------
    ValueError
        If the mutation rate is outside ``[0, 1]``.
    """

    if not 0.0 <= rate <= 1.0:
        raise ValueError("the mutation rate must be within [0, 1]")
    used = numpy.unique(indices)
    if len(used) < 2:
        # a single present value has nothing to change into
        return indices
    mask = rng.random(indices.shape[0]) < rate
    positions = numpy.flatnonzero(mask)
    count = len(positions)
    changed = 0
    if count:
        current = indices[positions]
        if matrix is None:
            weights = numpy.ones((count, len(used)))
        else:
            # every candidate redraws from the other present values
            # with the weights of the matrix row of its current
            # value; the rows without mass on another present value
            # cannot mutate
            weights = matrix[current[:, None], used]
        # the current value itself is never the target
        weights[
            numpy.arange(count), numpy.searchsorted(used, current)
        ] = 0.0
        totals = weights.sum(axis=1)
        draws = current.copy()
        valid = totals > 0
        if valid.any():
            rows = weights[valid]
            rows = rows / totals[valid, None]
            boundaries = numpy.cumsum(rows, axis=1)
            picks = (
                boundaries < rng.random(int(valid.sum()))[:, None]
            ).sum(axis=1)
            draws[valid] = used[numpy.minimum(picks, len(used) - 1)]
        hits = draws != current
        indices[positions[hits]] = draws[hits]
        changed = int(numpy.count_nonzero(hits))
    logger.debug(
        "mutated %d of %d characters over %d present values",
        changed, indices.shape[0], len(used),
    )
    return indices


def _generate(
    values: dict,
    n_categories: list[int],
    transition_matrices: list[numpy.ndarray],
    rate: float,
    rng: numpy.random.Generator,
    change: bool,
) -> tuple[dict, int | None, int | None, int | None]:
    """Generate the values of one child of a split dialect.

    The child copies the parent. When ``change`` is set — the child is
    the right one of the split — it applies one rule sound change:
    the feature is drawn with equal probability among all features,
    the replaced source value among the values the dialect actually
    holds, and the target with the probabilities of the transition
    matrix row of the source — the target never equals the source —
    and it may or may not merge two of the values the dialect
    holds into one. Independently of the rule change, every character
    of every feature then mutates with probability ``rate`` into
    another value that the feature holds.


    Parameters
    ----------
    values : dict
        The value code of every character of the parent per feature.
    n_categories : list[int]
        Number of value categories of each feature.
    transition_matrices : list[numpy.ndarray]
        The transition probability matrix of each feature, with the
        rows summing to one; the caller fills the missing ones with
        the uniform distribution over the other values.
    rate : float
        Mutation probability of each character for each feature.
    rng : numpy.random.Generator
        The random generator.
    change : bool
        Whether the child applies the rule sound change.

    Returns
    -------
    tuple[dict, int | None, int | None, int | None]
        The value code of every character of the child per feature
        and, when the child applies a rule change, the feature index
        and the source and target value codes of the change; all
        three are ``None`` when it does not.

    Raises
    ------
    ValueError
        If the mutation rate is outside ``[0, 1]`` or the drawn
        target of the rule change equals its source.
    """

    child = {f: v.copy() for f, v in values.items()}
    f = source = target = None
    if change:
        # the feature of the rule change is drawn with equal
        # probability among all features
        f = int(rng.integers(len(n_categories)))
        used = numpy.unique(child[f])
        # the source is a value the dialect actually holds
        source = int(used[rng.integers(0, len(used))])
        # the target is any value of the vocabulary, drawn with the
        # row as given, so the change may or may not merge two
        # present values
        target = int(
            rng.choice(n_categories[f], p=transition_matrices[f][source])
        )
        if target == source:
            raise ValueError(
                "the drawn target of the rule change equals its source"
            )
        child[f][child[f] == source] = target
        if len(used) >= 2 and len(numpy.unique(child[f])) == 1:
            logger.warning(
                "the rule change %d > %d leaves feature %d with a single "
                "value, which later rule changes may still shift "
                "but mutations cannot touch any more",
                source, target, f,
            )
    # every character of every feature mutates independently into
    # another value that the feature holds
    for feature, values_f in child.items():
        _mutate(values_f, rate, rng, transition_matrices[feature])
    return child, f, source, target


def _dialect_ids(numbers: numpy.ndarray) -> numpy.ndarray:
    """Return the legacy level-local ID of every global tree number.

    The ID is ``d<level>_<n>``, where the level is the generation of
    the dialect — ``floor(log2(number + 1))``, computed exactly from
    the exponent of the floating-point representation — and ``n`` is
    the zero-based position of the dialect within its level.

    Parameters
    ----------
    numbers : numpy.ndarray
        Global binary-tree number of each dialect.

    Returns
    -------
    numpy.ndarray
        One object-dtype ID string per number.
    """

    numbers = numpy.asarray(numbers)
    _, exponents = numpy.frexp(numbers + 1)
    levels = exponents - 1
    return numpy.char.add(
        numpy.char.add("d", levels.astype(str)),
        numpy.char.add("_", (numbers + 1 - (1 << levels)).astype(str)),
    ).astype(object)


def gen_dialects(
    n_characters: int,
    n_features: int | None = None,
    n_categories: list[int] | None = None,
    values: numpy.ndarray | None = None,
    n_iter: int = 5,
    max_dialects: int = 10,
    mutation_rate: float = 0.01,
    alpha: float = 1.0,
    transition_matrices: list[numpy.ndarray | None] | None = None,
    seed: int | numpy.random.Generator | None = None,
) -> tuple[numpy.ndarray, numpy.ndarray, list[tuple]]:
    """Simulate dialect formation from a common ancestor.

    The ancestor holds one value of every feature for every
    character — either given outright through ``values`` or sampled
    from a Dirichlet-drawn value distribution. Over ``n_iter``
    generations the dialects split into children of which the right
    one carries one rule sound change — one value of one randomly
    selected feature is replaced with another value of the feature
    vocabulary, which may or may not merge two of the values the
    dialect holds into one — and each child carries per-character
    mutations that touch every feature independently, while the
    population is capped at ``max_dialects``: every live dialect
    conceives two children, and a random selection keeps
    ``max_dialects`` of the conceived children when the population
    exceeds the cap.

    The lineages are settled first: ``dialect_numbers`` records the
    global binary-tree number of every dialect conceived in every
    generation, cut children included, as one ordered array — the
    generation of a dialect follows from its number, so no
    per-generation information is lost. The values are generated
    afterwards, only for the lineages surviving to the deepest level —
    the modern ones and their ancestors. ``survived_numbers`` holds
    the global binary-tree number of each surviving node — the parent
    number follows from the node's own as ``(number - 1) // 2``.
    ``survived_changes`` records the sound change each node carries
    from its parent as one row of the feature index and the source and
    target value codes, or all ``-1`` for the root and unchanged
    branches. Use :func:`sample_chars` to draw the incomplete
    character coverage of real dialect data.

    The values are simulated as integer codes, one vocabulary of the
    codes ``0 ... size - 1`` per feature; translating the codes to
    and from actual value names is the caller's responsibility.

    Parameters
    ----------
    n_characters : int
        Number of characters of the ancestor dialect.
    n_features : int | None
        Number of features; inferred from ``n_categories`` or
        ``values`` when omitted.
    n_categories : list[int] | None
        Number of value categories of each feature. One of
        ``n_categories`` and ``values`` must be given or
        inferable.
    values : numpy.ndarray | None
        The value code of every character for every feature as a 2-D
        integer array of shape ``(n_characters, n_features)`` with
        the codes ``0 ... size - 1`` of each feature vocabulary; it
        seeds the ancestor dialect directly instead of sampling the
        values from the Dirichlet distribution, in which case
        ``alpha`` is unused. The values need not cover the whole
        vocabulary, and when ``n_categories`` is omitted the
        vocabulary size of a feature is inferred as the largest
        observed code plus one.
    n_iter : int, default=5
        Number of generations.
    max_dialects : int, default=10
        Maximum number of live dialects per generation.
    mutation_rate : float, default=0.01
        Probability that one character mutates one feature in one
        split; the probability that a character keeps its ancestral
        value decays as ``(1 - mutation_rate) ** n_iter``,
        so keep the rate small for simulations of many generations.
    alpha : float, default=1.0
        Concentration of the Dirichlet distribution of the ancestor
        value distribution; smaller values concentrate the characters
        on fewer values. Unused when ``values`` is given.
    transition_matrices : list[numpy.ndarray | None] | None
        Transition probability matrix of each feature as an array of
        shape ``(len(vocabulary), len(vocabulary))``, where entry
        ``(i, j)`` is the probability that a change of the value
        ``i`` draws the value ``j`` as its target. The rows must sum
        to one and are used exactly as given: a non-zero diagonal
        warns, and a draw that hits the source itself raises
        ``ValueError``. The whole argument or any single
        matrix may be ``None``, which fills it with the uniform
        distribution over the other values of the vocabulary before
        the simulation starts. The matrices steer both the rule
        sound changes, whose targets may be any value of the
        vocabulary, and the mutations, whose targets are restricted
        to the values the dialect holds.
    seed : int | numpy.random.Generator | None
        Seed of the random generator, or the generator itself, for
        reproducibility.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray, numpy.ndarray]
        The global binary-tree number of every dialect conceived in
        every generation, cut children included, as one ordered
        one-dimensional integer array, the numbers of the dialects
        whose lineages survive to the deepest level as one ordered
        one-dimensional integer array, their value codes as an
        integer array of shape
        ``(n_dialects, n_characters, n_features)``, and the
        incoming sound change of each surviving node as one row of
        the 2-D integer array of shape ``(n_dialects, 3)`` holding
        the feature index and the source and target value codes, all
        ``-1`` for the root and unchanged branches. The three
        surviving arrays align by position.

    Raises
    ------
    ValueError
        If a structural argument is out of range, if the vocabulary
        and value arguments disagree with ``n_features`` or
        ``n_characters``, if the number of features or the number of
        categories cannot be inferred from anywhere, if the given
        arguments disagree about the number of features, if
        ``values`` is not a non-negative integer array or holds a
        code that is out of its feature vocabulary, or if a
        transition matrix does not match its feature or its rows do
        not sum to one.
    """

    # the number of features is inferred from whichever argument
    # carries it and never sampled
    n_categories_size = None if n_categories is None else len(n_categories)
    values_size = None
    if values is not None:
        values = numpy.asarray(values)
        if values.ndim != 2:
            raise ValueError(
                "values must be a 2-D array of shape "
                "(n_characters, n_features)"
            )
        if not numpy.issubdtype(values.dtype, numpy.integer):
            raise ValueError("values must hold integer value codes")
        if (values < 0).any():
            raise ValueError("value codes must be non-negative")
        values_size = values.shape[1]
    if n_features is None:
        if n_categories_size is None and values_size is None:
            raise ValueError(
                "cannot infer the number of features; give one of "
                "n_features, n_categories or values"
            )
        if (
            n_categories_size is not None
            and values_size is not None
            and n_categories_size != values_size
        ):
            raise ValueError(
                "the given vocabulary or value arguments disagree "
                "about the number of features"
            )
        n_features = (
            n_categories_size
            if n_categories_size is not None else values_size
        )
    elif (
        n_categories_size is not None and n_categories_size != n_features
    ) or (values_size is not None and values_size != n_features):
        raise ValueError(
            "n_features disagrees with the given vocabulary or "
            "value arguments"
        )

    if n_categories is None:
        if values is None:
            raise ValueError(
                "cannot infer the number of categories; give one of "
                "n_categories or values"
            )
        # the vocabulary of a feature covers exactly the codes up to
        # the largest observed one
        n_categories = [
            int(values[:, f].max()) + 1 for f in range(n_features)
        ]
    elif len(n_categories) != n_features:
        raise ValueError(
            "n_categories disagrees with the number of features"
        )

    if values is not None:
        if values.shape[0] != n_characters:
            raise ValueError("n_characters disagrees with values")
        # every code must fall into its feature vocabulary
        for f, size in enumerate(n_categories):
            if (values[:, f] >= size).any():
                raise ValueError(
                    f"the values of the feature {f} hold a code "
                    f"outside its vocabulary of {size} values"
                )

    if min(n_categories) < 2:
        raise ValueError("every feature needs at least 2 values")
    if max_dialects < 1:
        raise ValueError("max_dialects must be at least 1")
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1")
    if not 0.0 < mutation_rate < 1.0:
        raise ValueError("mutation_rate must be within (0, 1)")
    if n_characters < 1:
        raise ValueError("n_characters must be at least 1")

    if transition_matrices is None:
        transition_matrices = [None] * n_features
    if len(transition_matrices) != n_features:
        raise ValueError(
            "transition_matrices must hold one matrix per feature"
        )
    # the missing matrices are filled with the uniform distribution
    # over the other values of the vocabulary before the simulation
    # starts, so the generation never samples a distribution on the
    # fly
    checked = []
    for f, matrix in enumerate(transition_matrices):
        if matrix is None:
            matrix = numpy.full(
                (n_categories[f], n_categories[f]),
                1.0 / (n_categories[f] - 1),
            )
            numpy.fill_diagonal(matrix, 0.0)
        else:
            matrix = numpy.asarray(matrix, dtype=float)
            if matrix.shape != (n_categories[f], n_categories[f]):
                raise ValueError(
                    f"the transition matrix of the feature "
                    f"{f} must be of shape "
                    f"({n_categories[f]}, {n_categories[f]})"
                )
            if not numpy.isfinite(matrix).all() or (matrix < 0).any():
                raise ValueError(
                    f"the transition matrix of the feature "
                    f"{f} must be finite and non-negative"
                )
            if not numpy.isclose(matrix.sum(axis=1), 1.0).all():
                raise ValueError(
                    f"the rows of the transition matrix of the "
                    f"feature {f} must sum to one"
                )
            if (numpy.diag(matrix) != 0).any():
                logger.warning(
                    "the transition matrix of the feature %s assigns "
                    "probability to the source itself; such a draw "
                    "of a rule change is rejected",
                    f,
                )
        checked.append(matrix)
    transition_matrices = checked

    if 1.0 - (1.0 - mutation_rate) ** n_iter > 0.5:
        logger.warning(
            "only %.0f%% of the characters keep their ancestral "
            "values after %d generations; consider a smaller "
            "mutation_rate",
            100.0 * (1.0 - mutation_rate) ** n_iter, n_iter,
        )

    rng = _as_rng(seed)
    logger.info(
        "simulating %d generations of %d characters, value "
        "inventories %s, population cap %d, mutation rate %s",
        n_iter, n_characters, n_categories, max_dialects, mutation_rate,
    )
    # phase 1: settle the lineages — every live dialect of a
    # generation conceives the left and the right child 2n+1 and 2n+2;
    # when the children outnumber max_dialects, a random selection of
    # the numbers survives and the rest die out without conceiving
    # children of their own. The relation of the dialects is fully
    # determined by their numbers, so this phase operates on numbers
    # alone
    numbers = numpy.zeros(1, dtype=numpy.int64)
    generations = [numbers]
    for gen in range(n_iter):
        children = numpy.concatenate((2 * numbers + 1, 2 * numbers + 2))
        if len(children) > max_dialects:
            children = rng.choice(children, size=max_dialects, replace=False)
        generations.append(children)
        numbers = generations[-1]
        logger.debug(
            "generation %d: %d children survive the cap",
            gen + 1, len(numbers),
        )
    # the generation of a dialect follows from its number, so the
    # conceived dialects of every generation fold into one ordered
    # array without losing the per-generation information
    dialect_numbers = numpy.sort(numpy.concatenate(generations))

    # phase 2: generate values only for the lineages surviving to the
    # deepest level and their ancestors; the parent number of a
    # dialect is always (number - 1) // 2, so every generation's
    # surviving numbers derive from the one below
    survived_numbers = [generations[-1]]
    for numbers in reversed(generations[:-1]):
        survived_numbers.append(numpy.unique((survived_numbers[-1] - 1) // 2))
    survived_numbers = numpy.sort(numpy.concatenate(survived_numbers))

    # the first surviving dialect is the ancestor; a node without an
    # incoming change holds -1 in all three columns
    generated = [
        _init_ancestor(n_characters, n_categories, alpha, rng, values)
    ]
    changes = numpy.full((len(survived_numbers), 3), -1, dtype=int)
    for i in range(1, len(survived_numbers)):
        number = int(survived_numbers[i])
        parent = int(
            numpy.searchsorted(survived_numbers, (number - 1) // 2)
        )
        (
            val, f, source, target,
        ) = _generate(
            generated[parent], n_categories,
            transition_matrices, mutation_rate, rng,
            change=number % 2 == 0,
        )
        if f is not None:
            changes[i] = (f, source, target)
            logger.debug(
                "%d: rule change %d > %d", number, source, target,
            )
        generated.append(val)

    # the generated value dictionaries align with the surviving
    # numbers by position, so they fold into the values array in order
    survived_values = numpy.empty(
        (len(generated), n_characters, n_features), dtype=int
    )
    for i, val in enumerate(generated):
        for f in range(n_features):
            survived_values[i, :, f] = val[f]
    logger.info("generated values for %d dialects", len(survived_numbers))
    return dialect_numbers, survived_numbers, survived_values, changes


def sample_chars(
    n_characters: int,
    n_dialects: int,
    beta_alpha: float = 5.0,
    beta_beta: float = 5.0,
    dirichlet_alpha: float = 1.0,
    seed: int | numpy.random.Generator | None = None,
) -> list[numpy.ndarray]:
    """Sample character positions for each modern dialect.

    Each dialect draws its coverage probability from the beta distribution
    ``Beta(beta_alpha, beta_beta)`` and its recorded count from the
    resulting beta-binomial distribution, and selects the characters
    with usage weights drawn once from the Dirichlet distribution
    ``Dirichlet(dirichlet_alpha, ...)`` — so some dialects record many
    more characters than others and the occurrences concentrate on a
    few characters.

    Parameters
    ----------
    n_characters : int
        Number of characters available in each dialect.
    n_dialects : int
        Number of modern dialects to sample.
    beta_alpha : float, default=5.0
        First shape parameter of the coverage beta distribution; the
        mean coverage of every dialect is
        ``beta_alpha / (beta_alpha + beta_beta)`` and, through
        ``beta_alpha + beta_beta``, the pair controls how much the
        recorded counts differ across the dialects. Must be positive.
    beta_beta : float, default=5.0
        Second shape parameter of the coverage beta distribution.
        Must be positive.
    dirichlet_alpha : float, default=1.0
        Concentration of the Dirichlet distribution that the usage
        weights of the characters are drawn once from and shared by
        all dialects; smaller values concentrate the recorded
        occurrences on a few characters, larger values even the
        weights out. Must be positive.
    seed : int | numpy.random.Generator | None
        Seed of the random generator, or the generator itself, for
        reproducibility.

    Returns
    -------
    list[numpy.ndarray]
        One integer index array per dialect, containing that dialect's
        sampled character positions. The arrays may have different
        lengths.

    Raises
    ------
    ValueError
        If a count or shape parameter is not positive.
    """

    if n_characters < 1 or n_dialects < 1:
        raise ValueError("n_characters and n_dialects must be positive")
    if beta_alpha <= 0.0 or beta_beta <= 0.0 or dirichlet_alpha <= 0.0:
        raise ValueError(
            "beta_alpha, beta_beta and dirichlet_alpha must be positive"
        )
    if beta_alpha / (beta_alpha + beta_beta) < 0.1:
        logger.warning(
            "the expected coverage beta_alpha / (beta_alpha + "
            "beta_beta) = %.2f is low: every modern dialect records "
            "only a small fraction of the characters",
            beta_alpha / (beta_alpha + beta_beta),
        )

    rng = _as_rng(seed)
    logger.info(
        "sampling %d dialects over %d characters with coverage "
        "Beta(%s, %s) and Dirichlet(%s) usage weights",
        n_dialects, n_characters,
        beta_alpha, beta_beta, dirichlet_alpha,
    )

    # the usage weights of the characters are drawn once and shared
    # by all dialects: they stand for the frequency of the characters
    # in the language, which every dialect shares
    weights = rng.dirichlet(numpy.full(n_characters, dirichlet_alpha))

    indices = []
    total = 0
    for dialect_index in range(n_dialects):
        size = int(
            rng.binomial(
                n_characters, rng.beta(beta_alpha, beta_beta)
            )
        )
        character_indices = rng.choice(
            n_characters, size=size, replace=False, p=weights
        )
        character_indices = numpy.sort(character_indices)
        indices.append(character_indices)
        total += size
        logger.debug(
            "dialect %d records %d characters",
            dialect_index, size,
        )

    logger.info(
        "sampled %d of %d dialect-character occurrences",
        total, n_dialects * n_characters,
    )
    return indices


def gen_dataset(
    n_initial: int | None = None,
    n_final: int | None = None,
    n_tone: int | None = None,
    n_characters: int | None = None,
    characters: list[str] | None = None,
    initials: list[str] | None = None,
    finals: list[str] | None = None,
    tones: list[str] | None = None,
    max_dialects: int = 16,
    n_dialects: int | None = None,
    n_iter: int = 4,
    mutation_rate: float = 0.01,
    alpha: float = 1.0,
    transition_matrices: list[numpy.ndarray | None] | None = None,
    sample_beta_alpha: float = 5.0,
    sample_beta_beta: float = 5.0,
    sample_dirichlet_alpha: float = 1.0,
    seed: int | numpy.random.Generator | None = None,
) -> tuple[datasets.Dataset, pandas.DataFrame]:
    """Simulate dialects with the initial, final and tone features.

    A thin wrapper of :func:`gen_dialects` and :func:`sample_chars`
    for the common three-feature case: the features are fixed to
    initial, final and tone, the sampled records of the modern
    dialects are packed into a dataset, and unnamed phones default to
    ``i1 ...``, ``f1 ...`` and ``t1 ...``.

    Parameters
    ----------
    n_initial : int | None
        Number of phones of the initial feature; inferred from
        ``initials`` when omitted.
    n_final : int | None
        Number of phones of the final feature; inferred from
        ``finals`` when omitted.
    n_tone : int | None
        Number of phones of the tone feature; inferred from ``tones``
        when omitted.
    n_characters : int | None
        Number of characters of the ancestor dialect; inferred from
        ``characters`` when omitted.
    characters : list[str] | None
        The written form of each character; required when
        ``n_characters`` is omitted.
    initials : list[str] | None
        The phone names of the initial feature; required when
        ``n_initial`` is omitted.
    finals : list[str] | None
        The phone names of the final feature; required when
        ``n_final`` is omitted.
    tones : list[str] | None
        The phone names of the tone feature; required when
        ``n_tone`` is omitted.
    max_dialects : int, default=16
        Maximum number of live dialects per generation.
    n_dialects : int | None, default=None
        Number of modern dialects uniformly drawn from the dialects
        returned by :func:`gen_dialects` before the character
        sampling; ``None`` keeps every modern dialect. Unlike
        ``max_dialects``, which caps the population during the
        dialect formation, this parameter thins out the modern
        dialects afterwards. When it exceeds the number of modern
        dialects, all of them are kept and a warning is logged.
    n_iter : int, default=4
        Number of generations.
    mutation_rate : float, default=0.01
        Probability that one character mutates the selected feature
        in one split; the probability that a character keeps its
        ancestral phone decays as ``(1 - mutation_rate) ** n_iter``,
        so keep the rate small for simulations of many generations.
    alpha : float, default=1.0
        Concentration of the Dirichlet distribution of the ancestor
        phone distribution; smaller values concentrate the characters
        on fewer phones.
    transition_matrices : list[numpy.ndarray | None] | None
        Transition probability matrices for the initial, final and
        tone inventories, in that order. Each matrix follows the
        format described by :func:`gen_dialects`.
    sample_beta_alpha : float, default=5.0
        First shape parameter of the coverage beta distribution of
        :func:`sample_chars`; the mean coverage of every modern
        dialect is
        ``sample_beta_alpha / (sample_beta_alpha + sample_beta_beta)``.
    sample_beta_beta : float, default=5.0
        Second shape parameter of the coverage beta distribution.
    sample_dirichlet_alpha : float, default=1.0
        Concentration of the Dirichlet distribution that the usage
        weights of the characters are drawn from.
    seed : int | numpy.random.Generator | None
        Seed of the random generator, or the generator itself, for
        reproducibility.

    Returns
    -------
    tuple[datasets.Dataset, pandas.DataFrame]
        The sampled records of the modern dialects as a dataset of
        long-form rows with the ``did``, ``cid``, ``character``,
        ``initial``, ``final`` and ``tone`` columns, and the
        phylogenetic tree as a DataFrame holding the ``did`` of every
        recorded dialect plus the dialect IDs of its ``parent``,
        ``left`` and ``right`` and the ``changed_feature``,
        ``source`` and ``target`` of the sound change it carries
        from its parent; the dialect IDs of the modern dialects, the
        deepest level, join the tree to the records.

    Raises
    ------
    ValueError
        If a count and its corresponding name list are both omitted,
        if a structural argument is out of range, or if an explicit
        character and phone list disagrees with its given count.
    """

    logger.info(
        "generating an initial-final-tone dataset with %s characters",
        len(characters) if characters is not None
        else n_characters,
    )

    inventories = []
    for size, names, feature in (
        (n_initial, initials, FEATURES[0]),
        (n_final, finals, FEATURES[1]),
        (n_tone, tones, FEATURES[2]),
    ):
        if names is not None:
            names = [str(phone) for phone in names]
            if size is not None and size != len(names):
                raise ValueError(
                    "the phone count disagrees with the given phone "
                    "list"
                )
            size = len(names)
        else:
            if size is None:
                raise ValueError(
                    f"cannot infer the {feature} vocabulary; give its "
                    f"count or phone list"
                )
            # unnamed phones auto-encode with the first letter of the
            # feature name: i1 ..., f1 ..., t1 ...
            names = [
                f"{feature[0]}{i + 1}" for i in range(size)
            ]
        inventories.append(names)

    if characters is not None:
        characters = [str(c) for c in characters]
        if (
            n_characters is not None
            and n_characters != len(characters)
        ):
            raise ValueError(
                "n_characters disagrees with the given character list"
            )
        n_characters = len(characters)
    if n_characters is None and characters is None:
        raise ValueError(
            "cannot infer the number of characters; give n_characters "
            "or characters"
        )

    if n_dialects is not None and n_dialects < 1:
        raise ValueError("n_dialects must be at least 1")

    rng = _as_rng(seed)
    (
        dialect_numbers, survived_numbers,
        survived_values, survived_changes,
    ) = gen_dialects(
        n_characters,
        len(FEATURES),
        n_categories=[len(names) for names in inventories],
        n_iter=n_iter,
        max_dialects=max_dialects,
        mutation_rate=mutation_rate,
        alpha=alpha,
        transition_matrices=transition_matrices,
        seed=rng,
    )
    # the value codes are decoded back to the phone names: the
    # vocabulary of gen_dialects shares the index order with the
    # phone inventory of each feature
    survived_values = numpy.stack([
        numpy.asarray(inventories[f])[survived_values[:, :, f]]
        for f in range(len(FEATURES))
    ], axis=2)
    # the tree covers every dialect conceived in every generation,
    # but only the surviving ones carry a sound change and values
    changes = numpy.full((len(dialect_numbers), 3), None, dtype=object)
    position = numpy.searchsorted(dialect_numbers, survived_numbers)
    # the change codes decode into the feature names and the phone
    # names of the inventories, gathered from one 2-D name table that
    # shares the code order of the vocabularies
    changed = survived_changes[:, 0] >= 0
    f = survived_changes[changed, 0]
    names = numpy.full(
        (len(FEATURES), max(map(len, inventories))), None, dtype=object
    )
    for i, inventory in enumerate(inventories):
        names[i, :len(inventory)] = inventory
    changes[position[changed], 0] = numpy.asarray(FEATURES, dtype=object)[f]
    changes[position[changed], 1] = names[f, survived_changes[changed, 1]]
    changes[position[changed], 2] = names[f, survived_changes[changed, 2]]
    # the parent of every conceived dialect is conceived as well, and
    # the membership of the children follows from the numbers alone
    numbers = dialect_numbers
    left_numbers = 2 * numbers + 1
    right_numbers = 2 * numbers + 2
    tree = pandas.DataFrame({
        "did": _dialect_ids(numbers),
        "parent": numpy.where(
            numbers > 0,
            _dialect_ids(numpy.maximum((numbers - 1) // 2, 0)), None,
        ),
        "left": numpy.where(
            numpy.isin(left_numbers, numbers), _dialect_ids(left_numbers),
            None,
        ),
        "right": numpy.where(
            numpy.isin(right_numbers, numbers), _dialect_ids(right_numbers),
            None,
        ),
        "changed_feature": changes[:, 0],
        "source": changes[:, 1],
        "target": changes[:, 2],
    })
    levels = numpy.floor(numpy.log2(survived_numbers + 1)).astype(int)
    modern_mask = levels == n_iter
    modern_numbers = survived_numbers[modern_mask]
    modern_values = survived_values[modern_mask]
    if n_dialects is not None:
        if n_dialects > len(modern_numbers):
            logger.warning(
                "n_dialects = %d exceeds the number of modern "
                "generated dialects; keeping all of them",
                n_dialects,
            )
        elif n_dialects < len(modern_numbers):
            chosen = rng.choice(
                len(modern_numbers), size=n_dialects, replace=False
            )
            logger.info(
                "sampled %d of %d generated dialects",
                n_dialects, len(modern_numbers),
            )
            modern_numbers = modern_numbers[chosen]
            modern_values = modern_values[chosen]

    cids = [f"c{i + 1}" for i in range(modern_values.shape[1])]
    if characters is None:
        characters = cids
    dids = _dialect_ids(modern_numbers)
    data = pandas.DataFrame({
        "did": numpy.repeat(dids, modern_values.shape[1]),
        "cid": numpy.tile(cids, len(dids)),
        "character": numpy.tile(characters, len(dids)),
        **{
            feature: modern_values[:, :, i].reshape(-1)
            for i, feature in enumerate(FEATURES)
        },
    })
    sampled_indices = sample_chars(
        n_characters=modern_values.shape[1],
        n_dialects=len(modern_numbers),
        beta_alpha=sample_beta_alpha, beta_beta=sample_beta_beta,
        dirichlet_alpha=sample_dirichlet_alpha, seed=rng,
    )
    flat_indices = numpy.concatenate([
        dialect_index * modern_values.shape[1] + character_indices
        for dialect_index, character_indices in enumerate(sampled_indices)
    ])
    data = data.iloc[flat_indices].reset_index(drop=True)
    dataset = datasets.Dataset(
        data=data,
        name="simulated",
        description="Simulated dialects descended from a common "
                    "ancestor.",
    )
    return dataset, tree
