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
every live dialect splits into a left and a right child, where the
left child inherits the parent unchanged and the right child merges
one value of one randomly selected feature into another (the
inventory of that feature shrinks by one); both the merged values
and the mutation targets are drawn from the values the dialect
actually holds. After the split, every character of every child
mutates every feature independently with probability
``mutation_rate``, changing it with equal probability to one of the
other values the child holds for that feature. The population is capped at
``max_dialects``: every live dialect conceives a left and a right
child, the conceived children are randomly cut down to the cap, and
only the surviving children are actually generated and recorded.
Each generated dialect has a global binary-tree number: the root is 0,
and a node's left and right children are numbered
``2 * number + 1`` and ``2 * number + 2``. Sound changes from parent
to child are returned alongside the dialect records.

:func:`gen_dialects` returns the global binary-tree numbers of all
generated dialects — a node's parent number follows from its own as
``(number - 1) // 2`` —, their values, and their incoming sound changes.
:func:`gen_dataset` rebuilds the phylogenetic tree from this metadata,
then samples character coverage for the modern dialects, mimicking
real dialect data whose recorded counts differ across dialects and
whose occurrences concentrate on a few characters.

The vocabulary values may be given explicit names through the
``vocabularies`` argument; unnamed ones auto-encode per feature as
``f<feature number>_<n>`` (e.g. ``f1_1 ...``). Dataset dialect IDs
keep the legacy ``d<level>_<n>`` format, where ``n`` is the node's
zero-based position within its level, derived from the global
binary-tree index.

Usage::

    from sincomp.simulation import gen_dataset, gen_dialects

    dataset, tree = gen_dataset(seed=42)
    dataset.get_data("d4_3")
    tree[tree["changed_feature"].notna()]

    numbers, values, changes = gen_dialects(
        100, n_vocabularies=[4, 3], seed=42
    )
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
    vocabularies: list[numpy.ndarray],
    alpha: float,
    rng: numpy.random.Generator,
    values: numpy.ndarray | None = None,
) -> dict:
    """Initialize the ancestor dialect.

    The values are either given outright through ``values`` or,
    when it is omitted, sampled from a Dirichlet-drawn value
    distribution of each feature.

    Parameters
    ----------
    n_characters : int
        Number of characters of the dialect.
    vocabularies : list[numpy.ndarray]
        The value names of each feature.
    alpha : float
        Concentration of the Dirichlet distribution the 1-of-m value
        distribution of each feature is drawn from; unused when
        ``values`` is given.
    rng : numpy.random.Generator
        The random generator.
    values : numpy.ndarray | None
        The real value of every character for every feature as a 2-D
        array of shape ``(n_characters, n_features)`` that seeds the
        ancestor dialect directly; the values need not cover the
        whole vocabulary, so a value that is never observed simply
        never merges.

    Returns
    -------
    dict
        The ancestor dialect with the value index of each character
        per feature, the current inventory sizes, global number zero,
        parent number -1 and no sound change.

    Raises
    ------
    ValueError
        If the given values contain a value that is missing from the
        corresponding vocabulary.
    """

    if values is None:
        indices = {}
        for f, names in enumerate(vocabularies):
            distribution = rng.dirichlet(numpy.full(len(names), alpha))
            indices[f] = rng.choice(
                len(names), size=n_characters, p=distribution
            )
    else:
        positions = [
            {v: i for i, v in enumerate(vocab)}
            for vocab in vocabularies
        ]
        try:
            indices = {
                f: numpy.array(
                    [position[v] for v in values[:, f]], dtype=int
                )
                for f, position in enumerate(positions)
            }
        except KeyError as error:
            raise ValueError(
                f"the values contain {error} which is missing from "
                "the corresponding vocabulary"
            ) from None

    return {
        "values": indices,
        "sizes": [len(names) for names in vocabularies],
        "number": 0, "parent": -1, "change": (None, None, None),
    }


def _mutate(
    indices: numpy.ndarray,
    rate: float,
    rng: numpy.random.Generator,
) -> numpy.ndarray:
    """Mutate the value of each character in place.

    Each character keeps its value with probability ``1 - rate`` and
    changes it with equal probability to one of the other values the
    dialect actually holds, so a mutation never introduces a value
    that the dialect does not use.

    The characters are selected as mutation candidates with
    probability ``k * rate / (k - 1)`` — slightly more than ``rate``
    — where ``k`` is the number of values the dialect holds, and
    every candidate redraws its value uniformly from those ``k``
    values; the redraws that hit the current value leave the
    character unchanged. A character therefore changes with
    probability ``rate`` to a value drawn uniformly from the other
    ``k - 1`` present values.

    Parameters
    ----------
    indices : numpy.ndarray
        The value indices of the characters, mutated in place.
    rate : float
        Mutation probability of each character; must not exceed
        ``(k - 1) / k`` for the ``k`` values the dialect holds.
    rng : numpy.random.Generator
        The random generator.

    Returns
    -------
    numpy.ndarray
        The mutated value indices.

    Raises
    ------
    ValueError
        If the present values cannot support the mutation rate.
    """

    used = numpy.unique(indices)
    if len(used) < 2:
        # a single present value has nothing to change into
        return indices
    p = len(used) * rate / (len(used) - 1)
    if p > 1:
        raise ValueError(
            f"the mutation rate exceeds what the {len(used)} present "
            "values support"
        )
    mask = rng.random(indices.shape[0]) < p
    count = int(numpy.count_nonzero(mask))
    changed = 0
    if count:
        draws = rng.choice(used, size=count)
        current = indices[mask]
        # fresh draws hitting the current value change nothing
        hits = draws != current
        indices[numpy.flatnonzero(mask)[hits]] = draws[hits]
        changed = int(numpy.count_nonzero(hits))
    logger.debug(
        "mutated %d of %d characters over %d present values",
        changed, indices.shape[0], len(used),
    )
    return indices


def _pick_feature(dialect: dict, rng: numpy.random.Generator) -> tuple:
    """Draw a feature of the dialect for a sound change.

    Only the features where the dialect still distinguishes at least
    two values take part — a feature with a single value has no room
    for a change — and the more values a feature still distinguishes,
    the more room a change has, so the feature is drawn with
    probability proportional to the number of values it holds.

    Parameters
    ----------
    dialect : dict
        The dialect whose features are drawn from.
    rng : numpy.random.Generator
        The random generator.

    Returns
    -------
    tuple
        The drawn feature and the values it currently holds.

    Raises
    ------
    ValueError
        If no feature distinguishes two values any more.
    """

    eligible = []
    present = []
    for f, values_f in dialect["values"].items():
        used = numpy.unique(values_f)
        if len(used) >= 2:
            eligible.append(f)
            present.append(used)
    if not eligible:
        raise ValueError(
            "no feature distinguishes two values any more; the "
            "vocabularies cannot support another generation"
        )
    counts = numpy.array([len(used) for used in present], dtype=float)
    index = int(rng.choice(len(eligible), p=counts / counts.sum()))
    return eligible[index], present[index]


def _generate(
    dialect: dict,
    vocabularies: list[numpy.ndarray],
    rate: float,
    rng: numpy.random.Generator,
    merge: bool,
) -> dict:
    """Generate one child of a split dialect.

    The child copies the parent. When ``merge`` is set it applies the
    rule sound change of the split — one value of one randomly
    selected feature merges into another, and the inventory of that
    feature shrinks by one. Independently of the rule change, every
    character of every feature then mutates with probability ``rate``
    into another value that the feature holds. The rule-change
    feature is drawn among the features where the dialect still
    distinguishes at least two values, with probability proportional
    to the number of values each one still distinguishes, and the
    source, the target and the mutation targets are drawn from the
    values the dialect actually holds, so a change never idles. The
    child carries its global binary-tree number and its parent's
    global number.

    Parameters
    ----------
    dialect : dict
        The parent dialect.
    vocabularies : list[numpy.ndarray]
        The value names of each feature.
    rate : float
        Mutation probability of each character for each feature.
    rng : numpy.random.Generator
        The random generator.
    merge : bool
        Whether the child applies the rule sound change.

    Returns
    -------
    dict
        The child dialect with its incoming sound change in
        ``child["change"]`` as a tuple of the feature index and the
        source and target value names, or all ``None`` when the child
        inherits unchanged.

    Raises
    ------
    ValueError
        If no feature distinguishes two values any more.
    """

    # Heap indexing uniquely identifies nodes, even when some nodes
    # were not generated under the population cap.
    child = {
        "values": {g: v.copy() for g, v in dialect["values"].items()},
        "sizes": list(dialect["sizes"]),
        "number": 2 * dialect["number"] + 1 + int(merge),
        "parent": dialect["number"],
        "change": (None, None, None),
    }

    if merge:
        f, used = _pick_feature(dialect, rng)
        # the source and the target are values the dialect actually
        # holds, so the merge never idles
        source = int(used[rng.integers(0, len(used))])
        others = used[used != source]
        target = int(others[rng.integers(0, len(others))])
        child["values"][f][child["values"][f] == source] = target
        child["sizes"][f] = dialect["sizes"][f] - 1
        child["change"] = (
            f, vocabularies[f][source], vocabularies[f][target],
        )
        logger.debug(
            "d%d: rule change %s > %s",
            child["number"], vocabularies[f][source],
            vocabularies[f][target],
        )

    # every character of every feature mutates independently into
    # another value that the feature holds
    for values_f in child["values"].values():
        _mutate(values_f, rate, rng)

    return child


def _build_tree(
    numbers: numpy.ndarray,
    changes: list[tuple],
    features: list[str],
) -> pandas.DataFrame:
    """Rebuild the phylogenetic tree from global dialect numbers.

    Parameters
    ----------
    numbers : numpy.ndarray
        Global binary-tree number of each dialect.
    changes : list[tuple]
        Incoming sound change of each dialect as a tuple of the
        feature index and the source and target value names, or all
        ``None`` for the root and unchanged branches, in the same row
        order as ``numbers``.
    features : list[str]
        Name of each feature, filling the ``changed_feature`` column.

    Returns
    -------
    pandas.DataFrame
        Tree table in creation order, holding the ``did`` of every
        dialect, the dialect IDs of its ``parent``, ``left`` and
        ``right``, and the ``changed_feature``, ``source`` and
        ``target`` of the sound change it carries from its parent,
        all ``None`` for the root and unchanged branches. The legacy
        ``d<level>_<n>`` IDs are derived from the global numbers,
        where ``n`` is the node's zero-based position within its
        level; children not generated under the population cap are
        absent.
    """

    number_set = set(numbers.tolist())

    rows = []
    changes_by_number = dict(zip(numbers.tolist(), changes))
    for number in numbers:
        number = int(number)
        change = changes_by_number.get(number)
        rows.append({
            "did": _dialect_id(number),
            "parent": None if number == 0
            else _dialect_id((number - 1) // 2),
            "left": _dialect_id(2 * number + 1)
            if 2 * number + 1 in number_set else None,
            "right": _dialect_id(2 * number + 2)
            if 2 * number + 2 in number_set else None,
            "changed_feature": None if change is None or change[0] is None
            else features[change[0]],
            "source": None if change is None else change[1],
            "target": None if change is None else change[2],
        })
    return pandas.DataFrame(
        rows,
        columns=[
            "did", "parent", "left", "right",
            "changed_feature", "source", "target",
        ],
    )


def _dialect_id(number: int) -> str:
    """Return the legacy level-local ID for a global tree number."""

    level = (number + 1).bit_length() - 1
    pos = number + 1 - 2 ** level
    return f"d{level}_{pos}"


def gen_dialects(
    n_characters: int,
    n_features: int | None = None,
    n_vocabularies: list[int] | None = None,
    vocabularies: list[numpy.ndarray] | None = None,
    values: numpy.ndarray | None = None,
    n_iter: int = 5,
    max_dialects: int = 10,
    mutation_rate: float = 0.01,
    alpha: float = 1.0,
    seed: int | numpy.random.Generator | None = None,
) -> tuple[numpy.ndarray, numpy.ndarray, list[tuple]]:
    """Simulate dialect formation from a common ancestor.

    The ancestor holds one value of every feature for every
    character — either given outright through ``values`` or sampled
    from a Dirichlet-drawn value distribution. Over ``n_iter``
    generations the dialects split into children of which the right
    one carries one rule sound change (a value merge) and each child
    carries per-character mutations that touch every feature
    independently, while the population is
    capped at ``max_dialects``: every live dialect conceives two
    children, a random selection keeps ``max_dialects`` of the
    conceived children, and only the selected ones are actually
    generated and recorded.

    The returned arrays cover every dialect that was generated.
    ``numbers`` holds the global binary-tree number of each node —
    the parent number follows from the node's own as
    ``(number - 1) // 2``, and nodes omitted by the population cap
    are represented by gaps in the numbers. ``changes`` records the
    sound change from each node's parent, or all ``None`` for the
    root and unchanged branches.

    One of ``n_vocabularies``, ``vocabularies`` and ``values`` must
    be given; the vocabulary values may be named explicitly through
    ``vocabularies``, and unnamed ones auto-encode per feature as
    ``f<feature number>_<n>`` (e.g. ``f1_1 ...``).

    Parameters
    ----------
    n_characters : int
        Number of characters of the ancestor dialect.
    n_features : int | None
        Number of features; inferred from ``n_vocabularies``,
        ``vocabularies`` or ``values`` when omitted.
    n_vocabularies : list[int] | None
        Value vocabulary size of each feature. One of
        ``n_vocabularies``, ``vocabularies`` and ``values`` must be
        given or inferable.
    vocabularies : list[numpy.ndarray] | None
        The value vocabulary of each feature as an array of value
        names; sequences of names are converted, and it defaults to
        the auto-encoded names.
    values : numpy.ndarray | None
        The real value of every character for every feature as a 2-D
        array of shape ``(n_characters, n_features)``; it seeds the
        ancestor dialect directly instead of sampling the values
        from the Dirichlet distribution, in which case ``alpha`` is
        unused. The values need not cover the whole vocabulary, and
        when ``vocabularies`` is omitted it is inferred from the
        observed values.
    n_iter : int, default=5
        Number of generations; must be smaller than the smallest
        value vocabulary so that the vocabularies never run out.
    max_dialects : int, default=10
        Maximum number of live dialects per generation.
    mutation_rate : float, default=0.01
        Probability that one character mutates the selected feature
        in one split; the probability that a character keeps its
        ancestral value decays as ``(1 - mutation_rate) ** n_iter``,
        so keep the rate small for simulations of many generations.
    alpha : float, default=1.0
        Concentration of the Dirichlet distribution of the ancestor
        value distribution; smaller values concentrate the characters
        on fewer values. Unused when ``values`` is given.
    seed : int | numpy.random.Generator | None
        Seed of the random generator, or the generator itself, for
        reproducibility.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray, list[tuple]]
        The global binary-tree number of each generated dialect as a
        one-dimensional integer array, its values as an array of
        shape ``(n_dialects, n_characters, n_features)``, and the
        incoming sound change of each node as a tuple of the feature
        index and the source and target value names, all ``None``
        for the root and unchanged branches.

    Raises
    ------
    ValueError
        If a structural argument is out of range, if the vocabulary
        and value arguments disagree with ``n_features`` or
        ``n_characters``, if the number of features or the
        vocabularies cannot be inferred from anywhere, or if the
        given arguments disagree about the number of features.
    """

    # the number of features is inferred from whichever argument
    # carries it and never sampled
    counts = {}
    if n_vocabularies is not None:
        counts["n_vocabularies"] = len(n_vocabularies)
    if vocabularies is not None:
        counts["vocabularies"] = len(vocabularies)
    if values is not None:
        values = numpy.asarray(values).astype(str)
        if values.ndim != 2:
            raise ValueError(
                "values must be a 2-D array of shape "
                "(n_characters, n_features)"
            )
        counts["values"] = values.shape[1]
    if n_features is None:
        if not counts:
            raise ValueError(
                "cannot infer the number of features; give one of "
                "n_features, n_vocabularies, vocabularies or values"
            )
        if len(set(counts.values())) > 1:
            raise ValueError(
                "the given vocabulary or value arguments disagree "
                "about the number of features"
            )
        n_features = next(iter(counts.values()))
    elif any(n != n_features for n in counts.values()):
        raise ValueError(
            "n_features disagrees with the given vocabulary or "
            "value arguments"
        )

    if vocabularies is not None:
        vocabularies = [
            numpy.asarray(vocab, dtype=str) for vocab in vocabularies
        ]
        if n_vocabularies is not None and (
            len(n_vocabularies) != n_features
            or any(
                size != len(vocab)
                for size, vocab in zip(n_vocabularies, vocabularies)
            )
        ):
            raise ValueError(
                "n_vocabularies disagrees with vocabularies"
            )
    elif n_vocabularies is not None:
        # unnamed values auto-encode per feature as f1_1 ...
        vocabularies = [
            numpy.asarray(
                [f"f{f + 1}_{i + 1}" for i in range(size)], dtype=str
            )
            for f, size in enumerate(n_vocabularies)
        ]
    elif values is not None:
        # the vocabulary is inferred from the observed values and
        # covers exactly them
        vocabularies = [
            numpy.unique(values[:, f]) for f in range(n_features)
        ]
    else:
        raise ValueError(
            "cannot infer the vocabularies; give one of "
            "vocabularies, n_vocabularies or values"
        )
    sizes = [len(vocab) for vocab in vocabularies]

    if values is not None and values.shape[0] != n_characters:
        raise ValueError("n_characters disagrees with values")

    if min(sizes) < 2:
        raise ValueError("every feature needs at least 2 values")
    if max_dialects < 1:
        raise ValueError("max_dialects must be at least 1")
    if n_iter < 1:
        raise ValueError("n_iter must be at least 1")
    if n_iter >= min(sizes):
        raise ValueError(
            "n_iter must be smaller than the smallest value inventory"
        )
    if not 0.0 < mutation_rate < 1.0:
        raise ValueError("mutation_rate must be within (0, 1)")
    if n_characters < 1:
        raise ValueError("n_characters must be at least 1")

    if min(sizes) - n_iter <= 2:
        logger.warning(
            "the smallest value inventory of %d values is only %d "
            "above the %d generations and may run out in deep "
            "lineages",
            min(sizes), min(sizes) - n_iter, n_iter,
        )
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
        n_iter, n_characters, sizes, max_dialects, mutation_rate,
    )
    dialects = [
        _init_ancestor(n_characters, vocabularies, alpha, rng, values)
    ]
    # live stores dialect-list positions; each node's global number is
    # independent of list position and follows binary-heap indexing.
    live = [0]

    for step in range(n_iter):
        # every live dialect conceives a left and a right child; the
        # children are only booked until a random selection keeps
        # max_dialects of them, and only the surviving children are
        # actually generated and recorded — the deleted ones are
        # implied by the gaps in the dialect IDs of the recorded tree
        booked = []
        for parent in live:
            booked.extend(((parent, False), (parent, True)))
        if len(booked) > max_dialects:
            chosen = set(
                rng.choice(len(booked), size=max_dialects,
                           replace=False).tolist()
            )
        else:
            chosen = set(range(len(booked)))
        logger.debug(
            "generation %d: %d live dialects conceive %d children, "
            "%d survive the cap",
            step + 1, len(live), len(booked), len(chosen),
        )

        # generate the surviving children only, grouped by parent;
        # the children of every parent stay consecutive in booked
        children = {}
        for i, (parent, is_right) in enumerate(booked):
            if i in chosen:
                children.setdefault(parent, []).append(is_right)
        live = []
        for parent, flags in children.items():
            if len(flags) == 2:
                left = _generate(
                    dialects[parent], vocabularies, mutation_rate, rng,
                    merge=False,
                )
                right = _generate(
                    dialects[parent], vocabularies, mutation_rate, rng,
                    merge=True,
                )
                dialects.extend((left, right))
                live.extend((len(dialects) - 2, len(dialects) - 1))
            else:
                # one child of the split was cut by the population cap
                # before generation: the survivor applies the rule
                # change exactly when it is the right child
                child = _generate(
                    dialects[parent], vocabularies, mutation_rate, rng,
                    merge=flags[0],
                )
                dialects.append(child)
                live.append(len(dialects) - 1)

    logger.info("%d dialects generated in %d generations",
                len(dialects), n_iter)
    numbers = numpy.asarray(
        [dialect["number"] for dialect in dialects], dtype=int
    )
    changes = [dialect["change"] for dialect in dialects]
    dialect_values = numpy.empty(
        (len(dialects), n_characters, n_features), dtype=object
    )
    for i, dialect in enumerate(dialects):
        for f, vocabulary in enumerate(vocabularies):
            dialect_values[i, :, f] = vocabulary[dialect["values"][f]]
    logger.info("generated values for %d dialects", len(numbers))
    return numbers, dialect_values, changes


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
        Number of generations; must be smaller than the smallest
        phone inventory so that the inventories never run out.
    mutation_rate : float, default=0.01
        Probability that one character mutates the selected feature
        in one split; the probability that a character keeps its
        ancestral phone decays as ``(1 - mutation_rate) ** n_iter``,
        so keep the rate small for simulations of many generations.
    alpha : float, default=1.0
        Concentration of the Dirichlet distribution of the ancestor
        phone distribution; smaller values concentrate the characters
        on fewer phones.
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

    values = []
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
        values.append(names)

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
    numbers, dialect_values, changes = gen_dialects(
        n_characters,
        len(FEATURES),
        vocabularies=values,
        n_iter=n_iter,
        max_dialects=max_dialects,
        mutation_rate=mutation_rate,
        alpha=alpha,
        seed=rng,
    )
    tree = _build_tree(numbers, changes, list(FEATURES))
    levels = numpy.asarray([
        (int(number) + 1).bit_length() - 1
        for number in numbers
    ])
    modern_mask = levels == n_iter
    modern_numbers = numbers[modern_mask]
    modern_values = dialect_values[modern_mask]
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
    dids = [_dialect_id(int(number)) for number in modern_numbers]
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
