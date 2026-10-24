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
only the surviving children are actually generated and recorded —
the deleted ones are implied by the gaps in the dialect IDs of the
recorded tree. The rule change of a split is recorded in the
phylogenetic tree, e.g. ``i5 > i2``.

:func:`gen_dialects` returns the complete data of every dialect that
was generated, together with the phylogenetic tree, whose deepest
level holds the modern dialects, whose shallower leaves are the
dialects that died out without modern descendants and whose internal
nodes are the historical ones.
:func:`sample_chars` then draws a character sample from the modern
dialects, mimicking the incomplete coverage of real dialect data:
the recorded counts differ across the dialects and the
occurrences concentrate on a few characters.
:func:`gen_dataset` chains the two calls for the common
initial-final-tone case and packs the result into a dataset.

Characters and feature values may be given explicit names; otherwise
they are auto-encoded: characters as ``c1 ...`` and, per feature, the
values as ``<first letter><n>`` (e.g. ``i1 ...`` for the initial
feature) or, when the features are unnamed too, as
``<first letter><feature number>_<n>`` (e.g. ``f1_1 ...`` with the
default features ``feature1, feature2 ...``). The dialect ID
``d<level>_<number>`` reflects the
position in the binary lineage: the root is ``d0_0`` and the left
and right children of a dialect number ``2 * number`` and
``2 * number + 1`` on the next level.

Usage::

    from sincomp.simulation import gen_dataset, gen_dialects, sample_chars

    dataset, tree = gen_dataset(seed=42)
    dataset.get_data("d4_3")
    tree[tree["change"] != ""]

    records, tree = gen_dialects(n_vocabularies=[4, 3], seed=42)
    sampled = sample_chars(records, tree, seed=7)
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"


import logging
import warnings

import numpy
import pandas

from sincomp import datasets


logger = logging.getLogger(__name__)

# the features of the initial-final-tone wrapper
FEATURES = ("initial", "final", "tone")
# default sizes of the initial, final and tone value inventories
DEFAULT_SIZES = (20, 20, 8)


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
    vocabularies: list[list[str]],
    alpha: float,
    rng: numpy.random.Generator,
    values: numpy.ndarray | None = None,
) -> dict:
    """Initialize the ancestor dialect.

    The readings are either given outright through ``values`` or,
    when it is omitted, sampled from a Dirichlet-drawn value
    distribution of each feature.

    Parameters
    ----------
    n_characters : int
        Number of characters of the dialect.
    vocabularies : list[list[str]]
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
        per feature, the current inventory sizes and the lineage
        fields of the root: level and number zero, no parent, no
        children and no sound change.

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
        "level": 0, "number": 0, "parent": None,
        "left": None, "right": None, "change": "",
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
    vocabularies: list[list[str]],
    rate: float,
    rng: numpy.random.Generator,
    merge: bool,
) -> tuple[dict, str]:
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
    child carries the level and the number of the next generation;
    the parent position is left to the caller.

    Parameters
    ----------
    dialect : dict
        The parent dialect.
    vocabularies : list[list[str]]
        The value names of each feature.
    rate : float
        Mutation probability of each character for each feature.
    rng : numpy.random.Generator
        The random generator.
    merge : bool
        Whether the child applies the rule sound change.

    Returns
    -------
    tuple[dict, str]
        The child dialect and the recorded sound change, or an empty
        string when the child inherits unchanged.

    Raises
    ------
    ValueError
        If no feature distinguishes two values any more.
    """

    # the child is the right child exactly when it applies the merge
    child = {
        "values": {g: v.copy() for g, v in dialect["values"].items()},
        "sizes": list(dialect["sizes"]),
        "level": dialect["level"] + 1,
        "number": 2 * dialect["number"] + int(merge),
        "parent": None, "left": None, "right": None, "change": "",
    }
    change = ""
    if merge:
        f, used = _pick_feature(dialect, rng)
        # the source and the target are values the dialect actually
        # holds, so the merge never idles
        source = int(used[rng.integers(0, len(used))])
        others = used[used != source]
        target = int(others[rng.integers(0, len(others))])
        child["values"][f][child["values"][f] == source] = target
        child["sizes"][f] = dialect["sizes"][f] - 1
        change = f"{vocabularies[f][source]} > {vocabularies[f][target]}"
        logger.debug(
            "d%d_%d: rule change %s",
            child["level"], child["number"], change,
        )
    # every character of every feature mutates independently into
    # another value that the feature holds
    for values_f in child["values"].values():
        _mutate(values_f, rate, rng)
    return child, change


def _build_tree(
    dialects: list[dict],
) -> pandas.DataFrame:
    """Assemble the phylogenetic tree table.

    Parameters
    ----------
    dialects : list[dict]
        One record per dialect with the ``values``, ``sizes``,
        ``level``, ``number``, ``parent``, ``left``, ``right`` and
        ``change``, where the ``parent``, ``left`` and ``right``
        references hold list positions.

    Returns
    -------
    pandas.DataFrame
        The tree table in creation order, holding the dialect ID
        ``did`` of every recorded dialect and the dialect IDs of its
        ``parent``, ``left`` and ``right`` — the root has no parent,
        and the children deleted by the population cap are not
        recorded. The dialects of the deepest level are the modern
        ones, the shallower leaves are those that died out without
        modern descendants, and the internal nodes are the historical
        ancestors.
    """

    dids = [
        f"d{record['level']}_{record['number']}"
        for record in dialects
    ]

    def did_of(index):
        return None if index is None else dids[index]

    rows = []
    for index, record in enumerate(dialects):
        rows.append({
            "did": dids[index],
            "parent": did_of(record["parent"]),
            "left": did_of(record["left"]),
            "right": did_of(record["right"]),
            "change": record["change"],
        })
    return pandas.DataFrame(
        rows,
        columns=["did", "parent", "left", "right", "change"],
    )


def gen_dialects(
    n_features: int | None = None,
    features: list[str] | None = None,
    n_vocabularies: list[int] | None = None,
    vocabularies: list[list[str]] | None = None,
    n_characters: int | None = None,
    characters: list[str] | None = None,
    values: numpy.ndarray | None = None,
    max_dialects: int = 16,
    n_iter: int = 4,
    mutation_rate: float = 0.01,
    alpha: float = 1.0,
    seed: int | numpy.random.Generator | None = None,
) -> tuple[pandas.DataFrame, pandas.DataFrame]:
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

    The records cover every dialect that was generated — nothing is
    sampled away; the nodes of the deepest level of the phylogenetic
    tree are the modern dialects, its shallower leaves are the
    dialects that died out without modern descendants, and the
    children deleted by the population cap are implied by the gaps in
    the dialect IDs. Use :func:`sample_chars` to draw the incomplete
    character coverage of real dialect data.

    Every structural argument may be omitted: the missing ones are
    inferred from the given ones, and an argument that cannot be
    inferred from anywhere is an error — the structural arguments are
    never sampled. Characters and vocabulary values may be named
    explicitly through the ``characters`` and ``vocabularies``
    arguments; unnamed ones are auto-encoded as ``c1 ...`` and, per
    feature, ``<first letter>1 ...`` or, when the features are
    unnamed too, ``<feature>_1 ...``.

    Parameters
    ----------
    n_features : int | None
        Number of features; inferred from ``features``,
        ``n_vocabularies``, ``vocabularies`` or ``values`` when
        omitted.
    features : list[str] | None
        Name of each feature, used as the record column names;
        defaults to ``feature1 ...``.
    n_vocabularies : list[int] | None
        Value vocabulary size of each feature; the vocabulary names
        auto-encode as ``<first letter>1 ...`` per named feature, or
        ``<feature>_1 ...`` when the features are unnamed too. One of
        ``n_vocabularies``, ``vocabularies`` and ``values`` must be
        given or inferable.
    vocabularies : list[list[str]] | None
        The value vocabulary of each feature; defaults to the
        auto-encoded names.
    n_characters : int | None
        Number of characters of the ancestor dialect; inferred from
        ``characters`` or ``values`` when omitted.
    characters : list[str] | None
        The written form of each character; defaults to the character
        IDs.
    values : numpy.ndarray | None
        The real value of every character for every feature as a 2-D
        array of shape ``(n_characters, n_features)``; it seeds the
        ancestor dialect directly instead of sampling the readings
        from the Dirichlet distribution, in which case ``alpha`` is
        unused. The values need not cover the whole vocabulary, and
        when ``vocabularies`` is omitted it is inferred from the
        observed values.
    max_dialects : int, default=16
        Maximum number of live dialects per generation.
    n_iter : int, default=4
        Number of generations; must be smaller than the smallest
        value vocabulary so that the vocabularies never run out.
    mutation_rate : float, default=0.01
        Probability that one character mutates the selected feature
        in one split; the probability that a character keeps its
        ancestral reading decays as ``(1 - mutation_rate) ** n_iter``,
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
    tuple[pandas.DataFrame, pandas.DataFrame]
        The complete records of every generated dialect of every
        generation as long-form rows with the ``did``, ``cid``,
        ``character`` and one column per feature, and the phylogenetic
        tree as a DataFrame holding the ``did`` of every recorded
        dialect plus the dialect IDs of its ``parent``, ``left`` and
        ``right`` and its ``change``; the dialect IDs of the deepest
        level mark the modern dialects.

    Raises
    ------
    ValueError
        If a structural argument is out of range, if the feature,
        vocabulary and character arguments disagree about a count, or
        if a structural argument cannot be inferred from anywhere.
    """

    # the number of features is inferred from whichever argument
    # carries it and never sampled
    counts = {}
    if features is not None:
        counts["features"] = len(features)
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
                "n_features, features, n_vocabularies, vocabularies "
                "or values"
            )
        if len(set(counts.values())) > 1:
            raise ValueError(
                "the given arguments disagree about the number of "
                "features"
            )
        n_features = next(iter(counts.values()))
    elif any(n != n_features for n in counts.values()):
        raise ValueError(
            "n_features disagrees with the given feature, vocabulary "
            "or value arguments"
        )

    features_given = features is not None
    if features is None:
        # unnamed features default to feature1 ..., so their values
        # auto-encode as f1_1 ... with the first letter of the name
        # followed by the feature number
        features = [f"feature{f + 1}" for f in range(n_features)]
    else:
        features = [str(feature) for feature in features]

    if vocabularies is not None:
        vocabularies = [
            [str(value) for value in vocab] for vocab in vocabularies
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
        if features_given:
            # the values of a named feature auto-encode with the
            # first letter of its name
            vocabularies = [
                [f"{name[0]}{i + 1}" for i in range(size)]
                for name, size in zip(features, n_vocabularies)
            ]
        else:
            # unnamed features keep the <first letter><feature
            # number>_<n> encoding
            vocabularies = [
                [
                    f"{name[0]}{f + 1}_{i + 1}"
                    for i in range(size)
                ]
                for f, (name, size) in enumerate(
                    zip(features, n_vocabularies)
                )
            ]
    elif values is not None:
        # the vocabulary is inferred from the observed values and
        # covers exactly them
        vocabularies = [
            sorted(set(values[:, f].tolist()))
            for f in range(n_features)
        ]
    else:
        raise ValueError(
            "cannot infer the vocabularies; give one of "
            "vocabularies, n_vocabularies or values"
        )
    sizes = [len(vocab) for vocab in vocabularies]

    if characters is not None:
        characters = [str(c) for c in characters]
    if characters is not None and n_characters is None:
        n_characters = len(characters)
    if values is not None:
        if n_characters is None:
            n_characters = values.shape[0]
        elif n_characters != values.shape[0]:
            raise ValueError("n_characters disagrees with values")
        if characters is not None and len(characters) != values.shape[0]:
            raise ValueError("characters disagree with values")
    if n_characters is None:
        raise ValueError(
            "cannot infer the number of characters; give one of "
            "n_characters, characters or values"
        )

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
        warnings.warn(
            f"the smallest value inventory of {min(sizes)} values is "
            f"only {min(sizes) - n_iter} above the {n_iter} "
            "generations and may run out in deep lineages"
        )
    if 1.0 - (1.0 - mutation_rate) ** n_iter > 0.5:
        warnings.warn(
            f"only {100.0 * (1.0 - mutation_rate) ** n_iter:.0f}% of "
            f"the characters keep their ancestral readings after "
            f"{n_iter} generations; consider a smaller mutation_rate"
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
    # the list position of a dialect is its creation order, and the
    # number doubles per level: left children take 2 * number and
    # right children 2 * number + 1
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
            index = len(dialects)
            if len(flags) == 2:
                left, _ = _generate(
                    dialects[parent], vocabularies, mutation_rate, rng,
                    merge=False,
                )
                right, change = _generate(
                    dialects[parent], vocabularies, mutation_rate, rng,
                    merge=True,
                )
                left["parent"] = right["parent"] = parent
                dialects.extend((left, right))
                dialects[parent]["left"] = index
                dialects[parent]["right"] = index + 1
                live.extend((index, index + 1))
            else:
                # one child of the split was cut by the population cap
                # before generation: the survivor applies the rule
                # change exactly when it is the right child
                child, change = _generate(
                    dialects[parent], vocabularies, mutation_rate, rng,
                    merge=flags[0],
                )
                child["parent"] = parent
                dialects.append(child)
                if flags[0]:
                    dialects[parent]["right"] = index
                else:
                    dialects[parent]["left"] = index
                live.append(index)
            dialects[parent]["change"] = change

    logger.info("%d dialects generated in %d generations",
                len(dialects), n_iter)
    tree = _build_tree(dialects)

    # the records cover every dialect of every generation, each with
    # the full character set
    cids = [f"c{i + 1}" for i in range(n_characters)]
    written = characters if characters is not None else cids
    frames = []
    for dialect in dialects:
        frames.append(pandas.DataFrame({
            "did": f"d{dialect['level']}_{dialect['number']}",
            "cid": cids,
            "character": written,
            **{
                features[f]: [
                    vocabularies[f][v] for v in dialect["values"][f]
                ]
                for f in range(len(features))
            },
        }))
    data = pandas.concat(frames, axis=0, ignore_index=True)
    logger.info("recorded %d rows over %d dialects", len(data), len(dialects))
    return data, tree


def sample_chars(
    data: pandas.DataFrame,
    tree: pandas.DataFrame,
    beta_alpha: float = 5.0,
    beta_beta: float = 5.0,
    dirichlet_alpha: float = 1.0,
    seed: int | numpy.random.Generator | None = None,
) -> pandas.DataFrame:
    """Sample the characters recorded by the modern dialects.

    Real dialect data rarely covers every character of every dialect;
    this function mimics the incomplete coverage by drawing a
    character subset from the modern dialects only, leaving the
    historical dialects of the records untouched. Each dialect draws
    its coverage probability from the beta distribution
    ``Beta(beta_alpha, beta_beta)`` and its recorded count from the
    resulting beta-binomial distribution, and selects the characters
    with usage weights drawn once from the Dirichlet distribution
    ``Dirichlet(dirichlet_alpha, ...)`` — so some dialects record many
    more characters than others and the occurrences concentrate on a
    few characters.

    Parameters
    ----------
    data : pandas.DataFrame
        The records returned by :func:`gen_dialects`.
    tree : pandas.DataFrame
        The phylogenetic tree returned by :func:`gen_dialects`; the
        dialects of its deepest level are the modern ones.
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
    pandas.DataFrame
        The records of the modern dialects restricted to the sampled
        characters.

    Raises
    ------
    ValueError
        If a shape parameter is not positive.
    """

    if beta_alpha <= 0.0 or beta_beta <= 0.0 or dirichlet_alpha <= 0.0:
        raise ValueError(
            "beta_alpha, beta_beta and dirichlet_alpha must be positive"
        )
    if beta_alpha / (beta_alpha + beta_beta) < 0.1:
        warnings.warn(
            "the expected coverage beta_alpha / (beta_alpha + "
            f"beta_beta) = {beta_alpha / (beta_alpha + beta_beta):.2f} "
            "is low: every modern dialect records only a small "
            "fraction of the characters"
        )

    rng = _as_rng(seed)
    # the modern dialects are the recorded dialects of the deepest
    # level: the children deleted by the population cap are not
    # recorded, and the shallower leaves died out without modern
    # descendants
    levels = tree["did"].str[1:].str.split("_").str[0].astype(int)
    modern = tree.loc[levels == levels.max(), "did"].tolist()
    sub = data[data["did"].isin(modern)]
    if sub.empty:
        logger.warning("no records of the modern dialects to sample")
        return sub.reset_index(drop=True)
    logger.info(
        "sampling %d modern dialects over %d characters with coverage "
        "Beta(%s, %s) and Dirichlet(%s) usage weights",
        len(modern), len(sub[sub["did"] == modern[0]]),
        beta_alpha, beta_beta, dirichlet_alpha,
    )

    # the characters of one modern dialect cover the full set, so the
    # first dialect fixes the cid order shared by all of them
    first = sub[sub["did"] == modern[0]]
    cids = first["cid"].tolist()
    position = {cid: i for i, cid in enumerate(cids)}
    positions = sub["cid"].map(position).to_numpy()
    dids = sub["did"].to_numpy()

    # the usage weights of the characters are drawn once and shared
    # by all dialects: they stand for the frequency of the characters
    # in the language, which every dialect shares
    weights = rng.dirichlet(numpy.full(len(cids), dirichlet_alpha))

    rows = numpy.zeros(len(sub), dtype=bool)
    for i, did in enumerate(modern):
        size = int(
            rng.binomial(len(cids), rng.beta(beta_alpha, beta_beta))
        )
        mask = numpy.zeros(len(cids), dtype=bool)
        mask[
            rng.choice(len(cids), size=size, replace=False, p=weights)
        ] = True
        rows |= (dids == did) & mask[positions]
        logger.debug("%s records %d characters", did, size)

    logger.info("kept %d of %d records", int(numpy.count_nonzero(rows)), len(sub))
    return sub[rows].reset_index(drop=True)


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
        Number of phones of the initial feature; derived from
        ``initials`` when given, otherwise 20.
    n_final : int | None
        Number of phones of the final feature; derived from
        ``finals`` when given, otherwise 20.
    n_tone : int | None
        Number of phones of the tone feature; derived from ``tones``
        when given, otherwise 8.
    n_characters : int | None
        Number of characters of the ancestor dialect; derived from
        ``characters`` when given, otherwise 3000.
    characters : list[str] | None
        The written form of each character; defaults to the character
        IDs.
    initials : list[str] | None
        The phone names of the initial feature; defaults to
        ``i1 ...``.
    finals : list[str] | None
        The phone names of the final feature; defaults to ``f1 ...``.
    tones : list[str] | None
        The phone names of the tone feature; defaults to ``t1 ...``.
    max_dialects : int, default=16
        Maximum number of live dialects per generation.
    n_dialects : int | None, default=None
        Number of dialects uniformly drawn from the dialects
        generated by :func:`gen_dialects` before the character
        sampling; ``None`` keeps every generated dialect. Unlike
        ``max_dialects``, which caps the population during the
        dialect formation, this parameter thins out the recorded
        dialects afterwards. When it exceeds the number of generated
        dialects, all of them are kept with a warning.
    n_iter : int, default=4
        Number of generations; must be smaller than the smallest
        phone inventory so that the inventories never run out.
    mutation_rate : float, default=0.01
        Probability that one character mutates the selected feature
        in one split; the probability that a character keeps its
        ancestral reading decays as ``(1 - mutation_rate) ** n_iter``,
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
        ``left`` and ``right`` and its ``change``; the dialect IDs of
        the modern dialects, the deepest level, join the tree to the
        records.

    Raises
    ------
    ValueError
        If a structural argument is out of range or the explicit
        character and phone lists disagree with the given counts.
    """

    logger.info(
        "generating an initial-final-tone dataset with %s characters",
        len(characters) if characters is not None
        else n_characters if n_characters is not None else 3000,
    )

    values = []
    for size, names, feature, default in (
        (n_initial, initials, FEATURES[0], DEFAULT_SIZES[0]),
        (n_final, finals, FEATURES[1], DEFAULT_SIZES[1]),
        (n_tone, tones, FEATURES[2], DEFAULT_SIZES[2]),
    ):
        if names is not None:
            names = [str(phone) for phone in names]
            if size is not None and size != len(names):
                raise ValueError(
                    "the phone count disagrees with the given phone "
                    "list"
                )
        else:
            # unnamed phones auto-encode with the first letter of the
            # feature name: i1 ..., f1 ..., t1 ...
            names = [
                f"{feature[0]}{i + 1}"
                for i in range(size if size is not None else default)
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
    if n_characters is None and characters is None:
        # gen_dialects no longer defaults the character count: the
        # initial-final-tone wrapper keeps its own default
        n_characters = 3000

    if n_dialects is not None and n_dialects < 1:
        raise ValueError("n_dialects must be at least 1")

    rng = _as_rng(seed)
    data, tree = gen_dialects(
        features=list(FEATURES),
        vocabularies=values,
        n_characters=n_characters,
        characters=characters,
        max_dialects=max_dialects,
        n_iter=n_iter,
        mutation_rate=mutation_rate,
        alpha=alpha,
        seed=rng,
    )
    if n_dialects is not None:
        if n_dialects > len(tree):
            warnings.warn(
                f"n_dialects = {n_dialects} exceeds the number of "
                f"generated dialects; keeping all of them"
            )
        # thin out the generated dialects by a uniform draw before the
        # character sampling; the tree rows keep the original dialect
        # IDs, so references to dialects that were not drawn stay in
        # the parent, left and right columns
        elif n_dialects < len(tree):
            chosen = rng.choice(len(tree), size=n_dialects, replace=False)
            kept = set(tree.iloc[chosen]["did"])
            logger.info(
                "sampled %d of %d generated dialects",
                n_dialects, len(tree),
            )
            tree = tree[tree["did"].isin(kept)].reset_index(drop=True)
            data = data[data["did"].isin(kept)].reset_index(drop=True)
    data = sample_chars(
        data, tree,
        beta_alpha=sample_beta_alpha, beta_beta=sample_beta_beta,
        dirichlet_alpha=sample_dirichlet_alpha, seed=rng,
    )
    dataset = datasets.Dataset(
        data=data,
        name="simulated",
        description="Simulated dialects descended from a common "
                    "ancestor.",
    )
    return dataset, tree
