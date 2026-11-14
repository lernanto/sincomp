# -*- coding: utf-8 -*-

"""
Dialect pronunciation datasets.

This module is the public entry point of :mod:`sincomp._internal.dataset`.
It loads the long-form pronunciation records of Chinese dialects -- one
row per dialect and character, with the character ID, the character
itself and the initial, final and tone -- and normalizes them
into a common schema, whatever the source format is.

Supported sources
-----------------

- Beida zihui: the dialect data of the 漢語方音字彙 (Peking
  University), https://github.com/lexibank/beidazihui
- CCR: the modern dialects of the Xiaoxue archival database,
  https://xiaoxue.iis.sinica.edu.tw/ccrdata/
- MCPDict: the modern dialects of the 汉字音典,
  https://mcpdict.sourceforge.io/
- zhongguoyuyan: the dialect data of the 中国语言资源保护工程采录
  展示平台, https://zhongguoyuyan.cn/

Usage
-----

The predefined datasets are exposed as ready-to-use module attributes
(:data:`beidazihui`, :data:`ccr`, :data:`mcpdict` and
:data:`zhongguoyuyan`); :func:`get` looks up a predefined dataset by
name, or loads a custom dataset from a CSV file or a directory of
per-dialect CSV files, and :func:`predefined` maps every registered
dataset. :func:`make_dataset` simulates a pronunciation dataset
instead of loading one: it evolves the initial, final and tone
features from a common ancestor along a binary phylogenetic tree and
returns the sampled records of the modern dialects, useful for
testing and benchmarking without real data. The dataset classes are
not exposed here: the concrete dataset classes remain internal,
so always create datasets through :func:`get`. The module
attributes :data:`cache_dir` and :data:`dataset_dir` locate the local
data cache, evaluated at import time from the ``SINCOMP_CACHE``
environment variable.

Examples
--------
>>> import sincomp.datasets
>>> dataset = sincomp.datasets.get("MCPDict")             # doctest: +SKIP
>>> dataset.get_data("1800長沙").head()                   # doctest: +SKIP
>>> custom = sincomp.datasets.get("path/to/dialects")     # doctest: +SKIP
>>> simulated = sincomp.datasets.make_dataset(            # doctest: +SKIP
...     n_initial=10, n_final=10, n_tone=4, n_characters=20
... )
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'


import logging
import os
import sys

import numpy
import pandas

from ._internal import dataset, simulation


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler())


cache_dir = os.getenv(
    'SINCOMP_CACHE',
    os.path.join(
        os.getenv(
            'LOCALAPPDATA',
            os.path.expanduser('~')
        ) if sys.platform.startswith('win')
        else os.getenv(
            'XDG_DATA_HOME',
            os.path.join(os.path.expanduser('~'), '.local', 'share')
        ),
        'sincomp'
    )
)
dataset_dir = os.path.join(cache_dir, 'datasets')

ccr = dataset.CCRDataset(os.path.join(dataset_dir, 'ccr'))
mcpdict = dataset.MCPDictDataset(os.path.join(dataset_dir, 'mcpdict'))
beidazihui = dataset.BeidazihuiDataset(os.path.join(dataset_dir, 'beidazihui'))
zhongguoyuyan = dataset.ZhongguoyuyanDataset(
    os.path.join(dataset_dir, 'zhongguoyuyan')
)
_datasets = [beidazihui, ccr, mcpdict, zhongguoyuyan]
_dataset_mapping = {
    'beidazihui': beidazihui,
    'CCR': ccr,
    'ccr': ccr,
    'xiaoxue': ccr,
    'MCPDict': mcpdict,
    'mcpdict': mcpdict,
    'zhongguoyuyan': zhongguoyuyan,
    'yubao': zhongguoyuyan
}


def predefined() -> list[dataset.Dataset]:
    """
    List all predefined datasets.

    Returns
    -------
    datasets : list of Dataset
        List of all predefined dataset objects.
    """

    return _datasets.copy()


def make_dataset(
    n_initial: int | None = None,
    n_final: int | None = None,
    n_tone: int | None = None,
    n_characters: int | None = None,
    characters: list[str] | None = None,
    initials: list[str] | None = None,
    finals: list[str] | None = None,
    tones: list[str] | None = None,
    n_dialects: int | None = None,
    n_iter: int = 5,
    max_dialects: int = 10,
    mutation_rate: float = 0.01,
    alpha: float = 1.0,
    transition_matrices: list[numpy.ndarray | None] | None = None,
    sample_beta_alpha: float = 5.0,
    sample_beta_beta: float = 5.0,
    sample_dirichlet_alpha: float = 1.0,
    seed: int | numpy.random.Generator | None = None,
) -> dataset.Dataset:
    """
    Simulate a dialect pronunciation dataset.

    This is a thin wrapper of
    :func:`sincomp._internal.simulation.gen_dataset`: it simulates the
    formation of dialects with the initial, final and tone features
    from a common ancestor along a binary phylogenetic tree and returns
    the packed long-form pronunciation dataset, discarding the lineage
    information the simulation also reports.

    Either give a count or the name list of each feature and the
    character set; when a name list is omitted, phones default to
    ``i1 ...``, ``f1 ...`` and ``t1 ...``, and when the character list
    is omitted, characters are unnamed.

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
    n_dialects : int | None, default=None
        Number of modern dialects uniformly drawn from the dialects
        returned by the simulation before the character sampling;
        ``None`` keeps every modern dialect. Unlike ``max_dialects``,
        which caps the population during the dialect formation, this
        parameter thins out the modern dialects afterwards.
    n_iter : int, default=5
        Number of generations.
    max_dialects : int, default=10
        Maximum number of live dialects per generation.
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
        tone inventories, in that order; ``None`` for a feature means
        uniform transitions.
    sample_beta_alpha : float, default=5.0
        First shape parameter of the coverage beta distribution of the
        character sampling; the mean coverage of every modern dialect
        is ``sample_beta_alpha / (sample_beta_alpha +
        sample_beta_beta)``.
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
    dataset.Dataset
        The simulated dataset of long-form pronunciation records with
        the ``did``, ``cid``, ``character``, ``initial``, ``final``
        and ``tone`` columns.

    Raises
    ------
    ValueError
        If a count and its corresponding name list are both omitted,
        if a structural argument is out of range, or if an explicit
        character and phone list disagrees with its given count.

    Examples
    --------
    >>> from sincomp.datasets import make_dataset
    >>> ds = make_dataset(n_initial=10, n_final=10, n_tone=4,
    ...                   n_characters=20)
    >>> ds.data.columns.tolist()
    ['did', 'cid', 'character', 'initial', 'final', 'tone']
    """

    return simulation.gen_dataset(
        n_initial=n_initial,
        n_final=n_final,
        n_tone=n_tone,
        n_characters=n_characters,
        characters=characters,
        initials=initials,
        finals=finals,
        tones=tones,
        n_dialects=n_dialects,
        n_iter=n_iter,
        max_dialects=max_dialects,
        mutation_rate=mutation_rate,
        alpha=alpha,
        transition_matrices=transition_matrices,
        sample_beta_alpha=sample_beta_alpha,
        sample_beta_beta=sample_beta_beta,
        sample_dirichlet_alpha=sample_dirichlet_alpha,
        seed=seed,
    )[0]


def get(name: str) -> dataset.Dataset | None:
    """Get a predefined dataset or load one from a local path.

    Parameters
    ----------
    name : str
        Name of a predefined dataset, or path to a local dataset
        directory or CSV file.

    Returns
    -------
    dataset.Dataset or None
        The matching predefined dataset or a dataset loaded from the
        local path. Returns ``None`` if ``name`` is neither a
        predefined dataset name nor an existing file or directory.

    Notes
    -----
    Predefined names are checked before interpreting ``name`` as a
    local path. A directory is loaded as a dataset of per-dialect CSV
    files; a CSV file is loaded as a single dataset.
    """

    try:
        return _dataset_mapping[name]

    except KeyError:
        # If it is not predefined, try loading the input as a local path.
        logger.info(
            f'{name} is not a predefined dataset, try loading data from files.'
        )

        if os.path.isdir(name):
            # Load a dataset from the files in the directory.
            return dataset.FileDataset(data_dir=name)

        elif os.path.isfile(name):
            # Load the CSV file and wrap it as a dataset.
            return dataset.Dataset(
                pandas.read_csv(name, dtype=str, encoding='utf-8'),
                name=os.path.splitext(os.path.basename(name))[0]
            )

        else:
            logger.warning(f'{name} not found!')
