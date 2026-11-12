# -*- coding: utf-8 -*-

"""
SinComp: a toolset for the computation and comparison of Chinese dialects.

SinComp is a Python toolset for loading, cleaning and comparing the
pronunciation data of Chinese dialects. It turns the reading
pronunciations (initials, finals and tones) collected by fieldwork
projects and digitized corpora into aligned, comparable feature tables,
computes dialect-to-dialect distances and similarities, learns vector
representations of phones, characters and dialects, and simulates the
evolution of dialect populations along a phylogenetic tree.

Main features
-------------

Datasets
    :mod:`sincomp.datasets` loads the long-form pronunciation records
    (one row per dialect and character) from several digitized sources,
    including the 漢語方音字彙 digitization (`Beida zihui
    <https://github.com/lexibank/beidazihui>`_), the modern dialects of
    the Xiaoxue archival database (`CCR
    <https://xiaoxue.iis.sinica.edu.tw/ccrdata/>`_), the `MCPDict
    <https://mcpdict.sourceforge.io/>`_ and the 中国语言资源保护工程
    采录展示平台 (`zhongguoyuyan <https://zhongguoyuyan.cn/>`_).
    Datasets may be downloaded lazily and cached locally; a directory
    of CSV files or a single CSV file works as a custom dataset.

Preprocessing
    The preprocessing functions clean up raw phone transcriptions,
    recognize and parse romanization schemes, convert long-form
    pronunciation records into the wide-form feature matrices used by
    the estimators, and fill in the missing readings.

Alignment
    :func:`align` aligns the characters of two dialects through their
    reading correspondences, which is the basis of most comparisons;
    :func:`align_no_cid` aligns datasets without shared character IDs.

Rule comparison
    :func:`load_rules` reads declarative sound-change rules from JSON
    files and :func:`compliance` scores how well every dialect of a
    dataset complies with each rule, yielding the rule-compliance
    tables used to draw isoglosses.

Similarity
    :func:`chi2` and :func:`entropy` measure the distance or
    similarity between dialects over aligned feature matrices.

Embeddings
    :class:`PhoneSimilarity` derives phone similarities from their
    distributional behavior; :class:`DialectVectorizer` and
    :class:`DialectEmbedding` embed dialects into continuous spaces;
    :class:`CharacterVectorizer` and :class:`CharacterEmbedding` do
    the same for characters.

Factorization
    :func:`factorize` and :func:`factorize_svd` decompose
    dialect-by-feature matrices into latent components, e.g. to
    recover the areal structure behind the correspondences.

Neural models
    :mod:`sincomp.nn` exposes scikit-learn compatible neural
    building blocks on top of TensorFlow (:class:`~sincomp.nn.Processor`,
    :class:`~sincomp.nn.EncoderBase` and :class:`~sincomp.nn.BilinearEncoder`).
    TensorFlow is an optional dependency; the names in that module are
    importable only when it is installed.

Simulation
    :func:`gen_dialects` and :func:`gen_dataset` simulate a population
    of dialects descending from a common ancestor along a binary
    phylogenetic tree, under a mutation model optionally steered by
    transition matrices learned from real data.

Command line
    The ``sincomp`` command exposes the main workflows as
    human-friendly subcommands; every subcommand is also available as
    a stable entry point under :mod:`sincomp.cli`.

Plotting
    :mod:`sincomp.plot` visualizes similarity matrices, dendrograms,
    maps and the simulated phylogenies (requires the optional
    ``plot`` extra).

Examples
--------
>>> import sincomp
>>> dataset = sincomp.datasets.get("MCPDict")          # doctest: +SKIP
>>> features = sincomp.transform(dataset.data)          # doctest: +SKIP
>>> similarity = sincomp.chi2(features)                 # doctest: +SKIP

Package layout
--------------

The implementation lives in the private package
:mod:`sincomp._internal`; only the names imported into this module, the
:mod:`sincomp.datasets` and :mod:`sincomp.nn` facades, together with the
:mod:`sincomp.cli` and :mod:`sincomp.plot` packages, are public. Names
not listed here may change at any time without notice.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'

# align: character alignment
from ._internal.align import align, align_no_cid

# compare: sound rule comparison
from ._internal.compare import compliance, load_rules

# datasets: dialect pronunciation datasets
from ._internal.dataset import Dataset, FileDataset

# embedding: pronunciation embeddings
from ._internal.embedding import (
    CharacterEmbedding,
    CharacterVectorizer,
    DialectEmbedding,
    DialectVectorizer,
    PhoneSimilarity,
)

# factorize: feature factorization
from ._internal.factorize import factorize, factorize_svd

# similarity: dialect similarity
from ._internal.similarity import chi2, entropy

# simulation: phylogenetic simulation
from ._internal.simulation import gen_dialects, gen_dataset
