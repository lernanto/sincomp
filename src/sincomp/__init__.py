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
    (one row per dialect and character) from several digitized
    sources, downloading and caching them lazily; a single CSV file
    or a directory of per-dialect CSV files works as a custom
    dataset, and simulated datasets can be generated randomly for
    testing and benchmarking without real data.

Preprocessing
    The preprocessing functions clean up raw phone transcriptions,
    recognize and parse romanization schemes, and convert long-form
    pronunciation records into the wide-form feature matrices used by
    the estimators.

Imputation
    :mod:`sincomp.impute` fills the missing values of the wide-form
    feature tables with scikit-learn compatible transformers.

Alignment
    :func:`align` aligns the characters of two dialects through their
    reading correspondences, which is the basis of most comparisons;
    :func:`align_no_cid` aligns datasets without shared character IDs.

Feature extraction
    The feature extraction estimators convert pronunciation tables
    into numeric features: they vectorize dialects and characters
    from their phone distributions, and score the compliance of every
    dialect with sets of sound-change rules.

Decomposition
    The decomposition estimators learn vector representations from
    the feature tables: they derive phone similarities from the
    distributional behavior of phones, embed dialects and characters
    into continuous spaces, and decompose dialect-by-feature matrices
    into latent components with alternating least squares, e.g. to
    recover the areal structure behind the reading correspondences.

Metrics
    :func:`~sincomp.metrics.dialect_similarities` computes the pairwise
    similarities between the dialects of a feature table, either by the
    normalized chi-square statistic or by the conditional entropy of
    their phone co-occurrences.

Neural models
    :mod:`sincomp.nn` exposes scikit-learn compatible neural
    building blocks on top of TensorFlow (:class:`~sincomp.nn.Processor`,
    :class:`~sincomp.nn.EncoderBase` and :class:`~sincomp.nn.BilinearEncoder`).
    TensorFlow is an optional dependency; the names in that module are
    importable only when it is installed.

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
>>> features = sincomp.preprocessing.long_to_wide(dataset.data)  # doctest: +SKIP
>>> similarities = sincomp.metrics.dialect_similarities(  # doctest: +SKIP
...     features, metric="chi2"
... )
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'

# align: character alignment
from ._internal.align import align, align_no_cid
