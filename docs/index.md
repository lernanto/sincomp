# SinComp User Guide

SinComp is a Python toolkit for loading, preprocessing, comparing, and
modeling Chinese dialect pronunciation data. This guide follows the data
workflow from source records to analyses:

```text
datasets -> preprocessing -> similarity / rules / alignment
         -> imputation / feature extraction / embeddings / models
```

## Choose a path

- **New users:** start with [Installation](installation.md) and
  [Getting started](getting-started.md); consult
  [Concepts and terminology](concepts.md) for key terms.
- **Using the command line:** see the [CLI guide](cli.md); use each
  command's built-in help for complete options.
- **Preparing data:** read [Datasets](datasets.md) and
  [Preprocessing](preprocessing.md).
- **Comparing dialects:** see [Similarity](similarity.md),
  [Rules](rules.md), and [Alignment](alignment.md).
- **Building features:** see [Feature extraction](feature-extraction.md),
  [Imputation](imputation.md), and [Decomposition](decomposition.md).
- **Training neural encoders:** see [Models](models.md).

## Chapters

| Chapter | Covers |
| --- | --- |
| [Concepts and terminology](concepts.md) | Phonetics and phonology, historical linguistics, dialectology, Chinese dialects, IPA, machine learning, and SinComp-specific terminology |
| [Command-line interface](cli.md) | Main commands, common workflows, and where to find detailed help |
| [Installation](installation.md) | Base installation, optional dependencies, and cache location |
| [Getting started](getting-started.md) | A reproducible end-to-end example |
| [Datasets](datasets.md) | Built-in and custom data sources, formats, and attribution |
| [Preprocessing](preprocessing.md) | Long-to-wide table conversion and multiple readings |
| [Similarity](similarity.md) | Chi-square and conditional-entropy dialect similarities |
| [Feature extraction](feature-extraction.md) | Phone similarity, dialect vectors, and character encodings |
| [Rules](rules.md) | Sound-change rule tables and dialect compliance scores |
| [Alignment](alignment.md) | Character-ID alignment across datasets |
| [Imputation](imputation.md) | Nearest-neighbor and factorization-based imputation |
| [Decomposition](decomposition.md) | SVD embeddings and masked matrix factorization |
| [Models](models.md) | TensorFlow processors and neural encoders |

## API reference

Most estimators expose scikit-learn-style `fit`, `transform`, and
`fit_transform` methods. Public API details are documented in the
docstrings of the linked modules:

- [Datasets](../src/sincomp/datasets.py)
- [Preprocessing](../src/sincomp/preprocessing.py)
- [Metrics](../src/sincomp/metrics.py)
- [Feature extraction](../src/sincomp/feature_extraction.py)
- [Imputation](../src/sincomp/impute.py)
- [Decomposition](../src/sincomp/decomposition.py)
- [Neural models](../src/sincomp/nn.py)
- [Character alignment](../src/sincomp/_internal/align.py), also exported
  as `sincomp.align` and `sincomp.align_no_cid`
