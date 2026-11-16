# SinComp

SinComp is a Python toolkit for processing and comparing Chinese dialect
pronunciation data. It provides dataset loaders, long-to-wide
preprocessing, dialect similarity measures, character alignment,
feature extraction, missing-value imputation, embeddings, and neural
models.

## Installation

SinComp requires Python 3.10 or later.

```shell
python -m pip install sincomp
```

Install optional dependencies for remote datasets or TensorFlow models
when needed:

```shell
python -m pip install "sincomp[datasets,models]"
```

See the [installation guide](docs/installation.md) for all extras and
cache configuration.

## Quick start

This example generates a small synthetic dataset, pivots it to a wide
table, and computes a pairwise dialect similarity matrix without
downloading external data:

```python
import sincomp.datasets
import sincomp.metrics
import sincomp.preprocessing

dataset = sincomp.datasets.make_dataset(
    initials=["p", "t", "k"],
    finals=["a", "i", "u"],
    tones=["1", "2", "3"],
    n_characters=27,
    n_dialects=5,
    n_iter=3,
    seed=0,
)

wide = sincomp.preprocessing.long_to_wide(
    dataset,
    index="cid",
    columns="did",
    values=["initial", "final", "tone"],
    aggfunc=lambda values: " ".join(values.dropna()),
    fill_value="",
)

similarities = sincomp.metrics.dialect_similarities(
    wide.to_numpy(dtype=object),
    feature_num=3,
)
print(similarities.shape)  # (5, 5)
```

For a real corpus, use `sincomp.datasets.get("MCPDict")`; accessing the
dataset's records may download and cache source files. See the
[getting-started guide](docs/getting-started.md) and
[dataset guide](docs/datasets.md).

## User guide

- [Guide index and learning paths](docs/index.md)
- [Command-line interface](docs/cli.md)
- [Concepts and terminology](docs/concepts.md)
- [Installation](docs/installation.md)
- [Getting started](docs/getting-started.md)
- [Datasets and data formats](docs/datasets.md)
- [Preprocessing](docs/preprocessing.md)
- [Dialect similarity](docs/similarity.md)
- [Feature extraction](docs/feature-extraction.md)
- [Sound-change rules](docs/rules.md)
- [Character alignment](docs/alignment.md)
- [Missing-value imputation](docs/imputation.md)
- [Embeddings and decomposition](docs/decomposition.md)
- [Neural models](docs/models.md)

Public API details are documented in the Python docstrings. For command
options and examples, use the built-in help described in the
[CLI guide](docs/cli.md). Plotting is not covered in this guide.
