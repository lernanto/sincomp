# Getting started

This walkthrough creates a small simulated dataset, pivots its long-form
records into a wide table, and computes pairwise similarities. It needs no
external data download.

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
print(similarities.shape)
```

The resulting matrix has one row and column per dialect, so its shape is
`(5, 5)`. The example combines repeated readings in a cell with spaces and
uses an object-dtype NumPy array for the string table.

## Use a built-in dataset

After installing the optional dataset dependencies, replace the simulated
dataset with a built-in dataset:

```python
dataset = sincomp.datasets.get("MCPDict")
data = dataset.data
```

Accessing `.data` may fetch and cache the dataset data. See
[Datasets](datasets.md) for aliases, caching, network requirements, and
dataset attributions.

## Next steps

- Use [Preprocessing](preprocessing.md) to choose row/column orientation
  and aggregation rules.
- Compare the measures in [Similarity](similarity.md).
- Learn the input conventions for estimators in
  [Feature extraction](feature-extraction.md) and
  [Decomposition](decomposition.md).
