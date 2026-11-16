# Dialect similarity

Use `sincomp.metrics.dialect_similarities()` to compare the
pronunciation-feature distributions of source and target dialects. It
returns a matrix with one row per source dialect and one column per target
dialect.

## Input layout

For a NumPy input, each row represents a character. Each dialect occupies
a contiguous block of `feature_num` columns (three by default for
initial, final, and tone). Cells contain whitespace-separated feature
values; empty strings represent missing values.

For example, the following array contains two dialect blocks:

```python
import numpy as np
from itertools import product

wide = np.array(
    [
        [initial, final, tone, initial, final, tone]
        for initial, final, tone in product(
            ("p", "t", "k"),
            ("a", "i", "u"),
            ("1", "2", "3"),
        )
    ],
    dtype=object,
)
```

## Choose a measure

```python
from sincomp.metrics import dialect_similarities

scores = dialect_similarities(wide, feature_num=3, metric="chi2")
print(scores.shape)  # (2, 2)
```

- `metric="chi2"` computes a normalized chi-square statistic. Larger
  values indicate greater predictive similarity.
- `metric="entropy"` computes conditional entropy. Smaller values mean
  that one dialect's feature values predict the other's more strongly.

When `dest` is omitted, the function compares `src` with itself. Pass a
second table as `dest` to compare two different sets of dialects. The
`feature_num` and `blocksize` arguments are forwarded to the underlying
measure. `n_jobs=None` runs sequentially; set `n_jobs` to a job count
supported by joblib to parallelize block calculations.

The chi-square measure is theoretically symmetric, although floating
point calculations may introduce small differences. Conditional entropy
is directional, so the order of source and target dialects matters.

## API

See [`sincomp.metrics`](../src/sincomp/metrics.py) for the function
docstring and its parameters.
