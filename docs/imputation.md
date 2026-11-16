# Missing-value imputation

The imputers in `sincomp.impute` fill missing cells in wide
pronunciation-feature tables. Use an empty string or `pandas.NA` to mark a
missing value.

## Choose an imputer

### `KNeighborsImputer`

Use nearest-neighbor imputation when characters with similar observed
values are likely to share the missing value. For each column, the
imputer finds similar characters using that feature's values and lets the
nearest observed characters vote, weighted inversely by distance. With
MultiIndex columns, columns are grouped by their last level so distinct
features are imputed separately.

```python
import pandas as pd
from sincomp.impute import KNeighborsImputer

wide = pd.DataFrame(
    [["b", "ei", "51"], ["p", "ɔ", pd.NA]],
    columns=["d1_initial", "d1_final", "d1_tone"],
)
filled = KNeighborsImputer(n_neighbors=1).fit_transform(wide)
```

### `FactorizedImputer`

Use factorized imputation when latent relationships across feature
columns may help predict missing values. It multi-hot encodes the cells
and learns character and feature-value embeddings.

- `algorithm="als"` uses masked alternating least squares; missing cells
  do not contribute to the factorization loss.
- `algorithm="svd"` applies truncated SVD to the concatenated encoding;
  missing cells are represented as zeros.

The fitted embeddings are used to find nearby characters for each
feature. Columns with too little observed variation to predict a value
are left unchanged.

```python
from sincomp.impute import FactorizedImputer

wide = pd.DataFrame(
    [["b ei 51", "p ei 24"], ["p ɔ 213", pd.NA]],
    columns=["d1_initial", "d1_final"],
)
filled = FactorizedImputer(
    algorithm="svd",
    n_components=2,
    n_neighbors=1,
).fit_transform(wide)
```

Both estimators follow the scikit-learn transformer interface and return
an imputed table with the input type and shape. Fitting records input
metadata; the imputation is performed by `transform()`.

See [`sincomp.impute`](../src/sincomp/impute.py) for all parameters and
fitted attributes.
