# Embeddings and decomposition

`sincomp.decomposition` provides estimators for dense representations of
dialects and characters, as well as factorization of wide
dialect-by-feature tables. Choose the method that matches your input:

| Estimator | Input | Method |
| --- | --- | --- |
| `DialectEmbedding` | Sparse phone-similarity vectors, one row per dialect | Truncated SVD |
| `CharacterEmbedding` | Sparse character multi-hot vectors | Truncated SVD |
| `Factorizer` | Wide string table, one row per character | Masked alternating least squares |

## Dialect embeddings

`DialectEmbedding` drops features whose means are outside the configured
range, projects the remaining sparse matrix into an SVD subspace, and
rescales the training embeddings. Its effective dimension is limited by
the number of dialects and input features. It also provides
`inverse_transform()`, which reconstructs the feature vectors and fills
dropped features with their training means.

Phone-similarity vectors can be created with `PhoneSimilarity` or
`DialectVectorizer`. This example uses a small synthetic sparse matrix:

```python
from scipy.sparse import csr_matrix
from sincomp.decomposition import DialectEmbedding

phone_vectors = csr_matrix(
    [
        [1, 0, 1, 0, 1, 0],
        [0, 1, 1, 0, 0, 1],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 0, 1, 1, 1],
    ],
    dtype=float,
)
dialect_vectors = DialectEmbedding(embedding_size=2).fit_transform(
    phone_vectors
)
```

## Character embeddings

`CharacterEmbedding` applies truncated SVD to sparse multi-hot character
vectors. It can reduce the dimensionality of a
`CharacterVectorizer` output:

```python
import pandas as pd
from sincomp.decomposition import CharacterEmbedding
from sincomp.feature_extraction import CharacterVectorizer

wide = pd.DataFrame(
    {
        "d1_initial": ["p", "t", "k"],
        "d1_final": ["a", "i", "u"],
        "d2_initial": ["p", "k", "t"],
        "d2_final": ["i", "u", "a"],
    },
    index=["c1", "c2", "c3"],
)
character_vectors = CharacterVectorizer().fit_transform(wide)
character_embeddings = CharacterEmbedding(embedding_size=2).fit_transform(
    character_vectors
)
```

## Masked factorization

`Factorizer` decomposes a character-by-feature table into character and
feature-value embeddings using alternating least squares. Each cell can
contain several whitespace-separated values. Empty cells and `NaN` are
missing observations and do not contribute to the loss. For a DataFrame
with `(dialect, feature)` MultiIndex columns, each dialect forms a block
of features.

```python
import pandas as pd
from sincomp.decomposition import Factorizer

wide = pd.DataFrame(
    [["b ei 51", "p ei 24"], ["p ɔ 213", "p ɔ 53"]],
    columns=pd.MultiIndex.from_tuples(
        [("d1", "initial"), ("d1", "final")]
    ),
)

factorizer = Factorizer(
    n_components=2,
    max_iter=100,
    random_state=0,
)
character_embeddings = factorizer.fit_transform(wide)
```

`Factorizer` supports coverage-based staging, iteration limits,
regularization, and reproducible initialization. Unlike
`FactorizedImputer`, it produces embeddings rather than filling missing
cells.

See [`sincomp.decomposition`](../src/sincomp/decomposition.py) for
estimator parameters, attributes, and methods.
