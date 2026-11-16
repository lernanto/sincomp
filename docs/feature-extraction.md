# Feature extraction

The estimators in `sincomp.feature_extraction` turn string-valued
pronunciation tables into sparse numeric representations. Choose an
estimator based on the entity in each input row:

| Estimator | Row represents | Result |
| --- | --- | --- |
| `PhoneSimilarity` | A dialect | Sparse flattened pairwise similarity features |
| `DialectVectorizer` | A dialect | Concatenated similarity features for selected feature groups |
| `CharacterVectorizer` | A character | Sparse multi-hot encoding of its phone features |

Cells may contain whitespace-separated feature values.

## Similarity features for dialects

`PhoneSimilarity` computes phone similarity vectors for the rows of an
input table. `DialectVectorizer` applies it to selected groups of columns
and concatenates the results. String selectors match column names by
literal substring; sequences and slices select columns by position.

```python
import pandas as pd
from sincomp.feature_extraction import DialectVectorizer

wide_by_dialect = pd.DataFrame(
    {
        "initial_c1": ["p", "t"],
        "initial_c2": ["t", "k"],
        "final_c1": ["a", "a"],
        "final_c2": ["i", "u"],
        "tone_c1": ["1", "2"],
        "tone_c2": ["2", "1"],
    },
    index=["dialect-a", "dialect-b"],
)

dialect_vectors = DialectVectorizer().fit_transform(wide_by_dialect)
```

With the default selectors, the columns are grouped by the substrings
`initial`, `final`, and `tone`. The output is a CSR sparse matrix with
one row per input dialect.

Use `PhoneSimilarity` directly when a single table of phone features is
already selected:

```python
from sincomp.feature_extraction import PhoneSimilarity

initials = wide_by_dialect[["initial_c1", "initial_c2"]].to_numpy(dtype=object)
initial_similarity = PhoneSimilarity().fit_transform(initials)
```

## Multi-hot character features

`CharacterVectorizer` expects one row per character and one column per
dialect feature. Each column is tokenized independently; observed tokens
become binary features.

```python
import pandas as pd
from sincomp.feature_extraction import CharacterVectorizer

wide_by_character = pd.DataFrame(
    {
        "d1_initial": ["p t", "t"],
        "d1_final": ["a", "i"],
        "d2_initial": ["p", "k"],
        "d2_final": ["a i", "u"],
    },
    index=["c1", "c2"],
)

character_vectors = CharacterVectorizer().fit_transform(wide_by_character)
```

The output is a sparse multi-hot matrix. Use it as input to
`CharacterEmbedding` when a dense low-dimensional representation is
needed; see [Decomposition](decomposition.md).

## Rule compliance

`DialectCompliance` scores dialects against a table of sound-change
rules. Its input is a wide DataFrame indexed by character IDs with
`(dialect, feature)` MultiIndex columns. Rule rows require `feature`,
`cid1`, and `cid2`; the two `cid` columns hold collections of character
IDs for the rule's two sides. With the default L2 normalization, scores
are cosine similarities in `[0, 1]`.

The rule workflow and a complete example are in [Rules](rules.md).

## API

See [`sincomp.feature_extraction`](../src/sincomp/feature_extraction.py)
for estimator parameters, fitted attributes, and method documentation.
