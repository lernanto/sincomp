# Sound-change rules

`DialectCompliance` measures how similarly each dialect realizes the two
character sets in a list of sound-change rules. It is useful when rules
are already specified and the goal is to compare dialects against those
rules.

## Rule table

Create a DataFrame with one rule per row and these required columns:

| Column | Meaning |
| --- | --- |
| `feature` | Feature to compare, such as `initial`, `final`, or `tone` |
| `cid1` | Character IDs in the first side of the rule |
| `cid2` | Character IDs in the second side of the rule |

The `cid1` and `cid2` cells should contain collections of character IDs.
The rule table's index is used as the output rule labels.

## Prepare the input table

The input is a wide DataFrame indexed by character ID, with a two-level
column index `(dialect, feature)`. `long_to_wide()` creates this layout from
long-form records, or it can be assembled directly:

```python
import pandas as pd

wide = pd.DataFrame(
    [["p", "p"], ["p", "t"], ["t", "t"], ["t", "p"]],
    index=["c1", "c2", "c3", "c4"],
    columns=pd.MultiIndex.from_product(
        [["dialect-a", "dialect-b"], ["initial"]]
    ),
)

from sincomp.feature_extraction import DialectCompliance

rules = pd.DataFrame(
    {
        "feature": ["initial"],
        "cid1": [["c1", "c2"]],
        "cid2": [["c3", "c4"]],
    },
    index=["rule-1"],
)
scores = DialectCompliance(rules).fit_transform(wide)
```

For long-form records, use
[`sincomp.preprocessing.long_to_wide()`](preprocessing.md) to create the
same column layout.

The result is a DataFrame with one row per dialect and one column per
rule. By default, each side's feature-value distribution is L2-normalized
before taking its inner product, which gives cosine similarity. Set
`norm=None` to skip normalization.

`fit()` validates the rule table and records the input shape;
`transform()` computes the scores. See
[`DialectCompliance`](../src/sincomp/feature_extraction.py) for the full
API.
