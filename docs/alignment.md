# Character alignment

Datasets may assign different character IDs to the same written
character, and may represent polyphonic characters differently.
`sincomp.align` aligns character IDs across datasets; `sincomp.align_no_cid`
annotates data that has no character IDs against a base dataset.

## Align datasets with character IDs

Call `align()` with one `(data, chars)` pair per dataset. Each `data`
table is long form and contains `did`, `cid`, `initial`, `final`, and
`tone`. If it also contains a `character` column, pass `None` for `chars`;
otherwise pass a Series mapping `cid` to character text. Repeated
characters in that mapping represent polyphonic IDs.

```python
from sincomp import align

aligned_characters, character_maps = align(
    (data_a, chars_a),
    (data_b, chars_b),
    encoder="svd",
    embedding_size=32,
)
```

The returned character table assigns a common row index to aligned
characters and retains the source character IDs. `character_maps` contains
one Series per input dataset, mapping its original IDs to the new IDs.
Only same-spelling polyphonic characters from different datasets are
eligible to be clustered together.

`encoder="svd"` uses SVD-based character embeddings. `encoder="fm"` uses
factorization with missing values. `distance_threshold` controls
clustering: a larger threshold allows more candidate polyphones to be
merged. Review the aligned mappings before using them as definitive
annotations.

## Align a dataset without character IDs

Use `align_no_cid()` when a base dataset has character IDs and a target
dataset has character text but no IDs:

```python
from sincomp import align_no_cid

target_dialects = [target_dialect_a, target_dialect_b]
annotations = align_no_cid(
    base_data,
    simplified_characters,
    traditional_characters,
    target_dialects,
)
```

`base_data` is a long-form table with `did`, `cid`, `initial`, `final`,
and `tone`. `simplified_characters` and `traditional_characters` are Series indexed by
the base character IDs; pass `None` for traditional forms when they are
unavailable. Each additional positional argument is a sequence of
per-dialect DataFrames without `cid`; each DataFrame must contain
`character`, `did`, `initial`, `final`, and `tone`. The result is grouped
first by target dataset and then by dialect; each dialect entry contains
assigned IDs and the corresponding character forms. Unmatched
characters receive label `-1`.

## API

Both functions are exported from the package root. Their implementation
and parameter details are documented in
[`sincomp._internal.align`](../src/sincomp/_internal/align.py).
