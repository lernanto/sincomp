# Preprocessing

`sincomp.preprocessing.long_to_wide()` pivots long-form pronunciation
records into a wide table for estimators and analysis. Use it when you
need one character per row, one dialect per row, or grouped dialect
features in columns.

## Long and wide forms

A long table typically has one row per sample, usually a
dialect-character record, and columns such as `did`, `cid`,
`character`, `initial`, `final`, and `tone`. A dialect-character pair
may have multiple samples, for example when it has multiple readings.
A wide table uses the selected `index` column for rows and the selected
`columns` column for column groups. When `values` contains multiple
fields, the output columns have levels for the selected group and
feature name.

For example, begin with a small long-form table:

```python
import pandas as pd

long = pd.DataFrame(
    [
        ("d1", "c1", "甲", "p", "a", "1"),
        ("d1", "c1", "甲", "b", "a", "1"),
        ("d1", "c2", "乙", "t", "i", "2"),
        ("d2", "c1", "甲", "b", "a", "1"),
        ("d2", "c2", "乙", "d", "u", "2"),
    ],
    columns=["did", "cid", "character", "initial", "final", "tone"],
)
```

## Choose the orientation

To create one row per character and columns grouped by dialect:

```python
from sincomp.preprocessing import long_to_wide

wide_by_character = long_to_wide(
    long,
    index="cid",
    columns="did",
    values=["initial", "final", "tone"],
    aggfunc="first",
    fill_value="",
)
```

To create one row per dialect and columns grouped by character, swap
`index` and `columns`:

```python
wide_by_dialect = long_to_wide(
    long,
    index="did",
    columns="cid",
    values=["initial", "final", "tone"],
    aggfunc="first",
    fill_value="",
)
```

## Multiple readings and missing values

The default `aggfunc="first"` keeps the first value when more than one
record maps to the same output cell. To preserve multiple readings,
aggregate the non-missing values into a whitespace-separated string:

```python
wide = long_to_wide(
    long,
    index="cid",
    columns="did",
    values=["initial", "final", "tone"],
    aggfunc=lambda values: " ".join(values.dropna()),
    fill_value="",
)
```

The output can then be consumed by estimators that tokenize whitespace-
separated feature values. Use an empty string for missing cells when the
next step expects string input. `long_to_wide()` also accepts the keyword
arguments of `pandas.DataFrame.pivot_table()`, including `fill_value`.

## Parameters

- `data`: a long-form DataFrame or a SinComp dataset object.
- `index`: column in the input that becomes the output row index.
- `columns`: column whose values define output column groups.
- `values`: input columns to pivot; `None` retains all value columns.
- `aggfunc`: aggregation for duplicate row/column combinations; defaults
  to `"first"`.
- `**kwargs`: passed to `pandas.DataFrame.pivot_table()`.

See the [`sincomp.preprocessing` API](../src/sincomp/preprocessing.py)
and the [datasets chapter](datasets.md) for table conventions.
