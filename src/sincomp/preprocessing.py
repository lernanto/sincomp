# -*- coding: utf-8 -*-

"""Preprocess dialect pronunciation tables."""

__author__ = "黄艺华 <lernanto@foxmail.com>"


import collections.abc

import pandas


def long_to_wide(
    data,
    index: str = "did",
    columns: str = "cid",
    values: list[str] | None = None,
    aggfunc: str | collections.abc.Callable = "first",
    **kwargs,
) -> pandas.DataFrame:
    """Convert long-form pronunciation records into a wide table.

    Parameters
    ----------
    data : pandas.DataFrame or Dataset
        Long-form pronunciation records. A dataset object is accepted
        through its ``data`` attribute.
    index : str, default="did"
        Column used for the output row index.
    columns : str, default="cid"
        Column used for the output column groups.
    values : list of str or None, default=None
        Columns to pivot. If ``None``, retain all value columns.
    aggfunc : str or callable, default="first"
        Aggregation function for duplicate index-column combinations.
    **kwargs
        Additional keyword arguments passed to
        :meth:`pandas.DataFrame.pivot_table`.

    Returns
    -------
    pandas.DataFrame
        Wide-form table. For multiple value columns, feature names are
        placed in the second column-index level.
    """
    if not isinstance(data, pandas.DataFrame):
        data = data.data

    output = data.pivot_table(
        values,
        index=index,
        columns=columns,
        aggfunc=aggfunc,
        sort=False,
        **kwargs,
    )

    if output.columns.nlevels > 1:
        output = output.swaplevel(axis=1).reindex(
            pandas.MultiIndex.from_product(
                (output.columns.levels[1], output.columns.levels[0])
            ),
            axis=1,
        )

    return output
