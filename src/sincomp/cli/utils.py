# -*- coding: utf-8 -*-

"""Load the sound-change rule tables for the CLI commands.

The CLI layer reads the rule files itself instead of calling the
implementation in :mod:`sincomp._internal.compare`, so the command
line interface does not depend on the private modules.
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"


import logging

import pandas


logger = logging.getLogger(__name__)


def load_rules(fname: str) -> pandas.DataFrame:
    """Load the sound-change rules from a JSON file.

    Parameters
    ----------
    fname : str
        Path of the rule file holding one rule per record with the
        ``feature``, ``cid1`` and ``cid2`` columns.

    Returns
    -------
    pandas.DataFrame
        Rule table indexed by the rule IDs; records without an ``id``
        field are indexed by their position.
    """

    rules = pandas.read_json(fname, orient="records", encoding="utf-8")
    rules["id"] = rules.get("id")
    rules["id"] = rules["id"].fillna(rules.index.to_series()).astype(str)
    rules.set_index("id", inplace=True)
    logger.debug("loaded %d rules from %s.", len(rules), fname)
    return rules
