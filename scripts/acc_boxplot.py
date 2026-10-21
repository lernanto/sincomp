# -*- coding: utf-8 -*-
"""Plot a boxplot of initial/final/tone prediction accuracy (seaborn).

Input is a CSV matching the ``phone prediction accuracy:`` table reported
by the evaluate module, one dialect per row:

    did,initial,final,tone
    dialect1,0.961,0.879,0.919
    ...

The ``did`` column is required; a header row is skipped and a trailing
``mean`` row is dropped. One boxplot PNG is written with one box each
for initial, final and tone, in percent.

Usage::

    python acc_boxplot.py --accuracy-file accuracy.csv --output-file acc_boxplot.png
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"

import argparse
import logging
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

FEATS = ["initial", "final", "tone"]


def load(acc_file: str) -> pd.DataFrame:
    """Load per-dialect accuracy rows from the evaluate output CSV.

    Parameters
    ----------
    acc_file : str
        Path to the accuracy CSV.

    Returns
    -------
    pandas.DataFrame
        Columns ``did`` plus one column per feature, values as float.
    """

    df = pd.read_csv(acc_file, dtype=str, keep_default_na=False)
    # Drop header and the trailing mean row, then convert to float
    df = df[~df.iloc[:, 0].isin(["mean", "std"])]
    vals = df.iloc[:, 1:1 + len(FEATS)].astype(float).reset_index(drop=True)
    vals.columns = FEATS
    vals.insert(0, "did", df.iloc[:, 0].reset_index(drop=True))
    return vals


def main() -> None:
    """Parse command-line arguments and write the boxplot PNG."""

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--accuracy-file", default="accuracy.csv",
                    help="path to the per-dialect accuracy CSV input file")
    ap.add_argument("--output-file", default="acc_boxplot.png",
                    help="path of the output boxplot PNG")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(message)s")

    df = load(args.accuracy_file)
    if df.empty:
        raise SystemExit(f"no data rows in {args.accuracy_file}")

    # Long form: one row per (dialect, feature) with accuracy in percent
    long = df.melt(id_vars="did", value_vars=FEATS,
                   var_name="feature", value_name="accuracy")
    long["accuracy"] *= 100

    sns.set_theme()  # default style; English labels need no CJK font
    # DejaVu Sans covers IPA symbols and subscript digits missing from Arial
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    sns.boxplot(data=long, x="feature", y="accuracy", ax=ax,
                order=FEATS, width=0.55)
    ax.set_xlabel("")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Prediction accuracy")
    lo = long["accuracy"].min()
    ax.set_ylim(max(0.0, np.floor((lo - 2) / 5) * 5), 100.5)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_file)),
                exist_ok=True)
    fig.savefig(args.output_file, dpi=200)
    logger.info(f"written {args.output_file} ({len(df)} dialects)")


if __name__ == "__main__":
    main()
