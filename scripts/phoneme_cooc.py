# -*- coding: utf-8 -*-
"""Plot pairwise phoneme co-occurrence heatmaps for initials, finals, tones.

Input is a ``reconstruction.csv`` produced by the reconstruct module. For
each of the three pairs (initial x final, initial x tone, final x tone),
count the number of characters c sharing both phonemes and map it to a
co-occurrence score::

    cooc(c) = 1 - pow(0.5, c)

c=0 -> 0, c=1 -> 0.5, c=2 -> 0.75, approaching 1.

Phonemes on each axis are taken from the data itself (sorted unique
values); before plotting they are reordered by hierarchical clustering
seriation on the co-occurrence matrix so that phonemes with similar
co-occurrence profiles are adjacent. Three PNG files are written:
``initial_final_cooc.png``, ``initial_tone_cooc.png`` and
``final_tone_cooc.png``.

Usage::

    python phoneme_cooc.py --reconstruction-file reconstruction.csv --output-dir .
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)

PAIRS = [("initial", "final"),
         ("initial", "tone"),
         ("final", "tone")]


def seriate(M: np.ndarray) -> list[int]:
    """Order matrix rows by co-occurrence similarity (average-link
    hierarchical clustering on cosine distance).

    Parameters
    ----------
    M : numpy.ndarray of shape (n, m)
        Co-occurrence count matrix; rows are reordered.

    Returns
    -------
    list of int
        Row indices in seriation order.
    """

    M = M.astype(float)
    norms = np.linalg.norm(M, axis=1)
    norms[norms == 0] = 1.0
    X = M / norms[:, None]
    # Cosine distance = 1 - row dot products; an all-zero row (phoneme
    # co-occurring with nothing) keeps distance 1 to every other row.
    D = 1.0 - np.clip(X @ X.T, 0.0, 1.0)
    np.fill_diagonal(D, 0.0)
    return list(leaves_list(linkage(squareform(D, checks=False), method="average")))


def draw(table: pd.DataFrame, feats: tuple[str, str],
         out_path: str) -> None:
    """Render one co-occurrence heatmap and save it as a PNG.

    Parameters
    ----------
    table : pandas.DataFrame
        Co-occurrence count matrix; rows = feats[0], columns = feats[1].
    feats : tuple of str
        The two feature names.
    out_path : str
        Output PNG path.
    """

    C = table.to_numpy()
    heat = 1.0 - np.power(0.5, C.astype(float))
    n_r, n_c = C.shape
    ri = seriate(C)
    ci = seriate(C.T)
    heat = heat[ri][:, ci]
    rl = [table.index[i] for i in ri]
    cl = [table.columns[j] for j in ci]

    fig, ax = plt.subplots(figsize=(0.20 * n_c + 1.5, 0.20 * n_r + 1.3))
    sns.heatmap(heat, ax=ax, cmap="coolwarm", vmin=0.0, vmax=1.0,
                xticklabels=cl, yticklabels=rl, square=True,
                cbar_kws={"shrink": 0.85, "label": "Co-occurrence"},
                linewidths=0.3, linecolor="white")
    ax.tick_params(axis="x", rotation=90, labelsize=10)
    ax.tick_params(axis="y", rotation=0, labelsize=10)
    ax.set_xlabel(feats[1])
    ax.set_ylabel(feats[0])
    ax.set_title(f"{feats[0]} x {feats[1]} co-occurrence")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    logger.info(f"written {out_path}  ({n_r}x{n_c})")


def main() -> None:
    """Parse command-line arguments and write the three heatmaps."""

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--reconstruction-file", default="reconstruction.csv",
                    help="path to the reconstruction.csv input file")
    ap.add_argument("--output-dir", default=".",
                    help="output directory, defaults to the current directory")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(message)s")

    sns.set_theme()  # default style; English labels need no CJK font
    # DejaVu Sans covers IPA symbols and subscript digits missing from Arial
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    os.makedirs(args.output_dir, exist_ok=True)

    # keep_default_na=False keeps empty cells as "" instead of NaN
    df = pd.read_csv(args.reconstruction_file, dtype=str,
                     keep_default_na=False)

    for fa, fb in PAIRS:
        pa = df[f"{fa}_phoneme"].str.strip()
        pb = df[f"{fb}_phoneme"].str.strip()
        valid = (pa != "") & (pb != "")
        table = pd.pivot_table(df[valid], index=pa[valid], columns=pb[valid],
                               aggfunc="size", fill_value=0)
        out = os.path.join(args.output_dir, f"{fa}_{fb}_cooc.png")
        draw(table, (fa, fb), out)


if __name__ == "__main__":
    main()
