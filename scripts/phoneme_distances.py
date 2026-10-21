# -*- coding: utf-8 -*-

"""Measure the cosine distance between two phonemes per dialect.

Reads ``predictions.csv`` as written by the reconstruct command. For
one feature and a pair of its reconstructed phonemes, the profile of a
phoneme in one dialect is the distribution of the recorded phone values
over the characters the phoneme governs. The cosine distance between
the two phoneme profiles is computed for every dialect and written to
the standard output as a table with the columns ``did``, ``name`` and
``distance``, one row per dialect, followed by the ``mean`` and ``std``
rows over the dialects. A dialect where either phoneme is absent holds
no distance and is skipped by the summary rows.

Usage::

    python scripts/phoneme_distances.py \
        --predictions-file predictions.csv \
        --feature initial --phonemes p pʰ
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"


import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FEATURES = ("initial", "final", "tone")


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float | None:
    """Compute the cosine distance between two vectors.

    Parameters
    ----------
    a, b : numpy.ndarray
        The vectors to compare; either may be all zeros.

    Returns
    -------
    float | None
        ``1 - cos(a, b)``, or ``None`` when either vector is all
        zeros and the cosine is undefined.
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0.0 or norm_b == 0.0:
        return None
    return 1.0 - float(a @ b) / (norm_a * norm_b)


def phoneme_profile(
    dialect: pd.DataFrame,
    feature: str,
    phoneme: str,
    values: list[str],
) -> np.ndarray:
    """Build the phone value distribution of one phoneme.

    Parameters
    ----------
    dialect : pandas.DataFrame
        The prediction rows of one dialect, holding the recorded
        phone column ``feature`` and the reconstructed phoneme column
        ``reconstructed_<feature>``.
    feature : str
        The feature the phoneme belongs to.
    phoneme : str
        The reconstructed phoneme.
    values : list[str]
        The phone values the distribution is taken over.

    Returns
    -------
    numpy.ndarray
        The character counts of each phone value, in the order of
        ``values``.
    """
    recorded = dialect.loc[
        dialect[f"reconstructed_{feature}"] == phoneme, feature
    ]
    counts = recorded.value_counts()
    return counts.reindex(values, fill_value=0).to_numpy(dtype=float)


def pair_distances(
    rows: pd.DataFrame,
    feature: str,
    phonemes: tuple[str, str],
) -> list[tuple[str, str, float | None]]:
    """Compute the phoneme pair distance for every dialect.

    Parameters
    ----------
    rows : pandas.DataFrame
        The prediction rows, holding the ``did``, ``name``, the
        recorded phone column ``feature`` and the reconstructed
        phoneme column ``reconstructed_<feature>``.
    feature : str
        The feature the phonemes belong to.
    phonemes : tuple[str, str]
        The two reconstructed phonemes to compare.

    Returns
    -------
    list[tuple[str, str, float | None]]
        One ``(did, name, distance)`` tuple per dialect, in the order
        of their first appearance; the distance is ``None`` when
        either phoneme is absent in the dialect.
    """
    result = []
    for (did, name), dialect in rows.groupby(["did", "name"], sort=False):
        values = sorted(set(dialect[feature]) - {""})
        a = phoneme_profile(dialect, feature, phonemes[0], values)
        b = phoneme_profile(dialect, feature, phonemes[1], values)
        result.append((did, name, cosine_distance(a, b)))
    return result


def format_table(
    records: list[tuple[str, str, float | None]],
) -> str:
    """Format the distance records as an aligned table.

    Parameters
    ----------
    records : list[tuple[str, str, float | None]]
        One ``(did, name, distance)`` tuple per dialect, followed by
        the ``mean`` and ``std`` rows; the distance is ``None`` when
        undefined.

    Returns
    -------
    str
        The table with the columns ``did``, ``name`` and ``distance``.
    """
    cells = [
        (did, name, "-" if d is None else f"{d:.4f}")
        for did, name, d in records
    ]
    width_did = max(len("did"), *(len(did) for did, _, _ in cells))
    width_name = max(len("name"), *(len(name) for _, name, _ in cells))
    width_distance = max(
        len("distance"), *(len(distance) for _, _, distance in cells)
    )
    lines = [
        f"{'did':<{width_did}}  {'name':<{width_name}}  "
        f"{'distance':>{width_distance}}",
        *(
            f"{did:<{width_did}}  {name:<{width_name}}  "
            f"{distance:>{width_distance}}"
            for did, name, distance in cells
        ),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Parse command-line arguments and print the distance table."""

    parser = argparse.ArgumentParser(
        description="Measure the cosine distance between two "
                    "phonemes of one feature per dialect."
    )
    parser.add_argument(
        "--predictions-file",
        default="predictions.csv",
        help="Predictions CSV file holding one record per dialect "
             "and character.",
    )
    parser.add_argument(
        "--feature",
        required=True,
        choices=FEATURES,
        help="The feature the phonemes belong to.",
    )
    parser.add_argument(
        "--phonemes",
        required=True,
        nargs=2,
        metavar="PHONEME",
        help="The two reconstructed phonemes to compare.",
    )
    args = parser.parse_args(argv)

    predictions_file = Path(args.predictions_file)
    # keep_default_na=False keeps empty cells as "" instead of NaN
    rows = pd.read_csv(
        predictions_file, dtype=str, keep_default_na=False,
        usecols=["did", "name", args.feature,
                 f"reconstructed_{args.feature}"],
    )
    if rows.empty:
        raise SystemExit(f"{predictions_file} holds no rows.")

    records = pair_distances(rows, args.feature, tuple(args.phonemes))
    # mean and std over the dialects where the distance is defined
    distances = [d for _, _, d in records if d is not None]
    if distances:
        series = pd.Series(distances)
        records.append(("mean", "", float(series.mean())))
        records.append(("std", "", float(series.std())))
    print(format_table(records))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
