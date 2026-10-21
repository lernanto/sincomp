# -*- coding: utf-8 -*-

"""Summarize the phonemes of a reconstruction result with their
characters.

Reads ``reconstruction.csv`` as written by the reconstruct command,
groups the characters of every feature by their phonemes and phones,
and writes one CSV table per feature, ``<feature>_phonemes.csv``,
with one row per phoneme: the phoneme, the number of characters it
governs, the distinct phones recorded for those characters (each
phone followed by its characters, ordered by character count in
decreasing order) and the characters themselves in the same order as
the phone list. Characters are ordered by their ``dialect_count``
from the reconstruction CSV, with ties keeping the order of their
appearance. The rows are ordered by optimal leaf seriation: the
profile of each phoneme is the concatenation of its co-occurrence
count vectors with the phonemes of the other two features, and the
phonemes are clustered with average linkage on cosine distance.

Usage::

    python scripts/extract_phonemes.py --reconstruction-file reconstruction.csv
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"


import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

FEATURES = ("initial", "final", "tone")


def collect_phonemes(
    rows: list[dict],
    feature: str,
    counts: dict[str, int],
) -> dict[str, dict[str, list[str]]]:
    """Group the characters of one feature by phoneme and phone.

    Characters without a phoneme or phone are skipped. Characters are
    kept without duplicates and ordered by ``dialect_count`` in
    decreasing order, ties keeping the order of their appearance.

    Parameters
    ----------
    rows : list[dict]
        The records of the reconstruction CSV file.
    feature : str
        The feature whose phonemes are collected.
    counts : dict[str, int]
        The number of dialects recording each character, read from the
        reconstruction CSV.

    Returns
    -------
    dict[str, dict[str, list[str]]]
        The characters of each phone of each phoneme, with phonemes
        and phones in the order of their first appearance.
    """

    phonemes = {}
    for record in rows:
        phoneme = (record.get(f"{feature}_phoneme") or "").strip()
        phone = (record.get(feature) or "").strip()
        character = (record.get("character") or "").strip()
        if not phoneme or not phone or not character:
            continue
        characters = phonemes.setdefault(phoneme, {}).setdefault(phone, [])
        if character not in characters:
            characters.append(character)
    for phones in phonemes.values():
        for characters in phones.values():
            # the stable sort keeps the appearance order of the ties
            characters.sort(key=lambda character: -counts.get(character, 0))
    return phonemes


def summarize_phonemes(
    phonemes: dict[str, dict[str, list[str]]],
) -> list[dict]:
    """Summarize the phonemes into sortable output rows.

    The phones of one phoneme are ordered by the number of their
    characters in decreasing order, ties broken by the phone itself,
    and the characters follow the order of their phones.

    Parameters
    ----------
    phonemes : dict[str, dict[str, list[str]]]
        The characters of each phone of each phoneme.

    Returns
    -------
    list[dict]
        One dict per phoneme with the ``phoneme``,
        ``character_count``, ``phones`` and ``characters``, sorted by
        the phoneme.
    """

    items = []
    for phoneme, phones in phonemes.items():
        ordered = sorted(
            phones.items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
        items.append({
            "phoneme": phoneme,
            "character_count": sum(
                len(characters) for _, characters in ordered
            ),
            "phones": ", ".join(phone for phone, _ in ordered),
            "characters": " ".join(
                "".join(characters) for _, characters in ordered
            ),
        })
    items.sort(key=lambda item: item["phoneme"])
    return items


def seriate(M: np.ndarray) -> list[int]:
    """Order matrix rows by similarity with hierarchical clustering.

    Average linkage on cosine distance between the row vectors.

    Parameters
    ----------
    M : numpy.ndarray of shape (n, m)
        The row vectors to order.

    Returns
    -------
    list[int]
        Row indices in leaf order.
    """

    M = M.astype(float)
    norms = np.linalg.norm(M, axis=1)
    norms[norms == 0] = 1.0
    X = M / norms[:, None]
    # Cosine distance = 1 - row dot products; an all-zero row keeps
    # distance 1 to every other row.
    D = 1.0 - np.clip(X @ X.T, 0.0, 1.0)
    np.fill_diagonal(D, 0.0)
    return list(leaves_list(linkage(squareform(D, checks=False), method="average")))


def feature_order(
    items: list[dict],
    rows: list[dict],
    feature: str,
) -> list[dict]:
    """Order the phoneme items of one feature by optimal leaf ordering.

    The profile of each phoneme is the concatenation of its
    co-occurrence count vectors with the phonemes of the other two
    features. Phonemes are clustered with average linkage on cosine
    distance and returned in leaf order, so phonemes with similar
    co-occurrence profiles are adjacent.

    Parameters
    ----------
    items : list[dict]
        The phoneme items as returned by ``summarize_phonemes``,
        sorted by the phoneme.
    rows : list[dict]
        The records of the reconstruction CSV file.
    feature : str
        The feature whose phonemes are ordered.

    Returns
    -------
    list[dict]
        The phoneme items in seriation order.
    """

    others = [f for f in FEATURES if f != feature]
    df = pd.DataFrame(rows)
    phonemes = [item["phoneme"] for item in items]
    blocks = []
    for other in others:
        pa = df[f"{feature}_phoneme"].fillna("").str.strip()
        pb = df[f"{other}_phoneme"].fillna("").str.strip()
        valid = (pa != "") & (pb != "")
        table = pd.pivot_table(
            df[valid].assign(**{feature: pa[valid], other: pb[valid]}),
            index=feature, columns=other, aggfunc="size", fill_value=0,
        )
        blocks.append(table.reindex(index=phonemes, fill_value=0))
    # Concatenated co-occurrence vectors; blocks may miss phonemes or
    # phoneme pairs absent from one of the two tables
    M = pd.concat(blocks, axis=1).reindex(index=phonemes).fillna(0).to_numpy()
    return [items[i] for i in seriate(M)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize the phonemes of a reconstruction "
                    "result with their characters, one CSV table per "
                    "feature."
    )
    parser.add_argument(
        "--reconstruction-file",
        default="reconstruction.csv",
        help="Reconstruction result file as written by the "
             "reconstruct command.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        type=Path,
        help="Directory the phoneme files are written to; defaults "
             "to the current directory.",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    reconstruction_file = Path(args.reconstruction_file)
    with open(
        reconstruction_file, encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise SystemExit(f"{reconstruction_file} holds no rows.")

    counts = {}
    for record in rows:
        character = (record.get("character") or "").strip()
        if not character:
            continue
        dialect_count = (record.get("dialect_count") or "").strip()
        if not dialect_count:
            raise SystemExit(
                f"{reconstruction_file} is missing dialect_count "
                f"for character {character!r}."
            )
        try:
            counts[character] = int(dialect_count)
        except ValueError as error:
            raise SystemExit(
                f"{reconstruction_file} has invalid dialect_count "
                f"{dialect_count!r} for character {character!r}."
            ) from error

    for feature in FEATURES:
        phonemes = collect_phonemes(rows, feature, counts)
        items = summarize_phonemes(phonemes)
        items = feature_order(items, rows, feature)
        output_path = output_dir / f"{feature}_phonemes.csv"
        with open(
            output_path, "w", encoding="utf-8", newline=""
        ) as output_file:
            writer = csv.writer(output_file)
            writer.writerow([
                "phoneme", "character_count", "phones",
                "characters",
            ])
            writer.writerows(
                [
                    item["phoneme"],
                    item["character_count"],
                    item["phones"],
                    item["characters"],
                ]
                for item in items
            )
        print(
            f"feature={feature} phonemes={len(items)} "
            f"written {output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
