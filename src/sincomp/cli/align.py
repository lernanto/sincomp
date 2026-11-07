"""Command-line interface for dialect data alignment.

This module implements the ``align`` subcommand family of the ``sincomp``
command. It provides two commands:

- ``align``: align the characters of several dialect datasets into a
  common character ID space, producing a merged dataset written as a
  directory of per-dialect CSV files together with the character mapping
  table, the character information and the dialect information.
- ``evaluate``: measure the accuracy of the polyphone alignment
  algorithm on a single dataset by splitting its dialects into two
  disjoint halves, aligning them as if their correspondence were
  unknown, and comparing the recovered character labels with the true
  shared character IDs.

The alignment algorithms themselves live in
:mod:`sincomp._internal.align`; this module only handles argument
parsing, orchestration and output formatting.
"""

import argparse
import json
import logging
import os
from typing import Sequence

import numpy
import pandas
import sklearn.model_selection

from .. import datasets
from .._internal.align import align, align_no_cid


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler())


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    """Add the options shared by all alignment subcommands.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        The subparser to add the common options to.
    """

    parser.add_argument(
        "-l", "--log-level", default="WARNING", help="logging level"
    )
    parser.add_argument(
        "-n",
        "--embedding-size",
        type=int,
        default=32,
        help="size of the character embeddings used to align polyphones",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the alignment command-line parser.

    Returns
    -------
    parser : argparse.ArgumentParser
        The argument parser of the ``align`` command family, with the
        ``align`` and ``evaluate`` subcommands registered.
    """

    parser = argparse.ArgumentParser(description="align dialect datasets")
    commands = parser.add_subparsers(dest="command", required=True)

    align_parser = commands.add_parser("align", help="align the given datasets")
    _add_common_options(align_parser)
    align_parser.add_argument(
        "--prefix",
        default="aligned",
        help="output path prefix of the aligned dataset",
    )
    align_parser.add_argument(
        "--charmap-output",
        default="charmap.csv",
        help="output file of the old-to-new character ID mapping table",
    )
    align_parser.add_argument(
        "--character-output",
        default=os.path.join("aligned", ".characters"),
        help="output file mapping the new character IDs to the original "
             "character IDs of each dataset",
    )
    align_parser.add_argument(
        "--dialect-output",
        default=os.path.join("aligned", ".dialects"),
        help="output file of the merged dialect information",
    )
    align_parser.add_argument("datasets", nargs="+", help="datasets to align")

    evaluate_parser = commands.add_parser(
        "evaluate",
        help="evaluate polyphone alignment accuracy",
    )
    _add_common_options(evaluate_parser)
    evaluate_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="json",
        help="output format of the result",
    )
    evaluate_parser.add_argument(
        "datasets", nargs=1, help="dataset used for evaluation"
    )

    return parser


def align_cmd(args: argparse.Namespace) -> None:
    """Align the polyphones of the given datasets.

    The datasets are loaded through :func:`sincomp.datasets.get`. Those
    providing character IDs are aligned with :func:`align`, which yields
    the unified character table and the old-to-new character ID mapping;
    those without character IDs are annotated against the aligned result
    with :func:`align_no_cid`. All valid readings are then remapped to
    the new dialect and character IDs and written out:

    - the character mapping table to ``args.charmap_output``
    - the unified character information to ``args.character_output``
    - the merged dialect information to ``args.dialect_output``
    - the aligned per-dialect pronunciation tables under
      ``<args.prefix>/<dataset name>/<new dialect ID>``

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments with the attributes ``datasets``
        (names or paths of the datasets to align), ``embedding_size``,
        ``prefix``, ``charmap_output``, ``character_output`` and
        ``dialect_output``.

    Returns
    -------
    None
        Results are written to the files given by ``args``.
    """

    names = []
    dialects = []
    withcid = []
    nocid = []
    for name in args.datasets:
        data = datasets.get(name)
        if data is None:
            continue

        names.append(data.name)
        dialects.append(data.dialects)

        # separate datasets providing character IDs from those without
        if "cid" in data.select(data.dialect_ids[:1]).columns:
            withcid.append((data, data.characters["character"]))
        else:
            nocid.append(data)

    logger.info(
        f"align datasets {', '.join(names)}, "
        f"embedding size = {args.embedding_size}..."
    )

    # assign new dialect IDs and build the (dataset, old did) -> new did map
    dialects = pandas.concat(dialects, axis=0, keys=names) \
        .reset_index(names=["dataset", "old_did"])
    dialects.set_index(dialects.index.astype(str).rename("did"), inplace=True)
    dialect_map = pandas.Series(
        dialects.index,
        index=pandas.MultiIndex.from_frame(dialects[["dataset", "old_did"]])
    )

    # align the polyphones and build the old-to-new character ID mapping
    names = [d.name for d, _ in withcid]
    logger.info(f"align datasets {', '.join(names)}...")
    chars, charmaps = align(*withcid, embedding_size=args.embedding_size)

    chars = chars.set_index(chars.index.astype(str).rename("cid")) \
        .rename(columns=dict(enumerate(names)))
    # merge the per-dataset mappings into a single mapping table
    charmap = pandas.concat(charmaps, axis=0, keys=names) \
        .astype(str) \
        .rename_axis(["dataset", "old_cid"]) \
        .rename("cid")

    logger.info(
        f"annotate datasets without character ID: "
        f"{', '.join([d.name for d in nocid])} ..."
    )

    # merge the aligned datasets under the new dialect and character IDs
    # into a single base dataset
    base = pandas.concat(
        [d.loc[:, ["initial", "final", "tone"]].assign(
            did=dialect_map[n][d.loc[:, "did"]].values,
            cid=charmap[n].reindex(d.loc[:, "cid"]).values
        ) for (d, _), n in zip(withcid, names)],
        axis=0,
        ignore_index=True
    ).dropna(subset=["did", "cid", "initial", "final", "tone"])
    label_list = align_no_cid(
        base,
        chars["simplified"],
        chars["traditional"] if chars["traditional"].notna().all() else None,
        *nocid
    )

    # additionally encode characters unique to the datasets without IDs
    unknown = []
    for labels in label_list:
        for lab, s, t in labels:
            unknown.append(
                pandas.DataFrame(
                    {"label": lab, "simplified": s, "traditional": t}
                )
            )

    if unknown:
        unknown = pandas.concat(unknown, axis=0, ignore_index=True)
        unknown = unknown.loc[
            unknown["label"].isna(),
            ["simplified", "traditional"]
        ] \
            .dropna() \
            .sort_values(["simplified", "traditional"]) \
            .drop_duplicates()
        unknown.set_index(
            pandas.Index(
                range(chars.shape[0], chars.shape[0] + unknown.shape[0]),
                dtype=str,
                name="cid"
            ),
            inplace=True
        )
        chars = pandas.concat([chars, unknown], axis=0)

        # assign the generated character IDs to the previously unlabeled chars
        unknown_map = unknown.reset_index() \
            .set_index(["simplified", "traditional"])["cid"]
        for labels in label_list:
            for lab, s, t in labels:
                unlabeled = lab == None  # noqa: E711
                lab[unlabeled] = unknown_map.reindex(
                    pandas.MultiIndex.from_arrays([s[unlabeled], t[unlabeled]])
                )

    path = os.path.abspath(args.charmap_output)
    logger.info(f"save {charmap.shape[0]} character mapping to {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    charmap.to_csv(path, encoding="utf-8", lineterminator="\n")

    # prefer the traditional glyph and fall back to the simplified one
    chars["character"] = chars["traditional"].where(
        chars["traditional"].notna(),
        chars["simplified"]
    )
    path = os.path.abspath(args.character_output)
    logger.info(f"save {chars.shape[0]} character information to {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    chars.to_csv(path, encoding="utf-8", lineterminator="\n")

    path = os.path.abspath(args.dialect_output)
    logger.info(f"save {dialects.shape[0]} dialect information to {path}...")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dialects.to_csv(path, encoding="utf-8", lineterminator="\n")

    # remap all valid readings of the datasets with character IDs to the
    # new dialect and character IDs, and save them per dialect
    for (dataset, _), m in zip(withcid, charmaps):
        for did, data in dataset.items():
            did = dialect_map[(dataset.name, did)]
            data = data.assign(did=did, cid=data["cid"].map(m)) \
                .reindex([
                    "did",
                    "cid",
                    "initial",
                    "final",
                    "tone",
                    "tone_category",
                    "note"
                ], axis=1) \
                .dropna(subset=["did", "cid"])

            if data.shape[0] > 0:
                data["cid"] = data["cid"].astype(int)

                dirname = os.path.abspath(
                    os.path.join(args.prefix, dataset.name)
                )
                path = os.path.join(dirname, str(did))
                os.makedirs(dirname, exist_ok=True)
                logger.info(f"save {data.shape[0]} aligned data to {path}...")
                data.to_csv(
                    path, index=False, encoding="utf-8", lineterminator="\n"
                )

    # assign character IDs to the datasets without character IDs and save
    for dataset, labels in zip(nocid, label_list):
        for (did, data), (cid, _, _) in zip(dataset.items(), labels):
            did = dialect_map[(dataset.name, did)]
            data = data.assign(did=did, cid=cid).reindex([
                "did",
                "cid",
                "initial",
                "final",
                "tone",
                "tone_category",
                "note"
            ], axis=1).dropna(subset=["did", "cid"])

            if data.shape[0] > 0:
                data["cid"] = data["cid"].astype(int)

                dirname = os.path.abspath(
                    os.path.join(args.prefix, dataset.name)
                )
                path = os.path.join(dirname, str(did))
                os.makedirs(dirname, exist_ok=True)
                logger.info(f"save {data.shape[0]} aligned data to {path}...")
                data.to_csv(
                    path, index=False, encoding="utf-8", lineterminator="\n"
                )


def evaluate_cmd(args: argparse.Namespace) -> dict[str, float] | None:
    """Evaluate the polyphone alignment accuracy on a single dataset.

    The dataset is split into two halves of dialects five times with a
    fixed random seed. In each round the character correspondence between
    the halves is assumed unknown: the first half evaluates the
    :func:`align` algorithm, where both halves provide character IDs, and
    the second half evaluates :func:`align_no_cid`, where only the first
    half does. The recovered character labels are compared with the true
    character IDs of the polyphones, and the mean and standard deviation
    of the accuracies over the five rounds are reported.

    Parameters
    ----------
    args : argparse.Namespace
        Parsed command-line arguments with the attributes ``datasets``
        (a single-element sequence naming the dataset to evaluate) and
        ``embedding_size``.

    Returns
    -------
    result : dict of str to float or None
        Dict with the keys ``cid_accuracy``, ``cid_std``,
        ``no_cid_accuracy`` and ``no_cid_std``, holding the mean accuracy
        and standard deviation of both evaluation settings; None if the
        dataset cannot be loaded.
    """

    if (dataset := datasets.get(args.datasets[0])) is None:
        return

    logger.info(
        f"evaluating alignment accuracy of polyphones for {dataset.name} ..."
    )

    rs = numpy.random.RandomState(30918341)
    chars = dataset.characters["character"]
    index = chars[chars.duplicated(False)].index
    accuracies = []

    for _ in range(5):
        # split the dataset into two halves of dialects
        dids1, dids2 = sklearn.model_selection.train_test_split(
            dataset.dialect_ids,
            test_size=0.5,
            random_state=rs
        )
        data1, data2 = dataset.select(dids1), dataset.select(dids2)

        chars1 = chars.reindex(data1.loc[:, "cid"].dropna().unique()).dropna()
        chars2 = chars.reindex(data2.loc[:, "cid"].dropna().unique()).dropna()
        _, (chars1, chars2) = align(
            (data1, chars1),
            (data2, chars2),
            embedding_size=args.embedding_size
        )

        # measure the alignment accuracy against the true character IDs
        label1 = chars1.reindex(index, fill_value=-1)
        label2 = chars2.reindex(index, fill_value=-1)
        acc = ((label1 >= 0) & (label2 >= 0) & (label1 == label2))

        if not acc.all():
            for i, r in dataset.characters.loc[index[~acc.values]] \
                    .assign(label1=label1, label2=label2) \
                    .sort_values("character") \
                    .iterrows():
                logger.info(
                    f"bad case: {i}: {r['character']}, "
                    f"label1 = {r['label1']}, label2 = {r['label2']}"
                )

        acc = acc.astype(int)
        logger.info(
            f"{dataset.name}: accuracy = "
            f"{acc.mean()}({acc.sum()}/{acc.count()})"
        )
        accuracies.append(acc.mean())

    cid_accuracy = numpy.mean(accuracies)
    cid_deviation = numpy.std(accuracies)

    accuracies = []

    for _ in range(5):
        # split the dataset into two halves of dialects
        dids1, dids2 = sklearn.model_selection.train_test_split(
            dataset.dialect_ids,
            test_size=0.5,
            random_state=rs
        )
        data1, data2 = dataset.select(dids1), dataset.select(dids2)

        chars1 = chars.reindex(data1.loc[:, "cid"].dropna().unique()).dropna()
        results, = align_no_cid(
            data1,
            chars1,
            None,
            data2,
            embedding_size=args.embedding_size
        )
        labels = numpy.concatenate([lab for lab, _, _ in results], axis=0)
        data2 = data2.assign(label=labels)[data2.loc[:, "cid"].isin(index)]
        acc = data2["cid"] == data2["label"]

        if not acc.all():
            bad = data2[~acc].sort_values(["did", "character"])
            for _, r in bad.iterrows():
                logger.info(
                    f"bad case: {r['did']} {r['character']}, "
                    f"cid = {r['cid']}, label = {r['label']}"
                )

        acc = acc.astype(int)
        logger.info(
            f"{dataset.name}: accuracy = "
            f"{acc.mean()}({acc.sum()}/{acc.count()})"
        )
        accuracies.append(acc.mean())

    no_cid_accuracy = numpy.mean(accuracies)
    no_cid_deviation = numpy.std(accuracies)
    return {
        "cid_accuracy": cid_accuracy,
        "cid_std": cid_deviation,
        "no_cid_accuracy": no_cid_accuracy,
        "no_cid_std": no_cid_deviation,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the alignment command-line interface.

    Parameters
    ----------
    argv : sequence of str or None, default=None
        Command-line arguments to parse; the system arguments are used
        when None.

    Returns
    -------
    exit_code : int
        The process exit code, 0 on success.
    """

    args = build_parser().parse_args(argv)
    logger.setLevel(getattr(logging, args.log_level.upper()))

    if args.command == "align":
        align_cmd(args)
    else:
        result = evaluate_cmd(args)
        if result is not None:
            if args.format == "json":
                print(json.dumps(result, ensure_ascii=False))
            else:
                print(
                    f"accuracy = {result['cid_accuracy']:.4f}±"
                    f"{result['cid_std']:.4f}"
                )
                print(
                    f"accuracy = {result['no_cid_accuracy']:.4f}±"
                    f"{result['no_cid_std']:.4f}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
