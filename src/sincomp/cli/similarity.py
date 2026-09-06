"""Command-line interface for dialect similarity calculation."""

import argparse
import os
from typing import Sequence

import pandas

from .. import datasets, preprocess, similarity


def build_parser() -> argparse.ArgumentParser:
    """Build the similarity command-line parser."""
    parser = argparse.ArgumentParser(description="计算方言之间的预测相似度")
    parser.add_argument(
        "-m",
        "--method",
        choices=("chi2", "entropy"),
        help="计算方言间相似度的方法，不指定时计算所有方法",
    )
    parser.add_argument(
        "-o",
        "--output",
        help="输出路径，单个结果时为文件名，否则为输出目录",
    )
    parser.add_argument(
        "dataset",
        nargs="*",
        default=("ccr",),
        help="方言数据集名称、数据文件或目录路径",
    )
    return parser


def _load_dataset(value: str):
    """Load a registered dataset, CSV file, or dataset directory."""
    try:
        return getattr(datasets, value), value
    except AttributeError:
        if os.path.isdir(value):
            return datasets.FileDataset(path=value), os.path.basename(value)
        return pandas.read_csv(value, dtype=str), os.path.splitext(os.path.basename(value))[0]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the similarity command-line interface."""
    args = build_parser().parse_args(argv)
    methods = (args.method,) if args.method else ("chi2", "entropy")

    for value in args.dataset:
        data, name = _load_dataset(value)
        data = preprocess.transform(
            data,
            index="cid",
            columns="did",
            values=["initial", "final", "tone"],
            aggfunc="first",
        ).fillna("")

        for method in methods:
            if len(args.dataset) > 1 or len(methods) > 1:
                output = os.path.join(
                    os.getcwd() if args.output is None else args.output,
                    f"{name}_{method}.csv",
                )
            else:
                output = (
                    os.path.join(os.getcwd(), f"{name}_{method}.csv")
                    if args.output is None
                    else args.output
                )

            print(f"compute {method} between {name} dialects -> {output}")
            os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
            result = getattr(similarity, method)(data, parallel=4)
            result.to_csv(output, lineterminator="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
