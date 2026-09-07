"""Command-line interface for dialect datasets."""

import argparse
import json
from typing import Sequence

import pandas

from .. import datasets as dataset_api


def build_parser() -> argparse.ArgumentParser:
    """Build the dataset command-line parser."""
    parser = argparse.ArgumentParser(description="查看方言数据集")
    commands = parser.add_subparsers(dest="command", required=True)

    list_parser = commands.add_parser("list", help="列出预定义数据集")
    list_parser.add_argument(
        "--format", choices=("text", "json"), default="json", help="结果输出格式"
    )

    dialects_parser = commands.add_parser("dialects", help="列出数据集中的方言")
    dialects_parser.add_argument("dataset", help="数据集名称、别名或本地路径")
    dialects_parser.add_argument("dialect_filter", nargs="?", help="方言筛选表达式")
    dialects_parser.add_argument(
        "--format", choices=("text", "json"), default="json", help="结果输出格式"
    )

    query_parser = commands.add_parser("query", help="查询数据集中的读音数据")
    query_parser.add_argument("dataset", help="数据集名称、别名或本地路径")
    query_parser.add_argument("dialect_filter", help="方言筛选表达式")
    query_parser.add_argument("data_filter", nargs="?", help="读音数据筛选表达式")
    query_parser.add_argument(
        "--format", choices=("text", "json"), default="json", help="结果输出格式"
    )

    return parser


def _print_list(output_format: str) -> None:
    names = dataset_api.list_datasets()
    if output_format == "json":
        print(json.dumps(names, ensure_ascii=False))
    else:
        print("\n".join(names))


def _print_dialects(
    name: str, dialect_filter: str | None, output_format: str
) -> int:
    dataset = dataset_api.get(name)
    if dataset is None:
        raise SystemExit(f"unknown dataset: {name}")

    dialects = dataset.dialects.copy()
    dialects.index.name = "did"
    if dialect_filter is not None:
        dialects = dialects.query(dialect_filter)
    if output_format == "json":
        print(dialects.reset_index().to_json(orient="records", force_ascii=False))
    else:
        print(dialects.to_string())
    return 0


def _print_query(
    name: str,
    dialect_filter: str,
    data_filter: str | None,
    output_format: str,
) -> int:
    dataset = dataset_api.get(name)
    if dataset is None:
        raise SystemExit(f"unknown dataset: {name}")

    dialects = dataset.dialects.reset_index(names="did")
    selected_dialects = dialects.query(dialect_filter)
    if len(selected_dialects) == 0:
        data = pandas.DataFrame()
    else:
        data = pandas.concat(
            [dataset.get_data(did) for did in selected_dialects["did"]],
            ignore_index=True,
        )
    if data_filter is not None:
        data = data.query(data_filter)

    if output_format == "json":
        print(data.to_json(orient="records", force_ascii=False))
    else:
        print(data.to_string(index=False))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dataset command-line interface."""
    args = build_parser().parse_args(argv)
    if args.command == "list":
        _print_list(args.format)
        return 0
    if args.command == "dialects":
        return _print_dialects(args.dataset, args.dialect_filter, args.format)
    return _print_query(
        args.dataset,
        args.dialect_filter,
        args.data_filter,
        args.format,
    )


if __name__ == "__main__":
    raise SystemExit(main())
