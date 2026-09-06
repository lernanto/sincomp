"""Command-line interface for dialect data alignment."""

import argparse
import json
import logging
import os
from typing import Sequence

from .. import align as alignment


def _add_common_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-l", "--log-level", default="WARNING", help="日志级别")
    parser.add_argument(
        "-n",
        "--embedding-size",
        type=int,
        default=32,
        help="用于对齐多音字的字向量长度",
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the alignment command-line parser."""
    parser = argparse.ArgumentParser(description="对齐方言数据集")
    commands = parser.add_subparsers(dest="command", required=True)

    align_parser = commands.add_parser("align", help="对齐指定的数据集")
    _add_common_options(align_parser)
    align_parser.add_argument("--prefix", default="aligned", help="对齐后的数据集输出路径前缀")
    align_parser.add_argument(
        "--charmap-output", default="charmap.csv", help="新旧字 ID 映射表输出文件"
    )
    align_parser.add_argument(
        "--character-output",
        default=os.path.join("aligned", ".characters"),
        help="对齐后的新字 ID 到各数据集的原字 ID 的映射文件",
    )
    align_parser.add_argument(
        "--dialect-output",
        default=os.path.join("aligned", ".dialects"),
        help="合并各数据集的方言信息文件",
    )
    align_parser.add_argument("datasets", nargs="+", help="要对齐的数据集列表")

    evaluate_parser = commands.add_parser("evaluate", help="评测多音字对齐准确率")
    _add_common_options(evaluate_parser)
    evaluate_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="json",
        help="结果输出格式",
    )
    evaluate_parser.add_argument("datasets", nargs=1, help="用于评测的数据集")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the alignment command-line interface."""
    args = build_parser().parse_args(argv)
    alignment.logger.setLevel(getattr(logging, args.log_level.upper()))

    if args.command == "align":
        alignment.main(args)
    else:
        result = alignment.evaluate(args)
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
