"""Command-line interfaces for SinComp."""

import argparse
import sys
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
	"""Build the human-friendly top-level command parser."""
	parser = argparse.ArgumentParser(description="汉语方言数据处理与比较工具")
	commands = parser.add_subparsers(dest="command")
	commands.add_parser("align", help="对齐方言数据集")
	commands.add_parser("compare", help="计算语音规则符合度")
	commands.add_parser("similarity", help="计算方言相似度")
	commands.add_parser("dataset", help="查看方言数据集")
	return parser


def main(argv: Sequence[str] | None = None) -> int:
	"""Dispatch a human-friendly command to its domain CLI."""
	if argv is None:
		argv = sys.argv[1:]

	parser = build_parser()
	if not argv or argv[0] in {"-h", "--help"}:
		parser.print_help()
		return 0

	command = argv[0]
	arguments = list(argv[1:])
	if command == "align":
		from . import align

		if len(arguments) > 0 and arguments[0] == "evaluate":
			return align.main(["evaluate", "--format", "text", *arguments[1:]])
		return align.main(["align", *arguments])
	if command == "compare":
		from . import compare

		return compare.main(arguments)
	if command == "similarity":
		from . import similarity

		return similarity.main(arguments)
	if command == "dataset":
		from . import dataset

		if len(arguments) > 0 and arguments[0] in {"list", "dialects", "query"}:
			arguments = [arguments[0], "--format", "text", *arguments[1:]]
		return dataset.main(arguments)

	parser.error(f"invalid choice: {command!r}")
	return 2
