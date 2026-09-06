"""Command-line interface for rule compliance comparison."""

import argparse
import logging
from typing import Sequence

import sklearn.preprocessing

from .. import compare, datasets, preprocess


def build_parser() -> argparse.ArgumentParser:
    """Build the rule comparison command-line parser."""
    parser = argparse.ArgumentParser(description="计算方言数据对语音规则的符合度")
    parser.add_argument("-l", "--log-level", default="WARNING", help="日志级别")
    parser.add_argument("-r", "--rule-file", default="rules.json", help="语音规则文件")
    parser.add_argument(
        "-n", "--norm", type=int, default=2, help="把规则符合度归一化到 [0, 1]"
    )
    parser.add_argument("dataset", help="指定输入方言数据集")
    parser.add_argument("output", nargs="?", help="输出文件名")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the rule comparison command-line interface."""
    args = build_parser().parse_args(argv)
    compare.logger.setLevel(getattr(logging, args.log_level.upper()))

    dataset = datasets.get(args.dataset)
    if dataset is None:
        raise SystemExit(f"unknown dataset: {args.dataset}")

    output = (
        f"{dataset.name}_compliance_l{args.norm}.csv"
        if args.output is None
        else args.output
    )
    rules = compare.load_rules(args.rule_file)
    encoder = sklearn.preprocessing.LabelEncoder()
    rules["feature_id"] = encoder.fit_transform(rules["feature"])

    data = dataset.data
    if "cid" not in data.columns:
        data = data.rename(columns={"character": "cid"}).dropna(subset="cid")
    data = preprocess.transform(
        data,
        index="cid",
        columns="did",
        values=encoder.classes_,
        aggfunc=lambda values: " ".join(values.dropna()),
    )

    result = compare.compliance(data, rules, norm=args.norm if args.norm > 0 else None)
    result.insert(0, "dataset", dataset.name)
    result.insert(1, "did", result.index)
    result.to_csv(output, index=False, encoding="utf-8", lineterminator="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
