"""Command-line interface for dialect classification."""

import argparse
import logging
from typing import Sequence

import joblib
import numpy
import pandas
import sklearn.compose
import sklearn.impute
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline

from .. import compare, datasets, preprocess


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler())


def train_classifier(
    rules: pandas.DataFrame,
    annotations: pandas.DataFrame,
    resample: int = 0,
    samples: int = 500,
    min_rate: float = 0.5,
    max_rate: float = 0.8,
) -> sklearn.pipeline.Pipeline:
    """Train a dialect classifier.

    Dialect pronunciations are resampled for data augmentation, and the best
    model hyperparameters are selected by cross-validation.

    Parameters
    ----------
    rules : pandas.DataFrame
        Pronunciation rules used to compute the rule compliance of dialects.
    annotations : pandas.DataFrame
        Annotated samples, one dialect per row, with columns:

        - ``dataset``: name of the source dataset;
        - ``did``: dialect ID;
        - ``stratefy``: optional labels for stratified splitting;
        - ``group``: optional groups for grouped splitting; samples of one
          group always fall in the same fold;
        - ``label``: dialect class.
    resample : int, default=0
        Number of resampled copies per dialect; 0 disables resampling.
    samples : int, default=500
        Number of characters sampled per dialect during resampling.
    min_rate : float, default=0.5
        Minimum sample size as a proportion of the character count of a
        dialect.
    max_rate : float, default=0.8
        Maximum sample size as a proportion of the character count of a
        dialect.

    Returns
    -------
    classifier : sklearn.pipeline.Pipeline
        The fitted dialect classifier.
    """

    feature_names = rules["feature"].unique().tolist()

    compliances = []
    indices = []
    for name, annotation in annotations.groupby("dataset", sort=False):
        data = datasets.get(name).select(annotation["did"]).data

        if "cid" not in data.columns:
            data = data.rename(columns={"character": "cid"})
            data.dropna(subset="cid", inplace=True)

        indices.extend(annotation.index)
        expanded_data = [data]

        if resample > 0:
            # Resample dialect pronunciations for augmentation, improving
            # model stability.
            indices.extend(annotation.index.repeat(resample))
            for did, dialect in data.groupby("did", sort=False):
                n = int(numpy.clip(
                    samples,
                    min_rate * dialect.shape[0],
                    max_rate * dialect.shape[0],
                ))
                for i in range(resample):
                    expanded_data.append(
                        dialect.sample(n).assign(did=f"{did}_{i}")
                    )

        transformed = preprocess.transform(
            pandas.concat(expanded_data, axis=0, ignore_index=True),
            index="cid",
            columns="did",
            values=feature_names,
            aggfunc=lambda values: " ".join(values.dropna()),
        )
        compliances.append(compare.compliance(transformed, rules))

    compliances = pandas.concat(compliances, axis=0)
    annotations = annotations.loc[indices]

    splitter = sklearn.model_selection.StratifiedGroupKFold()
    classifier = sklearn.pipeline.make_pipeline(
        sklearn.compose.make_column_transformer(
            ("passthrough", compliances.columns)
        ),
        # impute missing features with the KNN algorithm
        sklearn.impute.KNNImputer(),
        # fit a linear classifier, selecting the best hyperparameters with
        # cross-validation
        sklearn.linear_model.LogisticRegressionCV(
            Cs=numpy.power(2.0, numpy.arange(-4, 5, 0.5)),
            fit_intercept=False,
            cv=list(splitter.split(
                compliances,
                annotations["stratefy"],
                annotations["group"],
            )),
            penalty="l1",
            solver="saga",
        ),
    ).fit(compliances, annotations["label"])

    logger.debug(
        f"best C = {classifier.steps[-1][1].C_[0]:.4f}"
    )
    return classifier


def train(args: argparse.Namespace) -> None:
    """Train a dialect classifier and save it to a file."""

    logger.debug(
        f"train dialect classifier, rule file = {args.rule_file}, "
        f"annotation file = {args.annotation_file}, "
        f"output file = {args.output_file}, resample = {args.resample}"
    )

    rules = compare.load_rules(args.rule_file)
    annotations = pandas.read_csv(
        args.annotation_file,
        dtype=str,
        comment="#",
        encoding="utf-8",
    )
    if "stratefy" not in annotations.columns:
        annotations["stratefy"] = annotations["label"]
    if "group" not in annotations.columns:
        annotations["group"] = (
            annotations["dataset"] + "_" + annotations["did"]
        )

    classifier = train_classifier(rules, annotations, args.resample)

    # Use the standard deviation of model weights as feature importance.
    weights = classifier.steps[-1][1].coef_
    if weights.shape[0] < 2:
        weights = numpy.concatenate(
            [numpy.zeros_like(weights), weights], axis=0
        )
    importances = numpy.std(weights, axis=0)
    order = numpy.argsort(importances)[::-1]
    logger.info("feature importances:")
    for i in order:
        logger.info(f'{rules.iloc[i]["name"]}: {importances[i]:.4f}')

    joblib.dump(classifier, args.output_file)


def evaluate(args: argparse.Namespace) -> None:
    """Cross-validate the accuracy of the dialect classifier."""

    logger.debug(
        f"cross evaluate dialect classifier, rule file = {args.rule_file}, "
        f"annotation file = {args.annotation_file}, "
        f"resample = {args.resample}"
    )

    rules = compare.load_rules(args.rule_file)
    annotations = pandas.read_csv(
        args.annotation_file,
        dtype=str,
        comment="#",
        encoding="utf-8",
    )
    if "stratefy" not in annotations.columns:
        annotations["stratefy"] = annotations["label"]
    if "group" not in annotations.columns:
        annotations["group"] = (
            annotations["dataset"] + "_" + annotations["did"]
        )

    feature_names = rules["feature"].unique().tolist()
    compliances = []
    indices = []
    for name, annotation in annotations.groupby("dataset", sort=False):
        data = datasets.get(name).select(annotation["did"]).data
        if "cid" not in data.columns:
            data = data.rename(columns={"character": "cid"})
            data.dropna(subset="cid", inplace=True)

        data = preprocess.transform(
            data,
            index="cid",
            columns="did",
            values=feature_names,
            aggfunc=lambda values: " ".join(values.dropna()),
        )
        compliances.append(compare.compliance(data, rules))
        indices.extend(annotation.index)

    compliances = pandas.concat(compliances, axis=0)
    annotations = annotations.loc[indices]

    splitter = sklearn.model_selection.StratifiedGroupKFold()
    accuracies = []
    for train_idx, test_idx in splitter.split(
        compliances, annotations["stratefy"], annotations["group"]
    ):
        classifier = train_classifier(
            rules, annotations.iloc[train_idx], args.resample
        )
        accuracies.append(sklearn.metrics.accuracy_score(
            annotations.iloc[test_idx]["label"],
            classifier.predict(compliances.iloc[test_idx]),
        ))

    print(
        f"cross validation with {annotations.shape[0]} samples, "
        f"accuracy = {numpy.mean(accuracies):.4f}"
        f"±{numpy.std(accuracies, ddof=1):.4f}"
    )


def predict(args: argparse.Namespace) -> None:
    """Predict dialect classes with a trained classifier."""

    logger.debug(
        f"predict dialect class, model file = {args.model_file}, "
        f"rule file = {args.rule_file}, input file = {args.input.name}, "
        f"output file = {args.output.name}"
    )

    classifier = joblib.load(args.model_file)
    if args.precomputed:
        # Load the precomputed rule compliance of dialects.
        compliances = pandas.read_csv(
            args.input, dtype={"dataset": str, "did": str}
        )
        samples = compliances.pop("did").to_frame()
        if "dataset" in compliances.columns:
            samples.insert(0, "dataset", compliances.pop("dataset"))

    else:
        # Compute the rule compliance of dialects from the raw data.
        rules = compare.load_rules(args.rule_file)
        samples = pandas.read_csv(
            args.input,
            dtype=str,
            comment="#",
            encoding="utf-8",
        )

        feature_names = rules["feature"].unique().tolist()
        compliances = []
        for name, items in samples.groupby("dataset", sort=False):
            data = datasets.get(name).select(items["did"]).data
            if "cid" not in data.columns:
                data = data.rename(columns={"character": "cid"})
                data.dropna(subset="cid", inplace=True)

            data = preprocess.transform(
                data,
                index="cid",
                columns="did",
                values=feature_names,
                aggfunc=lambda values: " ".join(values.dropna()),
            )
            compliances.append(
                compare.compliance(data, rules).set_index(items.index)
            )

        compliances = pandas.concat(compliances, axis=0)
        compliances = compliances.loc[samples.index]

    probabilities = pandas.DataFrame(
        classifier.predict_proba(compliances),
        columns=classifier.steps[-1][1].classes_,
    )
    pandas.concat([samples, probabilities], axis=1).to_csv(
        args.output, index=False
    )


def build_parser() -> argparse.ArgumentParser:
    """Build the dialect classifier command-line parser."""

    parser = argparse.ArgumentParser(
        description="根据指定的规则训练模型对方言分类"
    )
    parser.add_argument(
        "-l", "--log-level", default="WARNING", help="日志级别"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="训练方言分类器")
    train_parser.add_argument(
        "--resample",
        type=int,
        default=5,
        help="对每个方言样本重采样的次数，为 0 不重采样",
    )
    train_parser.add_argument(
        "rule_file", help="语音规则文件 (JSON)"
    )
    train_parser.add_argument(
        "annotation_file", help="标注样本文件 (CSV)"
    )
    train_parser.add_argument(
        "output_file",
        nargs="?",
        default="dialect_classifier.bz2",
        help="模型输出文件",
    )
    train_parser.set_defaults(func=train)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="交叉验证方言分类器准确率"
    )
    evaluate_parser.add_argument(
        "--resample",
        type=int,
        default=5,
        help="对每个方言样本重采样的次数，为 0 不重采样",
    )
    evaluate_parser.add_argument(
        "rule_file", help="语音规则文件 (JSON)"
    )
    evaluate_parser.add_argument(
        "annotation_file", help="标注样本文件 (CSV)"
    )
    evaluate_parser.set_defaults(func=evaluate)

    predict_parser = subparsers.add_parser(
        "predict", help="使用已训练的分类器预测方言分类"
    )
    predict_parser.add_argument(
        "-p",
        "--precomputed",
        action="store_true",
        default=False,
        help="输入为预计算的规则符合度文件",
    )
    predict_parser.add_argument(
        "-m",
        "--model-file",
        default="dialect_classifier.bz2",
        help="模型文件",
    )
    predict_parser.add_argument(
        "-r", "--rule-file", help="语音规则文件 (JSON)"
    )
    predict_parser.add_argument(
        "input",
        nargs="?",
        default="-",
        type=argparse.FileType("r", encoding="utf-8"),
        help="待分类方言样本文件 (CSV)",
    )
    predict_parser.add_argument(
        "output",
        nargs="?",
        default="-",
        type=argparse.FileType("w", encoding="utf-8"),
        help="预测结果输出文件 (CSV)",
    )
    predict_parser.set_defaults(func=predict)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dialect classifier command-line interface."""

    args = build_parser().parse_args(argv)
    logger.setLevel(getattr(logging, args.log_level.upper()))
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
