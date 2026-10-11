"""Command-line interface for dialect classification.

The classifier distinguishes the classes of dialects either by their
compliance with a set of pronunciation rules or by their dialect
vectors — the pairwise pronunciation similarities of the characters
embedded into dense vectors with
:class:`~sincomp.embedding.DialectEmbedding`.

train
    Train a dialect classifier on the annotated samples and save it
    to a file. The features are the rule compliances of the dialects
    when a rule file is given with ``--rule-file``, and the dialect
    vectors otherwise; the characters whose dialect coverage falls
    below ``--min-coverage`` are excluded from the vectors, and the
    vectors are embedded into ``--embedding-size`` dimensions. Every
    dialect may be resampled ``--resample`` times for augmentation,
    and the best hyperparameters of the linear classifier are
    selected by cross-validation.

evaluate
    Cross-validate the accuracy of the classifier on the annotated
    samples: the annotations are split into stratified groups, every
    fold trains a classifier on the training annotations and predicts
    the test annotations from their dialect data, and the mean and the standard
    deviation of the fold accuracies are printed to the standard
    output.

predict
    Predict the classes of the dialect samples with a trained
    classifier and write the class probabilities of every sample.
    The features are computed from the raw data with ``--rule-file``
    or the vectorizer stored in the model, or read from the input
    directly with ``--precomputed`` in the rule mode.

The annotated sample file is a CSV with one dialect per row and the
columns ``dataset`` (name of the source dataset), ``did`` (dialect
ID) and ``label`` (dialect class), plus the optional ``stratefy``
(labels for stratified splitting) and ``group`` (groups for grouped
splitting; samples of one group always fall in the same fold)
columns.

Usage::

    classifier train --rule-file rules.json --annotation-file annotations.csv
    classifier evaluate --annotation-file annotations.csv
    classifier predict --model-file model.bz2 --input-file samples.csv

All subcommands accept a JSON ``--config`` file providing the
command-line defaults; command-line values take precedence over the
config. Use one flat object with option names (without leading dashes)
and positional argument names as keys. Each subcommand uses only the
keys matching its arguments and ignores the others.
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"

import argparse
import json
import logging
import os
import sys
from typing import Iterator, Sequence, TextIO

import joblib
import numpy
import pandas
import sklearn.compose
import sklearn.impute
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline

from .. import compare, datasets, embedding, preprocess


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.addHandler(logging.StreamHandler())

FEATURES = ("initial", "final", "tone")


def filter_coverage(
    data: pandas.DataFrame,
    min_coverage: float,
) -> pandas.DataFrame:
    """Filter the characters meeting the dialect coverage threshold.

    The dialect coverage of one character is the proportion of the
    dialects in the data that record a pronunciation of the character.

    Parameters
    ----------
    data : pandas.DataFrame
        Long-format dialect data with ``did``, ``cid`` and segment
        columns.
    min_coverage : float
        Minimum dialect coverage of the kept characters.

    Returns
    -------
    pandas.DataFrame
        The rows of the data whose character meets the threshold.
    """

    feature_names = [f for f in FEATURES if f in data.columns]
    recorded = data[data[feature_names].notna().any(axis=1)]
    coverage = (
        recorded.groupby("cid")["did"].nunique() / data["did"].nunique()
    )
    characters = coverage[coverage >= min_coverage].index
    return data[data["cid"].isin(characters)]


def build_wide(data: pandas.DataFrame) -> pandas.DataFrame:
    """Build the wide phone table of the dialect vector features.

    One row per dialect and one ``cid_feature`` column per character
    and segment feature, as expected by
    :class:`~sincomp.embedding.DialectVectorizer`.

    Parameters
    ----------
    data : pandas.DataFrame
        Long-format dialect data with ``did``, ``cid`` and segment
        columns.

    Returns
    -------
    pandas.DataFrame
        The wide table with flattened ``cid_feature`` columns.
    """

    wide = preprocess.transform(
        data,
        index="did",
        columns="cid",
        values=[f for f in FEATURES if f in data.columns],
        aggfunc=lambda values: " ".join(values.dropna()),
    )
    wide.columns = [f"{c}_{f}" for c, f in wide.columns.to_flat_index()]
    # a character not recorded by a dialect holds an empty phone and
    # contributes no pronunciation similarities
    return wide.fillna("")


def resample_data(
    dataset: datasets.Dataset,
    annotation: pandas.DataFrame,
    resample: int,
    samples: int = 1000,
    min_rate: float = 0.5,
    max_rate: float = 0.8,
    characters: pandas.Index | numpy.ndarray | None = None,
) -> Iterator[tuple[pandas.DataFrame, pandas.Index]]:
    """Yield the rounds of the resampled pronunciations.

    Every round holds one resampled copy of each annotated dialect —
    ``samples`` characters clipped to the ``min_rate`` and
    ``max_rate`` proportions of the dialect, with the dialect ID
    suffixed with the round number. Resampling improves the stability
    of the model trained on few dialects.

    Parameters
    ----------
    dataset : datasets.Dataset
        Dataset containing the dialect pronunciations.
    annotation : pandas.DataFrame
        The annotated samples of the dialects to resample.
    resample : int
        Number of resampled copies per dialect.
    samples : int, default=1000
        Number of characters sampled per dialect copy.
    min_rate : float, default=0.5
        Minimum sample size as a proportion of the character count of
        the dialect.
    max_rate : float, default=0.8
        Maximum sample size as a proportion of the character count of
        the dialect.
    characters : pandas.Index, optional
        Character IDs to retain in each copy, used when the training
        data has already been filtered by character coverage.

    Yields
    ------
    data : pandas.DataFrame
        The resampled pronunciations of one round.
    indices : pandas.Index
        The annotation indexes of the dialects of the round, aligned
        with the dialect order.
    """

    for i in range(resample):
        copies = []
        for did in annotation["did"].drop_duplicates():
            dialect = dataset.select([did]).data
            if "cid" not in dialect.columns:
                dialect = dialect.rename(columns={"character": "cid"})
                dialect = dialect.dropna(subset="cid")
            if characters is not None:
                dialect = dialect[dialect["cid"].isin(characters)]
            n = int(numpy.clip(
                samples,
                min_rate * dialect.shape[0],
                max_rate * dialect.shape[0],
            ))
            copies.append(dialect.sample(n).assign(did=f"{did}_{i}"))
        yield (
            pandas.concat(copies, axis=0, ignore_index=True),
            annotation.index,
        )


def train_rule(
    annotations: pandas.DataFrame,
    resample: int = 0,
    samples: int = 1000,
    min_rate: float = 0.5,
    max_rate: float = 0.8,
    *,
    rules: pandas.DataFrame,
) -> sklearn.pipeline.Pipeline:
    """Train a dialect classifier on rule compliance features.

    Dialect pronunciations are resampled for data augmentation, and the
    best model hyperparameters are selected by cross-validation.

    Parameters
    ----------
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
    samples : int, default=1000
        Number of characters sampled per dialect during resampling.
    min_rate : float, default=0.5
        Minimum sample size as a proportion of the character count of a
        dialect.
    max_rate : float, default=0.8
        Maximum sample size as a proportion of the character count of a
        dialect.
    rules : pandas.DataFrame
        Pronunciation rules used to compute the rule compliance of
        dialects.
    rules : pandas.DataFrame
        Pronunciation rules used to compute the rule compliance of
        dialects.

    Returns
    -------
    classifier : sklearn.pipeline.Pipeline
        The fitted dialect classifier.
    """

    feature_names = rules["feature"].unique().tolist()

    compliances = []
    indices = []
    for name, annotation in annotations.groupby("dataset", sort=False):
        dataset = datasets.get(name)
        data = dataset.select(annotation["did"]).data

        if "cid" not in data.columns:
            data = data.rename(columns={"character": "cid"})
            data.dropna(subset="cid", inplace=True)

        transformed = preprocess.transform(
            data,
            index="cid",
            columns="did",
            values=feature_names,
            aggfunc=lambda values: " ".join(values.dropna()),
        )
        compliances.append(compare.compliance(transformed, rules))
        indices.extend(annotation.index)

        for round_data, round_indices in resample_data(
            dataset, annotation, resample, samples, min_rate, max_rate
        ):
            transformed = preprocess.transform(
                round_data,
                index="cid",
                columns="did",
                values=feature_names,
                aggfunc=lambda values: " ".join(values.dropna()),
            )
            compliances.append(compare.compliance(transformed, rules))
            indices.extend(round_indices)

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

    _log_best_cv_score(classifier.steps[-1][1])
    return classifier


def _log_best_cv_score(
    classifier: sklearn.linear_model.LogisticRegressionCV,
) -> None:
    """Log the cross-validation accuracy for the selected regularization."""

    best_c = numpy.asarray(classifier.C_).flat[0]
    best_index = numpy.abs(classifier.Cs_ - best_c).argmin()
    scores = next(iter(classifier.scores_.values()))
    best_scores = numpy.asarray(scores)[:, best_index]
    logger.info(
        "cross-validation accuracy = %.4f±%.4f",
        best_scores.mean(),
        best_scores.std(),
    )


def train_embedding(
    annotations: pandas.DataFrame,
    resample: int = 0,
    samples: int = 1000,
    min_rate: float = 0.5,
    max_rate: float = 0.8,
    min_coverage: float = 0.5,
    embedding_size: int = 128,
) -> sklearn.pipeline.Pipeline:
    """Train a dialect classifier on dialect vector features.

    The characters whose dialect coverage reaches ``min_coverage`` are
    kept, the pairwise pronunciation similarity of the kept characters
    is vectorized with :class:`~sincomp.embedding.DialectVectorizer`
    and embedded into dense dialect vectors with
    :class:`~sincomp.embedding.DialectEmbedding`, and the best model
    hyperparameters are selected by cross-validation. Every dialect
    may be resampled for augmentation; a resampled copy holds the
    sampled characters only, and the pairs involving the characters
    missing from a copy contribute no similarities. The annotated
    samples must come from one dataset — the dialect vector features
    are only comparable within one dataset. The vectorizer and the
    embedding are fitted on the dialects before the resampling; every
    round of the resampled dialects is transformed separately and the
    resulting embeddings are concatenated into the training features,
    so the memory footprint of the transformation stays low.

    Parameters
    ----------
    annotations : pandas.DataFrame
        Annotated samples, one dialect per row, with the same columns
        as :func:`train_rule`.
    resample : int, default=0
        Number of resampled copies per dialect; 0 disables resampling.
    samples : int, default=1000
        Number of characters sampled per dialect during resampling.
    min_rate : float, default=0.5
        Minimum sample size as a proportion of the character count of a
        dialect.
    max_rate : float, default=0.8
        Maximum sample size as a proportion of the character count of a
        dialect.
    min_coverage : float, default=0.5
        Minimum dialect coverage of the characters used by the dialect
        vector features.
    embedding_size : int, default=128
        Dimension of the embedded dialect vectors.

    Returns
    -------
    classifier : sklearn.pipeline.Pipeline
        The fitted dialect classifier.
    """

    dataset_names = annotations["dataset"].unique()
    if len(dataset_names) > 1:
        raise ValueError(
            "the dialect vector features are only comparable within "
            "one dataset, but the samples cover "
            f"{', '.join(map(str, dataset_names))}"
        )
    dataset = datasets.get(dataset_names[0])
    data = dataset.select(annotations["did"]).data
    if "cid" not in data.columns:
        data = data.rename(columns={"character": "cid"})
        data.dropna(subset="cid", inplace=True)
    data = filter_coverage(data, min_coverage)
    wide = build_wide(data)

    vectorizer = embedding.DialectVectorizer(
        column_selectors=list(FEATURES)
    )
    dialect_embedding = embedding.DialectEmbedding(
        embedding_size=embedding_size
    )
    feature_pipeline = sklearn.pipeline.make_pipeline(
        vectorizer,
        dialect_embedding,
    )

    features = [feature_pipeline.fit_transform(wide)]
    indices = []
    indices.extend(annotations.index)
    for round_data, round_indices in resample_data(
        dataset,
        annotations,
        resample,
        samples,
        min_rate,
        max_rate,
        characters=data["cid"].unique(),
    ):
        round_wide = build_wide(round_data).reindex(
            columns=vectorizer.feature_names_in_, fill_value=""
        )
        features.append(feature_pipeline.transform(round_wide))
        indices.extend(round_indices)

    features = numpy.concatenate(features, axis=0)
    annotations = annotations.loc[indices]

    splitter = sklearn.model_selection.StratifiedGroupKFold()
    classifier = sklearn.pipeline.make_pipeline(
        vectorizer,
        dialect_embedding,
        # fit a linear classifier, selecting the best hyperparameters
        # with cross-validation
        sklearn.linear_model.LogisticRegressionCV(
            Cs=numpy.power(2.0, numpy.arange(-4, 5, 0.5)),
            fit_intercept=False,
            cv=list(splitter.split(
                features,
                annotations["stratefy"],
                annotations["group"],
            )),
            penalty="l1",
            solver="saga",
        ),
    )
    # the vectorizer and the embedding are already fitted on the
    # pre-resampling dialects, only the linear classifier is fitted
    # on the features of the expanded data
    classifier.steps[-1][1].fit(features, annotations["label"])

    _log_best_cv_score(classifier.steps[-1][1])
    return classifier


def load_annotations(path: str) -> pandas.DataFrame:
    """Load the annotated samples and fill the splitting columns.

    Parameters
    ----------
    path : str
        Annotated sample file (CSV).

    Returns
    -------
    pandas.DataFrame
        The annotated samples, one dialect per row, with the
        ``stratefy`` and ``group`` columns filled with defaults when
        missing.
    """

    annotations = pandas.read_csv(
        path,
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
    return annotations


def train(args: argparse.Namespace) -> None:
    """Train a dialect classifier and save it to a file.

    The features are the rule compliances of the dialects when
    ``args.rule_file`` is given, and the dialect vectors of
    :func:`train_embedding` otherwise.
    """

    model_file = args.model_file
    logger.debug(
        f"train dialect classifier, rule file = {args.rule_file}, "
        f"annotation file = {args.annotation_file}, "
        f"model file = {model_file}, "
        f"resample = {args.resample}"
    )

    rules = compare.load_rules(args.rule_file) if args.rule_file else None
    annotations = load_annotations(args.annotation_file)

    if rules is None:
        classifier = train_embedding(
            annotations,
            resample=args.resample,
            samples=args.samples,
            min_coverage=args.min_coverage,
            embedding_size=args.embedding_size,
        )
    else:
        classifier = train_rule(
            annotations,
            resample=args.resample,
            samples=args.samples,
            rules=rules,
        )

    if rules is None:
        # report the variance explained by the dialect vector embedding
        ratios = classifier.steps[1][1].explained_variance_ratio_
        logger.info(
            f"dialect vector embedding explains {numpy.sum(ratios):.4f} "
            f"of the variance over {len(ratios)} components"
        )
    else:
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

    os.makedirs(
        os.path.dirname(os.path.abspath(model_file)),
        exist_ok=True,
    )
    joblib.dump(classifier, model_file)


def evaluate(args: argparse.Namespace) -> None:
    """Cross-validate the accuracy of the dialect classifier.

    The annotated samples are split into stratified groups, every fold
    trains a classifier on the training samples with
    :func:`train_rule` or :func:`train_embedding` and predicts the
    classes of the test samples. The per-fold metrics and their mean
    and standard deviation are printed to the standard output as a
    table: the precision, the recall and the F1 macro-averaged over
    the two classes for a binary classification, and the accuracy for
    a multiclass classification. The features are the rule
    compliances of the dialects when ``args.rule_file`` is given, and
    the dialect vectors of :func:`train_embedding` otherwise.
    """

    logger.debug(
        f"cross evaluate dialect classifier, rule file = {args.rule_file}, "
        f"annotation file = {args.annotation_file}, "
        f"resample = {args.resample}"
    )

    rules = compare.load_rules(args.rule_file) if args.rule_file else None
    annotations = load_annotations(args.annotation_file)

    splitter = sklearn.model_selection.StratifiedGroupKFold()
    binary = annotations["label"].nunique() == 2
    accuracies = []
    precisions = []
    recalls = []
    f1s = []
    for train_idx, test_idx in splitter.split(
        annotations,
        annotations["stratefy"],
        annotations["group"],
    ):
        train_samples = annotations.iloc[train_idx]
        test_samples = annotations.iloc[test_idx]
        if rules is None:
            classifier = train_embedding(
                train_samples,
                resample=args.resample,
                samples=args.samples,
                min_coverage=args.min_coverage,
                embedding_size=args.embedding_size,
            )
            probabilities = predict_embedding(classifier, test_samples)
        else:
            classifier = train_rule(
                train_samples,
                resample=args.resample,
                samples=args.samples,
                rules=rules,
            )
            _, probabilities = predict_rule(
                classifier,
                test_samples,
                rules,
            )
        predicted = probabilities.columns.to_numpy()[
            probabilities.to_numpy().argmax(axis=1)
        ]
        expected = test_samples["label"]
        accuracies.append(sklearn.metrics.accuracy_score(expected, predicted))
        if binary:
            # macro-averaged over the two classes, which are equally
            # informative in the dialect classification
            precisions.append(sklearn.metrics.precision_score(
                expected, predicted, average="macro", zero_division=0
            ))
            recalls.append(sklearn.metrics.recall_score(
                expected, predicted, average="macro", zero_division=0
            ))
            f1s.append(sklearn.metrics.f1_score(
                expected, predicted, average="macro", zero_division=0
            ))

    if binary:
        report = pandas.DataFrame(
            {
                "accuracy": accuracies,
                "precision": precisions,
                "recall": recalls,
                "f1": f1s,
            },
            index=pandas.Index(range(1, len(accuracies) + 1), name="fold"),
        )
    else:
        report = pandas.DataFrame(
            {"accuracy": accuracies},
            index=pandas.Index(range(1, len(accuracies) + 1), name="fold"),
        )
    report.loc["mean"] = report.mean()
    report.loc["std"] = report.iloc[:-1].std()
    print(
        f"cross validation with {annotations.shape[0]} samples, "
        f"accuracy = {numpy.mean(accuracies):.4f}"
    )
    print(report.to_string(float_format=lambda value: f"{value:.4f}"))


def predict_rule(
    classifier: sklearn.pipeline.Pipeline,
    annotations: pandas.DataFrame,
    rules: pandas.DataFrame,
) -> tuple[pandas.DataFrame, pandas.DataFrame]:
    """Predict dialect classes with a rule mode classifier.

    Rule compliance features are computed from the dialect data of the
    annotated samples.

    Parameters
    ----------
    classifier : sklearn.pipeline.Pipeline
        The trained rule mode classifier.
    annotations : pandas.DataFrame
        Samples to classify, with ``dataset`` and ``did`` columns.
    rules : pandas.DataFrame
        Pronunciation rules used to compute the rule compliance.

    Returns
    -------
    samples : pandas.DataFrame
        The sample table without the feature columns.
    probabilities : pandas.DataFrame
        The class probability of each sample.
    """

    feature_names = rules["feature"].unique().tolist()
    compliances = []
    for name, items in annotations.groupby("dataset", sort=False):
        data = datasets.get(name).select(items["did"]).data
        if "cid" not in data.columns:
            data = data.rename(columns={"character": "cid"})
            data.dropna(subset="cid", inplace=True)

        transformed = preprocess.transform(
            data,
            index="cid",
            columns="did",
            values=feature_names,
            aggfunc=lambda values: " ".join(values.dropna()),
        )
        compliances.append(
            compare.compliance(transformed, rules).set_index(items.index)
        )
    features = pandas.concat(compliances, axis=0).loc[annotations.index]
    probabilities = pandas.DataFrame(
        classifier.predict_proba(features),
        columns=classifier.steps[-1][1].classes_,
        index=annotations.index,
    )
    return annotations, probabilities


def predict_embedding(
    classifier: sklearn.pipeline.Pipeline,
    annotations: pandas.DataFrame,
) -> pandas.DataFrame:
    """Predict the class probabilities of the samples with an
    embedding mode classifier.

    The dialect vectors of the samples are computed from the raw data
    with the vectorizer stored in the classifier.

    Parameters
    ----------
    classifier : sklearn.pipeline.Pipeline
        The trained embedding mode classifier.
    annotations : pandas.DataFrame
        Samples to classify, with ``dataset`` and ``did`` columns.

    Returns
    -------
    probabilities : pandas.DataFrame
        The class probability of each sample, indexed like the
        annotations.
    """

    dataset_names = annotations["dataset"].unique()
    if len(dataset_names) > 1:
        raise ValueError(
            "the dialect vector features are only comparable within "
            "one dataset, but the samples cover "
            f"{', '.join(map(str, dataset_names))}"
        )
    data = datasets.get(dataset_names[0]).select(
        annotations["did"]
    ).data
    if "cid" not in data.columns:
        data = data.rename(columns={"character": "cid"})
        data.dropna(subset="cid", inplace=True)
    features = build_wide(data).reindex(
        columns=classifier.steps[0][1].feature_names_in_,
        fill_value="",
    )
    return pandas.DataFrame(
        classifier.predict_proba(features),
        columns=classifier.steps[-1][1].classes_,
        index=annotations.index,
    )


def predict(args: argparse.Namespace) -> None:
    """Predict dialect classes with a trained classifier.

    Dispatches to :func:`predict_embedding` when the model was
    trained on the dialect vector features and to
    :func:`predict_rule` otherwise.
    """

    logger.debug(
        f"predict dialect class, model file = {args.model_file}, "
        f"rule file = {args.rule_file}, "
        f"input file = {args.input_file.name}, "
        f"prediction file = {args.prediction_file.name}"
    )

    classifier = joblib.load(args.model_file)
    samples = pandas.read_csv(
        args.input_file,
        dtype=str,
        comment="#",
        encoding="utf-8",
    )
    if isinstance(classifier.steps[0][1], embedding.DialectVectorizer):
        probabilities = predict_embedding(classifier, samples)
    else:
        if args.precomputed:
            features = samples
            annotations = features.pop("did").to_frame()
            if "dataset" in features.columns:
                annotations.insert(0, "dataset", features.pop("dataset"))
            probabilities = pandas.DataFrame(
                classifier.predict_proba(features),
                columns=classifier.steps[-1][1].classes_,
                index=annotations.index,
            )
            samples = annotations
        else:
            if not args.rule_file:
                raise SystemExit(
                    "a rule file is required to compute the rule "
                    "compliance features of the raw data"
                )
            rules = compare.load_rules(args.rule_file)
            samples, probabilities = predict_rule(
                classifier, samples, rules
            )

    pandas.concat([samples, probabilities], axis=1).to_csv(
        args.prediction_file, index=False
    )


def _open_prediction_output(path: str) -> TextIO:
    """Open the prediction output path, creating its parent directory."""

    if path == "-":
        return sys.stdout
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    return open(path, "w", encoding="utf-8")


def load_config(path: str) -> dict:
    """Load flat command-line defaults from a JSON configuration file."""

    with open(path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a JSON object.")
    return config


CONFIG_HELP = (
    "JSON configuration file providing the command-line defaults; "
    "command-line values take precedence over the config. Use a flat "
    "JSON object with option names (without leading dashes) and "
    "positional argument names as keys; unused keys are ignored."
)


def build_parser(config: dict | None = None) -> argparse.ArgumentParser:
    """Build the dialect classifier command-line parser.

    The top-level description reuses the module docstring, and every
    subcommand reuses the docstring of its handler function as the
    detailed description.

    Parameters
    ----------
    config : dict, default=None
        Flat configuration values applied as argument defaults.
        Each subcommand uses only values matching its arguments.
    """

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-l", "--log-level", default="WARNING",
        help="Logging verbosity.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    config_defaults = {
        key.replace("-", "_"): value
        for key, value in (config or {}).items()
    }
    if "log_level" in config_defaults:
        parser.set_defaults(log_level=config_defaults["log_level"])

    def apply_config_defaults(command_parser: argparse.ArgumentParser) -> None:
        actions = {
            action.dest: action for action in command_parser._actions
        }
        defaults = {}
        for dest, value in config_defaults.items():
            action = actions.get(dest)
            if action is None or dest in {"help", "config"}:
                continue
            if not action.option_strings and action.nargs is None:
                action.nargs = "?"
            defaults[dest] = value
        command_parser.set_defaults(**defaults)

    train_parser = subparsers.add_parser(
        "train",
        help="Train the dialect classifier and save it to a file.",
        description=train.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    train_parser.add_argument("--config", help=CONFIG_HELP)
    train_parser.add_argument(
        "-r", "--rule-file",
        default=None,
        help="Pronunciation rule file (JSON); the dialect vector "
             "features are used when omitted.",
    )
    train_parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.5,
        help="Minimum dialect coverage of the characters used by the "
             "dialect vector features.",
    )
    train_parser.add_argument(
        "--embedding-size",
        type=int,
        default=128,
        help="Dimension of the embedded dialect vectors.",
    )
    train_parser.add_argument(
        "--resample",
        type=int,
        default=5,
        help="Number of resampled copies per dialect sample; 0 "
             "disables resampling.",
    )
    train_parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of characters sampled per dialect copy during "
             "resampling.",
    )
    train_parser.add_argument(
        "--annotation-file", help="Annotated sample file (CSV)."
    )
    train_parser.add_argument(
        "-m",
        "--model-file",
        default="dialect_classifier.bz2",
        help="Model output file.",
    )
    train_parser.set_defaults(func=train)
    apply_config_defaults(train_parser)

    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Cross-validate the accuracy of the dialect classifier.",
        description=evaluate.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    evaluate_parser.add_argument("--config", help=CONFIG_HELP)
    evaluate_parser.add_argument(
        "-r", "--rule-file",
        default=None,
        help="Pronunciation rule file (JSON); the dialect vector "
             "features are used when omitted.",
    )
    evaluate_parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.5,
        help="Minimum dialect coverage of the characters used by the "
             "dialect vector features.",
    )
    evaluate_parser.add_argument(
        "--embedding-size",
        type=int,
        default=128,
        help="Dimension of the embedded dialect vectors.",
    )
    evaluate_parser.add_argument(
        "--resample",
        type=int,
        default=5,
        help="Number of resampled copies per dialect sample; 0 "
             "disables resampling.",
    )
    evaluate_parser.add_argument(
        "--samples",
        type=int,
        default=1000,
        help="Number of characters sampled per dialect copy during "
             "resampling.",
    )
    evaluate_parser.add_argument(
        "--annotation-file", help="Annotated sample file (CSV)."
    )
    evaluate_parser.set_defaults(func=evaluate)
    apply_config_defaults(evaluate_parser)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict dialect classes with a trained classifier.",
        description=predict.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    predict_parser.add_argument("--config", help=CONFIG_HELP)
    predict_parser.add_argument(
        "-p",
        "--precomputed",
        action="store_true",
        default=False,
        help="The input holds precomputed rule compliances instead "
             "of raw dialect samples; only valid in the rule mode.",
    )
    predict_parser.add_argument(
        "-m",
        "--model-file",
        default="dialect_classifier.bz2",
        help="Model file.",
    )
    predict_parser.add_argument(
        "-r", "--rule-file", help="Pronunciation rule file (JSON)."
    )
    predict_parser.add_argument(
        "--input-file",
        default="-",
        type=argparse.FileType("r", encoding="utf-8"),
        help="Dialect sample file to classify (CSV).",
    )
    predict_parser.add_argument(
        "--prediction-file",
        default="-",
        type=_open_prediction_output,
        help="Prediction file (CSV).",
    )
    predict_parser.set_defaults(func=predict)
    apply_config_defaults(predict_parser)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the dialect classifier command-line interface.

    A flat JSON configuration file may be given with ``--config``
    anywhere in the command line. Each subcommand uses only its
    matching config values; command-line values take precedence.
    """

    if argv is None:
        argv = sys.argv[1:]

    # the configuration file may appear anywhere in the command line
    # and provides the defaults of the subcommand parsers
    preliminary_parser = argparse.ArgumentParser(add_help=False)
    preliminary_parser.add_argument("--config")
    preliminary_args, _ = preliminary_parser.parse_known_args(argv)
    config = (
        load_config(preliminary_args.config)
        if preliminary_args.config is not None
        else None
    )

    args = build_parser(config).parse_args(argv)
    if args.command in {"train", "evaluate"} and args.annotation_file is None:
        raise SystemExit(
            f"the {args.command} command requires --annotation-file"
        )
    logger.setLevel(getattr(logging, args.log_level.upper()))
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
