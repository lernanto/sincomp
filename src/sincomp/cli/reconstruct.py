# -*- coding: utf-8 -*-

"""Command-line interface for proto-segment reconstruction.

The command uses all selected data to learn character and segment
embeddings, then clusters the segment embeddings.

When no annotations are supplied, reconstruction also reports frequent modern
dialect pronunciations for each cluster.
"""

__author__ = '黄艺华 <lernanto@foxmail.com>'

import argparse
import json
import logging
import os
from typing import Sequence

import numpy
import pandas
from scipy import linalg
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline

from .. import datasets, embedding, preprocess


logger = logging.getLogger(__name__)

FEATURES = ("initial", "final", "tone")


def select_dialects(
    dataset: datasets.Dataset,
    dialect_ids: Sequence[str]
) -> numpy.ndarray:
    """Validate and return the requested dialect IDs.

    Parameters
    ----------
    dataset : sincomp.datasets.Dataset
        Dataset containing the available dialect metadata.
    dialect_ids : Sequence[str]
        Requested dialect identifiers.

    Returns
    -------
    numpy.ndarray
        Validated dialect identifiers.

    Raises
    ------
    ValueError
        If a dialect is unknown or no dialect is selected.
    """

    available = pandas.Index(dataset.dialects.index)
    selected = pandas.Index(dialect_ids)
    missing = selected.difference(available)
    if len(missing) > 0:
        raise ValueError(
            f"Unknown dialect IDs: {', '.join(map(str, missing))}"
        )
    if len(selected) == 0:
        raise ValueError("At least one dialect must be selected.")
    return selected.to_numpy()


def fit_encoder(
    wide: pandas.DataFrame,
    embedding_size: int
) -> Pipeline:
    """Fit the character vectorizer and character embedding.

    Parameters
    ----------
    wide : pandas.DataFrame
        Wide character-by-dialect segment table.
    embedding_size : int
        Maximum dimension of the character embedding.

    Returns
    -------
    sklearn.pipeline.Pipeline
        Fitted character vectorizer and embedding pipeline.
    """

    encoder = make_pipeline(
        embedding.CharacterVectorizer(),
        embedding.CharacterEmbedding(embedding_size=embedding_size)
    )
    encoder.fit(wide)
    return encoder


def get_feature_embeddings(
    encoder: Pipeline,
    wide: pandas.DataFrame
) -> dict[str, numpy.ndarray]:
    """Extract one dense embedding matrix for each aligned segment type.

    Parameters
    ----------
    encoder : sklearn.pipeline.Pipeline
        Fitted character vectorizer and embedding pipeline.
    wide : pandas.DataFrame
        Wide table transformed by ``encoder``.

    Returns
    -------
    dict[str, numpy.ndarray]
        Dense character embeddings keyed by segment type.
    """

    vectorizer = encoder.named_steps["charactervectorizer"]
    embedding_step = encoder.named_steps["characterembedding"]
    vocabularies = vectorizer.vocabularies_
    limits = numpy.cumsum(
        [0] + [len(vocabulary) for vocabulary in vocabularies]
    )
    char_embeddings = encoder.transform(wide)

    output = {}
    for feature in FEATURES:
        column_indices = [
            i for i, name in enumerate(vectorizer.feature_names_in_)
            if str(name).endswith(f"_{feature}")
        ]
        component = numpy.concatenate(
            [
                embedding_step.components_[:, limits[i]:limits[i + 1]]
                for i in column_indices
            ],
            axis=1
        )
        u, singular_values, _ = linalg.svd(component, full_matrices=False)
        output[feature] = char_embeddings @ (
            u * singular_values[None, :]
        )
    return output


def map_clusters(
    cluster_ids: numpy.ndarray,
    labels: pandas.Series,
    index: pandas.Index
) -> pandas.Series:
    """Assign the majority observed label to each cluster.

    Parameters
    ----------
    cluster_ids : numpy.ndarray
        Cluster identifier for each character.
    labels : pandas.Series
        Optional observed segment labels indexed by character ID.
    index : pandas.Index
        Character index corresponding to ``cluster_ids``.

    Returns
    -------
    pandas.Series
        Majority label assigned to each cluster, or missing when unavailable.
    """

    labels = labels.reindex(index)
    mask = labels.notna()
    output = pandas.Series(index=index, dtype=object)
    if mask.any():
        mapping = labels[mask].groupby(cluster_ids[mask]).agg(
            lambda values: values.mode().iloc[0]
        )
        output[:] = mapping.reindex(cluster_ids).to_numpy()
    return output


def cluster_phone_distributions(
    data: pandas.DataFrame,
    result: pandas.DataFrame
) -> pandas.DataFrame:
    """Return the full phone distribution of each cluster.

    Each long-form row with a non-empty segment value counts as one
    recorded sample for a character, so a character recorded in several
    dialects contributes one sample per dialect. The distribution of a
    cluster is dialect independent: it aggregates the phones of all
    the characters of the character set, no matter which dialect
    recorded them.

    Parameters
    ----------
    data : pandas.DataFrame
        Long-form dialect data with ``did``, ``cid`` and segment columns;
        characters without a character ID must have received their
        written form as ID.
    result : pandas.DataFrame
        Cluster identifiers indexed by character ID, with one
        ``<feature>_cluster`` column per segment type.

    Returns
    -------
    pandas.DataFrame
        Phone counts indexed by feature, cluster and phone, with
        ``sample_count`` (recorded samples of the phone),
        ``character_count`` (distinct characters) and ``total``
        (recorded samples of the cluster).
    """

    rows = []
    for feature in FEATURES:
        clusters = result[f"{feature}_cluster"]
        phones = data[["did", "cid", feature]].rename(
            columns={feature: "phone"}
        )
        phones = phones[
            phones["phone"].notna() & (phones["phone"] != "")
        ]
        phones["cluster"] = phones["cid"].map(clusters)
        phones = phones.dropna(subset=["cluster"])
        phones["cluster"] = phones["cluster"].astype(int)
        # count each phone at most once per recorded sample
        phones = phones.drop_duplicates(["cluster", "did", "cid", "phone"])
        totals = phones.groupby("cluster").size()
        samples = phones.groupby(["cluster", "phone"]).size()
        characters = phones.groupby(["cluster", "phone"])[
            "cid"
        ].nunique()
        for (cluster_id, phone), count in samples.items():
            rows.append({
                "feature": feature,
                "cluster": int(cluster_id),
                "phone": phone,
                "sample_count": int(count),
                "character_count": int(
                    characters.loc[(cluster_id, phone)]
                ),
                "total": int(totals.loc[cluster_id])
            })
    return pandas.DataFrame(
        rows,
        columns=[
            "feature",
            "cluster",
            "phone",
            "sample_count",
            "character_count",
            "total"
        ]
    ).set_index(["feature", "cluster", "phone"]).sort_index()


def assign_cluster_phones(
    distributions: pandas.DataFrame,
    min_phone_proportion: float = 0.1
) -> pandas.Series:
    """Assign one phone to each cluster.

    Cluster phones are assigned greedily so that different clusters
    prefer different phones, following the phoneme identification rules
    of the machine-assisted reconstruction method: clusters are
    processed by decreasing number of recorded samples, and each cluster
    claims its most frequent phone covering at least
    ``max(min_phone_proportion * total, 2)`` recorded samples that has
    not been claimed by another cluster. When every remaining phone of a
    cluster has already been claimed, the cluster is assigned its most
    frequent remaining phone with a numeric suffix to distinguish it
    from the other clusters sharing that phone.

    Parameters
    ----------
    distributions : pandas.DataFrame
        Phone distributions as returned by
        :func:`cluster_phone_distributions`.
    min_phone_proportion : float, default=0.1
        Minimum proportion of recorded samples covering a phone to be
        considered for assignment.

    Returns
    -------
    pandas.Series
        Assigned phone of each cluster, indexed by feature and cluster.
    """

    if not 0 <= min_phone_proportion <= 1:
        raise ValueError("min_phone_proportion must be between 0 and 1.")

    assignments = {}
    claims: dict[str, dict[str, list]] = {}
    totals = distributions.groupby(level=[0, 1])["total"].first()
    for key in sorted(totals.index, key=lambda key: (-totals[key], key)):
        claimed = claims.setdefault(key[0], {})
        group = distributions.xs(key, level=[0, 1])
        threshold = max(min_phone_proportion * group["total"].iloc[0], 2)
        phones = group[group["sample_count"] >= threshold].index
        for phone in phones:
            claimed.setdefault(phone, []).append(key)
            assignments[key] = phone
            break
        else:
            if len(phones) > 0:
                # every remaining phone is claimed; share the most
                # frequent one
                phone = phones[0]
                claimed[phone].append(key)
                assignments[key] = phone
    # distinguish clusters sharing a phone with subscript digits,
    # starting at ₁ for the first claimant
    subscript_digits = "₀₁₂₃₄₅₆₇₈₉"
    for phones in claims.values():
        for phone, claimants in phones.items():
            if len(claimants) > 1:
                for n, key in enumerate(claimants, start=1):
                    assignments[key] = phone + "".join(
                        subscript_digits[int(d)] for d in str(n)
                    )
    index = pandas.MultiIndex.from_tuples(
        assignments.keys(), names=["feature", "cluster"]
    )
    return pandas.Series(
        assignments.values(), index=index, name="phone"
    )


def summarize_cluster_phones(
    distributions: pandas.DataFrame,
    min_phone_proportion: float = 0.1
) -> pandas.DataFrame:
    """Summarize frequent modern dialect phones for each cluster.

    This is a display-oriented view of the phone distributions: only
    phones covering at least ``max(min_phone_proportion * total, 2)``
    recorded samples of the cluster are shown.

    Parameters
    ----------
    distributions : pandas.DataFrame
        Phone distributions as returned by
        :func:`cluster_phone_distributions`.
    min_phone_proportion : float, default=0.1
        Minimum proportion of recorded samples covering a phone to be
        displayed.

    Returns
    -------
    pandas.DataFrame
        Phone distribution indexed by feature and cluster.
    """

    if not 0 <= min_phone_proportion <= 1:
        raise ValueError("min_phone_proportion must be between 0 and 1.")

    rows = []
    for key, group in distributions.groupby(level=[0, 1], sort=False):
        feature, cluster_id = key
        threshold = max(min_phone_proportion * group["total"].iloc[0], 2)
        selected = group[
            group["sample_count"] >= threshold
        ].reset_index("phone")
        rows.extend({
            "feature": feature,
            "cluster": int(cluster_id),
            "phone": row.phone,
            "character_count": int(row.character_count),
            "sample_count": int(row.sample_count)
        } for _, row in selected.iterrows())
    return pandas.DataFrame(
        rows,
        columns=[
            "feature",
            "cluster",
            "phone",
            "character_count",
            "sample_count"
        ]
    ).set_index(["feature", "cluster"])


def apply_cluster_phones(
    result: pandas.DataFrame,
    assignments: pandas.Series
) -> pandas.DataFrame:
    """Add each cluster's assigned phone to the character result.

    Parameters
    ----------
    result : pandas.DataFrame
        Cluster identifiers indexed by character ID.
    assignments : pandas.Series
        Assigned phone of each cluster as returned by
        :func:`assign_cluster_phones`.

    Returns
    -------
    pandas.DataFrame
        Copy of the result with one column per feature holding the
        assigned phone of the character's cluster.
    """

    output = result.copy()
    for feature in FEATURES:
        labels = assignments.xs(feature, level="feature")
        output[feature] = output[f"{feature}_cluster"].map(labels)
    return output


def assign_unclustered_clusters(
    embeddings: numpy.ndarray,
    cluster_ids: numpy.ndarray,
    unclustered: numpy.ndarray,
) -> numpy.ndarray:
    """Assign clusters to unclustered characters character by character.

    Characters marked in ``unclustered`` (low coverage characters and
    members of clusters that are too small to keep) are each assigned
    to the cluster of their nearest clustered character with a
    nearest-neighbor classifier. The characters are assigned
    independently of each other, so the result does not depend on the
    order of the characters.

    Parameters
    ----------
    embeddings : numpy.ndarray
        Character embeddings for one segment type.
    cluster_ids : numpy.ndarray
        Cluster identifier of each character; characters marked in
        ``unclustered`` may hold any placeholder value and are
        replaced.
    unclustered : numpy.ndarray
        Boolean mask of characters without a cluster.

    Returns
    -------
    numpy.ndarray
        Cluster identifier for each character after assignment.

    Raises
    ------
    ValueError
        If every character is unclustered, so that there is no cluster
        to assign to.
    """

    if unclustered.all():
        raise ValueError("Every character is unclustered.")
    if not unclustered.any():
        return cluster_ids
    classifier = KNeighborsClassifier(
        n_neighbors=1, metric="cosine"
    ).fit(embeddings[~unclustered], cluster_ids[~unclustered])
    assigned = cluster_ids.copy()
    assigned[unclustered] = classifier.predict(embeddings[unclustered])
    return assigned


def dialect_coverage(wide: pandas.DataFrame) -> pandas.Series:
    """Return the proportion of dialects with data for each character.

    Parameters
    ----------
    wide : pandas.DataFrame
        Wide character-by-dialect segment table with one
        ``<dialect>_<feature>`` column per dialect and segment type.

    Returns
    -------
    pandas.Series
        Fraction of dialects in which each character has at least one
        non-empty segment value.
    """

    dialect_ids = {
        str(column).rsplit("_", 1)[0] for column in wide.columns
    }
    counts = pandas.Series(0, index=wide.index, dtype=int)
    for did in dialect_ids:
        columns = [
            column for column in wide.columns
            if str(column).startswith(f"{did}_")
        ]
        counts += (wide[columns] != "").any(axis=1).astype(int)
    return counts / len(dialect_ids)


def covered_characters(
    data: pandas.DataFrame,
    min_dialect_coverage: float
) -> pandas.Index:
    """Return the characters whose dialect coverage meets the minimum.

    Parameters
    ----------
    data : pandas.DataFrame
        Long-form dialect data with ``did``, ``cid`` and segment columns.
    min_dialect_coverage : float
        Minimum proportion of dialects in which a character must have
        data.

    Returns
    -------
    pandas.Index
        Character IDs meeting the minimum dialect coverage.
    """

    if not 0 <= min_dialect_coverage <= 1:
        raise ValueError("min_dialect_coverage must be between 0 and 1.")
    segments = data[list(FEATURES)]
    recorded = data[(segments.notna() & (segments != "")).any(axis=1)]
    coverage = recorded.groupby("cid")["did"].nunique() / data[
        "did"
    ].nunique()
    return coverage[coverage >= min_dialect_coverage].index


def cluster(
    data: pandas.DataFrame,
    covered: pandas.Index,
    annotations: pandas.DataFrame | None = None,
    embedding_size: int = 128,
    min_cluster_size: int = 3,
    max_cluster_distance: float = 0.1,
    max_clusters: int = 200
) -> pandas.DataFrame:
    """Cluster the segment phones of the characters.

    The long-form data is first converted to a wide character-by-dialect
    table. Characters outside ``covered`` are excluded from embedding
    fitting and clustering, but are still assigned to the nearest cluster
    and receive cluster phones. Clusters with fewer than
    ``min_cluster_size`` characters are merged into other clusters,
    except for characters with annotations, which keep their cluster.

    Parameters
    ----------
    data : pandas.DataFrame
        Long-form dialect data with ``did``, ``cid`` and segment columns;
        characters without a character ID must have received their
        written form as ID.
    covered : pandas.Index
        Character IDs that participate in embedding fitting and
        clustering, as returned by :func:`covered_characters`.
    annotations : pandas.DataFrame, default=None
        Optional manually assigned segment labels indexed by character ID.
    embedding_size : int, default=128
        Maximum dimension of the character embedding.
    min_cluster_size : int, default=3
        Minimum number of characters per cluster; smaller clusters are
        merged into other clusters.
    max_cluster_distance : float, default=0.1
        Maximum cosine distance within a cluster for agglomerative
        clustering.
    max_clusters : int, default=200
        Maximum number of clusters per segment type; when the
        distance-based clustering yields more clusters, the embeddings
        are re-clustered into exactly this number of clusters.

    Returns
    -------
    pandas.DataFrame
        Cluster identifiers and, when available, mapped segment labels,
        indexed by character ID.

    Raises
    ------
    ValueError
        If ``min_cluster_size`` is smaller than 1 or no character is
        covered.
    """

    if min_cluster_size < 1:
        raise ValueError("min_cluster_size must be at least 1.")

    # Convert the long-form data to a wide character-by-dialect table
    # with one "<dialect>_<feature>" column per dialect and segment,
    # joining the phones of each character in each dialect.
    valid = data.dropna(subset=["did", "cid"])
    wide = preprocess.transform(
        valid,
        index="cid",
        columns="did",
        values=list(FEATURES),
        aggfunc=lambda values: " ".join(values.dropna()),
        fill_value=""
    )
    wide.columns = [
        f"{did}_{feature}"
        for did, feature in wide.columns.to_flat_index()
    ]
    logger.info(
        "wide table: %d characters × %d segment columns",
        len(wide), wide.shape[1]
    )

    fit_mask = pandas.Series(wide.index.isin(covered), index=wide.index)
    if not fit_mask.any():
        raise ValueError("No character is covered.")
    logger.info(
        "fitting character embedding (size<=%d) on %d of %d characters",
        embedding_size, int(fit_mask.sum()), len(wide)
    )
    encoder = fit_encoder(wide[fit_mask], embedding_size)
    logger.info("embedding fitted; computing feature embeddings")
    feature_embeddings = get_feature_embeddings(encoder, wide)

    result = pandas.DataFrame(index=wide.index)
    fit_array = fit_mask.to_numpy()
    for feature, values in feature_embeddings.items():
        logger.info(
            "clustering %s embeddings (%d segments)",
            feature, int(fit_array.sum())
        )
        # Average-linkage agglomerative clustering groups the characters
        # whose segment embeddings are within the cosine distance
        # threshold.
        cluster_ids = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=max_cluster_distance
        ).fit_predict(values[fit_array])
        if len(set(cluster_ids)) > max_clusters:
            # too many clusters; re-cluster into the maximum number
            logger.info(
                "%s: %d clusters exceed the maximum %d; "
                "re-clustering into %d clusters",
                feature, len(set(cluster_ids)), max_clusters, max_clusters
            )
            cluster_ids = AgglomerativeClustering(
                n_clusters=max_clusters,
                metric="cosine",
                linkage="average"
            ).fit_predict(values[fit_array])
        logger.info(
            "%s: %d clusters at cosine distance %.2f",
            feature, len(set(cluster_ids)), max_cluster_distance
        )

        known_mask = None
        if annotations is not None and feature in annotations:
            known_mask = (
                annotations[feature].reindex(wide.index).notna().to_numpy()
            )
        # characters of clusters that are too small to keep are treated
        # as unclustered and assigned character by character, like
        # low coverage characters
        assigned = numpy.full(len(values), -1, dtype=cluster_ids.dtype)
        assigned[fit_array] = cluster_ids
        unclustered = ~fit_array
        low_count = int(unclustered.sum())
        counts = pandas.Series(cluster_ids).value_counts()
        small = counts[counts < min_cluster_size].index.to_numpy()
        if len(small) > 0:
            members = numpy.isin(cluster_ids, small)
            if known_mask is not None:
                members &= ~known_mask[fit_array]
            unclustered[fit_array] = members

        if unclustered.any():
            logger.debug(
                "%s: dissolving %d small clusters; assigning %d "
                "characters without a cluster (%d low coverage, "
                "%d from small clusters)",
                feature, len(small), int(unclustered.sum()), low_count,
                int(unclustered.sum()) - low_count,
            )
        assigned = assign_unclustered_clusters(
            values, assigned, unclustered
        )
        result[f"{feature}_cluster"] = assigned

        if annotations is None or feature not in annotations:
            continue
        result[feature] = map_clusters(
            assigned,
            annotations[feature],
            wide.index
        )
        logger.debug(
            "%s: mapped %d clusters to %d annotation labels",
            feature, len(set(assigned)), result[feature].nunique()
        )

    return result


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    """Add the shared reconstruction arguments.

    Parameters
    ----------
    parser : argparse.ArgumentParser
        Parser to which the arguments are added.

    Returns
    -------
    None
        The parser is modified in place.
    """

    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity for tracing the reconstruction process."
    )
    parser.add_argument("--dataset", default="MCPDict")
    parser.add_argument(
        "--dialects",
        nargs="+",
        required=False,
        help="Dialect IDs to use; all selected dialect data are used."
    )
    parser.add_argument("--annotation-file", default=None)
    parser.add_argument(
        "--min-dialect-coverage",
        type=float,
        default=0.5,
        help=(
            "Minimum proportion of dialects in which a character must "
            "have data to be used for fitting; characters below the "
            "threshold are still assigned to clusters and phones."
        )
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=3,
        help=(
            "Minimum number of characters per cluster; smaller clusters "
            "are merged into other clusters unless their characters have "
            "annotations."
        )
    )
    parser.add_argument(
        "--max-cluster-distance",
        type=float,
        default=0.1,
        help=(
            "Maximum cosine distance within a cluster for clustering."
        )
    )
    parser.add_argument(
        "--max-clusters",
        type=int,
        default=200,
        help=(
            "Maximum number of clusters per segment type; when the "
            "distance-based clustering yields more clusters, the "
            "embeddings are re-clustered into this number of clusters."
        )
    )
    parser.add_argument(
        "--min-phone-proportion",
        type=float,
        default=0.1,
        help=(
            "Minimum proportion of recorded samples covering a phone, "
            "for assignment and display."
        )
    )


def load_config(path: str) -> dict:
    """Load command-line defaults from a JSON configuration file."""

    with open(path, encoding="utf-8") as config_file:
        config = json.load(config_file)
    if not isinstance(config, dict):
        raise ValueError("Configuration must contain a JSON object.")
    return config


def build_parser(config: dict | None = None) -> argparse.ArgumentParser:
    """Build the reconstruction command-line parser.

    Parameters
    ----------
    config : dict, default=None
        Configuration values applied as argument defaults.

    Returns
    -------
    argparse.ArgumentParser
        Parser for the reconstruction command.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="JSON configuration file")
    add_common_arguments(parser)
    parser.add_argument("--output-dir", default="reconstruction")
    parser.add_argument("--embedding-size", type=int, default=128)

    if config is not None:
        sections = {"defaults", "reconstruct", "evaluate"}
        defaults = {
            key: value for key, value in config.items()
            if key not in sections
        }
        defaults.update(config.get("defaults", {}))
        if not isinstance(defaults, dict):
            raise ValueError(
                "Configuration sections must contain JSON objects."
            )
        parser.set_defaults(**defaults)
    return parser


def load_annotations(path: str | None) -> pandas.DataFrame | None:
    """Load annotations indexed by character ID.

    Parameters
    ----------
    path : str or os.PathLike or None
        Path to a CSV file with a ``cid`` index column. If None, no annotations
        are loaded.

    Returns
    -------
    pandas.DataFrame or None
        Annotation table indexed by character ID.
    """

    if path is None:
        return None
    annotations = pandas.read_csv(path, dtype=str)
    if "cid" not in annotations:
        if "character" not in annotations:
            raise ValueError(
                "Annotations without cid must contain a character column."
            )
        annotations = annotations.assign(cid=annotations["character"])
    annotations = annotations.set_index("cid")
    logger.info("loaded %d annotations from %s", len(annotations), path)
    return annotations


def main(argv: Sequence[str] | None = None) -> int:
    """Run the reconstruction command-line interface.

    Parameters
    ----------
    argv : Sequence[str], default=None
        Command-line arguments; defaults to ``sys.argv``.

    Returns
    -------
    int
        Exit code; results are written to the selected output directory.
    """

    preliminary_parser = argparse.ArgumentParser(add_help=False)
    preliminary_parser.add_argument("--config")
    preliminary_args, _ = preliminary_parser.parse_known_args(argv)
    config = None
    if preliminary_args.config is not None:
        config = load_config(preliminary_args.config)
    parser = build_parser(config)
    args = parser.parse_args(argv)
    if not args.dialects:
        parser.error("the following arguments are required: --dialects")

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info(
        "dataset=%s dialects=%d", args.dataset, len(args.dialects)
    )

    os.makedirs(args.output_dir, exist_ok=True)
    dataset = datasets.get(args.dataset)
    dialect_ids = select_dialects(dataset, args.dialects)
    data = dataset.select(dialect_ids).data
    has_character_ids = "cid" in data.columns
    if not has_character_ids:
        # without character IDs, characters act as their own IDs
        data = data.assign(cid=data["character"])
    annotations = load_annotations(args.annotation_file)
    covered = covered_characters(data, args.min_dialect_coverage)
    logger.info(
        "covering %d of %d characters at dialect coverage >= %.2f",
        len(covered), data["cid"].nunique(), args.min_dialect_coverage
    )

    result = cluster(
        data,
        covered,
        annotations=annotations,
        embedding_size=args.embedding_size,
        min_cluster_size=args.min_cluster_size,
        max_cluster_distance=args.max_cluster_distance,
        max_clusters=args.max_clusters
    )

    if has_character_ids:
        characters = (
            dataset.characters["character"]
            .reindex(result.index)
            .to_numpy()
        )
    else:
        characters = result.index.to_numpy()
    result.insert(0, "character", characters)

    if annotations is None:
        # assign cluster phones and propagate them to characters
        covered_data = data[data["cid"].isin(covered)]
        distributions = cluster_phone_distributions(covered_data, result)
        assignments = assign_cluster_phones(
            distributions, args.min_phone_proportion
        )
        logger.info("assigned phones to %d clusters", len(assignments))
        result = apply_cluster_phones(result, assignments)
        cluster_output = summarize_cluster_phones(
            distributions, args.min_phone_proportion
        ).reset_index()
        cluster_output.to_csv(
            clusters_file := os.path.join(
                args.output_dir, "cluster_phones.csv"
            ),
            columns=[
                "feature",
                "cluster",
                "phone",
                "character_count",
                "sample_count"
            ],
            index=False,
            encoding="utf-8"
        )
        logger.info("wrote %s", clusters_file)
        assignments.rename_axis(["feature", "cluster"]).reset_index().to_csv(
            prons_file := os.path.join(args.output_dir, "clusters.csv"),
            index=False,
            encoding="utf-8"
        )
        logger.info("wrote %s", prons_file)

    output = result
    output.index.name = "cid"
    output_file = os.path.join(args.output_dir, "reconstruction.csv")
    output.to_csv(output_file, encoding="utf-8")
    logger.info("wrote %s", output_file)
    print(
        f"Reconstructed {len(result)} characters "
        f"from {len(dialect_ids)} dialects of {args.dataset}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
