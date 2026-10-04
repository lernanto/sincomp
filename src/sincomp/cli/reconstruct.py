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
import sys
from typing import Sequence

import numpy
import pandas
from scipy import linalg
from sklearn.cluster import AgglomerativeClustering
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

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
        ``<feature>_cluster`` column per segment type; characters
        without a cluster (marked with ``-1``) contribute no samples.

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
        # characters without a cluster contribute no samples
        phones = phones[phones["cluster"] >= 0]
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


def claim_group_phones(
    groups: dict,
    priority: list,
    min_phone_proportion: float = 0.1
) -> dict:
    """Assign one phone to each group by greedy claiming.

    Following the phoneme identification rules of the machine-assisted
    reconstruction method, groups are processed in the given priority
    order, and each group claims its most frequent phone covering at
    least ``max(min_phone_proportion * total, 2)`` recorded samples —
    where ``total`` is the number of recorded samples of the group —
    that has not been claimed by an earlier group. When every eligible
    phone has already been claimed, the group shares its most frequent
    phone, distinguished with a subscript digit.

    Parameters
    ----------
    groups : dict
        Recorded samples of each phone per group, keyed by group.
    priority : list
        Group keys in the order in which they claim phones.
    min_phone_proportion : float, default=0.1
        Minimum proportion of recorded samples covering a phone for it
        to be eligible for claiming.

    Returns
    -------
    dict
        Phone of each group, keyed by group; shared phones carry
        subscript digits.
    """

    assignments = {}
    claims: dict[str, list] = {}
    for group in priority:
        counts = groups[group]
        if not counts:
            continue
        threshold = max(min_phone_proportion * sum(counts.values()), 2)
        eligible = [
            phone for phone, n in sorted(
                counts.items(), key=lambda item: -item[1]
            )
            if n >= threshold
        ]
        for phone in eligible:
            if phone not in claims:
                break
        else:
            # every eligible phone is claimed; share the most frequent
            # phone of the group
            phone = max(counts.items(), key=lambda item: item[1])[0]
        claims.setdefault(phone, []).append(group)
        assignments[group] = phone
    for phone, claimants in claims.items():
        if len(claimants) > 1:
            for n, group in enumerate(claimants, start=1):
                assignments[group] = f"{phone}{_subscript(n)}"
    return assignments


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
    cluster has already been claimed, the cluster shares its most
    frequent phone with the other clusters claiming it, distinguished by
    a subscript digit.

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
    totals = distributions.groupby(level=[0, 1])["total"].first()
    for feature in FEATURES:
        keys = [key for key in totals.index if key[0] == feature]
        groups = {
            key: {
                phone: int(n) for phone, n in distributions.xs(
                    key, level=[0, 1]
                ).groupby("phone")["sample_count"].sum().items()
            }
            for key in keys
        }
        priority = sorted(keys, key=lambda key: (-totals[key], key))
        assignments.update(
            claim_group_phones(groups, priority, min_phone_proportion)
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
        total = int(group["total"].iloc[0])
        threshold = max(min_phone_proportion * total, 2)
        counts = group.groupby("phone")["sample_count"].sum()
        characters = group.groupby("phone")["character_count"].sum()
        frequent = counts[counts >= threshold].sort_values(ascending=False)
        rows.extend({
            "feature": feature,
            "cluster": int(cluster_id),
            "phone": phone,
            "character_count": int(characters[phone]),
            "sample_count": int(counts[phone])
        } for phone in frequent.index)
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
    phones: pandas.Series,
    suffix: str = ""
) -> pandas.DataFrame:
    """Add each cluster's phone or phoneme to the character result.

    Parameters
    ----------
    result : pandas.DataFrame
        Cluster identifiers indexed by character ID.
    phones : pandas.Series
        Phone or phoneme of each cluster as returned by
        :func:`assign_cluster_phones` or :func:`merge_cluster_phonemes`.
    suffix : str, default=""
        Suffix appended to the added column names (e.g. ``"_phoneme"``).

    Returns
    -------
    pandas.DataFrame
        Copy of the result with one column per feature holding the phone
        or phoneme of the character's cluster.
    """

    output = result.copy()
    for feature in FEATURES:
        labels = phones.xs(feature, level="feature")
        output[f"{feature}{suffix}"] = output[f"{feature}_cluster"].map(
            labels
        )
    return output


def map_annotation_phones(
    result: pandas.DataFrame,
    annotations: pandas.DataFrame
) -> pandas.Series:
    """Assign each cluster the annotated phones of its characters.

    The annotations provide the phone of each cluster: characters of
    one cluster must share one phone, so each cluster takes the mode of
    the annotated phones of its characters, and clusters sharing the
    same phone are distinguished with subscript digits, larger
    clusters first.

    Parameters
    ----------
    result : pandas.DataFrame
        Cluster identifiers indexed by character ID, with one
        ``<feature>_cluster`` column per segment type.
    annotations : pandas.DataFrame
        Manually assigned segment labels indexed by character ID.

    Returns
    -------
    pandas.Series
        Phone of each cluster, indexed by feature and cluster.
    """

    phones = {}
    for feature in FEATURES:
        annotated = annotations[feature].reindex(result.index)
        clusters = result[f"{feature}_cluster"]
        mask = annotated.notna() & (clusters >= 0)
        modes = annotated[mask].groupby(clusters[mask]).agg(
            lambda values: values.mode().iloc[0]
        )
        sizes = clusters[clusters >= 0].value_counts()
        ordered = sorted(
            modes.index, key=lambda cluster: (-int(sizes[cluster]), cluster)
        )
        claims: dict[str, list] = {}
        for cluster in ordered:
            claims.setdefault(modes.loc[cluster], []).append(cluster)
        for phone, cluster_ids in claims.items():
            shared = (
                (f"{phone}{_subscript(n)}", cluster)
                for n, cluster in enumerate(cluster_ids, start=1)
            ) if len(cluster_ids) > 1 else (
                (phone, cluster) for cluster in cluster_ids
            )
            for label, cluster in shared:
                phones[(feature, int(cluster))] = label
    index = pandas.MultiIndex.from_tuples(
        phones.keys(), names=["feature", "cluster"]
    )
    return pandas.Series(phones.values(), index=index, name="phone")


def _subscript(number: int) -> str:
    """Return the subscript form of a decimal number."""
    digits = "₀₁₂₃₄₅₆₇₈₉"
    return "".join(digits[int(digit)] for digit in str(number))


def merge_cluster_phonemes(
    original: pandas.DataFrame,
    embeddings: dict[str, numpy.ndarray],
    distributions: pandas.DataFrame,
    max_merge_tolerance: int = 0,
    max_merge_distance: float = 0.5,
    min_phone_proportion: float = 0.1
) -> pandas.Series:
    """Merge clusters into phonemes by complementary distribution.

    Following the phoneme merging rules of the machine-assisted
    reconstruction method, two clusters of one feature are assigned the
    same phoneme when their character sets are in complementary
    distribution — there is no combination of the clusters of all the
    other features whose character sets both contain characters beyond
    ``max_merge_tolerance`` — and the average cosine distance between
    the phone vectors of their characters is at most
    ``max_merge_distance``. Only the characters of the kept clusters
    participate in the merging; characters without a cluster (low
    coverage characters and members of clusters that are too small to
    keep) are assigned to clusters and inherit their phonemes after the
    merging. Merging is greedy across all features: at every
    step the closest feasible pair among the clusters of all features
    is merged, so every choice is made against the latest environments
    and does not depend on the order of the features. After the
    merging, the phoneme value of each group is claimed greedily from
    the phone distribution of its member clusters, and groups sharing
    a value are distinguished with subscript digits.

    Parameters
    ----------
    original : pandas.DataFrame
        Cluster identifiers indexed by character ID, with one
        ``<feature>_cluster`` column per segment type; characters
        without a cluster are marked with ``-1``.
    embeddings : dict[str, numpy.ndarray]
        Dense phone vectors of each character keyed by segment type,
        as returned by :func:`reconstruct`.
    distributions : pandas.DataFrame
        Phone distributions of the kept clusters as returned by
        :func:`cluster_phone_distributions`; consulted only after the
        merging, to claim the phoneme value of each group from the
        phone distribution of its member clusters.
    max_merge_tolerance : int, default=0
        Maximum number of co-occurring characters tolerated for two
        clusters to be considered complementary.
    max_merge_distance : float, default=0.5
        Maximum average cosine distance between the phone vectors of
        two clusters to merge them.
    min_phone_proportion : float, default=0.1
        Minimum proportion of recorded samples covering a phone for it
        to be eligible as the phoneme label of a group.

    Returns
    -------
    pandas.Series
        Phoneme of each cluster, indexed by feature and cluster.
    """

    # only characters whose clusters are kept for all features
    # participate in the merging
    cluster_columns = [f"{feature}_cluster" for feature in FEATURES]
    clustered_mask = (
        original[cluster_columns] >= 0
    ).all(axis=1).to_numpy()
    covered_positions = numpy.flatnonzero(clustered_mask)
    covered_index = original.index[covered_positions]
    covered_original = original.loc[covered_index]
    if covered_index.empty:
        raise ValueError(
            "No character has all its features clustered; lower the "
            "minimum cluster size or increase the data coverage."
        )

    # environment of each cluster: the number of covered characters
    # belonging to each combination of the clusters of the other
    # features; the character sets are determined by the cluster
    # identifiers alone and do not depend on the dialect phones
    environments = {}
    for feature in FEATURES:
        others = [f"{g}_cluster" for g in FEATURES if g != feature]
        counts = covered_original.groupby(
            [f"{feature}_cluster"] + others
        ).size()
        for (cluster_id, *rest), n in counts.items():
            combo = ",".join(str(value) for value in rest)
            environments.setdefault((feature, int(cluster_id)), {})[
                combo
            ] = int(n)

    # one group per kept cluster to start with
    cluster_ids = {
        feature: sorted(pandas.unique(covered_original[f"{feature}_cluster"]))
        for feature in embeddings
    }

    # union-find state persists across steps so that already merged
    # clusters are not merged again
    parents = {
        feature: {cluster: cluster for cluster in cluster_ids[feature]}
        for feature in cluster_ids
    }

    def find(feature: str, cluster: int) -> int:
        parent = parents[feature]
        while parent[cluster] != cluster:
            parent[cluster] = parent[parent[cluster]]
            cluster = parent[cluster]
        return cluster

    other_features = {
        feature: [g for g in FEATURES if g != feature]
        for feature in FEATURES
    }

    # unit phone vectors of the covered characters per feature; the
    # average pairwise cosine distance between the covered characters
    # of two groups is 1 minus the cosine of the angle between the sums
    # of their unit vectors, so no pairwise matrix is needed
    unit_vectors = {}
    for feature, values in embeddings.items():
        vectors = values[covered_positions]
        norms = numpy.linalg.norm(vectors, axis=1, keepdims=True)
        unit_vectors[feature] = vectors / numpy.maximum(norms, 1e-12)

    def build_state(feature: str) -> dict:
        """Merging state of the groups of one feature."""

        members = {}
        env_counts = {}
        for cluster in cluster_ids[feature]:
            root = find(feature, cluster)
            members.setdefault(root, []).append(cluster)
            counts = env_counts.setdefault(root, {})
            for combo, n in environments.get((feature, cluster), {}).items():
                counts[combo] = counts.get(combo, 0) + int(n)
        roots = sorted(members)
        cluster_of = original.loc[covered_index, f"{feature}_cluster"]
        positions = {
            root: numpy.flatnonzero(cluster_of.to_numpy() == root)
            for root in roots
        }
        sizes = numpy.array(
            [len(positions[root]) for root in roots], dtype=float
        )
        # average pairwise distance between the covered characters of
        # each pair of groups
        sums = numpy.vstack([
            unit_vectors[feature][positions[root]].sum(axis=0)
            for root in roots
        ])
        group_distances = 1.0 - sums @ sums.T / numpy.maximum(
            numpy.outer(sizes, sizes), 1e-12
        )
        return {
            "members": members,
            "env_counts": env_counts,
            "roots": roots,
            "pos": {root: i for i, root in enumerate(roots)},
            "sizes": sizes,
            "active": numpy.ones(len(roots), dtype=bool),
            "distances": group_distances,
        }

    states = {
        feature: build_state(feature) for feature in cluster_ids
    }
    logger.info(
        "merging clusters into phonemes from %s",
        {feature: len(cluster_ids[feature]) for feature in cluster_ids},
    )

    def env_key(feature: str, combo: str) -> tuple:
        """Canonical environment key: roots of the other features."""

        return tuple(
            find(g, int(cluster))
            for g, cluster in zip(other_features[feature], combo.split(","))
        )

    def conflicts(feature: str, root1: int, root2: int) -> int:
        """Covered characters co-occurring with both groups."""

        env_counts = states[feature]["env_counts"]
        counts1 = {}
        for combo, n in env_counts[root1].items():
            key = env_key(feature, combo)
            counts1[key] = counts1.get(key, 0) + n
        counts2 = {}
        for combo, n in env_counts[root2].items():
            key = env_key(feature, combo)
            counts2[key] = counts2.get(key, 0) + n
        if len(counts1) > len(counts2):
            counts1, counts2 = counts2, counts1
        return sum(
            min(n, counts2[key])
            for key, n in counts1.items()
            if key in counts2
        )

    # greedily merge the closest feasible pair among the clusters of
    # all features; merging one feature changes the environments of the
    # others, so every choice is made against the latest state. Group
    # distances start from the precomputed character distances and are
    # updated by the average-linkage rule, and conflicts are only
    # evaluated for candidate pairs in increasing distance until the
    # first feasible pair is found.
    while True:
        best = None
        for feature in FEATURES:
            state = states[feature]
            active = numpy.flatnonzero(state["active"])
            if len(active) < 2:
                continue
            group_distances = state["distances"]
            sub = group_distances[numpy.ix_(active, active)]
            ii, jj = numpy.triu_indices(len(active), k=1)
            order = numpy.argsort(sub[ii, jj], kind="stable")
            for index in order:
                distance = float(sub[ii[index], jj[index]])
                if distance > max_merge_distance:
                    break
                r1 = state["roots"][active[ii[index]]]
                r2 = state["roots"][active[jj[index]]]
                if conflicts(feature, r1, r2) <= max_merge_tolerance:
                    if best is None or distance < best[0]:
                        best = (distance, feature, r1, r2)
                    break
        if best is None:
            break
        distance, feature, r1, r2 = best
        state = states[feature]
        members = state["members"]
        env_counts = state["env_counts"]
        logger.debug(
            "merge %s clusters %s and %s (conflict=%d, distance=%.3f)",
            feature, members[r1], members[r2],
            conflicts(feature, r1, r2), distance,
        )
        parents[feature][r2] = r1
        members[r1].extend(members.pop(r2))
        for combo, n in env_counts.pop(r2).items():
            env_counts[r1][combo] = (
                env_counts[r1].get(combo, 0) + n
            )
        # average-linkage update of the group distances
        p1, p2 = state["pos"][r1], state["pos"][r2]
        s1, s2 = state["sizes"][p1], state["sizes"][p2]
        group_distances = state["distances"]
        group_distances[p1, :] = (
            s1 * group_distances[p1, :] + s2 * group_distances[p2, :]
        ) / (s1 + s2)
        group_distances[:, p1] = group_distances[p1, :]
        state["sizes"][p1] = s1 + s2
        state["active"][p2] = False

    logger.info(
        "merging converged: %s phoneme groups",
        {feature: len(states[feature]["members"]) for feature in FEATURES},
    )

    # after the merging, claim each group's phoneme from the phone
    # distribution of its member clusters, larger groups first; a phone
    # is eligible only when it covers at least the same minimum number
    # of samples as the phone assignment
    labels = {}
    for feature in FEATURES:
        members = states[feature]["members"]
        sizes = original[f"{feature}_cluster"].value_counts()
        priority = sorted(
            members,
            key=lambda root: (
                -sum(sizes[c] for c in members[root]),
                min(members[root]),
            ),
        )
        group_counts = {}
        for root, member_ids in members.items():
            counts = group_counts.setdefault(root, {})
            for cluster in member_ids:
                key = (feature, cluster)
                if key not in distributions.index:
                    continue
                phones = distributions.xs(
                    key, level=["feature", "cluster"]
                )
                for phone, n in phones.groupby("phone")[
                    "sample_count"
                ].sum().items():
                    counts[phone] = counts.get(phone, 0) + int(n)
        assigned = claim_group_phones(
            group_counts, priority, min_phone_proportion
        )
        logger.debug(
            "%s: labeled %d phoneme groups", feature, len(assigned)
        )
        for root, phone in assigned.items():
            for cluster in members[root]:
                labels[(feature, cluster)] = phone

    index = pandas.MultiIndex.from_tuples(
        labels.keys(), names=["feature", "cluster"]
    )
    return pandas.Series(labels.values(), index=index, name="phoneme")


def assign_unclustered_characters(
    result: pandas.DataFrame,
    embeddings: dict[str, numpy.ndarray]
) -> pandas.DataFrame:
    """Assign clusters to unclustered characters character by character.

    Characters without a cluster identifier (low coverage characters
    and members of clusters that are too small to keep, marked with
    ``-1``) are each assigned to the cluster of their nearest clustered
    character with a nearest-neighbor classifier. The characters are
    assigned independently of each other, so the result does not depend
    on the order of the characters. They then inherit the phones and
    phonemes of their new clusters.

    Parameters
    ----------
    result : pandas.DataFrame
        Cluster identifiers indexed by character ID, with one
        ``<feature>_cluster`` column per segment type; characters
        without a cluster are marked with ``-1``.
    embeddings : dict[str, numpy.ndarray]
        Dense phone vectors of each character keyed by segment type,
        aligned with the result index, as returned by :func:`cluster`.

    Returns
    -------
    pandas.DataFrame
        Copy of the result with every character assigned to a cluster.

    Raises
    ------
    ValueError
        If every character of a feature is unclustered, so that there
        is no cluster to assign to.
    """

    output = result.copy()
    for feature, values in embeddings.items():
        column = f"{feature}_cluster"
        cluster_ids = output[column].to_numpy()
        unclustered = cluster_ids < 0
        if unclustered.all():
            raise ValueError(f"Every {feature} character is unclustered.")
        if not unclustered.any():
            continue
        classifier = KNeighborsClassifier(
            n_neighbors=1, metric="cosine"
        ).fit(values[~unclustered], cluster_ids[~unclustered])
        cluster_ids = cluster_ids.copy()
        cluster_ids[unclustered] = classifier.predict(values[unclustered])
        output[column] = cluster_ids
    return output


def predict_annotated_phonemes(
    result: pandas.DataFrame,
    annotations: pandas.DataFrame
) -> pandas.DataFrame:
    """Predict the annotated phoneme of each character from its clusters.

    The annotations are treated as already merged phonemes, but a
    cluster does not necessarily map to a single phoneme: the phoneme
    may depend on the other features (conditional divergence), so a
    decision tree is trained on the annotated characters to predict the
    phoneme of every character from the identifiers of the clusters of
    all its features.

    Parameters
    ----------
    result : pandas.DataFrame
        Cluster identifiers indexed by character ID, with one
        ``<feature>_cluster`` column per segment type.
    annotations : pandas.DataFrame
        Manually assigned segment labels indexed by character ID.

    Returns
    -------
    pandas.DataFrame
        Copy of the result with one ``<feature>_phoneme`` column per
        feature holding the predicted phoneme of each character.
    """

    output = result.copy()
    cluster_columns = [f"{feature}_cluster" for feature in FEATURES]
    for feature in FEATURES:
        known = annotations[feature].reindex(result.index)
        mask = known.notna()
        classifier = make_pipeline(
            OneHotEncoder(handle_unknown="ignore"),
            DecisionTreeClassifier()
        ).fit(output.loc[mask, cluster_columns], known[mask])
        output[f"{feature}_phoneme"] = classifier.predict(
            output[cluster_columns]
        )
    return output


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
    fitting and clustering, except for annotated characters, which
    always participate whatever their coverage is. Clusters with fewer
    than ``min_cluster_size`` characters are dissolved, except for
    their characters with annotations, which keep their cluster. When
    annotations are available, clusters without any annotated character
    are dissolved as well, since they would have no phone. Characters
    without a cluster are marked with ``-1``; they are assigned to
    clusters, phones and phonemes after the phoneme merging, see
    :func:`assign_unclustered_characters`.

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
    tuple[pandas.DataFrame, dict[str, numpy.ndarray]]
        Cluster identifiers and, when available, mapped segment labels,
        indexed by character ID, together with the dense segment
        embeddings of each character keyed by segment type.

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
    if annotations is not None:
        # annotated characters join the fitting and the clustering
        # whatever their dialect coverage is
        columns = [f for f in FEATURES if f in annotations.columns]
        annotated = (
            annotations[columns]
            .notna()
            .any(axis=1)
            .reindex(wide.index)
            .fillna(False)
            .astype(bool)
        )
        fit_mask |= annotated
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
        # characters without a cluster (low coverage characters and
        # members of dissolved clusters) are marked with -1 and assigned
        # to clusters, phones and phonemes after the phoneme merging;
        # annotated characters keep their cluster, and when annotations
        # are available, clusters without any annotated character have
        # no phone and are dissolved entirely
        assigned = numpy.full(len(values), -1, dtype=cluster_ids.dtype)
        counts = pandas.Series(cluster_ids).value_counts()
        members = numpy.isin(
            cluster_ids, counts[counts < min_cluster_size].index.to_numpy()
        )
        if known_mask is not None:
            members &= ~known_mask[fit_array]
            annotated = pandas.Series(known_mask[fit_array]).groupby(
                cluster_ids
            ).any()
            members |= ~numpy.isin(
                cluster_ids, annotated[annotated].index.to_numpy()
            )
        if members.any():
            dissolved = pandas.Series(cluster_ids[members]).nunique()
            cluster_ids = cluster_ids.copy()
            cluster_ids[members] = -1
            logger.debug(
                "%s: dissolving %d clusters (%d characters)",
                feature, dissolved, int(members.sum()),
            )
        assigned[fit_array] = cluster_ids
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

    return result, feature_embeddings


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
    parser.add_argument(
        "--max-merge-distance",
        type=float,
        default=0.5,
        help=(
            "Maximum cosine distance between the mean phone vectors "
            "of two clusters to merge them into one phoneme."
        )
    )
    parser.add_argument(
        "--max-merge-tolerance",
        type=int,
        default=0,
        help=(
            "Maximum number of co-occurring characters tolerated for two "
            "clusters to be considered complementary."
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


def reconstruct(argv: Sequence[str] | None = None) -> int:
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

    result, feature_embeddings = cluster(
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
        # assign cluster phones, merge them into phonemes and propagate
        # both to characters
        covered_data = data[data["cid"].isin(covered)]
        distributions = cluster_phone_distributions(covered_data, result)
        assignments = assign_cluster_phones(
            distributions, args.min_phone_proportion
        )
        logger.info("assigned phones to %d clusters", len(assignments))
        phonemes = merge_cluster_phonemes(
            result, feature_embeddings, distributions,
            args.max_merge_tolerance, args.max_merge_distance,
            args.min_phone_proportion
        )
        # characters without a cluster are assigned to clusters and
        # inherit their phones and phonemes after the merging
        result = assign_unclustered_characters(result, feature_embeddings)
        result = apply_cluster_phones(result, assignments)
        result = apply_cluster_phones(result, phonemes, suffix="_phoneme")
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
        clusters = assignments.rename_axis(
            ["feature", "cluster"]
        ).reset_index()
        keys = clusters.set_index(["feature", "cluster"]).index
        clusters.insert(3, "phoneme", keys.map(phonemes))
        clusters.to_csv(
            prons_file := os.path.join(args.output_dir, "clusters.csv"),
            index=False,
            encoding="utf-8"
        )
        logger.info("wrote %s", prons_file)
    else:
        # the annotations provide the cluster phones and act as already
        # merged phonemes; characters without a cluster are assigned
        # first, then the phoneme of each character is predicted from
        # the identifiers of its clusters
        result = assign_unclustered_characters(result, feature_embeddings)
        result = apply_cluster_phones(
            result, map_annotation_phones(result, annotations)
        )
        result = predict_annotated_phonemes(result, annotations)

    output = result
    output.index.name = "cid"
    output_file = os.path.join(args.output_dir, "reconstruction.csv")
    output.to_csv(output_file, encoding="utf-8")
    logger.info("wrote %s", output_file)
    print(
        f"Reconstructed {len(result)} characters "
        f"from {len(dialect_ids)} dialects of {args.dataset}."
    )
    for feature in FEATURES:
        print(
            f"{feature}: {result[feature + '_cluster'].nunique()} clusters / "
            f"{result[feature + '_phoneme'].nunique()} phonemes"
        )
    return 0


def extract_tree_rules(classifier: Pipeline) -> list[dict]:
    """Convert a trained one-hot decision tree into readable rules.

    Every root-to-leaf path of the tree becomes one rule: the
    ``conditions`` hold the branch decisions on the reconstructed
    phonemes (a bare category means the phoneme must match it, a
    ``!``-prefixed category means it must not), the ``phone`` is the
    majority class of the leaf and ``support`` the number of training
    samples reaching it. Because the values of one feature are
    mutually exclusive, the negative conditions of a feature are
    dropped when the path contains a positive condition of the same
    feature: the positive condition already implies them. Otherwise
    the negative conditions are rewritten as their complement among
    the observed values of the feature when that complement has fewer
    conditions, and a path whose negative conditions cover every
    observed value of a feature is dropped, since no sample can
    satisfy it.

    Parameters
    ----------
    classifier : sklearn.pipeline.Pipeline
        A fitted pipeline of a one-hot encoder and a decision tree.

    Returns
    -------
    list[dict]
        One rule per satisfiable leaf of the tree.
    """

    onehot = classifier.steps[0][1]
    estimator = classifier.steps[-1][1]
    if hasattr(estimator, "best_estimator_"):
        # a grid-search meta estimator: use the refitted best tree
        estimator = estimator.best_estimator_
    tree = estimator.tree_
    classes = estimator.classes_
    columns = [
        (feature, str(category))
        for feature, categories in zip(FEATURES, onehot.categories_)
        for category in categories
    ]

    def format_conditions(conditions: dict[int, list[str]]) -> dict:
        grouped: dict[str, list[str]] = {}
        for column, parts in conditions.items():
            if parts:
                feature = columns[column][0]
                grouped.setdefault(feature, []).extend(parts)

        # the values of one feature are mutually exclusive: a positive
        # condition implies all the negative conditions of the feature,
        # and negative conditions can be rewritten as their complement
        # among the observed values when the complement has fewer
        # conditions; a complement covering every observed value makes
        # the path unsatisfiable and drops the rule
        formatted = {}
        for feature, parts in sorted(grouped.items()):
            positives = [part for part in parts if not part.startswith("!")]
            if positives:
                selected = positives
            else:
                index = FEATURES.index(feature)
                negatives = {p[1:] for p in parts}
                remaining = sorted(
                    set(onehot.categories_[index]) - negatives
                )
                if not remaining:
                    return None
                if len(remaining) < len(negatives):
                    selected = remaining
                else:
                    selected = [f"!{value}" for value in sorted(negatives)]
            formatted[f"reconstructed_{feature}"] = "&".join(
                sorted(selected)
            )
        return formatted

    rules = []

    def walk(node: int, conditions: dict[int, list[str]]) -> None:
        if tree.children_left[node] == -1:
            # tree.value holds normalized class proportions; the number
            # of training samples is taken from n_node_samples
            values = tree.value[node][0]
            conditions = format_conditions(conditions)
            if conditions is None:
                # no sample can reach this leaf: the path excludes
                # every observed value of a feature
                return
            rules.append({
                "conditions": conditions,
                "phone": str(classes[int(values.argmax())]),
                "support": int(tree.n_node_samples[node]),
            })
            return
        column = tree.feature[node]
        category = columns[column][1]
        # the tree branches on ``x <= 0.5`` for one-hot columns, so the
        # left subtree is the absent case and the right subtree the
        # present case
        left = {key: list(parts) for key, parts in conditions.items()}
        right = {key: list(parts) for key, parts in conditions.items()}
        left.setdefault(column, []).append(f"!{category}")
        right.setdefault(column, []).append(category)
        walk(tree.children_left[node], left)
        walk(tree.children_right[node], right)

    walk(0, {})
    return rules


def train_pronunciation_rules(
    reconstruction: pandas.DataFrame,
    samples: pandas.DataFrame
) -> tuple[dict[str, list], dict[str, Pipeline | None]]:
    """Train the pronunciation rules of one descendant dialect.

    For every feature, a decision tree is trained to predict the
    dialect phone of a character from the reconstructed phonemes of
    all its features, and the tree is converted into readable rules.
    Every recorded phone is one training sample, so a character with
    several phones in the dialect contributes one sample per phone.

    Parameters
    ----------
    reconstruction : pandas.DataFrame
        Reconstruction result indexed by character ID, with one
        ``<feature>_phoneme`` column per segment type, as written by
        the reconstruct command.
    samples : pandas.DataFrame
        Long-form phones of the dialect; the ``cid`` column maps each
        phone to its character.

    Returns
    -------
    tuple[dict[str, list], dict[str, Pipeline | None]]
        The rules of each feature — every rule holds the ``conditions``
        on the reconstructed phonemes, the resulting ``phone`` and the
        ``support`` (the number of training samples) — together with
        the fitted classifier of each feature; the classifier is None
        when the classes of fewer than two phones occur at least
        twice and the rules reduce to a single majority phone.
    """

    phoneme_columns = [f"{feature}_phoneme" for feature in FEATURES]

    # the reconstructed phonemes of every recorded phone; a character
    # with several phones repeats once per phone
    rows = reconstruction.reindex(samples["cid"].to_numpy())
    rows.index = samples.index

    rules = {}
    classifiers: dict[str, Pipeline | None] = {}

    for feature in FEATURES:
        mask = (
            samples[feature].notna().to_numpy()
            & rows[phoneme_columns].notna().all(axis=1).to_numpy()
        )
        x = rows.loc[mask, phoneme_columns].astype(str)
        y = samples.loc[mask, feature].astype(str)

        if len(x) == 0:
            rules[feature] = []
            classifiers[feature] = None
            continue

        # samples with a singleton class are likely noise: they can
        # never dominate a leaf of two or more samples and would only
        # disturb the split selection, so they are excluded from
        # training; the pruning strength is tuned on the classes with
        # at least 5 samples, so that the stratified cross-validation
        # keeps at least 5 folds, while the classes with 2 to 4 samples
        # still participate in the final training
        counts = y.value_counts()
        common = y.isin(counts[counts >= 2].index)
        x_fit, y_fit = x[common], y[common]
        parameters = {}

        fit_counts = y_fit.value_counts()
        tune = y_fit.isin(fit_counts[fit_counts >= 5].index)
        tune_counts = y_fit[tune].value_counts()
        if len(tune_counts) >= 2:
            # stratified folds cannot have more splits than the least
            # populated class
            folds = min(10, int(tune_counts.min()))
            x_tune, y_tune = x_fit[tune], y_fit[tune]

            # enumerate the distinct pruned trees along the
            # cost-complexity pruning path instead of a blind grid:
            # every candidate alpha induces a different subtree and the
            # path is adaptive to the dataset
            encoder = OneHotEncoder(handle_unknown="ignore")
            x_encoded = encoder.fit_transform(x_tune)
            path = DecisionTreeClassifier(
                min_samples_leaf=2
            ).cost_complexity_pruning_path(x_encoded, y_tune)
            alphas = path.ccp_alphas
            step = max(1, len(alphas) // 8)
            candidates = alphas[::step].tolist()
            logger.debug(
                "%s: tuning ccp_alpha over %d candidates from a pruning "
                "path of %d splits on %d of %d samples",
                feature, len(candidates), len(alphas), len(y_tune),
                len(y_fit),
            )
            search = GridSearchCV(
                DecisionTreeClassifier(min_samples_leaf=2),
                param_grid={"ccp_alpha": candidates},
                cv=folds
            ).fit(x_encoded, y_tune)
            parameters = {"ccp_alpha": float(search.best_params_["ccp_alpha"])}
            logger.debug(
                "%s: best ccp_alpha=%.6g, cv accuracy %.3f over %d folds",
                feature, parameters["ccp_alpha"], search.best_score_, folds,
            )

        if len(y_fit) >= 2:
            classifier = make_pipeline(
                OneHotEncoder(handle_unknown="ignore"),
                DecisionTreeClassifier(
                    min_samples_leaf=2, **parameters
                ),
            ).fit(x_fit, y_fit)
            rules[feature] = extract_tree_rules(classifier)
            classifiers[feature] = classifier
            supports = sorted(rule["support"] for rule in rules[feature])
            logger.info(
                "%s: %d rules over %d of %d samples "
                "(%d singleton samples excluded), support %d..%d",
                feature, len(supports), len(y_fit), len(y),
                len(y) - len(y_fit), supports[0], supports[-1],
            )
        else:
            # every phone is a singleton (or none): a single rule
            # with the majority phone covers the whole feature
            rules[feature] = [{
                "conditions": {},
                "phone": y.mode().iloc[0],
                "support": int(len(y)),
            }]
            classifiers[feature] = None

    return rules, classifiers


def build_inference_parser(
    config: dict | None = None
) -> argparse.ArgumentParser:
    """Build the inference subcommand parser.

    Parameters
    ----------
    config : dict, default=None
        Configuration values applied as argument defaults; the
        ``inference`` section applies to this subcommand only, the
        other sections and top-level keys follow the reconstruct
        command.

    Returns
    -------
    argparse.ArgumentParser
        Parser for the inference subcommand.
    """

    parser = argparse.ArgumentParser(
        prog="reconstruct inference",
        description=(
            "Infer the pronunciation rules of descendant dialects from the "
            "reconstructed phonemes with decision trees"
        )
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="Logging verbosity for tracing the inference process."
    )
    parser.add_argument("--config", help="JSON configuration file")
    parser.add_argument("--dataset", default="MCPDict")
    parser.add_argument(
        "--dialects",
        nargs="+",
        required=False,
        help="Descendant dialects whose phones train the rules."
    )
    parser.add_argument(
        "--reconstruction-file",
        default="reconstruction.csv",
        help=(
            "Reconstruction result file as written by the reconstruct "
            "command."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="reconstruction",
        help="Directory the rule files are written to."
    )
    if config is not None:
        sections = {"defaults", "reconstruct", "evaluate", "inference"}
        defaults = {
            key: value for key, value in config.items()
            if key not in sections
        }
        defaults.update(config.get("defaults", {}))
        if not isinstance(defaults, dict):
            raise ValueError(
                "Configuration sections must contain JSON objects."
            )
        defaults.update(config.get("inference", {}))
        parser.set_defaults(**defaults)
    return parser


def inference(argv: Sequence[str] | None = None) -> int:
    """Run the inference command-line interface.

    For every selected descendant dialect, a decision tree is trained
    to predict the phones of the dialect from the reconstructed
    phonemes. The rules of every dialect are written into
    ``<output-dir>/rules.csv`` with leading ``did`` and ``feature``
    columns marking the dialect and the initial, final and tone rules,
    the predicted phones of every character of every dialect are
    written into ``<output-dir>/predictions.csv``, and a run report
    with the accuracy of the predictions against the recorded phones
    is reported to the standard output.

    Parameters
    ----------
    argv : Sequence[str], default=None
        Command-line arguments; defaults to ``sys.argv``.

    Returns
    -------
    int
        Exit code; the rule files and the prediction table are written
        to the output directory.
    """

    preliminary_parser = argparse.ArgumentParser(add_help=False)
    preliminary_parser.add_argument("--config")
    preliminary_args, _ = preliminary_parser.parse_known_args(argv)
    config = None
    if preliminary_args.config is not None:
        config = load_config(preliminary_args.config)
    parser = build_inference_parser(config)
    args = parser.parse_args(argv)
    if not args.dialects:
        parser.error("the following arguments are required: --dialects")
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    reconstruction = pandas.read_csv(
        args.reconstruction_file, index_col="cid"
    )
    dataset = datasets.get(args.dataset)
    dialect_ids = select_dialects(dataset, args.dialects)
    phoneme_columns = [f"{feature}_phoneme" for feature in FEATURES]
    valid = reconstruction[phoneme_columns].notna().all(axis=1)
    os.makedirs(args.output_dir, exist_ok=True)

    if "name" in dataset.dialects.columns:
        names = dataset.dialects["name"]
    else:
        names = pandas.Series(dialect_ids, index=dialect_ids)

    predictions = []
    accuracies = []
    rule_frames = []
    rule_table = []

    for did in dialect_ids:
        # iterate over the dialects; each dialect is selected once and
        # every recorded phone is one row, so a character with
        # several phones contributes one row per phone
        samples = dataset.select([did]).data
        if "cid" not in samples.columns:
            # datasets without character IDs use the written form;
            # unattested written forms (e.g. the placeholder □) have no
            # character ID and cannot be predicted
            samples = samples.assign(cid=samples["character"])
        # phones without a character cannot be aligned with the
        # reconstruction and are not predictable
        samples = samples.dropna(subset=["cid"])

        rules, classifiers = train_pronunciation_rules(
            reconstruction, samples
        )
        rule_table.append({
            "did": did,
            "name": names.loc[did],
            **{feature: len(rules[feature]) for feature in FEATURES},
        })
        logger.info(
            "%s: %d phones, %d rules",
            did,
            len(samples),
            sum(len(rules[feature]) for feature in FEATURES),
        )

        # one block of rules per dialect, with leading did and feature
        # columns marking the dialect and the initial, final and tone
        # rules; the condition columns hold the branch decisions on the
        # reconstructed phonemes
        for feature in FEATURES:
            frame = pandas.DataFrame({
                f"reconstructed_{reconstructed_feature}": [
                    rule["conditions"].get(
                        f"reconstructed_{reconstructed_feature}", ""
                    )
                    for rule in rules[feature]
                ]
                for reconstructed_feature in FEATURES
            })
            frame.insert(0, "feature", feature)
            frame.insert(0, "did", did)
            frame["phone"] = [rule["phone"] for rule in rules[feature]]
            frame["support"] = [rule["support"] for rule in rules[feature]]
            rule_frames.append(frame)

        # predict the phones of every character from the reconstructed
        # phonemes
        predicted = pandas.DataFrame(
            index=reconstruction.index, columns=FEATURES, dtype=object
        )
        for feature in FEATURES:
            classifier = classifiers[feature]
            if classifier is None:
                predicted[feature] = rules[feature][0]["phone"]
            else:
                predicted.loc[valid, feature] = classifiers[feature].predict(
                    reconstruction.loc[valid, phoneme_columns].astype(str)
                )

        # the recorded phones of the dialect for comparison, aligned
        # with the reconstructed phonemes and the predicted phones
        # through the character IDs
        reconstructed = reconstruction.reindex(samples["cid"].to_numpy())
        predicted_rows = predicted.reindex(samples["cid"].to_numpy())
        characters = (
            reconstructed["character"]
            if "character" in reconstructed.columns
            else pandas.Series(
                samples["cid"].to_numpy(), index=samples.index
            )
        )
        # the notes of the raw data, if the data provides a note column
        notes = samples["note"].fillna("") if "note" in samples.columns else ""

        # one row per recorded phone: the recorded phones first,
        # then the reconstructed phonemes and the predicted phones,
        # with the notes for manual checking
        frame = pandas.DataFrame({
            "did": did,
            "name": names.loc[did],
            "cid": samples["cid"].to_numpy(),
            "character": characters.to_numpy(),
            "initial": samples["initial"].to_numpy(),
            "final": samples["final"].to_numpy(),
            "tone": samples["tone"].to_numpy(),
            "note": notes,
            "reconstructed_initial": reconstructed[
                "initial_phoneme"
            ].to_numpy(),
            "reconstructed_final": reconstructed[
                "final_phoneme"
            ].to_numpy(),
            "reconstructed_tone": reconstructed[
                "tone_phoneme"
            ].to_numpy(),
            "predicted_initial": predicted_rows["initial"].to_numpy(),
            "predicted_final": predicted_rows["final"].to_numpy(),
            "predicted_tone": predicted_rows["tone"].to_numpy(),
        })
        predictions.append(frame)

        # report the accuracy against the recorded phones
        accuracy = {"did": did}

        for feature in FEATURES:
            mask = (
                samples[feature].notna().to_numpy()
                & predicted_rows[feature].notna().to_numpy()
            )
            accuracy[feature] = (
                float(
                    (
                        predicted_rows.loc[mask, feature].to_numpy()
                        == samples.loc[mask, feature].to_numpy()
                    ).mean()
                )
                if mask.any()
                else float("nan")
            )
        accuracies.append(accuracy)

    combined = pandas.concat(predictions, ignore_index=True)
    predictions_file = os.path.join(args.output_dir, "predictions.csv")
    combined.to_csv(predictions_file, index=False, encoding="utf-8")
    logger.info("wrote %s", predictions_file)
    rules_file = os.path.join(args.output_dir, "rules.csv")
    pandas.concat(rule_frames, ignore_index=True).to_csv(
        rules_file, index=False, encoding="utf-8"
    )
    logger.info("wrote %s", rules_file)
    report = pandas.DataFrame(accuracies).set_index("did")
    report.loc["mean"] = report.mean()
    print(
        f"Inferred pronunciation rules of {len(dialect_ids)} dialects "
        f"into {args.output_dir}."
    )
    print(f"dialects: {len(dialect_ids)}")
    print(f"characters: {combined['cid'].nunique()}")
    print(f"samples: {len(combined)}")
    print("rules per dialect:")
    rule_frame = pandas.DataFrame(rule_table).set_index(["did", "name"])
    rule_frame["all"] = rule_frame.sum(axis=1)
    print(rule_frame.to_string())
    print("phone prediction accuracy:")
    print(report.to_string(float_format=lambda value: f"{value:.3f}"))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch the reconstruction command-line interface.

    The reconstruct subcommand is the default, so that the
    human-friendly command line keeps working without a subcommand; the
    inference subcommand infers the pronunciation rules of descendant dialects
    from a reconstruction result.
    """

    if argv is None:
        argv = sys.argv[1:]
    argv = list(argv)
    if argv and argv[0] == "reconstruct":
        argv = argv[1:]
    if argv and argv[0] == "inference":
        return inference(argv[1:])
    return reconstruct(argv)


if __name__ == "__main__":
    raise SystemExit(main())
