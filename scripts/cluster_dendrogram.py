# -*- coding: utf-8 -*-

"""Draw the dendrogram of one cluster of a clustering tree.

Reads a ``<feature>_tree.csv`` file as written by the reconstruct
command, locates the subtree of the given cluster IDs — the topmost
merge node labeled with the cluster — and draws its dendrogram into
``<feature>_cluster_<cluster>_dendrogram.png`` with
``scipy.cluster.hierarchy.dendrogram``. When the subtree holds more
leaves than the branch limit, the dendrogram is truncated in the
``lastp`` mode: the lower subtrees are collapsed into single leaves,
annotated with their representative characters — the first two
leaves — followed by the character count, while the drawn leaves
hold one character each.

Usage::

    python scripts/cluster_dendrogram.py --tree-file initial_tree.csv \
        --cluster 75
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"


import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy
import pandas as pd
from scipy.cluster.hierarchy import dendrogram

# the tree files hold Chinese characters, which need a CJK font
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False


def _int(text: str) -> int | None:
    """Return the node ID of a CSV cell, ``None`` when it is empty.

    Parameters
    ----------
    text : str
        The raw CSV cell.

    Returns
    -------
    int | None
        The node ID, or ``None`` for an empty cell.
    """

    text = text.strip()
    return int(float(text)) if text else None


def read_tree(path: Path) -> dict[int, dict]:
    """Read a tree CSV file into a mapping of nodes.

    Parameters
    ----------
    path : Path
        Path to the tree CSV file.

    Returns
    -------
    dict[int, dict]
        One dict per node with the ``parent``, ``left``, ``right``,
        ``distance``, ``size``, ``cid``, ``character`` and ``cluster``.
    """

    nodes = {}
    # keep_default_na=False keeps empty cells as "" instead of NaN
    records = pd.read_csv(path, dtype=str, keep_default_na=False)
    for record in records.to_dict("records"):
        cluster = record["cluster"].strip()
        try:
            # the cluster labels may be written as floats
            number = float(cluster)
            if number.is_integer():
                cluster = str(int(number))
        except ValueError:
            pass
        cid = record["cid"].strip()
        nodes[_int(record["node"])] = {
            "parent": _int(record["parent"]),
            "left": _int(record["left"]),
            "right": _int(record["right"]),
            "distance": (
                float(record["distance"])
                if record["distance"].strip() else 0.0
            ),
            "size": int(float(record["size"])),
            "cid": cid,
            # fall back to the character ID when the tree file holds
            # no written forms (older tree files)
            "character": record.get("character", "").strip() or cid,
            "cluster": cluster,
        }
    return nodes


def cluster_root(nodes: dict[int, dict], cluster: str) -> int:
    """Return the root node of the subtree of one cluster.

    The cluster is drawn from the topmost merge node labeled with the
    cluster, below which every node belongs to the cluster; a cluster
    of one single character is its only leaf.

    Parameters
    ----------
    nodes : dict[int, dict]
        The tree nodes as returned by ``read_tree``.
    cluster : str
        The cluster ID.

    Returns
    -------
    int
        The node ID of the subtree root.

    Raises
    ------
    ValueError
        If the cluster holds no subtree.
    """

    merges = [
        node_id for node_id, node in nodes.items()
        if node["cluster"] == cluster and node["left"] is not None
    ]
    if merges:
        return max(merges, key=lambda node_id: nodes[node_id]["distance"])
    leaves = [
        node_id for node_id, node in nodes.items()
        if node["cluster"] == cluster
    ]
    if len(leaves) == 1:
        return leaves[0]
    raise ValueError(f"cluster {cluster} holds no subtree.")


def subtree_leaves(
    nodes: dict[int, dict],
    root: int,
    cache: dict[int, list[int]],
) -> list[int]:
    """Return the leaves under a node in left-to-right order.

    Parameters
    ----------
    nodes : dict[int, dict]
        The tree nodes as returned by ``read_tree``.
    root : int
        The node whose leaves are collected.
    cache : dict[int, list[int]]
        Memoization of the collected leaves per node.

    Returns
    -------
    list[int]
        The node IDs of the leaves under ``root``.
    """

    if root in cache:
        return cache[root]
    leaves = []
    stack = [root]
    while stack:
        node_id = stack.pop()
        node = nodes[node_id]
        if node["left"] is None:
            leaves.append(node_id)
        else:
            stack.append(node["right"])
            stack.append(node["left"])
    cache[root] = leaves
    return leaves


def build_linkage(
    nodes: dict[int, dict],
    root: int,
) -> tuple[numpy.ndarray, dict[int, str], int]:
    """Build the linkage matrix of one cluster subtree.

    The leaves of the subtree are renumbered from 0 in left-to-right
    order, and the merges follow in the order of their node IDs,
    which is the order of their distances, as the linkage matrix
    format requires.

    Parameters
    ----------
    nodes : dict[int, dict]
        The tree nodes as returned by ``read_tree``.
    root : int
        The node ID of the subtree root.

    Returns
    -------
    tuple[numpy.ndarray, dict[int, str], int]
        The linkage matrix, the label of each displayed node — one
        character for a leaf and the representative characters with
        the character count for a collapsed subtree — and the number
        of leaves.
    """

    leaves = subtree_leaves(nodes, root, {})
    new_ids = {leaf: i for i, leaf in enumerate(leaves)}
    stack = [root]
    internal = []
    while stack:
        node_id = stack.pop()
        node = nodes[node_id]
        if node["left"] is not None:
            internal.append(node_id)
            stack.append(node["left"])
            stack.append(node["right"])
    # the node IDs increase with the merge distance, so the renumbered
    # merges satisfy the increasing-distance requirement of the
    # linkage matrix format
    internal.sort()
    new_ids.update({
        old: len(leaves) + i for i, old in enumerate(internal)
    })

    leaf_labels = {}
    for leaf in leaves:
        leaf_labels[new_ids[leaf]] = nodes[leaf]["character"]

    linkage = numpy.zeros((len(internal), 4))
    collapsed_labels = {}
    for i, old in enumerate(internal):
        node = nodes[old]
        members = subtree_leaves(nodes, old, {})
        linkage[i] = (
            new_ids[node["left"]],
            new_ids[node["right"]],
            node["distance"],
            len(members),
        )
        chars = "".join(
            nodes[leaf]["character"] for leaf in members
        )
        # both characters fit into the label when the subtree holds
        # only two, the count marks the incomplete lists only
        collapsed_labels[len(leaves) + i] = (
            chars if len(chars) <= 2 else f"{chars[:2]}\n{len(chars)}"
        )
    return (
        linkage,
        {**leaf_labels, **collapsed_labels},
        len(leaves),
    )


def draw_dendrogram(
    nodes: dict[int, dict],
    root: int,
    title: str,
    output_path: Path,
    max_leaves: int,
) -> None:
    """Draw the dendrogram of the cluster subtree into a PNG file.

    Parameters
    ----------
    nodes : dict[int, dict]
        The tree nodes as returned by ``read_tree``.
    root : int
        The node ID of the subtree root.
    title : str
        The plot title.
    output_path : Path
        Path of the output PNG file.
    max_leaves : int
        Maximum number of branches drawn before the lower subtrees
        are collapsed.
    """

    linkage, node_labels, leaf_count = build_linkage(nodes, root)
    if not len(linkage):
        print(f"  single character cluster, nothing to draw")
        return
    fig, ax = plt.subplots(
        figsize=(min(max(8.0, 0.4 * min(max_leaves, leaf_count)), 48.0), 5.6)
    )
    dendrogram(
        linkage,
        ax=ax,
        truncate_mode="lastp",
        p=max_leaves,
        leaf_label_func=lambda node_id: node_labels[node_id],
        leaf_rotation=0,
        show_leaf_counts=False,
        color_threshold=numpy.inf,
    )
    ax.tick_params(axis="x", labelsize=10)
    ax.set_ylabel("cosine distance")
    ax.set_title(title)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    """Parse command-line arguments and draw the cluster dendrogram.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments; defaults to ``sys.argv``.

    Returns
    -------
    int
        Exit code; the dendrogram is written to the output directory.
    """

    parser = argparse.ArgumentParser(
        description="Draw the dendrogram of one cluster of a "
                    "clustering tree."
    )
    parser.add_argument(
        "--tree-file",
        required=True,
        help="Tree CSV file of one feature as written by the "
             "reconstruct command.",
    )
    parser.add_argument(
        "--cluster",
        required=True,
        help="The single cluster ID whose subtree is drawn.",
    )
    parser.add_argument(
        "--max-leaves",
        type=int,
        default=50,
        help="Maximum number of branches drawn before the lower "
             "subtrees are collapsed into branches annotated with "
             "their representative characters.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        type=Path,
        help="Directory the dendrograms are written to; defaults "
             "to the current directory.",
    )
    args = parser.parse_args(argv)

    tree_file = Path(args.tree_file)
    if not tree_file.exists():
        raise SystemExit(f"{tree_file} does not exist.")
    nodes = read_tree(tree_file)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature = tree_file.name.removesuffix("_tree.csv")

    try:
        root = cluster_root(nodes, args.cluster)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 1
    output_path = args.output_dir / f"{feature}_cluster_{args.cluster}_dendrogram.png"
    print(
        f"{feature} cluster {args.cluster}: {nodes[root]['size']} "
        f"characters, at most {args.max_leaves} branches drawn"
    )
    draw_dendrogram(
        nodes, root,
        f"{feature} cluster {args.cluster} ({nodes[root]['size']} "
        "characters)",
        output_path, args.max_leaves,
    )
    print(f"  written {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
