# -*- coding: utf-8 -*-

"""Turn the inferred pronunciation rules of one dialect into tables
of sound changes.

Reads a rules CSV file (the rules of all dialects, with ``did`` and
``feature`` columns) and a predictions CSV file from an inference
output, selects one dialect, merges the rules that share the source
phones and the target phone and differ in the condition of one feature
only, matches every merged rule against the reconstructed phonemes of
the characters and writes one CSV table per feature,
``<did>_<feature>_rules.csv``, with one row per rule: the sound change
formula, the number of characters that follow the rule, the characters
themselves and the exception characters with their deviant phones.

The rules of one feature are ordered by the optimal leaf seriation of
their target phones: the profile of each phone is the concatenation of
its co-occurrence count vectors with the phones of the other two
features, counted over the rows of the target dialect only, and the
phones are clustered with average linkage on cosine distance.

Usage::

    python scripts/extract_rules.py --rule-file rules.csv \
        --prediction-file predictions.csv --dialect 太原
"""

__author__ = "黄艺华 <lernanto@foxmail.com>"


import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform

FEATURES = ("initial", "final", "tone")
RECONSTRUCTED = {
    "initial": "reconstructed_initial",
    "final": "reconstructed_final",
    "tone": "reconstructed_tone",
}
RECORDED = {"initial": "initial", "final": "final", "tone": "tone"}
FEATURE_NAMES = {"initial": "声母", "final": "韵母", "tone": "声调"}

Condition = list[list[tuple[bool, str]]]


def parse_conditions(cell: str | None) -> Condition | None:
    """Parse a condition cell into a disjunction of conjunctions.

    The cell holds a disjunction (``|``) of conjunctions (``&``) of
    atoms, where an atom is a category the reconstructed phoneme must
    match or a ``!``-prefixed category it must not match. The values
    of one feature are mutually exclusive, so the positive atoms of a
    conjunction are alternatives of one another: the conjunction holds
    either the values the phoneme may take or the values it may not
    take.

    Parameters
    ----------
    cell : str | None
        The raw condition cell, or ``None``.

    Returns
    -------
    list[list[tuple[bool, str]]] | None
        A list of disjuncts of ``(negated, value)`` pairs, or ``None``
        for an empty cell.
    """

    cell = (cell or "").strip()
    if not cell:
        return None
    disjunctions = []
    for disjunct in cell.split("|"):
        atoms = []
        for atom in disjunct.split("&"):
            atom = atom.strip()
            if atom.startswith("!"):
                value = atom[1:].strip()
                if value:
                    atoms.append((True, value))
            elif atom:
                atoms.append((False, atom))
        if atoms:
            disjunctions.append(atoms)
    return disjunctions or None


def conditions_match(conditions: Condition, value: str) -> bool:
    """Return whether a disjunction of conjunctions matches a value.

    The positive atoms of a conjunction are alternatives of one
    another: the conjunction matches when the value is one of its
    positive atoms, if any, that is not excluded by its negated
    atoms, or, when it holds no positive atom, when the value is none
    of its negated atoms.

    Parameters
    ----------
    conditions : list[list[tuple[bool, str]]]
        A disjunction of conjunctions of ``(negated, value)`` pairs.
    value : str
        The value to test.

    Returns
    -------
    bool
        Whether any disjunct matches the value.
    """

    for disjunct in conditions:
        positives = [v for negated, v in disjunct if not negated]
        negatives = {v for negated, v in disjunct if negated}
        if positives:
            if value in positives and value not in negatives:
                return True
        elif value not in negatives:
            return True
    return False


def condition_shape(
    conditions: Condition,
) -> tuple[str, list[str] | str]:
    """Classify conditions into a shape for rendering.

    Parameters
    ----------
    conditions : list[list[tuple[bool, str]]]
        A disjunction of conjunctions of ``(negated, value)`` pairs.

    Returns
    -------
    tuple[str, list[str] | str]
        ``("inc", values)`` for a pure disjunction or conjunction of
        positive atoms (the values in their original order),
        ``("exc", values)`` for a pure conjunction of negative atoms,
        and ``("raw", text)`` for the original expression otherwise.
    """

    if len(conditions) == 1 and all(
        not negated for negated, _ in conditions[0]
    ):
        return "inc", [value for _, value in conditions[0]]
    if len(conditions) > 1 and all(
        len(disjunct) == 1 and not disjunct[0][0]
        for disjunct in conditions
    ):
        return "inc", [disjunct[0][1] for disjunct in conditions]
    if len(conditions) == 1 and all(
        negated for negated, _ in conditions[0]
    ):
        return "exc", [value for _, value in conditions[0]]
    text = "|".join(
        "&".join(("!" + value) if negated else value
                 for negated, value in disjunct)
        for disjunct in conditions
    )
    return "raw", text


def load_rules(path: Path, did: str) -> list[dict]:
    """Load the rules of one dialect from a rules CSV file.

    The file holds the rules of all dialects with ``did`` and
    ``feature`` columns; only the rows of ``did`` are kept.

    Parameters
    ----------
    path : Path
        Path to the rules CSV file.
    did : str
        The dialect ID whose rules are loaded.

    Returns
    -------
    list[dict]
        One dict per rule with the source ``row`` number, the
        ``feature`` the rule belongs to, the parsed ``conditions``
        keyed by feature and the target ``phone``.
    """

    rules = []
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row, record in enumerate(csv.DictReader(f), 1):
            if record["did"].strip() != did:
                continue
            conditions = {}
            for key in FEATURES:
                parsed = parse_conditions(record[RECONSTRUCTED[key]])
                if parsed is not None:
                    conditions[key] = parsed
            rules.append({
                "row": row,
                "feature": record["feature"].strip(),
                "conditions": conditions,
                "phone": record["phone"].strip(),
            })
    return rules


def canonical_condition(condition: Condition | None) -> tuple | None:
    """Return a hashable canonical form of one parsed condition.

    Parameters
    ----------
    condition : list[list[tuple[bool, str]]] | None
        A parsed condition, or ``None``.

    Returns
    -------
    tuple | None
        The condition with sorted atoms and disjuncts, or ``None``.
    """

    if condition is None:
        return None
    return tuple(
        tuple(sorted(disjunct))
        for disjunct in condition
    )


def union_conditions(a: Condition, b: Condition) -> Condition:
    """Return the disjunction of two conditions, without duplicates.

    Parameters
    ----------
    a, b : list[list[tuple[bool, str]]]
        The conditions to unite.

    Returns
    -------
    list[list[tuple[bool, str]]]
        The disjuncts of both conditions, duplicates removed.
    """

    union = list(a)
    seen = {canonical_condition([disjunct]) for disjunct in a}
    for disjunct in b:
        key = canonical_condition([disjunct])
        if key not in seen:
            seen.add(key)
            union.append(disjunct)
    return union


def simplify_condition(
    condition: Condition,
    domain: list[str],
) -> Condition | None:
    """Simplify a condition against the observed domain values.

    Every condition is equivalent to the set of the domain values it
    matches, so the condition is rebuilt from the matched values: a
    condition matching no value stays empty, a condition matching
    every observed value becomes unconditional (``None``), a
    condition excluding fewer values than it matches is rewritten as
    the conjunction of the excluded values, and any other condition
    as the disjunction of the matched values, all in the order of the
    domain.

    Parameters
    ----------
    condition : list[list[tuple[bool, str]]]
        The condition to simplify.
    domain : list[str]
        The values observed for the feature.

    Returns
    -------
    list[list[tuple[bool, str]]] | None
        The simplified condition, ``None`` when it is unconditional.
    """

    matched = [v for v in domain if conditions_match(condition, v)]
    if not matched:
        return []
    if len(matched) == len(domain):
        return None
    excluded = [v for v in domain if v not in set(matched)]
    if len(excluded) < len(matched):
        return [[(True, v) for v in excluded]]
    return [[(False, v)] for v in matched]


def merge_rules(
    rules: list[dict],
    domain: dict[str, list[str]] | None = None,
) -> list[dict]:
    """Merge rules that share the source phones and the target phone.

    Two rules of one group merge when their conditions differ in the
    condition of one feature only, or not at all: the differing
    conditions are united, e.g. ``A > a / _B`` and ``A > a / _C``
    merge into ``A > a / _B,C``, and when one rule leaves the
    differing feature unconstrained the union is unconstrained as
    well. The union is simplified against the observed domain of the
    feature: a union matching every value, like ``!n|n``, is dropped,
    a union excluding fewer values than it matches, like ``!k&!n|k``,
    is rewritten as the excluded values, and a union matching no
    value drops the whole rule. The merging is repeated until no pair
    of a group can be merged further, so that chains like ``_B``,
    ``_C`` and ``_D`` collapse into one rule.

    Parameters
    ----------
    rules : list[dict]
        The raw rules of one dialect.
    domain : dict[str, list[str]] | None
        The observed values of each feature; ``None`` disables the
        simplification of united conditions.

    Returns
    -------
    list[dict]
        The merged rules in the order of the first occurrence of their
        group.
    """

    def simplify(conditions, feature):
        if domain is None:
            return conditions
        return simplify_condition(conditions, domain[feature])

    groups = defaultdict(list)
    for rule in rules:
        # rules of different features never merge, and the group key
        # holds the feature to keep their unconditional rules apart
        key = (
            rule["feature"],
            canonical_condition(rule["conditions"].get(rule["feature"])),
            rule["phone"],
        )
        groups[key].append(rule)

    merged = []
    for group in groups.values():
        group = list(group)
        progress = True
        while progress:
            progress = False
            group = [rule for rule in group if rule is not None]
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    a = group[i]["conditions"]
                    b = group[j]["conditions"]
                    differing = [
                        key for key in set(a) | set(b)
                        if canonical_condition(a.get(key))
                        != canonical_condition(b.get(key))
                    ]
                    if len(differing) > 1:
                        continue
                    if len(differing) == 1:
                        feature = differing[0]
                        conditions = dict(a)
                        if feature in a and feature in b:
                            union = simplify(
                                union_conditions(a[feature], b[feature]),
                                feature,
                            )
                            if union is None:
                                # the union matches every value
                                conditions.pop(feature, None)
                            elif not union:
                                # the union matches no value: the
                                # rules cover no character at all
                                group[i] = None
                                group[j] = None
                                progress = True
                                break
                            else:
                                conditions[feature] = union
                        else:
                            # one rule leaves the feature
                            # unconstrained, so does the union
                            conditions.pop(feature, None)
                        group[i] = {
                            **group[i], "conditions": conditions
                        }
                    # identical conditions merge into a duplicate
                    del group[j]
                    progress = True
                    break
                if progress:
                    break
        merged.extend(rule for rule in group if rule is not None)
    return merged


def source_phones(
    rule: dict,
    domain: dict[str, list[str]],
) -> list[str]:
    """Return the source phones of a rule in the given domain.

    The source phones are the values of the rule's own feature covered
    by its condition; a rule without a condition covers the whole
    domain, and a rule covered by no value keeps the placeholder ``?``.

    Parameters
    ----------
    rule : dict
        The rule whose source phones are rendered.
    domain : dict[str, list[str]]
        The observed values of each feature.

    Returns
    -------
    list[str]
        The source phones of the rule.
    """

    feature = rule["feature"]
    condition = rule["conditions"].get(feature)
    if condition is None:
        source = list(domain[feature])
    else:
        kind, values = condition_shape(condition)
        if kind == "inc":
            source = values
        elif kind == "exc":
            source = [v for v in domain[feature] if v not in values]
        else:
            source = [v for v in domain[feature]
                      if conditions_match(condition, v)]
    if not source:
        source = ["?"]
    return source


def render_environment(rule: dict) -> str:
    """Render the conditions on the other features as the environment.

    The conditions are placed around the underscore marking the
    position of the target in the syllable (initial, final, tone),
    e.g. ``_aɣ̃53₁``.

    Parameters
    ----------
    rule : dict
        The rule whose environment is rendered.

    Returns
    -------
    str
        The rendered environment, an empty string when the rule is
        unconditional.
    """

    feature = rule["feature"]
    environment = {}
    for other in FEATURES:
        if other != feature and other in rule["conditions"]:
            environment[other] = condition_shape(rule["conditions"][other])

    def group(shape):
        kind, values = shape
        if kind == "inc":
            return ",".join(values)
        if kind == "exc":
            return "-" + ",".join(values)
        return values

    # the three features follow the syllable order initial, final and
    # tone, so the tone is written right after the final, as in a
    # phonetic transcription of the conditioning syllable
    parts = []
    if feature == "initial":
        if "final" in environment:
            parts.append("_" + group(environment["final"]))
        if "tone" in environment:
            if not parts:
                parts.append("_")
            parts.append(group(environment["tone"]))
    elif feature == "final":
        if "initial" in environment:
            parts.append(group(environment["initial"]) + "_")
        if "tone" in environment:
            if not parts:
                parts.append("_")
            parts.append(group(environment["tone"]))
    else:
        if "initial" in environment:
            parts.append(group(environment["initial"]) + "_")
        if "final" in environment:
            parts.append(group(environment["final"]))
    return "".join(parts)


def render_formula(
    rule: dict,
    domain: dict[str, list[str]],
) -> str:
    """Render a rule as a sound change formula.

    The source phones are the values of the rule's own feature covered
    by its condition, and the conditions of the other two features are
    rendered as the environment around the underscore marking the
    position of the target in the syllable (initial, final, tone),
    e.g. ``p,pʰ > pf / _aɣ̃53₁``.

    Parameters
    ----------
    rule : dict
        The rule to render.
    domain : dict[str, list[str]]
        The observed values of each feature.

    Returns
    -------
    str
        The rendered formula.
    """

    formula = ",".join(source_phones(rule, domain)) + " > " + rule["phone"]
    environment = render_environment(rule)
    if environment:
        formula += " / " + environment
    return formula


def rule_matches(row: dict, rule: dict) -> bool | None:
    """Match a prediction row against a rule.

    Parameters
    ----------
    row : dict
        One record of the predictions CSV file.
    rule : dict
        The rule to match against.

    Returns
    -------
    bool | None
        ``True`` when the row satisfies every condition of the rule
        and its recorded phone equals the target phone, ``False`` when
        it satisfies the conditions with a deviant phone, and ``None``
        when the conditions do not apply.
    """

    for feature, condition in rule["conditions"].items():
        if not conditions_match(condition, row[RECONSTRUCTED[feature]]):
            return None
    recorded = row[RECORDED[rule["feature"]]]
    if not recorded.strip():
        return None
    return recorded == rule["phone"]


def format_exceptions(exceptions: dict[str, set[str]]) -> str:
    """Format the exception characters grouped by their deviant phone.

    Parameters
    ----------
    exceptions : dict[str, set[str]]
        The deviant phones of each exception character.

    Returns
    -------
    str
        The phone groups rendered as ``phone:characters`` sequences.
    """

    groups = defaultdict(list)
    for character, phones in exceptions.items():
        for phone in sorted(phones):
            groups[phone].append(character)
    return " ".join(
        phone + ":" + "".join(characters)
        for phone, characters in groups.items()
    )


def seriate(M: np.ndarray) -> list[int]:
    """Order matrix rows by similarity with hierarchical clustering.

    Average linkage on cosine distance between the row vectors.

    Parameters
    ----------
    M : numpy.ndarray of shape (n, m)
        The row vectors to order.

    Returns
    -------
    list[int]
        Row indices in leaf order.
    """
    M = M.astype(float)
    norms = np.linalg.norm(M, axis=1)
    norms[norms == 0] = 1.0
    X = M / norms[:, None]
    # Cosine distance = 1 - row dot products; an all-zero row keeps
    # distance 1 to every other row.
    D = 1.0 - np.clip(X @ X.T, 0.0, 1.0)
    np.fill_diagonal(D, 0.0)
    return list(leaves_list(linkage(squareform(D, checks=False), method="average")))


def phone_order(rows: list[dict], feature: str) -> list[str]:
    """Order the phones of one feature by optimal leaf ordering.

    The profile of each phone is the concatenation of its
    co-occurrence count vectors with the phones of the other two
    features, counted over the given rows only (typically the rows of
    the target dialect). Phones are clustered with average linkage on
    cosine distance and returned in leaf order.

    Parameters
    ----------
    rows : list[dict]
        The records whose phones are counted.
    feature : str
        The feature whose phones are ordered.

    Returns
    -------
    list[str]
        The phones in seriation order.
    """
    others = [f for f in FEATURES if f != feature]
    df = pd.DataFrame(rows)
    pa = df[feature].fillna("").str.strip()
    phones = sorted(set(pa) - {""})
    blocks = []
    for other in others:
        pb = df[other].fillna("").str.strip()
        valid = (pa != "") & (pb != "")
        table = pd.pivot_table(
            df[valid].assign(**{feature: pa[valid], other: pb[valid]}),
            index=feature, columns=other, aggfunc="size", fill_value=0,
        )
        blocks.append(table.reindex(index=phones, fill_value=0))
    # Concatenated co-occurrence vectors; blocks may miss phones or
    # phone pairs absent from one of the two tables
    M = pd.concat(blocks, axis=1).reindex(index=phones).fillna(0).to_numpy()
    return [phones[i] for i in seriate(M)]


def main(argv: list[str] | None = None) -> int:
    """Parse command-line arguments and write the rule tables."""

    parser = argparse.ArgumentParser(
        description="Turn the inferred pronunciation rules of one "
                    "dialect into a CSV table of sound changes per "
                    "feature."
    )
    parser.add_argument(
        "--rule-file",
        required=True,
        help="Rules CSV file holding the rules of all dialects with "
             "did and feature columns.",
    )
    parser.add_argument(
        "--prediction-file",
        required=True,
        help="Predictions CSV file holding one record per dialect and "
             "character.",
    )
    parser.add_argument(
        "--dialect",
        required=True,
        help="Dialect ID to report, matched against the did column.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        type=Path,
        help="Directory the rule tables are written to; defaults "
             "to the current directory.",
    )
    args = parser.parse_args(argv)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rule_file = Path(args.rule_file)
    prediction_file = Path(args.prediction_file)
    rules = load_rules(rule_file, args.dialect)
    if not rules:
        sys.exit(f"{rule_file} holds no rules for did {args.dialect!r}.")

    # the number of dialects recording each character, used to order
    # the characters of a rule by how widely they are attested
    recordings = defaultdict(set)
    dialect_rows = []
    with open(prediction_file, encoding="utf-8-sig", newline="") as f:
        for record in csv.DictReader(f):
            recordings[record["character"]].add(record["did"])
            if record["did"] == args.dialect:
                dialect_rows.append(record)
    recordings = {
        character: len(dialects)
        for character, dialects in recordings.items()
    }
    if not dialect_rows:
        sys.exit(
            f"{prediction_file} holds no rows for did "
            f"{args.dialect!r}."
        )

    # the domain of each feature: the reconstructed phonemes of the
    # dialect in first-seen order
    domain = {feature: [] for feature in FEATURES}
    seen = {feature: set() for feature in FEATURES}
    for record in dialect_rows:
        for feature in FEATURES:
            value = record[RECONSTRUCTED[feature]]
            if value and value not in seen[feature]:
                seen[feature].add(value)
                domain[feature].append(value)

    # merge the raw rules first, then match the characters once per
    # merged rule
    rules = merge_rules(rules, domain)

    for feature in FEATURES:
        # rank of each target phone in the optimal leaf order
        phone_rank = {
            phone: rank
            for rank, phone in enumerate(phone_order(dialect_rows, feature))
        }
        feature_rules = [
            rule for rule in rules if rule["feature"] == feature
        ]
        rows = []
        for rule in feature_rules:
            hits = []
            exceptions = defaultdict(set)
            for record in dialect_rows:
                matched = rule_matches(record, rule)
                if matched is None:
                    continue
                character = record["character"]
                if matched:
                    hits.append(character)
                else:
                    exceptions[character].add(record[RECORDED[feature]])
            hit_characters = sorted(
                set(hits),
                key=lambda character: (
                    -recordings.get(character, 0), character
                ),
            )
            rows.append({
                "rule": render_formula(rule, domain),
                # the rows are ordered by the target phone seriation,
                # then the source phones, the conditions and the
                # character count
                "source": ",".join(source_phones(rule, domain)),
                "phone": rule["phone"],
                "condition": render_environment(rule),
                "character_count": len(hit_characters),
                "characters": "".join(hit_characters),
                "exceptions": format_exceptions(exceptions),
            })
        rows.sort(key=lambda row: (
            phone_rank.get(row["phone"], len(phone_rank)),
            row["source"], row["phone"], row["condition"],
            -row["character_count"],
        ))

        output_path = output_dir / f"{args.dialect}_{feature}_rules.csv"
        with open(
            output_path, "w", encoding="utf-8", newline=""
        ) as output_file:
            writer = csv.writer(output_file)
            writer.writerow([
                "rule", "character_count", "characters", "exceptions",
            ])
            for row in rows:
                writer.writerow([
                    row["rule"],
                    row["character_count"],
                    row["characters"],
                    row["exceptions"],
                ])
        print(
            f"dialect={args.dialect} feature={feature} "
            f"rules={len(rows)} samples={len(dialect_rows)} "
            f"written {output_path}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
