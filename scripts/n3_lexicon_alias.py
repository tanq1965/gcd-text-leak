#!/usr/bin/env python3
"""Build N3 C/D TextGCD lexicons and exact-name coverage reports.

Run from the repository root, for example:
    python scripts/n3_lexicon_alias.py --ds cub --names names/cub.csv \
        --tags third_party/textgcd/Lexicon/Lexicon_tags.csv \
        --attrs third_party/textgcd/Lexicon/Lexicon_attributes.csv --out .

The original TextGCD lexicon files are one-column CSVs without headers.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

NEEDED = ("label", "class_name", "name_norm", "is_old")

# Exact aliases approved in N3 review.
# Do not replace exact matching with substring matching.

CUB_ALIASES = {
    "whip poor will": {
        "eastern whip poor will",
        "mexican whip poor will",
    },
    "mockingbird": {
        "northern mockingbird",
    },
    "white pelican": {
        "american white pelican",
    },
    "cardinal": {
        "northern cardinal",
    },
}

AIRCRAFT_ALIASES = {
    "a320": {"airbus a320 family"},
    "dc 8": {"douglas dc 8"},
    "erj 145": {"embraer erj 145 family"},
    "md 80": {"mcdonnell douglas md 80"},
    "747 400": {"boeing 747 400"},
    "c 130": {"lockheed c 130 hercules"},
    "dc 6": {"douglas dc 6"},
    "challenger 600": {"bombardier challenger 600"},
    "metroliner": {"fairchild c 26 metroliner"},
}

ALIAS_REASONS = {
    "eastern whip poor will": "AOU 2010 split",
    "mexican whip poor will": "AOU 2010 split",
    "northern mockingbird": "only common North American mockingbird",
    "american white pelican": "CUB White Pelican species name",
    "northern cardinal": "CUB Cardinal species name",

    "airbus a320 family": "A320 family name",
    "douglas dc 8": "manufacturer prefix for DC-8",
    "embraer erj 145 family": "ERJ 145 family name",
    "mcdonnell douglas md 80": "manufacturer prefix for MD-80",
    "boeing 747 400": "manufacturer prefix for 747-400",
    "lockheed c 130 hercules": "full name for C-130",
    "douglas dc 6": "manufacturer prefix for DC-6",
    "bombardier challenger 600": "manufacturer prefix for Challenger 600",
    "fairchild c 26 metroliner": "military derivative of Metroliner",
}


def normalize(text: str) -> str:
    """Normalize dataset class names and lexicon name prefixes."""
    text = re.sub(r"^\s*\d{3}\.", "", str(text))
    text = text.casefold().replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def aliases(name: str, ds: str) -> set[str]:
    """Return exact normalized names and review-approved aliases."""
    base = normalize(name)
    result = {base}

    if ds == "aircraft":
        result.add(base + " aircraft")

        if base.endswith(" aircraft"):
            result.add(base.removesuffix(" aircraft").strip())

        result.update(AIRCRAFT_ALIASES.get(base, set()))

    elif ds == "cub":
        result.update(CUB_ALIASES.get(base, set()))

    return {item for item in result if item}


def read_one_column(path: Path) -> list[str]:
    """Read TextGCD's headerless one-column CSV, including a valid empty file."""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        values: list[str] = []
        for line_number, row in enumerate(csv.reader(handle), 1):
            if len(row) != 1:
                raise ValueError(
                    f"{path}:{line_number}: expected 1 CSV column, got {len(row)}. "
                    "Check quoting; TextGCD lexicons must not have a header."
                )
            values.append(row[0])
    return values


def write_one_column(path: Path, values: list[str]) -> None:
    """Write TextGCD-compatible headerless one-column CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for value in values:
            writer.writerow([value])


def write_dicts(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    """Write an audit/coverage CSV with a header, even when it has no rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore"
        )
        writer.writeheader()
        writer.writerows(rows)


def load_names(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(NEEDED) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} missing columns: {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path}: empty names file")

    labels: list[int] = []
    old_flags: list[bool] = []
    for row in rows:
        try:
            label = int(row["label"])
        except (TypeError, ValueError):
            raise ValueError(f"{path}: invalid label {row.get('label')!r}")
        flag = (row.get("is_old") or "").strip().casefold()
        if flag not in {"1", "0", "true", "false"}:
            raise ValueError(f"{path}: invalid is_old value {flag!r}")
        row["label"] = label
        row["name_norm"] = normalize(row.get("name_norm") or row["class_name"])
        row["is_old"] = flag in {"1", "true"}
        if not row["name_norm"]:
            raise ValueError(f"{path}: blank normalized class name at label {label}")
        labels.append(label)
        old_flags.append(row["is_old"])

    if labels != list(range(len(rows))):
        raise ValueError(f"{path}: labels must be ordered and contiguous from 0")
    if old_flags != sorted(old_flags, reverse=True):
        raise ValueError(f"{path}: Old classes (is_old=1) must precede New classes (is_old=0)")
    return rows


def tag_name(prompt: str) -> str:
    """Extract the class name from a TextGCD tag prompt."""
    match = re.match(r"^\s*a photo of\s+(?:a|an)\s+(.+?)\s*$", prompt, re.I)
    return normalize(match.group(1) if match else prompt)


def attr_parts(sentence: str) -> tuple[str | None, str | None]:
    """Return (normalized class prefix, description) for an 'X which Y' sentence."""
    match = re.match(r"^\s*(.+?)\s+which\s+(.+?)\s*$", sentence, re.I)
    if match:
        return normalize(match.group(1)), match.group(2).strip()
    return None, None

# Original source row numbers reviewed by Tan.
RESIDUAL_NAME_ROWS = {
    "aircraft": {
        "C": {
            17747: "erj 135",
        },
        "D": {
            17585: "cessna 172",
            17747: "erj 135",
        },
    },
}


def contains_exact_name(text: str, name: str) -> bool:
    """Match a complete normalized name, not an arbitrary substring."""
    normalized_text = re.sub(
        r"[^a-z0-9]+", " ", text.casefold()
    ).strip()

    normalized_name = re.sub(
        r"[^a-z0-9]+", " ", name.casefold()
    ).strip()

    return (
        f" {normalized_name} "
        in f" {normalized_text} "
    )


def residual_name(
    ds: str,
    version: str,
    source_line: int,
    processed_text: str,
) -> str | None:
    """Return a reviewed residual name; fail if source changed."""
    reviewed = (
        RESIDUAL_NAME_ROWS
        .get(ds, {})
        .get(version, {})
    )

    name = reviewed.get(source_line)

    if name is None:
        return None

    if not contains_exact_name(processed_text, name):
        raise ValueError(
            f"{ds}/{version}: source line {source_line} "
            f"no longer contains reviewed residual name {name!r}. "
            "Inspect the source before changing the rule."
        )

    return name


def alias_audit(
    classes: list[dict[str, Any]],
    ds: str,
) -> list[dict[str, Any]]:
    rows = []

    if ds == "cub":
        alias_map = CUB_ALIASES
    elif ds == "aircraft":
        alias_map = AIRCRAFT_ALIASES
    else:
        alias_map = {}

    for class_row in classes:
        base = class_row["name_norm"]
        approved = sorted(alias_map.get(base, set()))

        if approved:
            rows.append({
                "class_name": class_row["class_name"],
                "aliases": approved,
                "reasons": {
                    alias: ALIAS_REASONS[alias]
                    for alias in approved
                },
            })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds", choices=["cub", "scars", "aircraft"], required=True)
    parser.add_argument("--names", type=Path, required=True)
    parser.add_argument("--tags", type=Path, required=True)
    parser.add_argument("--attrs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("."), help="Output root")
    parser.add_argument(
        "--d-strip-all",
        action="store_true",
        help="D: strip X which on unrelated lexicon classes too; use only with an agreed policy",
    )
    args = parser.parse_args()

    for input_path in (args.names, args.tags, args.attrs):
        if not input_path.is_file():
            parser.error(f"Input file not found: {input_path}")

    classes = load_names(args.names)
    tags = read_one_column(args.tags)
    attrs = read_one_column(args.attrs)

    tag_keys = {tag_name(tag) for tag in tags}
    attr_prefixes = [attr_parts(attribute)[0] for attribute in attrs]
    attr_keys = set(filter(None, attr_prefixes))

    old_keys: set[str] = set()
    new_keys: set[str] = set()
    key_to_classes: dict[str, list[str]] = defaultdict(list)

    for class_row in classes:
        class_aliases = aliases(class_row["name_norm"], args.ds)
        for alias in class_aliases:
            key_to_classes[alias].append(class_row["class_name"])
        (old_keys if class_row["is_old"] else new_keys).update(class_aliases)

    collisions = {key: values for key, values in key_to_classes.items() if len(values) > 1}
    if collisions:
        raise ValueError(f"Ambiguous class name/alias mapping(s): {collisions}")

    # Coverage measures the ORIGINAL TextGCD lexicon (B), not the filtered C/D files.
    coverage: list[dict[str, Any]] = []
    for class_row in classes:
        class_aliases = aliases(class_row["name_norm"], args.ds)
        in_tag = bool(class_aliases & tag_keys)
        in_attribute = bool(class_aliases & attr_keys)
        coverage.append(
            {
                "label": class_row["label"],
                "class_name": class_row["class_name"],
                "name_norm": class_row["name_norm"],
                "is_old": int(class_row["is_old"]),
                "in_tag": int(in_tag),
                "in_attribute": int(in_attribute),
                "in_either": int(in_tag or in_attribute),
            }
        )

    coverage_root = args.out / "coverage"
    write_dicts(
        coverage_root / f"{args.ds}.csv",
        coverage,
        ["label", "class_name", "name_norm", "is_old", "in_tag", "in_attribute", "in_either"],
    )

    summary: list[dict[str, Any]] = []
    for group in ("All", "Old", "New"):
        members = [
            row
            for row in coverage
            if group == "All" or bool(row["is_old"]) == (group == "Old")
        ]
        count = len(members)
        tag_count = sum(row["in_tag"] for row in members)
        attribute_count = sum(row["in_attribute"] for row in members)
        either_count = sum(row["in_either"] for row in members)
        summary.append(
            {
                "group": group,
                "classes": count,
                "in_tag": tag_count,
                "tag_pct": round(100 * tag_count / count, 2) if count else "",
                "in_attribute": attribute_count,
                "attribute_pct": round(100 * attribute_count / count, 2) if count else "",
                "in_either": either_count,
                "either_pct": round(100 * either_count / count, 2) if count else "",
            }
        )
    write_dicts(
        coverage_root / f"{args.ds}_summary.csv",
        summary,
        [
            "group",
            "classes",
            "in_tag",
            "tag_pct",
            "in_attribute",
            "attribute_pct",
            "in_either",
            "either_pct",
        ],
    )

    # C: remove New class tags and strip New class prefixes from "X which Y".
    c_tags: list[str] = []
    c_removed: list[dict[str, Any]] = []
    for line_number, tag in enumerate(tags, 1):
        key = tag_name(tag)
        if key in new_keys:
            c_removed.append(
                {
                    "file": "Lexicon_tags.csv",
                    "line": line_number,
                    "text": tag,
                    "matched_class": " | ".join(key_to_classes[key]),
                    "rule": "alias" if key in ALIAS_REASONS else "C_remove_New_tag",
                    "reason": ALIAS_REASONS.get(key, ""),
                }
            )
        else:
            c_tags.append(tag)

        c_attrs: list[str] = []
    c_modified: list[dict[str, Any]] = []

    for line_number, (sentence, prefix) in enumerate(
        zip(attrs, attr_prefixes), 1
    ):
        is_new_prefix = prefix in new_keys

        if is_new_prefix:
            _, description = attr_parts(sentence)

            if not description:
                raise ValueError(
                    f"Could not strip class prefix at line {line_number}"
                )

            candidate = description
        else:
            candidate = sentence

        # Remove reviewed residual-name attributes.
        residual = residual_name(
            args.ds, "C", line_number, candidate
        )

        if residual is not None:
            c_removed.append({
                "file": "Lexicon_attributes.csv",
                "line": line_number,
                "text": sentence,
                "matched_class": " | ".join(
                    key_to_classes[residual]
                ),
                "rule": "residual_name",
                "reason": "New class name remains in description",
            })
            continue

        if is_new_prefix:
            c_attrs.append(candidate)

            c_modified.append({
                "file": "Lexicon_attributes.csv",
                "line": line_number,
                "text": sentence,
                "new_text": candidate,
                "matched_class": " | ".join(
                    key_to_classes[prefix]
                ),
                "rule": (
                    "alias"
                    if prefix in ALIAS_REASONS
                    else "C_strip_New_prefix_X_which"
                ),
                "reason": ALIAS_REASONS.get(prefix, ""),
            })
        else:
            c_attrs.append(sentence)

        # D removes all tags by definition.
    # Only removed attributes are logged in D/removed.csv.
    d_removed: list[dict[str, Any]] = []
    d_attrs: list[str] = []
    d_modified: list[dict[str, Any]] = []

    target_keys = old_keys | new_keys

    for line_number, (sentence, prefix) in enumerate(
        zip(attrs, attr_prefixes), 1
    ):
        should_strip = (
            prefix is not None
            and (args.d_strip_all or prefix in target_keys)
        )

        if should_strip:
            _, description = attr_parts(sentence)

            if not description:
                raise ValueError(
                    f"Could not strip class prefix at line {line_number}"
                )

            candidate = description
        else:
            candidate = sentence

        # Remove reviewed residual-name attributes.
        residual = residual_name(
            args.ds, "D", line_number, candidate
        )

        if residual is not None:
            d_removed.append({
                "file": "Lexicon_attributes.csv",
                "line": line_number,
                "text": sentence,
                "matched_class": " | ".join(
                    key_to_classes[residual]
                ),
                "rule": "residual_name",
                "reason": "Class name remains in description",
            })
            continue

        if should_strip:
            d_attrs.append(candidate)

            d_modified.append({
                "file": "Lexicon_attributes.csv",
                "line": line_number,
                "text": sentence,
                "new_text": candidate,
                "matched_class": " | ".join(
                    key_to_classes.get(prefix, [])
                ),
                "rule": (
                    "alias"
                    if prefix in ALIAS_REASONS
                    else (
                        "D_strip_all_X_which"
                        if args.d_strip_all
                        else "D_strip_dataset_X_which"
                    )
                ),
                "reason": ALIAS_REASONS.get(prefix, ""),
            })
        else:
            d_attrs.append(sentence)

    for version, output_tags, output_attrs, removed, modified in (
        ("C", c_tags, c_attrs, c_removed, c_modified),
        ("D", [], d_attrs, d_removed, d_modified),
    ):
        output_root = args.out / "lexicon" / args.ds / version
        write_one_column(output_root / "tags.csv", output_tags)
        write_one_column(output_root / "attributes.csv", output_attrs)
        write_dicts(
            output_root / "removed.csv",
            removed,
            ["file", "line", "text", "matched_class", "rule", "reason"],
        )
        write_dicts(
            output_root / "modified.csv",
            modified,
            ["file", "line", "text", "new_text", "matched_class", "rule", "reason"],
        )

    # Guard the two experimental conditions against accidental class-name leakage
    # in the exact tag/prefix fields that this N3 transformation is designed to remove.
    assert all(tag_name(tag) not in new_keys for tag in c_tags), "C still contains a New-class tag"
    assert all(attr_parts(sentence)[0] not in new_keys for sentence in c_attrs), (
        "C still contains a New-class attribute prefix"
    )
    assert not read_one_column(args.out / "lexicon" / args.ds / "D" / "tags.csv"), (
        "D tags must be empty"
    )
    assert all(attr_parts(sentence)[0] not in target_keys for sentence in d_attrs), (
        "D still contains a dataset-class attribute prefix"
    )

    audit = {
        "dataset": args.ds,
        "classes": len(classes),
        "old": sum(bool(row["is_old"]) for row in classes),
        "new": sum(not bool(row["is_old"]) for row in classes),
        "original_tags": len(tags),
        "original_attributes": len(attrs),
        "C_removed_tags": sum(
            row["file"] == "Lexicon_tags.csv"
            for row in c_removed
        ),
        "C_removed_attributes": sum(
            row["file"] == "Lexicon_attributes.csv"
            for row in c_removed
        ),
        "C_modified_attributes": len(c_modified),

        "D_removed_tags": len(tags),
        "D_removed_attributes": sum(
            row["file"] == "Lexicon_attributes.csv"
            for row in d_removed
        ),
        "D_modified_attributes": len(d_modified),
        "D_scope": "all_attribute_prefixes" if args.d_strip_all else "dataset_classes_only",
        "approved_aliases": alias_audit(classes, args.ds),
        "unmatched_classes": [
            row["class_name"] for row in coverage if not row["in_either"]
        ],
        "review_note": (
            "Unmatched classes require manual review; substring candidates are leads, "
            "not approved aliases. Checks do not detect every embedded name in free text."
        ),
    }
    audit_path = coverage_root / f"{args.ds}_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with audit_path.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        handle.write(
            json.dumps(
                audit,
                ensure_ascii=False,
                indent=2,
            ) + "\n"
        )
    print(json.dumps({"summary": summary, "audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
