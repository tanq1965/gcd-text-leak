#!/usr/bin/env python3
"""Review substring candidates for classes not matched by N3 exact/alias coverage.

This report is deliberately a manual-review aid. A substring hit is not sufficient
evidence to add an alias (e.g. "Tree Sparrow" vs "American Tree Sparrow").
Run from the repository root:
    python scripts/review_unmatched.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

DATASETS = ("cub", "scars", "aircraft")
FIELDS = [
    "class_name",
    "label",
    "is_old",
    "tag_hits",
    "attribute_hits",
    "tag_example",
    "attribute_example",
]


def normalize_search(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(text).casefold()).strip()


def read_one_column(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        values = []
        for line_number, row in enumerate(csv.reader(handle), 1):
            if len(row) != 1:
                raise ValueError(f"{path}:{line_number}: expected one CSV column")
            values.append(row[0])
        return values


def write_candidates(dataset: str) -> tuple[int, int]:
    names_path = Path("names") / f"{dataset}.csv"
    audit_path = Path("coverage") / f"{dataset}_audit.json"
    tags_path = Path("third_party/textgcd/Lexicon/Lexicon_tags.csv")
    attrs_path = Path("third_party/textgcd/Lexicon/Lexicon_attributes.csv")

    with names_path.open("r", encoding="utf-8-sig", newline="") as handle:
        names = list(csv.DictReader(handle))
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    unmatched = set(audit["unmatched_classes"])
    tags = read_one_column(tags_path)
    attrs = read_one_column(attrs_path)

    normalized_tags = [normalize_search(text) for text in tags]
    normalized_attrs = [normalize_search(text) for text in attrs]
    results = []

    for row in names:
        class_name = row["class_name"]
        if class_name not in unmatched:
            continue
        name = normalize_search(row["name_norm"])
        needle = f" {name} "
        tag_hits = [
            text for text, normalized in zip(tags, normalized_tags)
            if needle in f" {normalized} "
        ]
        attr_hits = [
            text for text, normalized in zip(attrs, normalized_attrs)
            if needle in f" {normalized} "
        ]
        results.append(
            {
                "class_name": class_name,
                "label": row["label"],
                "is_old": row["is_old"],
                "tag_hits": len(tag_hits),
                "attribute_hits": len(attr_hits),
                "tag_example": tag_hits[0] if tag_hits else "",
                "attribute_example": attr_hits[0] if attr_hits else "",
            }
        )

    output_path = Path("coverage") / f"{dataset}_unmatched_candidates.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(results)

    candidate_count = sum(
        row["tag_hits"] > 0 or row["attribute_hits"] > 0 for row in results
    )
    print(
        f"[{dataset}] unmatched={len(results)}, substring candidates={candidate_count}; "
        f"saved {output_path}"
    )
    return len(results), candidate_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ds", choices=DATASETS, action="append", help="Dataset; repeat to select multiple")
    args = parser.parse_args()
    datasets = args.ds or list(DATASETS)
    for dataset in datasets:
        write_candidates(dataset)


if __name__ == "__main__":
    main()
