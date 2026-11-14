#!/usr/bin/env python3
"""Prepare names/{cub,aircraft,scars}.csv for the N3 lexicon audit.

Run from the repository root:
    python scripts/prepare_n3_names.py
    python scripts/prepare_n3_names.py --cars-meta "D:\\path\\to\\devkit\\cars_meta.mat"

CUB and Aircraft names are read from repository split/classmap files. Stanford Cars
requires the original devkit's cars_meta.mat because the classmap stores raw_label.
"""
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path, PurePosixPath
from typing import Any

EXPECTED = {
    "cub": (200, 100),
    "aircraft": (100, 50),
    "scars": (196, 98),
}


def normalize(text: str) -> str:
    text = re.sub(r"^\s*\d{3}\.", "", str(text))
    text = text.casefold().replace("-", " ").replace("_", " ")
    return re.sub(r"\s+", " ", text).strip()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def int_value(row: dict[str, str], field: str, file: str) -> int:
    try:
        return int(row[field])
    except (KeyError, TypeError, ValueError):
        raise ValueError(f"{file}: invalid integer in column {field!r}: {row.get(field)!r}")


def validate_rows(dataset: str, rows: list[dict[str, str]]) -> None:
    expected_total, expected_old = EXPECTED[dataset]
    if len(rows) != expected_total:
        raise ValueError(f"{dataset}: expected {expected_total} classes, got {len(rows)}")

    labels = [int(row["label"]) for row in rows]
    if labels != list(range(expected_total)):
        raise ValueError(f"{dataset}: labels must be ordered and contiguous from 0")

    old_flags = [int(row["is_old"]) for row in rows]
    if any(flag not in (0, 1) for flag in old_flags):
        raise ValueError(f"{dataset}: is_old must contain only 0/1")
    if sum(old_flags) != expected_old or old_flags != [1] * expected_old + [0] * expected_old:
        raise ValueError(
            f"{dataset}: expected {expected_old} Old classes followed by "
            f"{expected_total - expected_old} New classes"
        )

    normalized = [row["name_norm"] for row in rows]
    if any(not value for value in normalized):
        raise ValueError(f"{dataset}: class names cannot normalize to blank")
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{dataset}: duplicate normalized class names")


def write_names(dataset: str, rows: list[dict[str, str]]) -> None:
    validate_rows(dataset, rows)
    output = Path("names") / f"{dataset}.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["label", "class_name", "name_norm", "is_old"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"[OK] {output}: {len(rows)} classes")


def classmap_by_label(dataset: str) -> dict[int, dict[str, str]]:
    path = Path("splits") / f"{dataset}_classmap.csv"
    rows = read_csv(path)
    result: dict[int, dict[str, str]] = {}
    for row in rows:
        label = int_value(row, "label", str(path))
        if label in result:
            raise ValueError(f"{path}: duplicate label {label}")
        old = int_value(row, "is_old", str(path))
        if old not in (0, 1):
            raise ValueError(f"{path}: is_old must be 0 or 1, got {old}")
        result[label] = {**row, "is_old": str(old)}
    expected_total = EXPECTED[dataset][0]
    if sorted(result) != list(range(expected_total)):
        raise ValueError(f"{path}: labels must cover 0..{expected_total - 1}")
    return result


def prepare_cub() -> None:
    """CUB class names come from the class-folder names in splits/cub.csv."""
    split_path = Path("splits/cub.csv")
    classmap = classmap_by_label("cub")
    split_rows = read_csv(split_path)
    observed: dict[int, dict[str, set[str]]] = {}

    for row in split_rows:
        label = int_value(row, "label", str(split_path))
        old_value = int_value(row, "is_old", str(split_path))
        if old_value not in (0, 1):
            raise ValueError(f"{split_path}: is_old must be 0 or 1")
        old = str(old_value)
        relative_path = row["path"].replace("\\", "/")
        folder = PurePosixPath(relative_path).parent.name
        if not folder:
            raise ValueError(
                f"{split_path}: cannot infer class folder from path {row['path']!r}"
            )
        observed.setdefault(label, {"names": set(), "old": set()})
        observed[label]["names"].add(folder)
        observed[label]["old"].add(old)

    if set(observed) != set(classmap):
        missing = sorted(set(classmap) - set(observed))
        extra = sorted(set(observed) - set(classmap))
        raise ValueError(f"CUB split/classmap labels differ; missing={missing}, extra={extra}")

    output = []
    for label in sorted(classmap):
        names = observed[label]["names"]
        old_values = observed[label]["old"]
        if len(names) != 1:
            raise ValueError(f"CUB label {label} maps to multiple folders: {sorted(names)}")
        if len(old_values) != 1 or next(iter(old_values)) != classmap[label]["is_old"]:
            raise ValueError(
                f"CUB label {label}: is_old in splits/cub.csv disagrees with cub_classmap.csv"
            )
        class_name = next(iter(names))
        output.append(
            {
                "label": str(label),
                "class_name": class_name,
                "name_norm": normalize(class_name),
                "is_old": classmap[label]["is_old"],
            }
        )
    write_names("cub", output)


def prepare_aircraft() -> None:
    """Aircraft class names must come from aircraft_classmap.csv, not variants.txt."""
    path = Path("splits/aircraft_classmap.csv")
    classmap = classmap_by_label("aircraft")
    rows = read_csv(path)
    output = []
    for row in rows:
        label = int_value(row, "label", str(path))
        class_name = (row.get("class_name") or "").strip()
        if not class_name:
            raise ValueError(f"{path}: class_name is blank for label {label}")
        output.append(
            {
                "label": str(label),
                "class_name": class_name,
                "name_norm": normalize(class_name),
                "is_old": classmap[label]["is_old"],
            }
        )
    output.sort(key=lambda row: int(row["label"]))
    write_names("aircraft", output)


def matlab_text(value: Any) -> str:
    """Convert common scipy representations of MATLAB cell-string values to text."""
    import numpy as np

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    if isinstance(value, str):
        return value.strip()

    array = np.asarray(value)
    if array.ndim == 0:
        return matlab_text(array.item())

    flattened = array.reshape(-1).tolist()
    if len(flattened) == 1:
        return matlab_text(flattened[0])

    # MATLAB char arrays may be exposed as an array of one-character strings.
    # Join before stripping so internal spaces are not lost.
    if all(isinstance(item, (str, bytes)) and len(item) <= 1 for item in flattened):
        parts = [
            item.decode("utf-8", errors="replace") if isinstance(item, bytes) else item
            for item in flattened
        ]
        return "".join(parts).strip()

    parts = [matlab_text(item) for item in flattened]
    return " ".join(part for part in parts if part).strip()


def prepare_scars(cars_meta: Path) -> None:
    try:
        import numpy as np
        import scipy.io
    except ImportError as exc:
        raise SystemExit(
            "Stanford Cars requires scipy and numpy in the active Python environment."
        ) from exc

    mat = scipy.io.loadmat(str(cars_meta), squeeze_me=True)
    if "class_names" not in mat:
        raise ValueError(
            f"{cars_meta} does not contain 'class_names'. Keys: {list(mat)}"
        )

    raw_names = np.asarray(mat["class_names"], dtype=object).reshape(-1)
    classmap_path = Path("splits/scars_classmap.csv")
    classmap_rows = read_csv(classmap_path)
    by_raw: dict[int, dict[str, str]] = {}
    for row in classmap_rows:
        raw_label = int_value(row, "raw_label", str(classmap_path))
        if raw_label in by_raw:
            raise ValueError(f"{classmap_path}: duplicate raw_label {raw_label}")
        old = int_value(row, "is_old", str(classmap_path))
        if old not in (0, 1):
            raise ValueError(f"{classmap_path}: is_old must be 0 or 1")
        by_raw[raw_label] = row

    expected_total = EXPECTED["scars"][0]
    if len(raw_names) != expected_total or sorted(by_raw) != list(range(expected_total)):
        raise ValueError(
            f"cars_meta.mat/classmap must contain raw_label 0..{expected_total - 1}; "
            f"got {len(raw_names)} names and {len(by_raw)} classmap rows"
        )

    output = []
    for raw_label, raw_name in enumerate(raw_names):
        row = by_raw[raw_label]
        class_name = matlab_text(raw_name)
        if not class_name:
            raise ValueError(f"cars_meta.mat: empty class name at raw_label {raw_label}")
        output.append(
            {
                "label": str(int_value(row, "label", str(classmap_path))),
                "class_name": class_name,
                "name_norm": normalize(class_name),
                "is_old": str(int_value(row, "is_old", str(classmap_path))),
            }
        )

    output.sort(key=lambda row: int(row["label"]))
    write_names("scars", output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cars-meta",
        type=Path,
        default=None,
        help="Path to Stanford Cars devkit/cars_meta.mat; if omitted, Cars is not regenerated.",
    )
    args = parser.parse_args()

    prepare_cub()
    prepare_aircraft()
    if args.cars_meta:
        if not args.cars_meta.is_file():
            raise FileNotFoundError(f"cars_meta.mat not found: {args.cars_meta}")
        prepare_scars(args.cars_meta)
    else:
        print("[WAIT] Skipped Stanford Cars: pass --cars-meta <path-to-cars_meta.mat>.")


if __name__ == "__main__":
    main()
