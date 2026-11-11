"""Export splits/{ds}.csv (train set of SimGCD's SSB split) plus splits/{ds}_classmap.csv.

Run from the repo root (on Kaggle, where the images are):
    python -m src.export_splits --dataset cub --data-root <dir that contains CUB_200_2011/>
`--data-root` is the dataset's own root as the SimGCD loader expects it:
cub -> contains CUB_200_2011/; scars -> contains devkit/, cars_train/, cars_test/; aircraft -> contains data/.
Split seed is fixed at 0: SimGCD's subsample_instances calls np.random.seed(0) itself, so it cannot be changed.
"""
from __future__ import annotations

import argparse
import os
import sys
from argparse import Namespace
from pathlib import Path

import pandas as pd

from src import tp_env

DATASETS = ("cub", "scars", "aircraft")
SPLIT_SEED = 0
PROP_TRAIN_LABELS = 0.5
EXPECTED_TRAIN_ROWS = {"cub": 5994, "scars": 8144, "aircraft": 6667}
ROOT_VAR = {"cub": "cub_root", "scars": "car_root", "aircraft": "aircraft_root"}
COLUMNS = ["image_id", "path", "label", "is_labeled", "is_old"]
CLASSMAP_COLUMNS = ["raw_label", "label", "is_old", "class_name"]


def config_roots(dataset: str, data_root: Path, repo_root: Path = tp_env.REPO_ROOT) -> dict[str, str]:
    """Values for the fake `config` module: only this dataset's root is real; osr_split_dir is absolute."""
    roots = {name: "<unused>" for name in tp_env.CONFIG_VARS}
    roots[ROOT_VAR[dataset]] = str(data_root)
    roots["osr_split_dir"] = str(repo_root / "third_party" / "simgcd" / "data" / "ssb_splits")
    return roots


def rel_posix(path: str, root: Path) -> str:
    """Path relative to `root`, with '/' separators."""
    return Path(os.path.relpath(path, root)).as_posix()


def subset_rows(subset: object, dataset: str, data_root: Path) -> list[tuple[str, int]]:
    """(path relative to dataset root, original 0-based class id) for every image of a SimGCD subset.

    Reads loader attributes directly (no image is opened).
    """
    if dataset == "cub":
        paths = [f"{subset.base_folder}/{p}" for p in subset.data["filepath"]]
        raw = (subset.data["target"].astype(int) - 1).tolist()
    elif dataset == "scars":
        paths = [rel_posix(p, data_root) for p in subset.data]
        raw = [int(t) - 1 for t in subset.target]
    elif dataset == "aircraft":
        paths = [rel_posix(p, data_root) for p, _ in subset.samples]
        raw = [int(t) for _, t in subset.samples]
    else:
        raise ValueError(f"unknown dataset {dataset!r}")
    return list(zip(paths, raw))


def class_names(dataset: str, subset: object) -> dict[int, str]:
    """{raw_label: class name} when the loader provides names (aircraft: `.classes`, '\\n' stripped), else {}."""
    if dataset == "aircraft":
        return {i: str(name).strip() for i, name in enumerate(subset.classes)}
    return {}


def build_splits(dataset: str, data_root: Path,
                 repo_root: Path = tp_env.REPO_ROOT) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    """Run SimGCD get_class_splits + get_datasets and return (splits, classmap, {n_old, n_new}).

    splits: columns COLUMNS, labelled rows first, each block sorted by image_id.
    classmap: columns CLASSMAP_COLUMNS (label = position in list(train_classes)+list(unlabeled_classes);
    class_name is filled for aircraft only, empty otherwise).
    """
    tp_env.add_third_party_paths(repo_root)
    tp_env.inject_config(config_roots(dataset, data_root, repo_root))
    from data.get_datasets import get_class_splits, get_datasets  # after config injection

    args = get_class_splits(Namespace(dataset_name=dataset, use_ssb_splits=True,
                                      prop_train_labels=PROP_TRAIN_LABELS))
    old_raw, new_raw = list(args.train_classes), list(args.unlabeled_classes)
    n_old, n_new = len(old_raw), len(new_raw)
    merged, _test, _unlabelled_test, _all = get_datasets(dataset, None, None, args)

    blocks = []
    for is_labeled, subset in ((1, merged.labelled_dataset), (0, merged.unlabelled_dataset)):
        rows = [(os.path.basename(p), p, int(subset.target_transform(raw)), is_labeled)
                for p, raw in subset_rows(subset, dataset, data_root)]
        block = pd.DataFrame(rows, columns=["image_id", "path", "label", "is_labeled"])
        blocks.append(block.sort_values(["image_id", "path"], kind="stable"))
    splits = pd.concat(blocks, ignore_index=True)
    splits["is_old"] = (splits["label"] < n_old).astype(int)
    splits = splits[COLUMNS]

    all_raw = [int(c) for c in old_raw + new_raw]
    names = class_names(dataset, merged.labelled_dataset)
    classmap = pd.DataFrame({"raw_label": all_raw, "label": range(len(all_raw)),
                             "is_old": [int(i < n_old) for i in range(len(all_raw))],
                             "class_name": [names.get(raw, "") for raw in all_raw]})
    return splits, classmap[CLASSMAP_COLUMNS], {"n_old": n_old, "n_new": n_new}


def self_check(splits: pd.DataFrame, classmap: pd.DataFrame, dataset: str, n_old: int, n_new: int,
               data_root: Path | None, expected_rows: int | None = None) -> list[tuple[str, str, str]]:
    """Self-check rows (name, value, status) with status in PASS / FAIL / WARN.

    WARN is printed but does not block writing. data_root=None skips the image-file existence check.
    """
    checks: list[tuple[str, str, str]] = []

    def add(name: str, value: object, ok: bool, warn_only: bool = False) -> None:
        checks.append((name, str(value), "PASS" if ok else ("WARN" if warn_only else "FAIL")))

    exp = EXPECTED_TRAIN_ROWS[dataset] if expected_rows is None else expected_rows
    add(f"rows (expected {exp})", len(splits), len(splits) == exp)
    add("classes Old/New = len(train_classes)/len(unlabeled_classes)", f"{n_old}/{n_new}", n_old > 0 and n_new > 0)
    dup = int(splits["image_id"].duplicated().sum())
    add("duplicate image_id", dup, dup == 0)
    bad = int(((splits["is_labeled"] == 1) & (splits["is_old"] == 0)).sum())
    add("is_labeled=1 outside Old classes", bad, bad == 0)
    old = splits[splits["is_old"] == 1]
    frac = float(old["is_labeled"].mean()) if len(old) else float("nan")
    add(f"labelled fraction among Old-class images (expected ~{PROP_TRAIN_LABELS})",
        f"{frac:.4f} ({int(splits['is_labeled'].sum())} labelled)", abs(frac - PROP_TRAIN_LABELS) < 0.01)
    add("labels cover 0..n_old+n_new-1", f"{splits['label'].nunique()} distinct",
        set(splits["label"].unique()) == set(range(n_old + n_new)))
    per_class = old[old["is_labeled"] == 1].groupby("label").size().reindex(range(n_old), fill_value=0)
    add("labelled images per Old class >= 1 (min / max)", f"{per_class.min()} / {per_class.max()}",
        per_class.min() >= 1, warn_only=True)
    add("classmap raw_label unique", classmap["raw_label"].is_unique, classmap["raw_label"].is_unique)
    if data_root is not None:
        missing = sum(1 for p in splits["path"] if not (data_root / p).is_file())
        add("image files missing", missing, missing == 0)
    return checks


def format_check_table(checks: list[tuple[str, str, str]]) -> str:
    """Plain-text PASS/FAIL/WARN table."""
    width = max(len(name) for name, _, _ in checks)
    return "\n".join(f"{status}  {name.ljust(width)}  {value}" for name, value, status in checks)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    """Deterministic CSV: fixed columns, LF line endings, no index."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments."""
    p = argparse.ArgumentParser(description="Export splits/{ds}.csv from SimGCD's SSB split.")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--data-root", required=True, type=Path, help="dataset root as the SimGCD loader expects it")
    p.add_argument("--out", type=Path, default=None, help="default: splits/{ds}.csv")
    p.add_argument("--classmap-out", type=Path, default=None, help="default: splits/{ds}_classmap.csv")
    p.add_argument("--no-check-files", action="store_true", help="skip the image-file existence check")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Build, self-check, and write both CSVs. Returns 1 (and writes nothing) if any check is FAIL."""
    a = parse_args(argv)
    out = a.out or tp_env.REPO_ROOT / "splits" / f"{a.dataset}.csv"
    classmap_out = a.classmap_out or out.with_name(f"{a.dataset}_classmap.csv")
    splits, classmap, info = build_splits(a.dataset, a.data_root)
    checks = self_check(splits, classmap, a.dataset, info["n_old"], info["n_new"],
                        None if a.no_check_files else a.data_root)
    print(format_check_table(checks))
    if any(status == "FAIL" for _, _, status in checks):
        print("FAIL: at least one check failed, nothing written")
        return 1
    write_csv(splits, out)
    write_csv(classmap, classmap_out)
    print(f"wrote {out} ({len(splits)} rows) and {classmap_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
