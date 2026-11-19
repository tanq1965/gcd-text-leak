#!/usr/bin/env python3
"""Read-only structural and Git checks before pushing N3.

Run from the repository root:
    python scripts/prepush_n3_check.py

Use --skip-git for data-only validation outside a Git clone. This script never
stages, commits, pushes, or modifies project files.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

EXPECTED = {
    "cub": (200, 100),
    "aircraft": (100, 50),
    "scars": (196, 98),
}
CUB_ALIAS_CLASS_NAMES = {
    "105.Whip_poor_Will",
    "091.Mockingbird",
    "101.White_Pelican",
    "017.Cardinal",
}

CUB_ALIAS_CLASS_NAMES.update({
    "067.Anna_Hummingbird",
    "193.Bewick_Wren",
    "023.Brandt_Cormorant",
    "009.Brewer_Blackbird",
    "115.Brewer_Sparrow",
    "093.Clark_Nutcracker",
    "061.Heermann_Gull",
    "123.Henslow_Sparrow",
    "146.Forsters_Tern",
    "141.Artic_Tern",
    "124.Le_Conte_Sparrow",
    "125.Lincoln_Sparrow",
    "098.Scott_Oriole",
    "180.Wilson_Warbler",
    "122.Harris_Sparrow",
})

AIRCRAFT_ALIAS_CLASS_NAMES = {
    "A320",
    "DC-8",
    "ERJ 145",
    "MD-80",
    "747-400",
    "C-130",
    "DC-6",
    "Challenger 600",
    "Metroliner",
}

class Checks:
    def __init__(self) -> None:
        self.passed = 0
        self.warned = 0
        self.failed = 0

    def report(self, status: str, message: str) -> None:
        colors = {"PASS": "\033[32m", "WARN": "\033[33m", "FAIL": "\033[31m"}
        reset = "\033[0m"
        print(f"{colors[status]}[{status}]{reset} {message}")
        if status == "PASS":
            self.passed += 1
        elif status == "WARN":
            self.warned += 1
        else:
            self.failed += 1

    def require(self, condition: bool, message: str) -> None:
        self.report("PASS" if condition else "FAIL", message)


def read_dict_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_one_column(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        values = []
        for line_number, row in enumerate(csv.reader(handle), 1):
            if len(row) != 1:
                raise ValueError(f"{path}:{line_number}: expected one CSV column, got {len(row)}")
            values.append(row[0])
        return values


def git_output(root: Path, *args: str) -> tuple[int, str]:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=False,
    )
    return result.returncode, (
        result.stdout.strip() or result.stderr.strip()
    )


def check_dataset(root: Path, checks: Checks, dataset: str) -> None:
    total, old_expected = EXPECTED[dataset]
    names_path = root / "names" / f"{dataset}.csv"
    names = read_dict_csv(names_path)
    checks.require(len(names) == total, f"{names_path.relative_to(root)} has {len(names)}/{total} classes")

    labels = [int(row["label"]) for row in names]
    checks.require(labels == list(range(total)), f"{dataset}: labels are contiguous and ordered")
    old_flags = [int(row["is_old"]) for row in names]
    checks.require(
        old_flags == [1] * old_expected + [0] * old_expected,
        f"{dataset}: Old/New order and counts are {old_expected}/{old_expected}",
    )
    normalized = [row["name_norm"].strip().casefold() for row in names]
    checks.require(
        all(normalized) and len(set(normalized)) == total,
        f"{dataset}: normalized names are nonblank and unique",
    )

    summary_path = root / "coverage" / f"{dataset}_summary.csv"
    summary = read_dict_csv(summary_path)
    by_group = {row["group"]: row for row in summary}
    checks.require(
        len(summary) == 3 and set(by_group) == {"All", "Old", "New"},
        f"{summary_path.relative_to(root)} contains All/Old/New rows",
    )
    expected_groups = {"All": total, "Old": old_expected, "New": total - old_expected}
    for group, expected_count in expected_groups.items():
        if group not in by_group:
            continue
        row = by_group[group]
        checks.require(
            int(row["classes"]) == expected_count,
            f"{dataset} {group}: coverage summary has {row['classes']} classes",
        )
        for field in ("tag_pct", "attribute_pct", "either_pct"):
            percentage = float(row[field])
            checks.require(
                0 <= percentage <= 100,
                f"{dataset} {group}: {field}={percentage:g}% is within 0–100%",
            )

    audit_path = root / "coverage" / f"{dataset}_audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8-sig"))
    checks.require(
        audit["classes"] == total
        and audit["old"] == old_expected
        and audit["new"] == total - old_expected,
        f"{dataset}: audit class counts agree with names/",
    )

    unmatched_path = root / "coverage" / f"{dataset}_unmatched_candidates.csv"
    unmatched_rows = read_dict_csv(unmatched_path)
    unmatched_count = len(audit["unmatched_classes"])
    checks.require(
        len(unmatched_rows) == unmatched_count,
        f"{dataset}: unmatched-candidate report has {len(unmatched_rows)}/{unmatched_count} rows",
    )

    source_tags = read_one_column(
        root / "third_party" / "textgcd" / "Lexicon" / "Lexicon_tags.csv"
    )
    source_attrs = read_one_column(
        root / "third_party" / "textgcd" / "Lexicon" / "Lexicon_attributes.csv"
    )

    for version in ("C", "D"):
        output_dir = root / "lexicon" / dataset / version
        tags_path = output_dir / "tags.csv"
        attrs_path = output_dir / "attributes.csv"
        removed_path = output_dir / "removed.csv"
        modified_path = output_dir / "modified.csv"

        tag_rows = read_one_column(tags_path)
        attr_rows = read_one_column(attrs_path)
        removed = read_dict_csv(removed_path)
        modified = read_dict_csv(modified_path)

        removed_tags = [
            row for row in removed
            if row["file"] == "Lexicon_tags.csv"
        ]

        removed_attrs = [
            row for row in removed
            if row["file"] == "Lexicon_attributes.csv"
        ]

        checks.require(
            len(modified) == audit[f"{version}_modified_attributes"],
            f"{dataset} {version}: modified audit count matches JSON",
        )

        if version == "C":
            checks.require(
                len(tag_rows)
                == len(source_tags) - len(removed_tags)
                and len(removed_tags) == audit["C_removed_tags"],
                f"{dataset} C: source/output tag counts agree",
            )

            checks.require(
                len(removed_attrs)
                == audit["C_removed_attributes"],
                f"{dataset} C: removed attribute count matches audit",
            )

        else:
            checks.require(
                tags_path.stat().st_size == 0
                and not tag_rows,
                f"{dataset} D: tags.csv is exactly zero bytes",
            )

            checks.require(
                not removed_tags
                and audit["D_removed_tags"] == len(source_tags),
                f"{dataset} D: all source tags are excluded",
            )

            checks.require(
                len(removed_attrs)
                == audit["D_removed_attributes"],
                f"{dataset} D: removed attribute count matches audit",
            )

            duplicates = [
                row for row in removed_attrs
                if row["rule"] == "D_deduplicate_after_strip"
            ]

            checks.require(
                audit["D_scope"] == "all_attribute_prefixes",
                f"{dataset} D: strip-all scope is enabled",
            )

            checks.require(
                len(duplicates) == audit["D_deduplicated_attributes"],
                f"{dataset} D: duplicate audit count matches JSON",
            )

            checks.require(
                len(attr_rows) == len(set(attr_rows)),
                f"{dataset} D: retained descriptions are unique",
            )

            checks.require(
                all(
                    int(row["line"]) > 0 and row["reason"].strip()
                    for row in duplicates
                ),
                f"{dataset} D: duplicate removals have source and reason",
            )

            checks.require(
                audit["D_removed_attributes"]
                == audit["D_removed_residual_attributes"]
                + audit["D_deduplicated_attributes"],
                f"{dataset} D: removed attribute categories agree",
            )

        checks.require(
            len(attr_rows)
            == len(source_attrs) - len(removed_attrs),
            f"{dataset} {version}: attribute row count is correct",
        )

        checks.require(
            len(removed) == len(removed_tags) + len(removed_attrs),
            f"{dataset} {version}: removed rows have valid file types",
        )

        bad_alias_rows = [
            row
            for row in removed + modified
            if row.get("rule") == "alias" and not row.get("reason", "").strip()
        ]

        residual_rows = [
            row for row in removed
            if row["rule"] == "residual_name"
        ]

        expected_residual = {
            ("aircraft", "C"): {17747},
            ("aircraft", "D"): {17585, 17747},
        }.get((dataset, version), set())

        checks.require(
            {int(row["line"]) for row in residual_rows}
            == expected_residual,
            f"{dataset} {version}: residual_name source rows match review",
        )

        checks.require(
            all(
                row["file"] == "Lexicon_attributes.csv"
                and row["reason"].strip()
                for row in residual_rows
            ),
            f"{dataset} {version}: residual_name rows are audited",
        )

        checks.require(
            not bad_alias_rows,
            f"{dataset} {version}: every alias audit row has a reason",
        )

    approved = {
        row["class_name"]
        for row in audit.get("approved_aliases", [])
    }

    if dataset == "cub":
        checks.require(
            approved == CUB_ALIAS_CLASS_NAMES,
            "CUB audit contains all 19 approved alias classes",
        )

    elif dataset == "aircraft":
        checks.require(
            approved == AIRCRAFT_ALIAS_CLASS_NAMES,
            "Aircraft audit contains all nine approved alias classes",
        )


def check_git(root: Path, checks: Checks) -> None:
    code, branch = git_output(root, "branch", "--show-current")
    if code != 0:
        checks.report("FAIL", "Cannot read current Git branch")
        return
    checks.require(
        bool(branch) and branch not in {"main", "master"},
        f"working on feature branch '{branch}' (not main/master)",
    )

    ignore_path = root / ".gitignore"
    ignore_text = ignore_path.read_text(encoding="utf-8-sig")
    for pattern in (
        "scratch/",
        ".venv/",
        "__pycache__/",
        ".pytest_cache/",
    ):
        checks.require(
            pattern in ignore_text,
            f".gitignore excludes {pattern}",
        )

    for label, args in (
        ("unstaged", ("diff", "--check")),
        ("staged", ("diff", "--cached", "--check")),
    ):
        code, output = git_output(root, *args)
        checks.require(code == 0, f"git diff --check ({label}) is clean" + (f": {output}" if code else ""))

    code, staged_text = git_output(root, "diff", "--cached", "--name-only")
    if code != 0:
        checks.report("FAIL", "Cannot inspect staged files")
        return
    staged = [line for line in staged_text.splitlines() if line]
    forbidden = [
        path
        for path in staged
        if any(
            part in path.split("/")
            for part in (".venv", "alias_test", "scratch", "rtg", "feats")
        )
        or any(part.startswith("backup_") for part in path.split("/"))
        or path.endswith((".npy", ".pkl", ".pt", ".pth", ".jpg", ".jpeg", ".png", ".zip", ".mat"))
    ]
    checks.require(
        not forbidden,
        "no backup, scratch, raw dataset, or model files are staged"
        + (f": {', '.join(forbidden)}" if forbidden else ""),
    )
    if staged:
        print("\nStaged files:")
        for path in staged:
            print(f"  {path}")
    else:
        checks.report("WARN", "Nothing is staged yet; this check never runs git add")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-git",
        action="store_true",
        help="Skip branch/staging checks (for data validation outside a Git clone).",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    checks = Checks()
    print("=== N3 data validation ===")
    try:
        for dataset in ("cub", "aircraft", "scars"):
            check_dataset(root, checks, dataset)
    except Exception as exc:
        checks.report("FAIL", f"Validation stopped: {type(exc).__name__}: {exc}")

    if not args.skip_git:
        print("\n=== Git pre-push checks ===")
        try:
            code, _ = git_output(root, "rev-parse", "--show-toplevel")
            if code != 0:
                checks.report("FAIL", "No Git repository found; use --skip-git for data-only checks")
            else:
                check_git(root, checks)
        except Exception as exc:
            checks.report("FAIL", f"Git check stopped: {type(exc).__name__}: {exc}")

    print(
        f"\nSummary: PASS={checks.passed}  WARN={checks.warned}  FAIL={checks.failed}"
    )
    if checks.failed:
        print("NOT READY: fix all FAIL items and rerun.")
        return 1
    print(
        "N3 structural checks passed. Confirm N2 skips tag embedding/retrieval for empty D/tags.csv."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
