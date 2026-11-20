#!/usr/bin/env python3
"""N8: name-retrieval and name-confusion rates of RTG output (RQ1, RQ4).

Run from the repository root after rtg/{ds}_{B,C}.pkl exist:
    python scripts/n8_retrieval.py                      # B and C, 3 datasets
    python scripts/n8_retrieval.py --versions B C D     # D has no tags, only attribute columns are meaningful

Metrics (per image, then averaged over All/Old/New images of the split):
- correct_top3   : the image's own class name (exact name or N3-approved alias) is one of its top-3 tags.
- top1_correct   : the top-1 tag is the own class name.
- top1_wrong     : "nhận nhầm tên" - the top-1 tag is the name of ANOTHER class of the same dataset.
                   Split further into top1_wrong_old / top1_wrong_new by the class it names.
- top1_other     : the top-1 tag is not a class name of this dataset (other species, generic tag...).
- no_tag         : RTG kept no tag (all scores under threshold, or version D).
  top1_correct + top1_wrong + top1_other + no_tag = 100%.
- attr_correct   : the "X which" prefix of one of the top-2 attributes is the own class name.
- covered_B      : the own class name is a tag in the ORIGINAL lexicon B (coverage/{ds}.csv, in_tag);
  correct_top3_covered = correct_top3 restricted to those images (separates "name available" from
  "name retrieved").

Name matching reuses scripts/n3_lexicon_alias.py (normalize, aliases, tag_name, attr_parts), so N8 counts
exactly what N3 counts as a class name.

Outputs (in --out, default retrieval/):
- {ds}_{ver}_per_image.csv : one row per image (input for N9 error analysis),
- n8_summary.csv           : one row per dataset x version x group,
- n8_summary.md            : RQ1 table for the report.
"""
from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import n3_lexicon_alias as n3  # noqa: E402

DATASETS = ("cub", "scars", "aircraft")
DS_TITLE = {"cub": "CUB", "scars": "Stanford Cars", "aircraft": "FGVC-Aircraft"}
GROUPS = ("All", "Old", "New")
PCT_COLS = ("correct_top3", "correct_top3_covered", "top1_correct", "top1_wrong", "top1_wrong_old",
            "top1_wrong_new", "top1_other", "no_tag", "attr_correct")


def build_name_index(names_csv: Path, ds: str) -> tuple[dict[str, int], dict[int, bool]]:
    """Map every normalized class name/alias -> label, and label -> is_old."""
    classes = n3.load_names(names_csv)
    key_to_label: dict[str, int] = {}
    for row in classes:
        for key in n3.aliases(row["name_norm"], ds):
            if key in key_to_label and key_to_label[key] != row["label"]:
                raise ValueError(f"{ds}: alias {key!r} maps to two classes")
            key_to_label[key] = row["label"]
    return key_to_label, {row["label"]: row["is_old"] for row in classes}


def load_split(split_csv: Path) -> list[dict]:
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label"] = int(row["label"])
        row["is_old"] = row["is_old"] in ("1", "True", "true")
    return rows


def load_covered(coverage_csv: Path) -> dict[int, bool]:
    with coverage_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        return {int(r["label"]): r["in_tag"] == "1" for r in csv.DictReader(handle)}


def classify_image(tags: list[str], attrs: list[str], label: int,
                   key_to_label: dict[str, int], is_old: dict[int, bool]) -> dict:
    """Per-image name metrics for one RTG record."""
    tag_labels = [key_to_label.get(n3.tag_name(t)) for t in tags]
    attr_labels = [key_to_label.get(n3.attr_parts(a)[0] or "") for a in attrs]
    top1 = tag_labels[0] if tag_labels else None
    if not tags:
        top1_kind = "no_tag"
    elif top1 == label:
        top1_kind = "correct"
    elif top1 is not None:
        top1_kind = "wrong_old" if is_old[top1] else "wrong_new"
    else:
        top1_kind = "other"
    return {
        "correct_top3": int(label in tag_labels[:3]),
        "top1_kind": top1_kind,
        "top1_label": "" if top1 is None else top1,
        "attr_correct": int(label in attr_labels[:2]),
    }


def evaluate(ds: str, ver: str, rtg_dir: Path, root: Path = ROOT) -> list[dict]:
    """Return per-image rows for dataset ds, RTG version ver."""
    key_to_label, is_old = build_name_index(root / "names" / f"{ds}.csv", ds)
    split = load_split(root / "splits" / f"{ds}.csv")
    covered = load_covered(root / "coverage" / f"{ds}.csv")
    with (rtg_dir / f"{ds}_{ver}.pkl").open("rb") as handle:
        rtg = pickle.load(handle)

    if [r["image_id"] for r in rtg] != [s["image_id"] for s in split]:
        raise ValueError(f"rtg/{ds}_{ver}.pkl: image order differs from splits/{ds}.csv")
    for s in split:
        if is_old[s["label"]] != s["is_old"]:
            raise ValueError(f"{ds}: is_old of label {s['label']} differs between names/ and splits/")

    rows = []
    for s, r in zip(split, rtg):
        m = classify_image(r["tags"], r["attributes"], s["label"], key_to_label, is_old)
        rows.append({
            "image_id": s["image_id"], "label": s["label"], "is_old": int(s["is_old"]),
            "is_labeled": s["is_labeled"], "covered_B": int(covered[s["label"]]), **m,
            "top1_tag": r["tags"][0] if r["tags"] else "",
        })
    return rows


def summarize(rows: list[dict], ds: str, ver: str) -> list[dict]:
    out = []
    for group in GROUPS:
        g = [r for r in rows if group == "All" or bool(r["is_old"]) == (group == "Old")]
        n = len(g)
        cov = [r for r in g if r["covered_B"]]
        pct = lambda k: round(100 * k / n, 2) if n else float("nan")  # noqa: E731
        kinds = [r["top1_kind"] for r in g]
        out.append({
            "dataset": ds, "version": ver, "group": group, "images": n,
            "covered_B_images": len(cov),
            "correct_top3": pct(sum(r["correct_top3"] for r in g)),
            "correct_top3_covered": round(100 * sum(r["correct_top3"] for r in cov) / len(cov), 2)
            if cov else float("nan"),
            "top1_correct": pct(kinds.count("correct")),
            "top1_wrong": pct(kinds.count("wrong_old") + kinds.count("wrong_new")),
            "top1_wrong_old": pct(kinds.count("wrong_old")),
            "top1_wrong_new": pct(kinds.count("wrong_new")),
            "top1_other": pct(kinds.count("other")),
            "no_tag": pct(kinds.count("no_tag")),
            "attr_correct": pct(sum(r["attr_correct"] for r in g)),
        })
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def to_markdown(summary: list[dict]) -> str:
    lines = [
        "# N8 — Truy xuất đúng tên và nhận nhầm tên (RQ1)", "",
        "Tỉ lệ % trên số ảnh của split (train, cả có nhãn và không nhãn). "
        "*Đúng top-3*: tên lớp đúng nằm trong top-3 tag. *Nhầm top-1*: tag đứng đầu là tên lớp khác "
        "trong cùng tập (tách theo lớp bị gọi tên là Old/New). *Đúng top-3 (có tên)*: chỉ tính ảnh của "
        "lớp có tên trong tag của lexicon B.", "",
        "| Tập | Bản | Nhóm | Ảnh | Đúng top-3 | Đúng top-3 (có tên) | Đúng top-1 | Nhầm top-1 "
        "| ↳ gọi tên Old | ↳ gọi tên New | Top-1 không phải tên lớp | Không tag | Thuộc tính đúng tên |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    f = lambda v: "–" if v != v else f"{v:.1f}"  # noqa: E731  (nan -> –)
    for r in summary:
        lines.append(
            f"| {DS_TITLE[r['dataset']]} | {r['version']} | {r['group']} | {r['images']} | "
            f"{f(r['correct_top3'])} | {f(r['correct_top3_covered'])} | {f(r['top1_correct'])} | "
            f"{f(r['top1_wrong'])} | {f(r['top1_wrong_old'])} | {f(r['top1_wrong_new'])} | "
            f"{f(r['top1_other'])} | {f(r['no_tag'])} | {f(r['attr_correct'])} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rtg-dir", type=Path, default=ROOT / "rtg")
    p.add_argument("--root", type=Path, default=ROOT, help="repo root holding names/, splits/, coverage/")
    p.add_argument("--out", type=Path, default=ROOT / "retrieval")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--versions", nargs="+", default=["B", "C"], choices=["B", "C", "D"])
    args = p.parse_args(argv)

    summary = []
    for ds in args.datasets:
        for ver in args.versions:
            rows = evaluate(ds, ver, args.rtg_dir, args.root)
            write_csv(args.out / f"{ds}_{ver}_per_image.csv", rows)
            summary += summarize(rows, ds, ver)
            if ver == "C":  # risk check in the plan: C must never retrieve a New class name
                new_hits = sum(r["correct_top3"] for r in rows if not r["is_old"])
                if new_hits:
                    print(f"CẢNH BÁO {ds}/C: {new_hits} ảnh New vẫn truy xuất được tên lớp đúng "
                          f"-> lọc tên còn sót, báo N3/N6", file=sys.stderr)
    write_csv(args.out / "n8_summary.csv", summary)
    (args.out / "n8_summary.md").write_text(to_markdown(summary), encoding="utf-8", newline="\n")
    print(to_markdown(summary))
    print(f"Đã ghi {args.out}/n8_summary.csv, n8_summary.md và *_per_image.csv")


if __name__ == "__main__":
    main()
