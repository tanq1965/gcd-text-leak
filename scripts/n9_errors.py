#!/usr/bin/env python3
"""N9: four error types of semi-supervised k-means, following SimGCD (RQ4).

Run from the repository root after results/{ds}_{variant}_s{seed}.json exist (N5, N7):
    python scripts/n9_errors.py                          # A, A+B, A+C on the 3 datasets
    python scripts/n9_errors.py --variants A B C A+B A+C A+D

Method (same scope and cluster->class mapping as ACC v2, see docs/data_formats.md):
- Only rows with is_labeled=0. One Hungarian assignment on those rows (scipy linear_sum_assignment on
  w.max() - w, w[pred, true]), exactly as SimGCD split_cluster_acc_v2. Each cluster gets one class.
- Each image is then: correct, or an error by (true class Old/New) x (mapped class Old/New):
    Old -> other Old  (old_to_old)     Old -> New  (old_to_new)
    New -> Old        (new_to_old)     New -> other New (new_to_new)
- Percentages are over the Old (resp. New) unlabelled images, so per group:
  correct + same-side error + cross-side error = 100, and "correct" equals ACC Old (resp. New).
  The script asserts this against all/old/new stored in every result JSON.
- Seeds: mean x100 and sample std (ddof=1), as scripts/summarize_results.py.

Optional link to N8 (RQ4, "the name is there but CLIP picks the wrong one"): if
retrieval/{ds}_{B,C}_per_image.csv exist (python scripts/n8_retrieval.py), the report adds, per dataset,
the error rate split by the N8 top-1 tag of the image, and the share of images whose top-1 tag names a
wrong class of the dataset that are clustered INTO that named class ("follows the wrong tag").
A and A+B use the B tags, A+C uses the C tags; A is the image-only reference.

Outputs: errors/n9_errors_seeds.csv, errors/n9_errors.csv, errors/n9_errors.md, figures/n9_errors.png.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("cub", "scars", "aircraft")
DS_TITLE = {"cub": "CUB", "scars": "Stanford Cars", "aircraft": "FGVC-Aircraft"}
DEFAULT_VARIANTS = ("A", "A+B", "A+C")
KINDS = ("old_correct", "old_to_old", "old_to_new", "new_correct", "new_to_old", "new_to_new")
TAGS_FOR = {"A": "B", "A+B": "B", "A+C": "C", "B": "B", "C": "C"}
ATOL = 1e-9


def load_split(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "image_id": np.array([r["image_id"] for r in rows]),
        "label": np.array([int(r["label"]) for r in rows]),
        "is_labeled": np.array([int(r["is_labeled"]) for r in rows]),
        "is_old": np.array([int(r["is_old"]) for r in rows]),
    }


def hungarian_map(y_true: np.ndarray, y_pred: np.ndarray) -> dict[int, int]:
    """cluster id -> class id, the same assignment as SimGCD split_cluster_acc_v2."""
    size = int(max(y_pred.max(), y_true.max())) + 1
    w = np.zeros((size, size), dtype=int)
    np.add.at(w, (y_pred, y_true), 1)
    rows, cols = linear_sum_assignment(w.max() - w)
    return {int(r): int(c) for r, c in zip(rows, cols)}


def error_types(split: dict, pred: np.ndarray) -> dict:
    """Per unlabelled image: mapped class and error kind; plus counts per kind."""
    u = np.flatnonzero(split["is_labeled"] == 0)
    y_true, y_pred = split["label"][u], np.asarray(pred)[u]
    old_classes = set(split["label"][split["is_old"] == 1].tolist())
    mapping = hungarian_map(y_true, y_pred)
    mapped = np.array([mapping[int(c)] for c in y_pred])
    true_old = split["is_old"][u] == 1
    mapped_old = np.array([int(c) in old_classes for c in mapped])
    correct = mapped == y_true
    kind = np.where(correct, np.where(true_old, "old_correct", "new_correct"),
                    np.where(true_old, np.where(mapped_old, "old_to_old", "old_to_new"),
                             np.where(mapped_old, "new_to_old", "new_to_new")))
    counts = {k: int((kind == k).sum()) for k in KINDS}
    counts.update(n_old=int(true_old.sum()), n_new=int((~true_old).sum()))
    return {"index": u, "mapped": mapped, "kind": kind, "counts": counts}


def check_against_json(counts: dict, record: dict, name: str) -> None:
    """Correct counts must reproduce ACC All/Old/New of the result file exactly."""
    n_all = counts["n_old"] + counts["n_new"]
    got = {"all": (counts["old_correct"] + counts["new_correct"]) / n_all,
           "old": counts["old_correct"] / counts["n_old"], "new": counts["new_correct"] / counts["n_new"]}
    for key, value in got.items():
        if abs(value - record[key]) > ATOL:
            raise ValueError(f"{name}: {key} from Hungarian = {value:.6f} but result JSON has {record[key]:.6f}")


def result_files(results: Path, ds: str, variant: str) -> list[tuple[int, Path]]:
    pattern = re.compile(rf"^{re.escape(ds)}_{re.escape(variant)}_s(\d+)\.json$")
    return sorted((int(m.group(1)), p) for p in results.glob(f"{ds}_*.json") if (m := pattern.match(p.name)))


def pct_row(counts: dict) -> dict:
    out = {}
    for k in KINDS:
        denom = counts["n_old"] if k.startswith("old") else counts["n_new"]
        out[k] = 100 * counts[k] / denom
    return out


def aggregate(seed_rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in seed_rows:
        groups.setdefault((row["dataset"], row["variant"]), []).append(row)
    out = []
    for (ds, variant), rows in groups.items():
        agg = {"dataset": ds, "variant": variant, "seeds": len(rows),
               "n_old": rows[0]["n_old"], "n_new": rows[0]["n_new"]}
        for k in KINDS:
            vals = np.array([r[f"{k}_pct"] for r in rows])
            agg[f"{k}_mean"] = round(float(vals.mean()), 4)
            agg[f"{k}_std"] = round(float(vals.std(ddof=1)) if len(vals) > 1 else 0.0, 4)
        out.append(agg)
    return out


def load_n8(n8_dir: Path, ds: str, ver: str) -> dict[str, dict] | None:
    path = n8_dir / f"{ds}_{ver}_per_image.csv"
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {r["image_id"]: r for r in csv.DictReader(handle)}


def n8_link(split: dict, info: dict, n8: dict[str, dict]) -> dict:
    """Error rate by N8 top-1 kind, and share of wrong-tag images clustered into the named class."""
    out: dict[str, float] = {}
    ids = split["image_id"][info["index"]]
    wrong = ~np.isin(info["kind"], ("old_correct", "new_correct"))
    top1 = np.array([n8[i]["top1_kind"] for i in ids])
    named = np.array([int(n8[i]["top1_label"]) if n8[i]["top1_label"] != "" else -1 for i in ids])
    for label, kinds in (("correct", ("correct",)), ("wrong", ("wrong_old", "wrong_new")), ("other", ("other", "no_tag"))):
        sel = np.isin(top1, kinds)
        out[f"n_{label}"] = int(sel.sum())
        out[f"err_{label}"] = 100 * float(wrong[sel].mean()) if sel.any() else float("nan")
    sel = np.isin(top1, ("wrong_old", "wrong_new"))
    out["follow_wrong_tag"] = 100 * float((info["mapped"][sel] == named[sel]).mean()) if sel.any() else float("nan")
    return out


def fmt(mean: float, std: float) -> str:
    return f"{mean:.1f} ± {std:.1f}"


def to_markdown(summary: list[dict], variants: list[str], link: list[dict]) -> str:
    by = {(r["dataset"], r["variant"]): r for r in summary}
    lines = [
        "# N9 — Bốn loại lỗi phân cụm (RQ4)", "",
        "Theo SimGCD. Chỉ ảnh không nhãn; dùng đúng phép ghép Hungarian của ACC v2 (một ánh xạ cụm → lớp). "
        "Tỉ lệ % trên số ảnh Old (hoặc New) không nhãn, nên mỗi nhóm: đúng + nhầm cùng phía + nhầm chéo phía = 100, "
        "và cột *đúng* bằng ACC Old/New của N5/N7 (script kiểm khớp từng file). Trung bình ± std qua 5 seed (ddof=1).", "",
        "## Bảng 1 — Thành phần lỗi", "",
        "| Tập | Bản | Old: đúng | Old → Old khác | Old → New | New: đúng | New → Old | New → New khác |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in DATASETS:
        for v in variants:
            r = by.get((ds, v))
            if r is None:
                continue
            lines.append(f"| {DS_TITLE[ds]} | {v} | " + " | ".join(fmt(r[f'{k}_mean'], r[f'{k}_std']) for k in KINDS) + " |")
    if "A" in variants:
        others = [v for v in variants if v != "A"]
        lines += ["", "## Bảng 2 — Thay đổi so với A (điểm %, theo trung bình)", "",
                  "Dương ở cột lỗi = bản ghép text làm loại lỗi đó tăng so với chỉ ảnh.", "",
                  "| Tập | So sánh | Old → Old khác | Old → New | New → Old | New → New khác |", "|---|---|---:|---:|---:|---:|"]
        for ds in DATASETS:
            base = by.get((ds, "A"))
            for v in others:
                r = by.get((ds, v))
                if base is None or r is None:
                    continue
                cells = [f"{r[f'{k}_mean'] - base[f'{k}_mean']:+.1f}" for k in ("old_to_old", "old_to_new", "new_to_old", "new_to_new")]
                lines.append(f"| {DS_TITLE[ds]} | {v} − A | " + " | ".join(cells) + " |")
    if link:
        lines += ["", "## Bảng 3 — Liên hệ N8: lỗi theo tag top-1 của ảnh", "",
                  "Ảnh không nhãn, chia theo tag top-1 của RTG (N8): *đúng tên*, *tên lớp khác trong tập*, *không phải tên lớp*. "
                  "A và A+B dùng tag bản B, A+C dùng tag bản C; A (chỉ ảnh) là mốc so sánh vì không dùng tag. "
                  "*Theo tag sai*: trong các ảnh có tag top-1 là tên lớp khác, % bị phân vào đúng lớp mà tag đó gọi tên. "
                  "Trung bình qua seed.", "",
                  "| Tập | Bản | Tag | Ảnh: đúng tên / tên lớp khác / khác | Lỗi % khi tag đúng tên | Lỗi % khi tag là tên lớp khác "
                  "| Lỗi % khi tag khác | Theo tag sai % |", "|---|---|---|---|---:|---:|---:|---:|"]
        for r in link:
            lines.append(f"| {DS_TITLE[r['dataset']]} | {r['variant']} | {r['tags']} | {r['n_correct']} / {r['n_wrong']} / {r['n_other']} | "
                         f"{r['err_correct']:.1f} | {r['err_wrong']:.1f} | {r['err_other']:.1f} | {r['follow_wrong_tag']:.1f} |")
    lines += ["", "Hình: `figures/n9_errors.png`."]
    return "\n".join(lines) + "\n"


def plot(summary: list[dict], variants: list[str], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"correct": "#2a78d6", "same": "#eb6834", "cross": "#1baf7a"}  # validated categorical slots 1-3
    by = {(r["dataset"], r["variant"]): r for r in summary}
    datasets = [ds for ds in DATASETS if any((ds, v) in by for v in variants)]
    fig, axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 3.8), sharey=True, squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        labels, parts = [], []
        for group, (ok, same, cross) in (("Old", ("old_correct", "old_to_old", "old_to_new")),
                                         ("New", ("new_correct", "new_to_new", "new_to_old"))):
            for v in variants:
                r = by.get((ds, v))
                if r is None:
                    continue
                labels.append(f"{v}\n{group}")
                parts.append((r[f"{ok}_mean"], r[f"{same}_mean"], r[f"{cross}_mean"]))
        x = np.arange(len(labels))
        bottom = np.zeros(len(labels))
        for idx, (name, key) in enumerate((("Đúng", "correct"), ("Nhầm cùng phía", "same"), ("Nhầm chéo phía", "cross"))):
            vals = np.array([p[idx] for p in parts])
            ax.bar(x, vals, bottom=bottom, width=0.7, color=colors[key], label=name, edgecolor="white", linewidth=1.5)
            bottom += vals
        ax.set_xticks(x, labels, fontsize=8)
        ax.set_title(DS_TITLE[ds], fontsize=10)
        ax.set_ylim(0, 100)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e5e0", linewidth=0.6)
        ax.set_axisbelow(True)
        if len(variants) <= 4:
            ax.axvline(len(labels) / 2 - 0.5, color="#c3c2b7", linewidth=0.8)
    axes[0][0].set_ylabel("% ảnh không nhãn của nhóm")
    handles, names = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, names, loc="upper center", ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="N9: four error types (SimGCD) from result JSONs.")
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--results", type=Path, default=None, help="default: <root>/results")
    p.add_argument("--n8-dir", type=Path, default=None, help="default: <root>/retrieval (optional)")
    p.add_argument("--out", type=Path, default=None, help="default: <root>/errors")
    p.add_argument("--figure", type=Path, default=None, help="default: <root>/figures/n9_errors.png")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--variants", nargs="+", default=list(DEFAULT_VARIANTS))
    a = p.parse_args(argv)
    results = a.results or a.root / "results"
    n8_dir = a.n8_dir or a.root / "retrieval"
    out = a.out or a.root / "errors"

    seed_rows, link_rows, missing = [], [], []
    for ds in a.datasets:
        split = load_split(a.root / "splits" / f"{ds}.csv")
        n8 = {ver: load_n8(n8_dir, ds, ver) for ver in ("B", "C")}
        for variant in a.variants:
            files = result_files(results, ds, variant)
            if not files:
                missing.append(f"{ds}_{variant}")
                continue
            link_acc: list[dict] = []
            for seed, path in files:
                record = json.loads(path.read_text(encoding="utf-8"))
                if len(record["pred"]) != len(split["label"]):
                    raise ValueError(f"{path.name}: pred has {len(record['pred'])} rows, splits has {len(split['label'])}")
                info = error_types(split, np.asarray(record["pred"], dtype=int))
                check_against_json(info["counts"], record, path.name)
                pct = pct_row(info["counts"])
                seed_rows.append({"dataset": ds, "variant": variant, "seed": seed, **info["counts"],
                                  **{f"{k}_pct": round(v, 4) for k, v in pct.items()}})
                tags = n8.get(TAGS_FOR.get(variant, ""))
                if tags is not None:
                    link_acc.append(n8_link(split, info, tags))
            if link_acc:
                avg = {k: float(np.mean([x[k] for x in link_acc])) for k in link_acc[0]}
                link_rows.append({"dataset": ds, "variant": variant, "tags": TAGS_FOR[variant], **avg,
                                  **{k: int(link_acc[0][k]) for k in ("n_correct", "n_wrong", "n_other")}})
    if not seed_rows:
        print("Không tìm thấy file kết quả nào", file=sys.stderr)
        return 1
    if missing:
        print(f"CẢNH BÁO: thiếu kết quả cho {', '.join(missing)}", file=sys.stderr)

    variants = [v for v in a.variants if any(r["variant"] == v for r in seed_rows)]
    summary = aggregate(seed_rows)
    write_csv(out / "n9_errors_seeds.csv", seed_rows)
    write_csv(out / "n9_errors.csv", summary)
    if link_rows:
        write_csv(out / "n9_n8_link.csv", [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()} for r in link_rows])
    md = to_markdown(summary, variants, link_rows)
    (out / "n9_errors.md").write_text(md, encoding="utf-8", newline="\n")
    plot(summary, variants, a.figure or a.root / "figures" / "n9_errors.png")
    print(md)
    print(f"Đã ghi {out}/n9_errors*.csv, n9_errors.md và hình. Mọi file khớp ACC của JSON ({len(seed_rows)} lần chạy).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
