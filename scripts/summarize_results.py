"""Summarize evaluation results from JSON files in results/.

Calculates Mean ± Std across seeds for All, Old, and New cluster accuracy (v2).
Can export to CSV, Markdown, and print formatted summary tables.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

DATASET_ORDER = ["cub", "scars", "aircraft"]
VARIANT_ORDER = [
    "A", "B", "A+B", "A+B_a0.25", "A+B_a0.75",
    "C", "A+C", "A+C_a0.25", "A+C_a0.75",
    "D", "A+D", "A+D_a0.25", "A+D_a0.75",
]


def parse_result_filename(filename: str) -> tuple[str, str, int] | None:
    """Parse {dataset}_{variant}_s{seed}.json into (dataset, variant, seed)."""
    match = re.match(r"^([a-zA-Z0-9]+)_(.+)_s(\d+)\.json$", filename)
    if not match:
        return None
    ds, variant, seed = match.groups()
    return ds, variant, int(seed)


def load_results(results_dir: str | Path) -> pd.DataFrame:
    """Read all result JSON files from results_dir into a DataFrame."""
    dir_path = Path(results_dir)
    if not dir_path.is_dir():
        return pd.DataFrame()

    records = []
    for p in sorted(dir_path.glob("*.json")):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"Warning: could not read {p}: {e}")
            continue

        cfg = data.get("config", {})
        parsed = parse_result_filename(p.name)

        ds = cfg.get("dataset") or (parsed[0] if parsed else "unknown")
        variant = cfg.get("variant") or (parsed[1] if parsed else "unknown")
        seed = cfg.get("seed") if "seed" in cfg else (parsed[2] if parsed else None)
        alpha = cfg.get("alpha")

        records.append({
            "dataset": str(ds),
            "variant": str(variant),
            "seed": int(seed) if seed is not None else -1,
            "alpha": float(alpha) if alpha is not None else None,
            "all": float(data["all"]),
            "old": float(data["old"]),
            "new": float(data["new"]),
            "n_empty_clusters": int(data.get("n_empty_clusters", 0)),
            "file": p.name,
            "path": str(p),
        })

    return pd.DataFrame(records)


def compute_summary(df: pd.DataFrame, as_percent: bool = True) -> pd.DataFrame:
    """Group by (dataset, variant), compute Mean ± Std for all, old, and new."""
    if df.empty:
        return pd.DataFrame()

    scale = 100.0 if as_percent else 1.0
    metric_cols = ["all", "old", "new"]

    rows = []
    grouped = df.groupby(["dataset", "variant"], sort=False)
    for (ds, variant), group in grouped:
        n_seeds = len(group)
        seeds_list = sorted(group["seed"].tolist())
        row = {
            "dataset": ds,
            "variant": variant,
            "n_seeds": n_seeds,
            "seeds": ",".join(str(s) for s in seeds_list),
        }

        for m in metric_cols:
            vals = group[m].to_numpy() * scale
            mean = float(np.mean(vals))
            std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            row[f"{m}_mean"] = mean
            row[f"{m}_std"] = std
            row[f"{m}_str"] = f"{mean:.2f} ± {std:.2f}"

        rows.append(row)

    res_df = pd.DataFrame(rows)

    # Sort logically according to group conventions
    def sort_key(row):
        ds_idx = DATASET_ORDER.index(row["dataset"]) if row["dataset"] in DATASET_ORDER else 999
        var_idx = VARIANT_ORDER.index(row["variant"]) if row["variant"] in VARIANT_ORDER else 999
        return (ds_idx, var_idx, row["variant"])

    res_df["_sort"] = res_df.apply(sort_key, axis=1)
    res_df = res_df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)
    return res_df


def format_table(summary_df: pd.DataFrame, title: str = "TỔNG HỢP KẾT QUẢ ĐÁNH GIÁ (ACC %)") -> str:
    """Format summary DataFrame as a readable text table."""
    if summary_df.empty:
        return "Không có dữ liệu kết quả."

    display_df = pd.DataFrame({
        "Tập": summary_df["dataset"].str.upper(),
        "Bản": summary_df["variant"],
        "Seeds": summary_df["n_seeds"].astype(str),
        "ACC All (%)": summary_df["all_str"],
        "ACC Old (%)": summary_df["old_str"],
        "ACC New (%)": summary_df["new_str"],
    })

    return f"=== {title} ===\n" + display_df.to_string(index=False)


def format_markdown(summary_df: pd.DataFrame) -> str:
    """Format summary DataFrame as Markdown table without external dependencies."""
    if summary_df.empty:
        return "_Không có dữ liệu kết quả._"

    headers = ["Dataset", "Variant", "Seeds", "ACC All (%)", "ACC Old (%)", "ACC New (%)"]
    rows = [
        [
            str(r["dataset"]).upper(),
            str(r["variant"]),
            str(r["n_seeds"]),
            str(r["all_str"]),
            str(r["old_str"]),
            str(r["new_str"]),
        ]
        for _, r in summary_df.iterrows()
    ]
    col_widths = [len(h) for h in headers]
    for row in rows:
        for i, val in enumerate(row):
            col_widths[i] = max(col_widths[i], len(val))

    header_line = "| " + " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers)) + " |"
    sep_line = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"
    data_lines = [
        "| " + " | ".join(val.ljust(col_widths[i]) for i, val in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([header_line, sep_line] + data_lines)


def format_latex(summary_df: pd.DataFrame) -> str:
    """Format summary DataFrame as LaTeX table rows (useful for report)."""
    if summary_df.empty:
        return "% Empty results"

    lines = [
        r"\begin{tabular}{llcccc}",
        r"\toprule",
        r"\textbf{Dataset} & \textbf{Variant} & \textbf{Seeds} & \textbf{All (\%)} & \textbf{Old (\%)} & \textbf{New (\%)} \\",
        r"\midrule",
    ]
    curr_ds = None
    for _, row in summary_df.iterrows():
        ds = row["dataset"].upper()
        ds_label = ds if ds != curr_ds else ""
        curr_ds = ds
        variant = str(row["variant"]).replace("_", r"\_")
        lines.append(
            f"{ds_label} & {variant} & {row['n_seeds']} & "
            f"${row['all_mean']:.2f} \\pm {row['all_std']:.2f}$ & "
            f"${row['old_mean']:.2f} \\pm {row['old_std']:.2f}$ & "
            f"${row['new_mean']:.2f} \\pm {row['new_std']:.2f}$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize results/*.json")
    parser.add_argument("--results_dir", type=Path, default=Path("results"), help="Folder containing result JSONs")
    parser.add_argument("--out_csv", type=Path, default=None, help="Path to save summary CSV")
    parser.add_argument("--out_md", type=Path, default=None, help="Path to save summary Markdown")
    parser.add_argument("--detailed_csv", type=Path, default=None, help="Path to save per-seed details (Phụ lục B)")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    df = load_results(args.results_dir)
    if df.empty:
        print(f"Không tìm thấy file kết quả nào trong: {args.results_dir}")
        return 0

    print(f"Đã đọc {len(df)} file kết quả từ {args.results_dir}")
    summary = compute_summary(df, as_percent=True)
    print("\n" + format_table(summary))

    if args.out_csv:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.out_csv, index=False, encoding="utf-8")
        print(f"Đã lưu bảng tổng hợp CSV: {args.out_csv}")

    if args.out_md:
        args.out_md.parent.mkdir(parents=True, exist_ok=True)
        args.out_md.write_text(format_markdown(summary), encoding="utf-8")
        print(f"Đã lưu bảng tổng hợp Markdown: {args.out_md}")

    if args.detailed_csv:
        args.detailed_csv.parent.mkdir(parents=True, exist_ok=True)
        # Export individual seed results (Phụ lục B)
        detailed_df = df.copy()
        for col in ["all", "old", "new"]:
            detailed_df[f"{col}_pct"] = (detailed_df[col] * 100).round(2)
        detailed_df.to_csv(args.detailed_csv, index=False, encoding="utf-8")
        print(f"Đã lưu kết quả chi tiết từng seed (Phụ lục B): {args.detailed_csv}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
