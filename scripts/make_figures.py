"""N10 phần 1: vẽ hình Chương 5 (F1 kết quả chính, F2 độ nhạy α) từ results/summary_n7.csv.

Mọi con số trên hình đọc từ file summary (đơn vị %, std ddof=1), không tính lại ACC.
Chạy từ gốc repo:
    python -X utf8 -m scripts.make_figures [--summary results/summary_n7.csv]
        [--detailed results/detailed_seeds_n7.csv] [--out-dir figures] [--check]

`--check`: tính lại mean/std (ddof=1) từ file detailed (từng seed), so với file summary
và với giá trị thực sự truyền vào hàm vẽ; in chênh lệch tối đa từng cặp.
Hình ghi PNG (300 dpi) và PDF (vector), metadata không chứa ngày giờ nên chạy lại cho file giống hệt.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

DATASETS = ["cub", "scars", "aircraft"]
DS_LABEL = {"cub": "CUB", "scars": "Stanford Cars", "aircraft": "FGVC-Aircraft"}
ALPHAS = [0.25, 0.5, 0.75]

# Tên biến thể thật trong summary: 13 tên.
ALL_VARIANTS = [
    "A", "B", "C", "D", "A+B", "A+C", "A+D",
    "A+B_a0.25", "A+B_a0.75", "A+C_a0.25", "A+C_a0.75", "A+D_a0.25", "A+D_a0.75",
]
F1_VARIANTS = ["A", "B", "C", "D", "A+B", "A+C", "A+D"]
# F2: tên biến thể thật theo từng α (α=0.5 là tên không hậu tố).
F2_SERIES = {
    "A+B": {0.25: "A+B_a0.25", 0.5: "A+B", 0.75: "A+B_a0.75"},
    "A+C": {0.25: "A+C_a0.25", 0.5: "A+C", 0.75: "A+C_a0.75"},
    "A+D": {0.25: "A+D_a0.25", 0.5: "A+D", 0.75: "A+D_a0.75"},
}

# Okabe-Ito (an toàn cho mù màu). Mỗi biến thể một màu cố định ở mọi hình;
# A+X dùng màu của X (B/C/D) và được phân biệt bằng hoa văn / kiểu nét.
COLOR = {
    "A": "#7f7f7f",
    "B": "#0072B2", "C": "#E69F00", "D": "#009E73",
    "A+B": "#0072B2", "A+C": "#E69F00", "A+D": "#009E73",
}
MARKER = {"A+B": "o", "A+C": "s", "A+D": "^"}
LINESTYLE = {"A+B": "-", "A+C": "--", "A+D": "-."}
METRIC_LABEL = {"new": "ACC New (%)", "all": "ACC All (%)", "old": "ACC Old (%)"}

TOL = 1e-9  # sai số float khi tính lại mean/std từ từng seed (thứ tự cộng khác nhau)
FONT = 10
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": FONT,
    "axes.labelsize": FONT + 1,
    "xtick.labelsize": FONT,
    "ytick.labelsize": FONT,
    "legend.fontsize": FONT,
    "axes.titlesize": FONT + 2,
    "pdf.fonttype": 42,
})


def load_summary(path: str | Path) -> pd.DataFrame:
    """Đọc summary và kiểm đủ 13 biến thể × 3 dataset (assert, không bịa số thiếu)."""
    df = pd.read_csv(path, dtype={"dataset": str, "variant": str})
    pairs = set(zip(df["dataset"], df["variant"]))
    expected = {(d, v) for d in DATASETS for v in ALL_VARIANTS}
    missing, extra = sorted(expected - pairs), sorted(pairs - expected)
    assert not missing, f"summary thiếu (dataset, biến thể): {missing}"
    assert not extra, f"summary thừa (dataset, biến thể): {extra}"
    assert len(df) == len(expected), "summary có dòng trùng (dataset, biến thể)"
    return df


def _cell(summary: pd.DataFrame, ds: str, variant: str, metric: str) -> tuple[float, float]:
    row = summary[(summary["dataset"] == ds) & (summary["variant"] == variant)]
    assert len(row) == 1, (ds, variant)
    return float(row[f"{metric}_mean"].iloc[0]), float(row[f"{metric}_std"].iloc[0])


def f1_data(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Dữ liệu F1: dataset, variant, mean, std (đúng giá trị sẽ vẽ)."""
    rows = []
    for ds in DATASETS:
        for v in F1_VARIANTS:
            m, s = _cell(summary, ds, v, metric)
            rows.append({"dataset": ds, "variant": v, "mean": m, "std": s})
    return pd.DataFrame(rows)


def f2_data(summary: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Dữ liệu F2: dataset, series (A+B/A+C/A+D hoặc A), alpha, variant thật, mean, std."""
    rows = []
    for ds in DATASETS:
        m, s = _cell(summary, ds, "A", metric)
        rows.append({"dataset": ds, "series": "A", "alpha": np.nan, "variant": "A",
                     "mean": m, "std": s})
        for series, amap in F2_SERIES.items():
            for a in ALPHAS:
                v = amap[a]
                m, s = _cell(summary, ds, v, metric)
                rows.append({"dataset": ds, "series": series, "alpha": a, "variant": v,
                             "mean": m, "std": s})
    return pd.DataFrame(rows)


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> list[Path]:
    """Ghi PNG 300 dpi + PDF vector, không metadata ngày giờ."""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    png, pdf = out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"
    fig.savefig(png, dpi=300, metadata={"Software": None})
    fig.savefig(pdf, metadata={"CreationDate": None, "Creator": None, "Producer": None})
    plt.close(fig)
    return paths + [png, pdf]


def plot_f1(data: pd.DataFrame, metric: str, out_dir: Path, stem: str) -> list[Path]:
    """F1: cột nhóm theo biến thể, một panel mỗi dataset, trục y 0–100 chung."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), sharey=True)
    x = np.arange(len(F1_VARIANTS))
    for ax, ds in zip(axes, DATASETS):
        sub = data[data["dataset"] == ds].set_index("variant").loc[F1_VARIANTS]
        for i, v in enumerate(F1_VARIANTS):
            mean, std = sub.loc[v, "mean"], sub.loc[v, "std"]
            hatch = "//" if v.startswith("A+") else None
            ax.bar(x[i], mean, width=0.7, color=COLOR[v], edgecolor="black",
                   linewidth=0.8, hatch=hatch, yerr=std, capsize=3,
                   error_kw={"elinewidth": 1.0, "ecolor": "black"})
            ax.text(x[i], min(mean + std + 1.0, 97.5), f"{mean:.1f}", ha="center",
                    va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels(F1_VARIANTS)
        ax.set_ylim(0, 100)
        ax.set_title(DS_LABEL[ds])
        ax.set_xlabel("Biến thể")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel(METRIC_LABEL[metric])
    handles = [
        Patch(facecolor="#7f7f7f", edgecolor="black"),
        tuple(Patch(facecolor=COLOR[v], edgecolor="black") for v in ("B", "C", "D")),
        Patch(facecolor="white", edgecolor="black", hatch="//"),
    ]
    labels = ["A: chỉ ảnh", "B, C, D: chỉ text",
              "A+B, A+C, A+D: ảnh + text, α = 0.5 (màu theo B, C, D)"]
    fig.legend(handles=handles, labels=labels, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.01),
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return _save(fig, out_dir, stem)


def plot_f2(data: pd.DataFrame, metric: str, out_dir: Path, stem: str) -> list[Path]:
    """F2: độ nhạy α, đường nối thẳng + marker, dải ±1 std, đường ngang cho A."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    for ax, ds in zip(axes, DATASETS):
        sub = data[data["dataset"] == ds]
        a = sub[sub["series"] == "A"].iloc[0]
        ax.axhline(a["mean"], color=COLOR["A"], linestyle=":", linewidth=2.0)
        ax.axhspan(a["mean"] - a["std"], a["mean"] + a["std"], color=COLOR["A"], alpha=0.15,
                   linewidth=0)
        for series in F2_SERIES:
            s = sub[sub["series"] == series].sort_values("alpha")
            ax.plot(s["alpha"], s["mean"], color=COLOR[series], marker=MARKER[series],
                    linestyle=LINESTYLE[series], linewidth=2.0, markersize=7)
            ax.fill_between(s["alpha"], s["mean"] - s["std"], s["mean"] + s["std"],
                            color=COLOR[series], alpha=0.18, linewidth=0)
        ax.set_xticks(ALPHAS)
        ax.set_xlim(0.2, 0.8)
        ax.set_xlabel("α (trọng số nhánh text)")
        ax.set_title(DS_LABEL[ds])
        ax.grid(linestyle=":", alpha=0.5)
    axes[0].set_ylabel(METRIC_LABEL[metric])
    handles = [Line2D([0], [0], color=COLOR["A"], linestyle=":", linewidth=2.0,
                      label="A (chỉ ảnh, mốc)")]
    for series in F2_SERIES:
        handles.append(Line2D([0], [0], color=COLOR[series], marker=MARKER[series],
                              linestyle=LINESTYLE[series], linewidth=2.0, markersize=7,
                              label=series))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return _save(fig, out_dir, stem)


# (tên file, hàm dữ liệu, hàm vẽ, metric)
FIGURES = [
    ("f1_main_new", f1_data, plot_f1, "new"),
    ("f1_main_all", f1_data, plot_f1, "all"),
    ("f1_main_old", f1_data, plot_f1, "old"),
    ("f2_alpha_new", f2_data, plot_f2, "new"),
    ("f2_alpha_all", f2_data, plot_f2, "all"),
]


def recompute_from_detailed(detailed: pd.DataFrame) -> pd.DataFrame:
    """mean và std (ddof=1) theo (dataset, variant) từ từng seed; đơn vị %."""
    rows = []
    for (ds, v), g in detailed.groupby(["dataset", "variant"]):
        row = {"dataset": ds, "variant": v, "n": len(g)}
        for m in ("all", "old", "new"):
            vals = g[m].to_numpy(dtype=float) * 100.0
            row[f"{m}_mean"] = float(np.mean(vals))
            row[f"{m}_std"] = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows).set_index(["dataset", "variant"])


def run_check(summary: pd.DataFrame, detailed: pd.DataFrame) -> int:
    """In bảng đã vẽ + chênh lệch tối đa từng cặp. Trả về 0 nếu mọi chênh lệch = 0."""
    rec = recompute_from_detailed(detailed)
    summ = summary.set_index(["dataset", "variant"])
    ok = True
    print(f"detailed: {len(detailed)} dòng; seed/nhóm: {sorted(int(n) for n in rec['n'].unique())}; "
          f"{detailed['variant'].nunique()} biến thể × {detailed['dataset'].nunique()} dataset")
    missing = [(d, v) for d in DATASETS for v in ALL_VARIANTS if (d, v) not in rec.index]
    for d, v in missing:
        print(f"CẢNH BÁO: thiếu {d}/{v} trong detailed, không so được")

    # (1) summary vs tính lại từ detailed, toàn bộ 39 nhóm × 3 metric × mean/std
    d_sr = 0.0
    for key in summ.index:
        if key not in rec.index:
            continue
        for m in ("all", "old", "new"):
            for st in ("mean", "std"):
                d_sr = max(d_sr, abs(summ.loc[key, f"{m}_{st}"] - rec.loc[key, f"{m}_{st}"]))
    print(f"\n[summary vs tính lại từ detailed] max|Δ| = {d_sr:.3e}")
    ok &= d_sr <= TOL

    # (2) giá trị truyền vào hàm vẽ vs summary và vs tính lại
    for stem, dfun, _, metric in FIGURES:
        data = dfun(summary, metric)
        d_ps = d_pr = 0.0
        for _, r in data.iterrows():
            key = (r["dataset"], r["variant"])
            d_ps = max(d_ps, abs(r["mean"] - summ.loc[key, f"{metric}_mean"]),
                       abs(r["std"] - summ.loc[key, f"{metric}_std"]))
            if key in rec.index:
                d_pr = max(d_pr, abs(r["mean"] - rec.loc[key, f"{metric}_mean"]),
                           abs(r["std"] - rec.loc[key, f"{metric}_std"]))
            else:
                print(f"CẢNH BÁO: {stem}: thiếu {key} trong detailed")
        print(f"[{stem}] {len(data)} giá trị vẽ | max|Δ| vẽ vs summary = {d_ps:.3e} | "
              f"vẽ vs detailed = {d_pr:.3e}")
        ok &= d_ps == 0.0 and d_pr <= TOL

    # (3) bảng giá trị đã vẽ
    for stem, dfun, _, metric in FIGURES:
        data = dfun(summary, metric)
        print(f"\n== {stem} ({METRIC_LABEL[metric]}) ==")
        if "series" in data.columns:
            tab = data.assign(cell=data.apply(lambda r: f"{r['mean']:.2f}±{r['std']:.2f}", axis=1))
            tab["col"] = tab["variant"]
            print(tab.pivot(index="dataset", columns="col", values="cell")
                  .reindex(DATASETS)[[v for v in ALL_VARIANTS if v in set(tab["col"])]]
                  .to_string())
        else:
            tab = data.assign(cell=data.apply(lambda r: f"{r['mean']:.2f}±{r['std']:.2f}", axis=1))
            print(tab.pivot(index="dataset", columns="variant", values="cell")
                  .reindex(DATASETS)[F1_VARIANTS].to_string())
    print("\nKẾT QUẢ --check:", "ĐẠT (vẽ vs summary = 0 tuyệt đối; tính lại từ detailed ≤ 1e-9, nhiễu làm tròn float)" if ok else "KHÔNG ĐẠT")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    """CLI: vẽ F1/F2 hoặc --check."""
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", default="results/summary_n7.csv")
    ap.add_argument("--detailed", default="results/detailed_seeds_n7.csv")
    ap.add_argument("--out-dir", default="figures")
    ap.add_argument("--check", action="store_true",
                    help="chỉ kiểm giá trị vẽ so với summary và detailed, không vẽ")
    args = ap.parse_args(argv)

    summary = load_summary(args.summary)
    if args.check:
        detailed = pd.read_csv(args.detailed, dtype={"dataset": str, "variant": str})
        return run_check(summary, detailed)
    out = Path(args.out_dir)
    for stem, dfun, pfun, metric in FIGURES:
        for p in pfun(dfun(summary, metric), metric, out, stem):
            print("wrote", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
