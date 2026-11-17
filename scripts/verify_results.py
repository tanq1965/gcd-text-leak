"""Independent verifier for the N5 result grid (results/*.json). Does NOT import src.* or third_party.

ACC definition used here (GCD; confirmed by reading third_party/simgcd/util/cluster_and_log_utils.py:26-79, not imported):
  * Only the rows with is_labeled=0 of splits/{ds}.csv take part; rows with labels are never scored.
  * w[c, t] = number of those rows with predicted cluster c and true label t; w is square, D x D with
    D = max(true label, cluster label) + 1.
  * ONE Hungarian assignment (scipy.optimize.linear_sum_assignment) on the whole matrix, maximising matched rows.
  * All = matched rows / rows; Old = matched rows among is_old=1 rows / is_old=1 rows; New likewise (is_old=0).
    All three use the same cluster -> class mapping, and counts are per sample (not averaged per class).

Checks: V1 pred length and labels | V2 ACC recomputed from pred+splits | V3 config consistency, alpha, seeds, commits |
V4 mean/std (ddof=1) against results/summary_n5.csv | V5 feature id files (only if feats/ exists).
V3 details: k, n_init, max_iter, tolerance and commit must be identical across the seeds of one (dataset, variant) (FAIL);
feats, feats2, splits are compared by file NAME only (directories may differ); device differences are a WARN; several commits
in the whole grid (but one per group) are a WARN that lists the commits and a `git diff --stat` command to confirm src/ is unchanged.
Exit code: 0 all fine, 1 any FAIL, 2 no result files. WARN never changes the exit code.

Run from the repo root:  python -m scripts.verify_results [--results_dir results] [--splits_dir splits] [--feats_dir feats]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except Exception:  # captured/replaced streams (e.g. under pytest) may not support it
        pass

FILE_RE = re.compile(r"^(?P<ds>[A-Za-z0-9]+)_(?P<variant>.+)_s(?P<seed>\d+)\.json$")
VARIANT_RE = re.compile(r"^(?P<base>[A-Z](?:\+[A-Z])?)(?:_a(?P<alpha>[0-9]*\.?[0-9]+))?$")
SAME_EXACT = ("k", "n_init", "max_iter", "tolerance", "commit")   # must be identical across the seeds of a (dataset, variant): FAIL
SAME_BASENAME = ("feats", "feats2", "splits")                      # only the file name is compared (directories may differ): FAIL
SAME_WARN = ("device",)                                            # a difference is only a WARN
CONFIG_REQUIRED = ("dataset", "variant", "seed", "k", "alpha", "n_init", "max_iter", "tolerance", "commit")
TITLES = {"V1": "pred: độ dài và nhãn hợp lệ", "V2": "ACC tính lại (Hungarian) khớp JSON",
          "V3": "nhất quán cấu hình, alpha, seed, commit", "V4": "mean ± std (ddof=1) khớp summary_n5.csv",
          "V5": "file ids của đặc trưng khớp splits"}


# ----------------------------------------------------------------------------------------------- ACC
def recompute_acc(y_true: np.ndarray, y_pred: np.ndarray, is_old: np.ndarray, method: str = "maximize") -> dict[str, float]:
    """All/Old/New ACC of (already unlabelled-only) rows with one Hungarian assignment on a square D x D count matrix.

    method="maximize": assignment on w with maximize=True; method="shifted": minimise (w.max() - w) (same optimum value,
    used only to detect ties between optimal assignments).
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    is_old = np.asarray(is_old, dtype=bool)
    d = int(max(y_true.max(), y_pred.max())) + 1
    w = np.zeros((d, d), dtype=np.int64)
    np.add.at(w, (y_pred, y_true), 1)
    if method == "maximize":
        rows, cols = linear_sum_assignment(w, maximize=True)
    else:
        rows, cols = linear_sum_assignment(w.max() - w)
    cluster_to_class = np.full(d, -1, dtype=np.int64)
    cluster_to_class[rows] = cols
    correct = cluster_to_class[y_pred] == y_true
    return {"all": float(correct.mean()),
            "old": float(correct[is_old].mean()) if is_old.any() else float("nan"),
            "new": float(correct[~is_old].mean()) if (~is_old).any() else float("nan")}


def basename(value) -> str | None:
    """File name of a path written with / or \\ (None stays None)."""
    return None if value is None else re.split(r"[\\/]", str(value))[-1]


def expected_alpha(variant: str):
    """alpha implied by a variant name: A, B -> None; A+B -> 0.5; A+B_a0.25 -> 0.25; unknown pattern -> 'unknown'."""
    m = VARIANT_RE.match(variant)
    if not m:
        return "unknown"
    if "+" in m.group("base"):
        return float(m.group("alpha")) if m.group("alpha") else 0.5
    return "unknown" if m.group("alpha") else None


# ----------------------------------------------------------------------------------------------- report
class Section:
    def __init__(self, key: str) -> None:
        self.key, self.checked, self.fails, self.warns, self.skipped = key, 0, [], [], None

    @property
    def status(self) -> str:
        return "BỎ QUA" if self.skipped else ("KHÔNG ĐẠT" if self.fails else "ĐẠT")


def read_splits(path: Path) -> dict:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    return {"image_id": [r["image_id"] for r in rows],
            "label": np.array([int(r["label"]) for r in rows], dtype=np.int64),
            "is_labeled": np.array([int(r["is_labeled"]) for r in rows], dtype=np.int64),
            "is_old": np.array([int(r["is_old"]) for r in rows], dtype=np.int64)}


def verify(results_dir: Path, splits_dir: Path, feats_dir: Path | None = None, summary_csv: Path | None = None,
           seeds: Sequence[int] = (0, 1, 2, 3, 4), tol: float = 1e-9) -> tuple[dict[str, Section], int]:
    """Run V1-V5 and return (sections, exit code)."""
    S = {k: Section(k) for k in TITLES}
    files = sorted(Path(results_dir).glob("*.json")) if Path(results_dir).is_dir() else []
    if not files:
        return S, 2
    splits_cache: dict[str, dict | None] = {}

    def get_splits(ds: str):
        if ds not in splits_cache:
            p = Path(splits_dir) / f"{ds}.csv"
            sp = read_splits(p) if p.is_file() else None
            splits_cache[ds] = sp
            if sp is not None:
                S["V1"].checked += 0
                labels = np.unique(sp["label"])
                if not np.array_equal(labels, np.arange(len(labels))):
                    S["V1"].fails.append(f"splits/{ds}.csv: nhãn không liên tục 0..K-1")
                mixed = [int(l) for l in labels if len(np.unique(sp["is_old"][sp["label"] == l])) > 1]
                if mixed:
                    S["V1"].fails.append(f"splits/{ds}.csv: lớp có cả Old lẫn New: {mixed[:5]}")
        return splits_cache[ds]

    recs = []   # (file name, ds, variant, seed, cfg, data, acc_recomputed|None)
    for p in files:
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            S["V1"].fails.append(f"{p.name}: không đọc được JSON ({type(e).__name__})")
            continue
        S["V1"].checked += 1
        cfg = data.get("config") if isinstance(data, dict) else None
        missing = [k for k in ("all", "old", "new", "pred", "config") if not isinstance(data, dict) or k not in data]
        if missing or not isinstance(cfg, dict):
            S["V1"].fails.append(f"{p.name}: thiếu khoá {missing or ['config']}")
            continue
        miss_cfg = [k for k in CONFIG_REQUIRED if k not in cfg]
        if miss_cfg:
            S["V1"].fails.append(f"{p.name}: config thiếu {miss_cfg}")
            continue
        ds = str(cfg["dataset"])
        sp = get_splits(ds)
        if sp is None:
            S["V1"].fails.append(f"{p.name}: không có {Path(splits_dir) / (ds + '.csv')}")
            recs.append((p.name, ds, str(cfg["variant"]), cfg["seed"], cfg, data, None))
            continue
        pred = data["pred"]
        n = len(sp["label"])
        acc = None
        ok_v1 = True
        if len(pred) != n:
            S["V1"].fails.append(f"{p.name}: len(pred)={len(pred)} khác số hàng splits {n}")
            ok_v1 = False
        else:
            arr = np.asarray(pred)
            if arr.dtype.kind not in "iu":
                S["V1"].fails.append(f"{p.name}: pred không phải số nguyên (dtype {arr.dtype})")
                ok_v1 = False
            else:
                k_true = int(np.unique(sp["label"]).size)
                if int(cfg["k"]) != k_true:
                    S["V1"].fails.append(f"{p.name}: config.k={cfg['k']} khác số lớp trong splits {k_true}")
                if arr.min() < 0 or arr.max() >= int(cfg["k"]):
                    S["V1"].fails.append(f"{p.name}: nhãn pred ngoài [0, {int(cfg['k']) - 1}] (min {arr.min()}, max {arr.max()})")
                    ok_v1 = False
                lab = sp["is_labeled"] == 1
                bad = int((arr[lab] != sp["label"][lab]).sum()) if ok_v1 else 0
                if bad:
                    S["V1"].warns.append(f"{p.name}: {bad} hàng có nhãn có pred khác nhãn thật")
        if data.get("pred_scope") != "all_rows":
            S["V1"].fails.append(f"{p.name}: pred_scope={data.get('pred_scope')!r}, kỳ vọng 'all_rows'")
        # V2
        if ok_v1:
            u = sp["is_labeled"] == 0
            args = (sp["label"][u], np.asarray(pred, dtype=np.int64)[u], sp["is_old"][u] == 1)
            acc = recompute_acc(*args)
            alt = recompute_acc(*args, method="shifted")
            S["V2"].checked += 1
            for m in ("all", "old", "new"):
                if not np.isfinite(acc[m]) or abs(acc[m] - float(data[m])) > tol:
                    S["V2"].fails.append(f"{p.name}: {m} JSON={float(data[m])!r} tính lại={acc[m]!r}")
            if any(abs(acc[m] - alt[m]) > tol for m in ("all", "old", "new")):
                S["V2"].warns.append(f"{p.name}: có nhiều nghiệm Hungarian tối ưu (All {acc['all']:.6f}/{alt['all']:.6f}, "
                                     f"Old {acc['old']:.6f}/{alt['old']:.6f}, New {acc['new']:.6f}/{alt['new']:.6f})")
        recs.append((p.name, ds, str(cfg["variant"]), cfg["seed"], cfg, data, acc))

    # V3
    groups: dict[tuple[str, str], list] = {}
    for name, ds, variant, seed, cfg, data, acc in recs:
        m = FILE_RE.match(name)
        if not m or (m.group("ds"), m.group("variant"), int(m.group("seed"))) != (ds, variant, int(seed)):
            S["V3"].fails.append(f"{name}: tên file không khớp config (dataset={ds}, variant={variant}, seed={seed})")
        exp = expected_alpha(variant)
        alpha = cfg["alpha"]
        if exp == "unknown":
            S["V3"].warns.append(f"{name}: không có quy tắc alpha cho biến thể {variant!r} (config.alpha={alpha})")
        elif (exp is None) != (alpha is None) or (exp is not None and abs(float(alpha) - exp) > 1e-12):
            S["V3"].fails.append(f"{name}: config.alpha={alpha} không khớp biến thể {variant!r} (kỳ vọng {exp})")
        groups.setdefault((ds, variant), []).append((name, int(seed), cfg))
    S["V3"].checked = len(recs)
    for (ds, variant), items in sorted(groups.items()):
        for fld in SAME_EXACT:
            vals = {json.dumps(c.get(fld), sort_keys=True) for _, _, c in items}
            if len(vals) > 1:
                S["V3"].fails.append(f"{ds}/{variant}: {fld} khác nhau giữa các seed: {sorted(vals)}")
        for fld in SAME_BASENAME:
            vals = {basename(c.get(fld)) for _, _, c in items}
            if len(vals) > 1:
                S["V3"].fails.append(f"{ds}/{variant}: tên file {fld} khác nhau giữa các seed: {sorted(map(str, vals))}")
        for fld in SAME_WARN:
            vals = {json.dumps(c.get(fld), sort_keys=True) for _, _, c in items}
            if len(vals) > 1:
                S["V3"].warns.append(f"{ds}/{variant}: {fld} khác nhau giữa các seed: {sorted(vals)}")
        have = sorted(s for _, s, _ in items)
        dup = sorted({s for s in have if have.count(s) > 1})
        missing = [s for s in seeds if s not in have]
        if dup:
            S["V3"].fails.append(f"{ds}/{variant}: trùng seed {dup}")
        if missing:
            S["V3"].fails.append(f"{ds}/{variant}: thiếu seed {missing} (có {have})")
        extra = [s for s in have if s not in seeds]
        if extra:
            S["V3"].warns.append(f"{ds}/{variant}: seed ngoài danh sách kỳ vọng {extra}")
    commits = sorted({str(c.get("commit")) for _, _, _, _, c, _, _ in recs})
    if len(commits) > 1:       # across groups this is only a WARN; a clash inside one group is already a FAIL above
        cmds = "; ".join(f"git diff --stat {commits[0]} {c} -- src/ scripts/" for c in commits[1:])
        S["V3"].warns.append(f"lưới dùng {len(commits)} commit khác nhau: {commits}. Tự xác nhận src/ không đổi bằng: {cmds}")

    # V4
    if summary_csv is None or not Path(summary_csv).is_file():
        S["V4"].skipped = f"không có {summary_csv}"
    else:
        summ = pd.read_csv(summary_csv)
        by_group: dict[tuple[str, str], list] = {}
        for name, ds, variant, seed, cfg, data, acc in recs:
            vals = acc if acc is not None else {m: float(data[m]) for m in ("all", "old", "new")}
            by_group.setdefault((ds, variant), []).append(vals)
        S["V4"].checked = len(by_group)
        csv_groups = {(r["dataset"], r["variant"]): r for _, r in summ.iterrows()}
        for g in sorted(set(by_group) - set(csv_groups)):
            S["V4"].fails.append(f"{g}: có trong results nhưng không có trong {Path(summary_csv).name}")
        for g in sorted(set(csv_groups) - set(by_group)):
            S["V4"].fails.append(f"{g}: có trong {Path(summary_csv).name} nhưng không có file kết quả")
        for g, lst in sorted(by_group.items()):
            if g not in csv_groups:
                continue
            row = csv_groups[g]
            if int(row["n_seeds"]) != len(lst):
                S["V4"].fails.append(f"{g}: n_seeds CSV={row['n_seeds']} tính lại={len(lst)}")
            for m in ("all", "old", "new"):
                v = np.array([x[m] for x in lst]) * 100.0
                mean, std = float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0
                for nm, got, want in ((f"{m}_mean", float(row[f"{m}_mean"]), mean), (f"{m}_std", float(row[f"{m}_std"]), std)):
                    if abs(got - want) > 1e-6:
                        S["V4"].fails.append(f"{g}: {nm} CSV={got!r} tính lại={want!r}")

    # V5
    if feats_dir is None or not Path(feats_dir).is_dir():
        S["V5"].skipped = f"không có thư mục {feats_dir}"
    else:
        for ds in sorted({r[1] for r in recs}):
            sp = get_splits(ds)
            if sp is None:
                continue
            for fname in (f"{ds}_img_ids.npy", f"{ds}_txt_B_ids.npy"):
                p = Path(feats_dir) / fname
                if not p.is_file():
                    S["V5"].warns.append(f"{fname}: không có file")
                    continue
                S["V5"].checked += 1
                ids = np.load(p, allow_pickle=False)
                if ids.dtype.kind != "U":
                    S["V5"].fails.append(f"{fname}: dtype {ids.dtype}, kỳ vọng chuỗi (<U)")
                elif len(ids) != len(sp["image_id"]) or ids.tolist() != sp["image_id"]:
                    S["V5"].fails.append(f"{fname}: ids không bằng splits/{ds}.csv theo đúng thứ tự (độ dài {len(ids)} vs {len(sp['image_id'])})")
        if S["V5"].checked == 0 and not S["V5"].fails:
            S["V5"].skipped = "feats/ không có file ids nào để kiểm"
    code = 1 if any(s.fails for s in S.values()) else 0
    return S, code


def format_report(S: dict[str, Section], max_lines: int = 30) -> str:
    out = ["=== KẾT QUẢ KIỂM LƯỚI (verify_results) ===", f"{'Mục':4s} {'Kết quả':10s} {'Đã kiểm':8s} Nội dung"]
    for k, s in S.items():
        out.append(f"{k:4s} {s.status:10s} {s.checked:<8d} {TITLES[k]}" + (f"  [{s.skipped}]" if s.skipped else ""))
    for k, s in S.items():
        for kind, msgs in (("LỖI", s.fails), ("CẢNH BÁO", s.warns)):
            if msgs:
                out.append(f"\n{k} {kind} ({len(msgs)}):")
                out += [f"  - {m}" for m in msgs[:max_lines]]
                if len(msgs) > max_lines:
                    out.append(f"  ... còn {len(msgs) - max_lines} dòng")
    return "\n".join(out)


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Independent verification of results/*.json")
    p.add_argument("--results_dir", type=Path, default=Path("results"))
    p.add_argument("--splits_dir", type=Path, default=Path("splits"))
    p.add_argument("--feats_dir", type=Path, default=Path("feats"))
    p.add_argument("--summary_csv", type=Path, default=None, help="default: <results_dir>/summary_n5.csv")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--tol", type=float, default=1e-9)
    a = p.parse_args(argv)
    S, code = verify(a.results_dir, a.splits_dir, a.feats_dir, a.summary_csv or a.results_dir / "summary_n5.csv", a.seeds, a.tol)
    if code == 2:
        print(f"Không tìm thấy file kết quả nào trong {a.results_dir}")
        return 2
    print(format_report(S))
    print("\nTỔNG: " + ("ĐẠT (có thể còn cảnh báo)" if code == 0 else "KHÔNG ĐẠT"))
    return code


if __name__ == "__main__":
    sys.exit(main())
