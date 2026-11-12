"""Semi-supervised k-means (GCD) on frozen features, scored with SimGCD's `v2` ACC.

Run from the repo root:
    python -m src.eval --dataset cub --variant A --feats feats/cub_img.npy --seed 0
    python -m src.eval --dataset cub --variant A+B --feats feats/cub_img.npy --feats2 feats/cub_txt_B.npy --alpha 0.5
Feature file X.npy is paired with X_ids.npy (or the legacy X.ids.txt).
Scores only the rows with is_labeled=0; mask = is_old of those rows.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from src import tp_env

SPLIT_COLUMNS = ["image_id", "path", "label", "is_labeled", "is_old"]
TOLERANCE = 1e-4          # as in GCD's test_kmeans_semi_sup
PAIRWISE_BATCH = 1024     # idem
DEFAULT_MAX_ITER = 200    # GCD bash_scripts/k_means.sh
DEFAULT_N_INIT = 10       # GCD k_means.py CLI default; bash_scripts use 100, reduced for compute (see N1 benchmark)
NORM_ATOL = 5e-3


def load_splits(path: Path) -> pd.DataFrame:
    """Read a splits CSV; check columns and that image_id is unique."""
    df = pd.read_csv(path, dtype={"image_id": str, "path": str})
    missing = [c for c in SPLIT_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if df["image_id"].duplicated().any():
        raise ValueError(f"{path}: duplicated image_id")
    return df[SPLIT_COLUMNS].copy()


def load_feats(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read X.npy [N, D] and its image IDs from X_ids.npy or legacy X.ids.txt."""
    feats = np.load(path)
    if feats.ndim != 2:
        raise ValueError(f"{path}: expected a 2-D array, got shape {feats.shape}")
    ids_npy = path.with_name(f"{path.stem}_ids.npy")
    ids_txt = path.with_suffix(".ids.txt")
    if ids_npy.is_file():
        ids = np.asarray(np.load(ids_npy, allow_pickle=False)).astype(str)
    elif ids_txt.is_file():
        ids = np.array(ids_txt.read_text(encoding="utf-8").splitlines())
    else:
        raise FileNotFoundError(f"missing image IDs: expected {ids_npy} or {ids_txt}")
    if ids.ndim != 1:
        raise ValueError(f"{ids_npy}: expected a 1-D image_id array, got shape {ids.shape}")
    if len(ids) != len(feats):
        raise ValueError(f"{path}: {len(feats)} rows but {len(ids)} ids")
    return feats.astype(np.float32), ids


def align_to_splits(feats: np.ndarray, ids: Sequence[str], split_ids: Sequence[str]) -> np.ndarray:
    """Reorder feature rows to the row order of splits; raise on any image_id mismatch or duplicate."""
    ids, split_ids = [str(i) for i in ids], [str(i) for i in split_ids]
    if len(ids) != feats.shape[0]:
        raise ValueError(f"{feats.shape[0]} feature rows but {len(ids)} ids")
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})[:5]
        raise ValueError(f"duplicated image_id in features, e.g. {dup}")
    if len(set(split_ids)) != len(split_ids):
        raise ValueError("duplicated image_id in splits")
    only_s, only_f = sorted(set(split_ids) - set(ids)), sorted(set(ids) - set(split_ids))
    if only_s or only_f:
        raise ValueError(f"image_id mismatch: {len(only_s)} in splits but not in features (e.g. {only_s[:5]}), "
                         f"{len(only_f)} in features but not in splits (e.g. {only_f[:5]})")
    pos = {i: k for k, i in enumerate(ids)}
    return feats[[pos[i] for i in split_ids]]


def check_unit_norm(x: np.ndarray, name: str, atol: float = NORM_ATOL) -> float:
    """Return max | ||row|| - 1 |; raise if it exceeds atol or is not finite (never re-normalises silently)."""
    dev = float(np.abs(np.linalg.norm(x.astype(np.float64), axis=1) - 1.0).max())
    if not np.isfinite(dev) or dev > atol:
        raise ValueError(f"{name}: rows are not L2-normalised (max | ||x|| - 1 | = {dev:.4g}, atol {atol})")
    return dev


def combine_features(f_img: np.ndarray, f_txt: np.ndarray, alpha: float) -> np.ndarray:
    """F = [sqrt(1-alpha) * F_I , sqrt(alpha) * F_T], shape [N, D_I + D_T]; inputs must be unit-norm."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    if f_img.shape[0] != f_txt.shape[0]:
        raise ValueError(f"row count differs: image {f_img.shape[0]} vs text {f_txt.shape[0]}")
    check_unit_norm(f_img, "image features")
    check_unit_norm(f_txt, "text features")
    out = np.concatenate([np.sqrt(1.0 - alpha) * f_img, np.sqrt(alpha) * f_txt], axis=1).astype(np.float32)
    check_unit_norm(out, "combined features")
    return out


def unlabelled_view(splits: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(row indices with is_labeled=0, their label as int, mask = is_old as bool)."""
    u_idx = np.flatnonzero(splits["is_labeled"].to_numpy() == 0)
    y_true = splits["label"].to_numpy()[u_idx].astype(int)
    mask = splits["is_old"].to_numpy()[u_idx].astype(bool)
    return u_idx, y_true, mask


def run_kmeans(feats: np.ndarray, splits: pd.DataFrame, k: int, seed: int, n_init: int = DEFAULT_N_INIT,
               max_iter: int = DEFAULT_MAX_ITER, device: str = "cpu") -> tuple[np.ndarray, dict[str, object]]:
    """GCD semi-supervised k-means (K_Means.fit_mix). Returns (pred [N] in splits order, run stats).

    fit_mix concatenates [labelled ; unlabelled] in the order given, so labels_[:n_l] are the labelled rows
    and labels_[n_l:] the unlabelled rows; they are scattered back with index arrays (splits need not be sorted).
    Raises RuntimeError if any cluster centre is NaN (typically an empty cluster).
    """
    tp_env.add_third_party_paths()
    import torch
    from methods.clustering.faster_mix_k_means_pytorch import K_Means

    is_lab = splits["is_labeled"].to_numpy() == 1
    l_idx, u_idx = np.flatnonzero(is_lab), np.flatnonzero(~is_lab)
    if len(l_idx) == 0 or len(u_idx) == 0:
        raise ValueError("need both labelled and unlabelled rows")
    torch.manual_seed(seed)
    dev = torch.device(device)
    l_feats = torch.from_numpy(np.ascontiguousarray(feats[l_idx], dtype=np.float32)).to(dev)
    u_feats = torch.from_numpy(np.ascontiguousarray(feats[u_idx], dtype=np.float32)).to(dev)
    l_targets = torch.from_numpy(splits["label"].to_numpy()[l_idx].astype(np.int64)).to(dev)

    km = K_Means(k=k, tolerance=TOLERANCE, max_iterations=max_iter, init="k-means++", n_init=n_init,
                 random_state=seed, n_jobs=None, pairwise_batch_size=PAIRWISE_BATCH, mode=None)
    km.fit_mix(u_feats, l_feats, l_targets)

    nan_rows = torch.isnan(km.cluster_centers_).any(dim=1).nonzero().flatten().tolist()
    if nan_rows:
        raise RuntimeError(f"k-means produced NaN cluster centres for cluster ids {nan_rows[:10]} "
                           f"(empty cluster?); result discarded")
    labels_ = km.labels_.cpu().numpy().astype(np.int64)
    pred = np.full(len(feats), -1, dtype=np.int64)
    pred[l_idx], pred[u_idx] = labels_[:len(l_idx)], labels_[len(l_idx):]
    stats = {"n_empty_clusters": int(k - len(np.unique(pred))), "max_iter": max_iter, "n_init": n_init,
             "tolerance": TOLERANCE, "device": device}
    return pred, stats


def score(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    """SimGCD log_accs_from_preds with eval_funcs=['v2'] (sample-weighted ACC): {all, old, new}."""
    tp_env.add_third_party_paths()
    from util.cluster_and_log_utils import log_accs_from_preds

    all_acc, old_acc, new_acc = log_accs_from_preds(
        y_true=np.asarray(y_true), y_pred=np.asarray(y_pred), mask=np.asarray(mask),
        eval_funcs=["v2"], save_name="n1", print_output=False)
    return {"all": float(all_acc), "old": float(old_acc), "new": float(new_acc)}


def evaluate(splits: pd.DataFrame, feats: np.ndarray, *, k: int, seed: int, n_init: int = DEFAULT_N_INIT,
             max_iter: int = DEFAULT_MAX_ITER, device: str = "cpu") -> dict[str, object]:
    """Cluster, then score on is_labeled=0 rows. Returns {all, old, new, pred (all N rows), pred_scope, run stats}."""
    if feats.shape[0] != len(splits):
        raise ValueError(f"{feats.shape[0]} feature rows vs {len(splits)} split rows")
    pred, stats = run_kmeans(feats, splits, k, seed, n_init, max_iter, device)
    u_idx, y_true, mask = unlabelled_view(splits)
    return {**score(y_true, pred[u_idx], mask), "pred": pred.tolist(), "pred_scope": "all_rows", **stats}


def git_commit(repo_root: Path = tp_env.REPO_ROOT) -> str:
    """Current commit hash, or 'unknown' if git is unavailable."""
    try:
        return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"],
                                       text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line arguments."""
    p = argparse.ArgumentParser(description="Semi-supervised k-means + v2 ACC on frozen features.")
    p.add_argument("--dataset", required=True, choices=("cub", "scars", "aircraft"))
    p.add_argument("--variant", required=True, help="name used in the output file, e.g. A or A+B")
    p.add_argument("--splits", type=Path, default=None, help="default: splits/{ds}.csv")
    p.add_argument("--feats", required=True, type=Path, help="image features X.npy (+ X.ids.txt)")
    p.add_argument("--feats2", type=Path, default=None, help="text features X.npy (+ X.ids.txt)")
    p.add_argument("--alpha", type=float, default=0.5, help="weight of the text part (only with --feats2)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-init", type=int, default=DEFAULT_N_INIT)
    p.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", type=Path, default=None, help="default: results/{ds}_{variant}_s{seed}.json")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Run one (dataset, variant, seed): print ACC All/Old/New and write the result JSON."""
    a = parse_args(argv)
    root = tp_env.REPO_ROOT
    splits = load_splits(a.splits or root / "splits" / f"{a.dataset}.csv")
    feats, ids = load_feats(a.feats)
    f = align_to_splits(feats, ids, splits["image_id"])
    norm_dev = {"feats": check_unit_norm(f, "image features")}
    if a.feats2 is not None:
        feats2, ids2 = load_feats(a.feats2)
        f2 = align_to_splits(feats2, ids2, splits["image_id"])
        norm_dev["feats2"] = check_unit_norm(f2, "text features")
        f = combine_features(f, f2, a.alpha)
    n_classes = int(splits["label"].nunique())
    if int(splits["label"].max()) + 1 != n_classes:
        raise ValueError("labels are not contiguous 0..K-1")
    res = evaluate(splits, f, k=n_classes, seed=a.seed, n_init=a.n_init, max_iter=a.max_iter, device=a.device)
    print(f"{a.dataset} {a.variant} seed {a.seed}: All {res['all']:.4f} | Old {res['old']:.4f} | New {res['new']:.4f}")
    record = {
        "all": res["all"], "old": res["old"], "new": res["new"],
        "pred": res["pred"], "pred_scope": res["pred_scope"], "n_empty_clusters": res["n_empty_clusters"],
        "config": {
            "dataset": a.dataset, "variant": a.variant, "splits": str(a.splits or f"splits/{a.dataset}.csv"),
            "feats": str(a.feats), "feats2": None if a.feats2 is None else str(a.feats2),
            "alpha": None if a.feats2 is None else a.alpha, "seed": a.seed, "k": n_classes,
            "max_iter": res["max_iter"], "n_init": res["n_init"], "tolerance": res["tolerance"],
            "device": res["device"], "norm_max_dev": norm_dev, "commit": git_commit(root),
        },
    }
    out = a.out or root / "results" / f"{a.dataset}_{a.variant}_s{a.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record), encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
