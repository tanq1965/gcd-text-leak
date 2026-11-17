"""Tests for scripts/verify_results.py on fake data (tests/fake_data.py); no GPU, no real results."""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.verify_results import expected_alpha, recompute_acc, verify
from tests.fake_data import make_blobs

ROOT = Path(__file__).resolve().parents[1]
K = 4
VARIANTS = [("A", None), ("A+B", 0.5), ("A+B_a0.25", 0.25)]
SEEDS = [0, 1, 2, 3, 4]


def brute_force_acc(y_true, y_pred, is_old, k=K):
    """Best cluster->class permutation by exhaustive search (independent of scipy)."""
    best = None
    for perm in itertools.permutations(range(k)):
        correct = np.array(perm)[y_pred] == y_true
        if best is None or correct.sum() > best.sum():
            best = correct
    return {"all": float(best.mean()), "old": float(best[is_old].mean()), "new": float(best[~is_old].mean())}


def noisy_pred(splits: pd.DataFrame, seed: int) -> np.ndarray:
    """Cluster ids = class ids, with ~15% of the unlabelled rows moved to a random cluster (labelled rows untouched)."""
    rng = np.random.default_rng(seed)
    pred = splits["label"].to_numpy().copy()
    lab = splits["is_labeled"].to_numpy() == 1
    flip = (rng.random(len(pred)) < 0.15) & ~lab
    pred[flip] = rng.integers(0, K, flip.sum())
    return pred


def write_grid(tmp_path: Path, mutate=None):
    splits, _ = make_blobs([40] * K, n_old=2, dim=8, seed=0)
    (tmp_path / "splits").mkdir()
    splits.to_csv(tmp_path / "splits/cub.csv", index=False)
    res = tmp_path / "results"
    res.mkdir()
    u = splits["is_labeled"].to_numpy() == 0
    for variant, alpha in VARIANTS:
        for seed in SEEDS:
            pred = noisy_pred(splits, 100 * seed + len(variant))
            acc = brute_force_acc(splits["label"].to_numpy()[u], pred[u], splits["is_old"].to_numpy()[u] == 1)
            rec = {**acc, "pred": pred.tolist(), "pred_scope": "all_rows", "n_empty_clusters": 0,
                   "config": {"dataset": "cub", "variant": variant, "splits": "splits/cub.csv", "feats": "f.npy", "feats2": None,
                              "alpha": alpha, "seed": seed, "k": K, "max_iter": 200, "n_init": 10, "tolerance": 0.0001,
                              "device": "cpu", "norm_max_dev": {}, "commit": "abc123"}}
            if mutate:
                mutate(rec, variant, seed)
            (res / f"cub_{variant}_s{seed}.json").write_text(json.dumps(rec), encoding="utf-8")
    return splits, res


def run_verify(tmp_path: Path, **kw):
    return verify(tmp_path / "results", tmp_path / "splits", kw.pop("feats_dir", None), kw.pop("summary_csv", None), **kw)


def failing(S):
    return {k: s.fails for k, s in S.items() if s.fails}


# ------------------------------------------------------------------------------------------- the ACC definition
def test_recompute_acc_known_case():
    y_true = np.array([0] * 10 + [1] * 2 + [2] * 4 + [3] * 2)
    y_pred = np.array([0] * 10 + [1] * 2 + [2] * 6)                  # New classes 2 and 3 merged into cluster 2
    is_old = y_true < 2
    r = recompute_acc(y_true, y_pred, is_old)
    assert (r["all"], r["old"], r["new"]) == pytest.approx((16 / 18, 1.0, 4 / 6))      # per sample, not per class
    assert recompute_acc(y_true, y_pred, ~is_old)["old"] == pytest.approx(4 / 6)       # same mapping, roles swapped


def test_recompute_acc_permuted_labels_give_100():
    y_true = np.repeat(np.arange(K), 25)
    perm = np.random.default_rng(0).permutation(K)
    r = recompute_acc(y_true, perm[y_true], y_true < 2)
    assert (r["all"], r["old"], r["new"]) == (1.0, 1.0, 1.0)


@pytest.mark.parametrize("seed", range(4))
def test_recompute_acc_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, K, 200)
    y_pred = np.where(rng.random(200) < 0.7, y_true, rng.integers(0, K, 200))
    is_old = y_true < 2
    got, want = recompute_acc(y_true, y_pred, is_old), brute_force_acc(y_true, y_pred, is_old)
    assert got == pytest.approx(want, abs=1e-12)


def test_expected_alpha():
    assert expected_alpha("A") is None and expected_alpha("B") is None
    assert expected_alpha("A+B") == 0.5 and expected_alpha("A+B_a0.25") == 0.25 and expected_alpha("A+B_a0.75") == 0.75
    assert expected_alpha("A+C") == 0.5
    assert expected_alpha("weird name") == "unknown"


# ------------------------------------------------------------------------------------------- grid checks
def test_clean_grid_passes(tmp_path):
    write_grid(tmp_path)
    S, code = run_verify(tmp_path)
    assert code == 0 and not failing(S), failing(S)
    assert S["V1"].checked == 15 and S["V2"].checked == 15 and S["V4"].skipped and S["V5"].skipped


def test_no_results_gives_exit_2(tmp_path):
    (tmp_path / "results").mkdir()
    assert run_verify(tmp_path)[1] == 2


def test_corrupted_acc_is_caught_by_cli(tmp_path):
    """A JSON whose `all` was edited (+0.01) must be reported, with the file and both values, and exit code 1."""
    write_grid(tmp_path)
    p = tmp_path / "results/cub_A+B_s2.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    original = rec["all"]
    rec["all"] = original + 0.01
    p.write_text(json.dumps(rec), encoding="utf-8")
    proc = subprocess.run([sys.executable, "-m", "scripts.verify_results", "--results_dir", str(tmp_path / "results"),
                           "--splits_dir", str(tmp_path / "splits"), "--feats_dir", str(tmp_path / "nofeats")],
                          cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert proc.returncode == 1, proc.stdout + proc.stderr
    assert "cub_A+B_s2.json" in proc.stdout and "KHÔNG ĐẠT" in proc.stdout
    assert repr(original + 0.01) in proc.stdout and repr(original) in proc.stdout


def test_clean_grid_exit_0_by_cli(tmp_path):
    write_grid(tmp_path)
    proc = subprocess.run([sys.executable, "-m", "scripts.verify_results", "--results_dir", str(tmp_path / "results"),
                           "--splits_dir", str(tmp_path / "splits"), "--feats_dir", str(tmp_path / "nofeats")],
                          cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                          env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "ĐẠT" in proc.stdout


def test_wrong_pred_length_is_caught(tmp_path):
    write_grid(tmp_path, lambda rec, v, s: rec.update(pred=rec["pred"][:-1]) if (v, s) == ("A", 1) else None)
    S, code = run_verify(tmp_path)
    assert code == 1 and any("len(pred)" in m and "cub_A_s1.json" in m for m in S["V1"].fails)


def test_pred_label_out_of_range_is_caught(tmp_path):
    def mut(rec, v, s):
        if (v, s) == ("A", 0):
            rec["pred"][5] = K                                   # cluster id == k is invalid
    write_grid(tmp_path, mut)
    S, code = run_verify(tmp_path)
    assert code == 1 and any("ngoài" in m for m in S["V1"].fails)


def test_alpha_not_matching_variant_is_caught(tmp_path):
    write_grid(tmp_path, lambda rec, v, s: rec["config"].update(alpha=0.5) if v == "A+B_a0.25" and s == 3 else None)
    S, code = run_verify(tmp_path)
    assert code == 1 and any("alpha" in m and "A+B_a0.25" in m for m in S["V3"].fails)


def test_missing_seed_is_caught(tmp_path):
    write_grid(tmp_path)
    (tmp_path / "results/cub_A+B_s4.json").unlink()
    S, code = run_verify(tmp_path)
    assert code == 1 and any("thiếu seed [4]" in m for m in S["V3"].fails)


def test_two_commits_in_one_group_and_other_config_are_caught(tmp_path):
    def mut(rec, v, s):
        if (v, s) == ("A", 2):
            rec["config"]["commit"] = "zzz999"
        if (v, s) == ("A+B", 1):
            rec["config"]["n_init"] = 100
    write_grid(tmp_path, mut)
    S, code = run_verify(tmp_path)
    msgs = " | ".join(S["V3"].fails)
    assert code == 1 and "cub/A: commit khác nhau giữa các seed" in msgs and "n_init khác nhau" in msgs


def test_feats_in_other_directory_with_same_basename_passes(tmp_path):
    def mut(rec, v, s):
        if v == "A+B":
            rec["config"]["feats"] = ("/kaggle/working/run1/feats/f.npy" if s % 2 else "C:\\work\\feats\\f.npy")
            rec["config"]["splits"] = "other/place/cub.csv" if s == 2 else "splits/cub.csv"
    write_grid(tmp_path, mut)
    S, code = run_verify(tmp_path)
    assert code == 0 and not failing(S), failing(S)


def test_feats_with_different_basename_fails(tmp_path):
    write_grid(tmp_path, lambda rec, v, s: rec["config"].update(feats="feats/OTHER.npy") if (v, s) == ("A+B", 3) else None)
    S, code = run_verify(tmp_path)
    assert code == 1 and any("tên file feats khác nhau" in m and "A+B" in m for m in S["V3"].fails)


def test_two_commits_in_the_same_group_fail(tmp_path):
    write_grid(tmp_path, lambda rec, v, s: rec["config"].update(commit="def456") if (v, s) == ("A+B_a0.25", 4) else None)
    S, code = run_verify(tmp_path)
    assert code == 1 and any("cub/A+B_a0.25: commit khác nhau giữa các seed" in m for m in S["V3"].fails)


def test_different_commits_across_groups_only_warn(tmp_path):
    write_grid(tmp_path, lambda rec, v, s: rec["config"].update(commit="def456") if v == "A+B" else None)
    S, code = run_verify(tmp_path)
    assert code == 0 and not failing(S), failing(S)
    warn = " | ".join(S["V3"].warns)
    assert "2 commit khác nhau" in warn and "abc123" in warn and "def456" in warn
    assert "git diff --stat abc123 def456 -- src/ scripts/" in warn


def test_device_difference_only_warns(tmp_path):
    write_grid(tmp_path, lambda rec, v, s: rec["config"].update(device="cuda") if (v, s) == ("A", 1) else None)
    S, code = run_verify(tmp_path)
    assert code == 0 and not failing(S) and any("device khác nhau" in m for m in S["V3"].warns)


def test_file_name_must_match_config(tmp_path):
    write_grid(tmp_path)
    (tmp_path / "results/cub_A+B_s0.json").rename(tmp_path / "results/cub_A+B_s9.json")      # name says seed 9, config says seed 0
    S, code = run_verify(tmp_path)
    assert code == 1 and any("tên file không khớp config" in m for m in S["V3"].fails)


def test_summary_csv_checked(tmp_path):
    write_grid(tmp_path)
    rows = []
    for variant, _ in VARIANTS:
        vals = {m: np.array([json.loads((tmp_path / f"results/cub_{variant}_s{s}.json").read_text())[m] for s in SEEDS]) * 100 for m in ("all", "old", "new")}
        row = {"dataset": "cub", "variant": variant, "n_seeds": len(SEEDS), "seeds": "0,1,2,3,4"}
        for m, v in vals.items():
            row[f"{m}_mean"], row[f"{m}_std"] = v.mean(), v.std(ddof=1)
        rows.append(row)
    csv_path = tmp_path / "results/summary_n5.csv"
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    S, code = run_verify(tmp_path, summary_csv=csv_path)
    assert code == 0 and S["V4"].checked == 3 and not S["V4"].skipped, failing(S)
    rows[1]["all_std"] += 0.5                                         # a wrong std (e.g. ddof=0 would also differ)
    pd.DataFrame(rows).to_csv(csv_path, index=False)
    S, code = run_verify(tmp_path, summary_csv=csv_path)
    assert code == 1 and any("all_std" in m for m in S["V4"].fails)


def test_feats_ids_checked(tmp_path):
    splits, _ = write_grid(tmp_path)
    feats = tmp_path / "feats"
    feats.mkdir()
    ids = splits["image_id"].to_numpy().astype(str)
    np.save(feats / "cub_img_ids.npy", ids)
    np.save(feats / "cub_txt_B_ids.npy", ids)
    S, code = run_verify(tmp_path, feats_dir=feats)
    assert code == 0 and S["V5"].checked == 2 and not S["V5"].skipped
    np.save(feats / "cub_txt_B_ids.npy", ids[::-1])                  # same ids, wrong order
    S, code = run_verify(tmp_path, feats_dir=feats)
    assert code == 1 and any("cub_txt_B_ids.npy" in m for m in S["V5"].fails)
