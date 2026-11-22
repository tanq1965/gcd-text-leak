"""Tests for scripts/n9_errors.py (N9, four error types)."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import n9_errors as n9  # noqa: E402


def make_split(labels, is_labeled, old_classes):
    return {
        "image_id": np.array([f"i{k}" for k in range(len(labels))]),
        "label": np.array(labels),
        "is_labeled": np.array(is_labeled),
        "is_old": np.array([int(c in old_classes) for c in labels]),
    }


# 4 classes: 0, 1 Old; 2, 3 New. Two labelled rows (ignored), eight unlabelled rows.
LABELS = [0, 1, 0, 0, 1, 1, 2, 2, 3, 3]
LABELED = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
# clusters equal classes except: row 3 (Old 0 -> cluster 1, Old), row 5 (Old 1 -> cluster 2, New),
# row 7 (New 2 -> cluster 0, Old), row 9 (New 3 -> cluster 2, other New). Labelled rows get any cluster.
PRED = [3, 3, 0, 1, 1, 2, 2, 0, 3, 2]


def test_four_error_types_on_a_hand_case():
    info = n9.error_types(make_split(LABELS, LABELED, {0, 1}), np.array(PRED))
    c = info["counts"]
    assert (c["n_old"], c["n_new"]) == (4, 4)
    assert (c["old_correct"], c["old_to_old"], c["old_to_new"]) == (2, 1, 1)
    assert (c["new_correct"], c["new_to_old"], c["new_to_new"]) == (2, 1, 1)
    assert list(info["index"]) == list(range(2, 10))  # labelled rows excluded


def test_percentages_per_group_sum_to_100():
    info = n9.error_types(make_split(LABELS, LABELED, {0, 1}), np.array(PRED))
    pct = n9.pct_row(info["counts"])
    assert pct["old_correct"] + pct["old_to_old"] + pct["old_to_new"] == pytest.approx(100)
    assert pct["new_correct"] + pct["new_to_old"] + pct["new_to_new"] == pytest.approx(100)


def _v2_numpy(y_true, y_pred, mask):
    """Inline copy of SimGCD split_cluster_acc_v2 (sample-weighted), for environments without torch."""
    from scipy.optimize import linear_sum_assignment
    d = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((d, d), dtype=int)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    ind = np.vstack(linear_sum_assignment(w.max() - w)).T
    ind_map = {j: i for i, j in ind}
    total = sum(w[i, j] for i, j in ind) / y_pred.size
    old = sum(w[ind_map[i], i] for i in set(y_true[mask])) / mask.sum()
    new = sum(w[ind_map[i], i] for i in set(y_true[~mask])) / (~mask).sum()
    return total, old, new


@pytest.mark.parametrize("seed", range(5))
def test_correct_counts_equal_v2_acc_on_random_predictions(seed):
    rng = np.random.default_rng(seed)
    k, n = 12, 600
    labels = rng.integers(0, k, n)
    labeled = (rng.random(n) < 0.3).astype(int)
    pred = np.where(rng.random(n) < 0.6, labels, rng.integers(0, k, n))
    split = make_split(labels.tolist(), labeled.tolist(), set(range(6)))
    info = n9.error_types(split, pred)
    c = info["counts"]
    u = labeled == 0
    total, old, new = _v2_numpy(labels[u], pred[u], split["is_old"][u] == 1)
    assert (c["old_correct"] + c["new_correct"]) / u.sum() == pytest.approx(total)
    assert c["old_correct"] / c["n_old"] == pytest.approx(old)
    assert c["new_correct"] / c["n_new"] == pytest.approx(new)


def test_matches_simgcd_v2_function_itself():
    pytest.importorskip("torch")
    from src import tp_env
    tp_env.add_third_party_paths()
    from util.cluster_and_log_utils import split_cluster_acc_v2

    rng = np.random.default_rng(7)
    labels = rng.integers(0, 10, 400)
    pred = np.where(rng.random(400) < 0.5, labels, rng.integers(0, 10, 400))
    split = make_split(labels.tolist(), [0] * 400, set(range(5)))
    c = n9.error_types(split, pred)["counts"]
    total, old, new = split_cluster_acc_v2(labels, pred, split["is_old"] == 1)
    assert (c["old_correct"] + c["new_correct"]) / 400 == pytest.approx(total)
    assert c["old_correct"] / c["n_old"] == pytest.approx(old)
    assert c["new_correct"] / c["n_new"] == pytest.approx(new)


def test_check_against_json_rejects_mismatch():
    info = n9.error_types(make_split(LABELS, LABELED, {0, 1}), np.array(PRED))
    n9.check_against_json(info["counts"], {"all": 0.5, "old": 0.5, "new": 0.5}, "ok.json")
    with pytest.raises(ValueError, match="old from Hungarian"):
        n9.check_against_json(info["counts"], {"all": 0.5, "old": 0.75, "new": 0.5}, "bad.json")


def test_n8_link_counts_images_following_the_wrong_tag():
    split = make_split(LABELS, LABELED, {0, 1})
    info = n9.error_types(split, np.array(PRED))
    # row 7 (New 2) is clustered into class 0; its top-1 tag names class 0 -> follows the wrong tag.
    n8 = {f"i{k}": {"top1_kind": "correct", "top1_label": str(LABELS[k])} for k in range(10)}
    n8["i7"] = {"top1_kind": "wrong_old", "top1_label": "0"}
    n8["i9"] = {"top1_kind": "wrong_new", "top1_label": "3"}  # row 9 clustered into 2, not 3
    n8["i5"] = {"top1_kind": "other", "top1_label": ""}
    link = n9.n8_link(split, info, n8)
    assert (link["n_correct"], link["n_wrong"], link["n_other"]) == (5, 2, 1)
    assert link["follow_wrong_tag"] == pytest.approx(50)
    assert link["err_wrong"] == pytest.approx(100) and link["err_other"] == pytest.approx(100)
    assert link["err_correct"] == pytest.approx(20)  # row 3 wrong among rows 2,3,4,6,8


def _repo(tmp_path: Path, runs: dict[str, list[int]]) -> Path:
    for sub in ("splits", "results", "retrieval"):
        (tmp_path / sub).mkdir()
    with (tmp_path / "splits" / "cub.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_id", "path", "label", "is_labeled", "is_old"])
        for k, (lab, isl) in enumerate(zip(LABELS, LABELED)):
            w.writerow([f"i{k}", "x", lab, isl, int(lab in (0, 1))])
    split = make_split(LABELS, LABELED, {0, 1})
    for name, pred in runs.items():
        c = n9.error_types(split, np.array(pred))["counts"]
        rec = {"all": (c["old_correct"] + c["new_correct"]) / 8, "old": c["old_correct"] / 4,
               "new": c["new_correct"] / 4, "pred": pred}
        (tmp_path / "results" / name).write_text(json.dumps(rec), encoding="utf-8")
    return tmp_path


def test_main_end_to_end_and_seed_aggregation(tmp_path, capsys):
    perfect = LABELS
    root = _repo(tmp_path, {"cub_A_s0.json": PRED, "cub_A_s1.json": perfect, "cub_A+B_s0.json": perfect})
    assert n9.main(["--root", str(root), "--datasets", "cub"]) == 0
    with (root / "errors" / "n9_errors.csv").open() as f:
        rows = {r["variant"]: r for r in csv.DictReader(f)}
    assert rows["A"]["seeds"] == "2" and float(rows["A"]["old_correct_mean"]) == pytest.approx(75)
    assert float(rows["A"]["old_correct_std"]) == pytest.approx(float(np.std([50, 100], ddof=1)), abs=1e-3)  # CSV keeps 4 decimals
    assert float(rows["A+B"]["old_correct_std"]) == 0.0
    assert (root / "figures" / "n9_errors.png").stat().st_size > 0
    err = capsys.readouterr().err
    assert "cub_A+C" in err  # default variant without results is reported, not silently skipped
    md = (root / "errors" / "n9_errors.md").read_text(encoding="utf-8")
    assert "A+B − A" in md and "Bảng 3" not in md  # no N8 files -> no link table


def test_main_rejects_result_that_does_not_match_its_acc(tmp_path):
    root = _repo(tmp_path, {"cub_A_s0.json": PRED})
    path = root / "results" / "cub_A_s0.json"
    rec = json.loads(path.read_text())
    rec["old"] = 0.99
    path.write_text(json.dumps(rec))
    with pytest.raises(ValueError, match="cub_A_s0.json"):
        n9.main(["--root", str(root), "--datasets", "cub"])


def test_result_files_do_not_mix_variants(tmp_path):
    for name in ("cub_A_s0.json", "cub_A+B_s0.json", "cub_A+B_a0.25_s0.json", "cub_A_s10.json"):
        (tmp_path / name).write_text("{}")
    assert [s for s, _ in n9.result_files(tmp_path, "cub", "A")] == [0, 10]
    assert [p.name for _, p in n9.result_files(tmp_path, "cub", "A+B")] == ["cub_A+B_s0.json"]
