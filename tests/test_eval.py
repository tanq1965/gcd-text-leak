"""Tests for src/eval.py and src/tp_env.py (synthetic data only)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import src.eval as ev
from src import tp_env
from tests.fake_data import make_blobs


def uneven_case() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Old classes 0 (10 imgs) and 1 (2) perfect; New classes 2 (4) and 3 (2) merged into cluster 2."""
    y_true = np.array([0] * 10 + [1] * 2 + [2] * 4 + [3] * 2)
    y_pred = np.array([0] * 10 + [1] * 2 + [2] * 6)
    return y_true, y_pred, y_true < 2


def test_permuted_labels_give_100():
    rng = np.random.default_rng(0)
    y_true = np.repeat(np.arange(6), [50, 40, 30, 30, 30, 20])
    y_pred = rng.permutation(6)[y_true]
    res = ev.score(y_true, y_pred, y_true < 3)
    assert res == pytest.approx({"all": 1.0, "old": 1.0, "new": 1.0})


def test_mask_orientation_and_sample_weighting():
    y_true, y_pred, mask = uneven_case()
    # sample-weighted v2: Hungarian gives 10+2+4 = 16 correct of 18
    assert ev.score(y_true, y_pred, mask) == pytest.approx({"all": 16 / 18, "old": 12 / 12, "new": 4 / 6})
    # swapping the mask swaps the roles of Old and New
    assert ev.score(y_true, y_pred, ~mask) == pytest.approx({"all": 16 / 18, "old": 4 / 6, "new": 1.0})
    # the class-averaged variant gives different numbers, so the two ways of averaging are told apart
    tp_env.add_third_party_paths()
    from util.cluster_and_log_utils import split_cluster_acc_v2_balanced
    bal_all, bal_old, bal_new = split_cluster_acc_v2_balanced(y_true, y_pred, mask)
    assert (bal_all, bal_old, bal_new) == pytest.approx((0.75, 1.0, 0.5))
    assert bal_all != pytest.approx(16 / 18)


def test_unlabelled_view_uses_unlabelled_rows_only():
    splits, _ = make_blobs([8, 8, 8, 8], n_old=2)
    u_idx, y_true, mask = ev.unlabelled_view(splits)
    assert (splits.loc[u_idx, "is_labeled"] == 0).all()
    assert np.array_equal(y_true, splits.loc[u_idx, "label"].to_numpy())
    assert np.array_equal(mask, splits.loc[u_idx, "is_old"].to_numpy().astype(bool))
    assert set(y_true[mask]) == {0, 1} and set(y_true[~mask]) == {2, 3}


def test_score_is_invariant_to_row_order():
    rng = np.random.default_rng(3)
    y_true = rng.integers(0, 5, size=300)
    y_pred = rng.integers(0, 5, size=300)
    mask = y_true < 2
    perm = rng.permutation(300)
    assert ev.score(y_true[perm], y_pred[perm], mask[perm]) == pytest.approx(ev.score(y_true, y_pred, mask))


def test_pipeline_is_invariant_to_feature_file_order():
    splits, feats = make_blobs([30, 30, 30, 30], n_old=2)
    ids = splits["image_id"].to_numpy()
    base = ev.evaluate(splits, feats, k=4, seed=0, n_init=3, max_iter=20)
    perm = np.random.default_rng(1).permutation(len(splits))
    # the feature file comes in another order: align back to the canonical splits order, then evaluate
    aligned = ev.align_to_splits(feats[perm], ids[perm], splits["image_id"])
    assert np.array_equal(aligned, feats)
    again = ev.evaluate(splits, aligned, k=4, seed=0, n_init=3, max_iter=20)
    assert np.array_equal(again["pred"], base["pred"])


def test_image_id_mismatch_raises():
    splits, feats = make_blobs([5, 5, 5, 5], n_old=2)
    ids = list(splits["image_id"])
    with pytest.raises(ValueError, match="image_id mismatch"):
        ev.align_to_splits(feats[:-1], ids[:-1], ids)                       # missing one id
    with pytest.raises(ValueError, match="image_id mismatch"):
        ev.align_to_splits(np.vstack([feats, feats[:1]]), ids + ["extra.jpg"], ids)   # unknown id
    with pytest.raises(ValueError, match="duplicated"):
        ev.align_to_splits(feats, ids[:-1] + [ids[0]], ids)                 # duplicate in features
    with pytest.raises(ValueError, match="duplicated"):
        ev.align_to_splits(feats, ids, ids[:-1] + [ids[0]])                 # duplicate in splits


def test_combine_features():
    rng = np.random.default_rng(0)

    def unit(n: int, d: int) -> np.ndarray:
        x = rng.standard_normal((n, d)).astype(np.float32)
        return x / np.linalg.norm(x, axis=1, keepdims=True)

    f_i, f_t = unit(10, 8), unit(10, 5)
    out = ev.combine_features(f_i, f_t, 0.25)
    assert out.shape == (10, 13)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
    np.testing.assert_allclose(out[:, :8], np.sqrt(0.75) * f_i, atol=1e-6)
    np.testing.assert_allclose(out[:, 8:], 0.5 * f_t, atol=1e-6)
    assert np.allclose(ev.combine_features(f_i, f_t, 0.0)[:, 8:], 0)
    assert np.allclose(ev.combine_features(f_i, f_t, 1.0)[:, :8], 0)
    with pytest.raises(ValueError, match="L2"):
        ev.combine_features(2 * f_i, f_t, 0.5)
    with pytest.raises(ValueError, match="alpha"):
        ev.combine_features(f_i, f_t, 1.5)


@pytest.mark.parametrize("seed", [0, 1])
def test_pipeline_on_separable_blobs(seed: int):
    splits, feats = make_blobs([40, 40, 40, 40], n_old=2)
    res = ev.evaluate(splits, feats, k=4, seed=seed, n_init=3, max_iter=30)
    assert (res["all"], res["old"], res["new"]) == pytest.approx((1.0, 1.0, 1.0))
    assert res["n_empty_clusters"] == 0 and len(res["pred"]) == len(splits) and res["pred_scope"] == "all_rows"
    assert (res["max_iter"], res["n_init"], res["device"]) == (30, 3, "cpu")


def test_cluster_with_a_single_point():
    """A New class with exactly one image. Contract: either a finite result or a RuntimeError about NaN,
    never a silently wrong number. (Tighten once the real behaviour is observed in N1-3.)"""
    splits, feats = make_blobs([20, 20, 20, 1], n_old=2)
    try:
        res = ev.evaluate(splits, feats, k=4, seed=0, n_init=3, max_iter=30)
    except RuntimeError as e:
        assert "NaN" in str(e)
    else:
        assert res["n_empty_clusters"] == 0 and np.isfinite([res["all"], res["old"], res["new"]]).all()


def test_nan_centres_raise(monkeypatch):
    tp_env.add_third_party_paths()
    import torch
    from methods.clustering import faster_mix_k_means_pytorch as m

    def fake_fit_mix(self, u_feats, l_feats, l_targets):
        self.labels_ = torch.zeros(len(u_feats) + len(l_feats), dtype=torch.long)
        self.cluster_centers_ = torch.full((self.k, u_feats.shape[1]), float("nan"))

    monkeypatch.setattr(m.K_Means, "fit_mix", fake_fit_mix)
    splits, feats = make_blobs([10, 10, 10, 10], n_old=2)
    with pytest.raises(RuntimeError, match="NaN"):
        ev.run_kmeans(feats, splits, k=4, seed=0, n_init=1, max_iter=2)


def test_textgcd_never_on_sys_path(monkeypatch):
    tp_env.add_third_party_paths()
    assert not any("textgcd" in Path(p).parts for p in sys.path)
    monkeypatch.syspath_prepend(str(tp_env.REPO_ROOT / "third_party" / "textgcd"))
    with pytest.raises(RuntimeError, match="textgcd"):
        tp_env.add_third_party_paths()
