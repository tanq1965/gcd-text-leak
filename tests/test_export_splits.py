"""Tests for src/export_splits.py on a fake CUB tree (runs the real SimGCD get_datasets, no images needed)."""
from __future__ import annotations

import os
from types import SimpleNamespace

import numpy as np
import pytest

from src import export_splits as ex
from tests.fake_data import make_fake_cub

N_CLASSES, N_TRAIN = 200, 6          # fake tree: 200 classes x 6 train images = 1200 train rows


@pytest.fixture(scope="module")
def fake_cub(tmp_path_factory):
    root = tmp_path_factory.mktemp("cub")
    make_fake_cub(root, n_classes=N_CLASSES, n_train=N_TRAIN, n_test=2)
    return root


def test_cub_export_invariants(fake_cub):
    splits, classmap, info = ex.build_splits("cub", fake_cub)
    n_old, n_new = info["n_old"], info["n_new"]
    checks = ex.self_check(splits, classmap, "cub", n_old, n_new, fake_cub, expected_rows=N_CLASSES * N_TRAIN)
    assert all(status != "FAIL" for _, _, status in checks), ex.format_check_table(checks)

    assert list(splits.columns) == ex.COLUMNS and n_old + n_new == N_CLASSES
    assert splits["is_labeled"].is_monotonic_decreasing                       # labelled block first
    for _, block in splits.groupby("is_labeled"):
        assert block["image_id"].is_monotonic_increasing                      # each block sorted by image_id
    assert int(splits["is_labeled"].sum()) == int(0.5 * n_old * N_TRAIN)
    assert (splits.loc[splits["is_labeled"] == 1, "label"] < n_old).all()
    assert (splits["image_id"] == splits["path"].map(os.path.basename)).all()
    assert splits["path"].str.startswith("CUB_200_2011/images/").all()
    assert list(classmap.columns) == ex.CLASSMAP_COLUMNS
    assert len(classmap) == N_CLASSES and int(classmap["is_old"].sum()) == n_old
    assert list(classmap["label"]) == list(range(N_CLASSES))
    assert (classmap["class_name"] == "").all()                               # CUB provides no names here


def test_cub_relabelling_matches_classmap(fake_cub):
    splits, classmap, _ = ex.build_splits("cub", fake_cub)
    raw = splits["image_id"].map(lambda s: int(s.split("_")[1][1:]))          # 'img_c012_3.jpg' -> 12
    expected = raw.map(classmap.set_index("raw_label")["label"])
    assert (splits["label"] == expected).all()                                # Old classes got the smaller labels


def test_export_twice_gives_identical_files(fake_cub, tmp_path):
    for name in ("a", "b"):
        splits, classmap, _ = ex.build_splits("cub", fake_cub)
        ex.write_csv(splits, tmp_path / f"{name}.csv")
        ex.write_csv(classmap, tmp_path / f"{name}_classmap.csv")
    assert (tmp_path / "a.csv").read_bytes() == (tmp_path / "b.csv").read_bytes()
    assert (tmp_path / "a_classmap.csv").read_bytes() == (tmp_path / "b_classmap.csv").read_bytes()


def test_main_fails_and_writes_nothing_when_a_check_fails(fake_cub, tmp_path):
    out = tmp_path / "splits" / "cub.csv"
    # the fake tree has 1200 rows, not the real 5994, so the row-count check fails
    assert ex.main(["--dataset", "cub", "--data-root", str(fake_cub), "--out", str(out)]) == 1
    assert not out.exists()


def test_warn_does_not_count_as_fail():
    """A labelled-images-per-Old-class minimum of 0 is a WARN, not a FAIL."""
    import pandas as pd

    splits = pd.DataFrame({"image_id": ["a", "b", "c", "d"], "path": ["a", "b", "c", "d"],
                           "label": [0, 0, 1, 2], "is_labeled": [1, 0, 0, 0], "is_old": [1, 1, 1, 0]})
    classmap = pd.DataFrame({"raw_label": [0, 1, 2], "label": [0, 1, 2], "is_old": [1, 1, 0], "class_name": ""})
    checks = ex.self_check(splits, classmap, "cub", n_old=2, n_new=1, data_root=None, expected_rows=4)
    status = {name: s for name, _, s in checks}
    assert status["labelled images per Old class >= 1 (min / max)"] == "WARN"
    assert "WARN" in ex.format_check_table(checks)


def test_class_names_aircraft_strips_newlines():
    sub = SimpleNamespace(classes=np.array(["707-320\n", "A300B4\n"]))
    assert ex.class_names("aircraft", sub) == {0: "707-320", 1: "A300B4"}


def test_class_names_empty_for_cub_and_cars():
    assert ex.class_names("cub", SimpleNamespace()) == {}
    assert ex.class_names("scars", SimpleNamespace()) == {}
