"""Fake data for tests: a tiny CUB tree (metadata + empty image files) and separable feature blobs."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def make_fake_cub(root: Path, n_classes: int = 200, n_train: int = 6, n_test: int = 2) -> None:
    """Create root/CUB_200_2011/{images.txt, image_class_labels.txt, train_test_split.txt, images/...}.

    Image files are empty (the SimGCD CUB loader only checks that they exist).
    image_id is 'img_c{raw class:03d}_{j}.jpg'; the first n_train images of each class are train.
    """
    base = root / "CUB_200_2011"
    lines_img, lines_cls, lines_split = [], [], []
    img_id = 0
    for c in range(n_classes):
        folder = f"{c + 1:03d}.Class{c + 1:03d}"
        (base / "images" / folder).mkdir(parents=True, exist_ok=True)
        for j in range(n_train + n_test):
            img_id += 1
            fname = f"img_c{c:03d}_{j}.jpg"
            (base / "images" / folder / fname).touch()
            lines_img.append(f"{img_id} {folder}/{fname}")
            lines_cls.append(f"{img_id} {c + 1}")
            lines_split.append(f"{img_id} {int(j < n_train)}")
    for name, lines in (("images.txt", lines_img), ("image_class_labels.txt", lines_cls),
                        ("train_test_split.txt", lines_split)):
        (base / name).write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_blobs(sizes: list[int], n_old: int, dim: int = 32, noise: float = 0.05,
               seed: int = 0) -> tuple[pd.DataFrame, np.ndarray]:
    """Well-separated unit-norm clusters; class c has sizes[c] points around axis c.

    Half of every Old class is labelled. Returns (splits in canonical order: labelled first then image_id,
    feats [N, dim] float32 in the same row order).
    """
    assert dim >= len(sizes)
    rng = np.random.default_rng(seed)
    labels, lab, feats = [], [], []
    for c, n in enumerate(sizes):
        center = np.zeros(dim)
        center[c] = 1.0
        x = center + noise * rng.standard_normal((n, dim))
        feats.append(x / np.linalg.norm(x, axis=1, keepdims=True))
        labels += [c] * n
        lab += [int(c < n_old and j % 2 == 0) for j in range(n)]
    feats_a = np.concatenate(feats).astype(np.float32)
    labels_a, lab_a = np.array(labels), np.array(lab)
    ids = np.array([f"img_{i:05d}.jpg" for i in range(len(labels))])
    df = pd.DataFrame({"image_id": ids, "path": "x/" + ids, "label": labels_a, "is_labeled": lab_a,
                       "is_old": (labels_a < n_old).astype(int)})
    order = np.lexsort((ids, 1 - lab_a))   # primary: labelled first; secondary: image_id
    return df.iloc[order].reset_index(drop=True), feats_a[order]
