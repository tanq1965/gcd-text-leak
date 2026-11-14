"""Tests for src/rtg.py with a tiny fake CLIP (no open_clip / GPU needed)."""
from __future__ import annotations

import json

import pandas as pd
import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src import rtg


class FakeClip(torch.nn.Module):
    """encode_text maps sentence i -> one-hot e_(i mod D); encode_image passes the vector through."""

    def __init__(self, dim: int = 4):
        super().__init__()
        self.dim = dim
        self.w = torch.nn.Parameter(torch.ones(1))
        self.logit_scale = torch.tensor(100.0)

    def encode_text(self, tokens: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.one_hot(tokens[:, 0] % self.dim, self.dim).float() * 2.0

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return images.reshape(images.shape[0], -1)[:, : self.dim] * 3.0


def fake_tokenize(texts):
    return torch.tensor([[int(t.split()[-1])] for t in texts])


def write_lexicon(path, rows):
    path.write_text("".join(f"{r}\n" for r in rows), encoding="utf-8")


def test_load_lexicon_empty_and_quoted(tmp_path):
    empty = tmp_path / "tags.csv"
    empty.write_bytes(b"")
    assert rtg.load_lexicon(empty) == []
    quoted = tmp_path / "attr.csv"
    quoted.write_text('"x which has a, b"\nplain\n', encoding="utf-8")
    assert rtg.load_lexicon(quoted) == ["x which has a, b", "plain"]


def test_retrieve_topk_order_and_threshold():
    img = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8]])
    txt = torch.eye(3)
    texts = ["a", "b", "c"]
    out = rtg.retrieve(img, txt, texts, logit_scale=1.0, k=2, threshold=0.5)
    assert out[0] == ["a"]             # second best score 0 < 0.5 dropped
    assert out[1] == ["c", "b"]        # sorted by decreasing score
    assert rtg.retrieve(img, torch.zeros((0, 0)), [], 1.0, 3, 0.2) == [[], []]


def test_cache_is_keyed_by_lexicon_content(tmp_path):
    model = FakeClip()
    lex = tmp_path / "attributes.csv"
    write_lexicon(lex, ["s 0", "s 1"])
    cache = tmp_path / "cache"
    f1 = rtg.load_or_encode(model, fake_tokenize, lex, ["s 0", "s 1"], "m", cache, torch.device("cpu"))
    write_lexicon(lex, ["s 2", "s 3"])  # same path, new content (B -> C mistake)
    f2 = rtg.load_or_encode(model, fake_tokenize, lex, ["s 2", "s 3"], "m", cache, torch.device("cpu"))
    assert not torch.equal(f1, f2)
    assert len(list(cache.glob("*.pt"))) == 2
    # tampered sidecar must be rejected, not silently reused
    pt = rtg.cache_path(cache, lex, rtg.file_sha256(lex), "m")
    meta = json.loads(pt.with_suffix(".json").read_text())
    meta["rows"] = 99
    pt.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        rtg.load_or_encode(model, fake_tokenize, lex, ["s 2", "s 3"], "m", cache, torch.device("cpu"))


def make_split(tmp_path, n=3):
    root = tmp_path / "data"
    (root / "imgs").mkdir(parents=True)
    rows = []
    for i in range(n):
        name = f"{i:07d}.jpg"
        Image.new("RGB", (4, 4), color=(i * 40, 0, 0)).save(root / "imgs" / name)
        rows.append({"image_id": name, "path": f"imgs/{name}", "label": i, "is_labeled": 0, "is_old": 0})
    split = tmp_path / "split.csv"
    pd.DataFrame(rows).to_csv(split, index=False)
    return split, root


def one_hot_transform(img):
    # image i (red = 40*i) -> one-hot e_i so the fake model "sees" sentence i
    i = img.getpixel((0, 0))[0] // 40
    return torch.nn.functional.one_hot(torch.tensor(i), 4).float()


def test_run_rtg_split_order_and_no_tag_mode(tmp_path):
    split, root = make_split(tmp_path)
    ds = rtg.SplitImageDataset_RTG(split, root, one_hot_transform)
    loader = DataLoader(ds, batch_size=2, shuffle=False)
    model = FakeClip()
    attrs = ["attr 0", "attr 1", "attr 2", "attr 3"]
    attr_feats = rtg.encode_texts(model, fake_tokenize, attrs, torch.device("cpu"))
    res = rtg.run_rtg(model, loader, [], torch.zeros((0, 0)), attrs, attr_feats,
                      torch.device("cpu"), num_tags=3, num_attributes=2, threshold=0.2)
    assert [r["image_id"] for r in res] == ds.df["image_id"].tolist()
    assert all(r["tags"] == [] for r in res)                      # version D: no tags
    assert [r["attributes"][0] for r in res] == ["attr 0", "attr 1", "attr 2"]


def test_split_rejects_id_not_matching_basename(tmp_path):
    split, root = make_split(tmp_path)
    df = pd.read_csv(split, dtype=str)
    df.loc[0, "image_id"] = "other.jpg"
    df.to_csv(split, index=False)
    with pytest.raises(ValueError):
        rtg.SplitImageDataset_RTG(split, root)


def test_real_splits_ids_are_basenames():
    for ds in rtg.DATASETS:
        d = rtg.SplitImageDataset_RTG(rtg.REPO_ROOT / "splits" / f"{ds}.csv", "unused")
        assert len(d) > 0


def test_lexicon_paths():
    t, a = rtg.lexicon_paths("B", "cub")
    assert t.name == "Lexicon_tags.csv" and a.name == "Lexicon_attributes.csv"
    t, a = rtg.lexicon_paths("D", "aircraft")
    assert t.as_posix().endswith("lexicon/aircraft/D/tags.csv")
