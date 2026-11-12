from __future__ import annotations

import pickle

import numpy as np
import pandas as pd
import pytest

from src import encode_text


def test_build_sentence_matches_textgcd_prompt():
    assert encode_text.build_sentence(
        "cub",
        ["a photo of a blue bird", "red wings"],
        ["has a long beak", "is small"],
    ) == (
        "A photo of a bird, most likely blue bird, probably red wings. "
        "Likely has a long beak. Perhaps is small."
    )
    assert encode_text.build_sentence("scars", ["a photo of a red sedan"], []) == (
        "A photo of a car, most likely a red sedan."
    )


def _write_inputs(tmp_path, rtg_data):
    split_path = tmp_path / "split.csv"
    pkl_path = tmp_path / "rtg.pkl"
    output_path = tmp_path / "feats" / "cub_txt.npy"
    pd.DataFrame({"image_id": ["one.jpg", "two.jpg"]}).to_csv(split_path, index=False)
    with pkl_path.open("wb") as handle:
        pickle.dump(rtg_data, handle)
    return split_path, pkl_path, output_path


def test_encode_rtg_missing_image_id_raises(tmp_path):
    split_path, pkl_path, output_path = _write_inputs(tmp_path, {"one.jpg": {"tags": [], "attributes": []}})
    with pytest.raises(KeyError, match="two.jpg"):
        encode_text.encode_rtg_pkl("cub", split_path, pkl_path, output_path)


def test_encode_rtg_empty_entry_uses_base_text_and_saves_ids(tmp_path, monkeypatch):
    split_path, pkl_path, output_path = _write_inputs(
        tmp_path,
        {"one.jpg": {"tags": [], "attributes": []}, "two.jpg": {"tags": [], "attributes": []}},
    )
    seen = []

    def fake_encode(sentences, batch_size=128, device=None):
        seen.extend(sentences)
        return np.ones((len(sentences), encode_text.FEATURE_DIM), dtype=np.float32)

    monkeypatch.setattr(encode_text, "encode_sentences", fake_encode)
    encode_text.encode_rtg_pkl("cub", split_path, pkl_path, output_path)
    assert seen == ["A photo of a bird.", "A photo of a bird."]
    assert np.array_equal(np.load(output_path.with_name("cub_txt_ids.npy"), allow_pickle=False),
                          np.array(["one.jpg", "two.jpg"], dtype=str))
