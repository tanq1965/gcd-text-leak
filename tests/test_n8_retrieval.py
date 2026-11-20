from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import n8_retrieval as n8  # noqa: E402

KEYS = {"737 200": 0, "737 200 aircraft": 0, "a320": 1, "airbus a320 family": 1, "a380": 2}
IS_OLD = {0: True, 1: True, 2: False}


def test_correct_in_top3_and_top1():
    m = n8.classify_image(["a photo of a 737-200 aircraft", "a photo of a A320"], [], 0, KEYS, IS_OLD)
    assert m["correct_top3"] == 1 and m["top1_kind"] == "correct"


def test_wrong_name_uses_alias_and_old_new_of_named_class():
    m = n8.classify_image(["a photo of an Airbus A320 family", "a photo of a 737 200"], [], 2, KEYS, IS_OLD)
    assert m["correct_top3"] == 0 and m["top1_kind"] == "wrong_old" and m["top1_label"] == 1
    m = n8.classify_image(["a photo of a A380"], [], 0, KEYS, IS_OLD)
    assert m["top1_kind"] == "wrong_new"


def test_other_no_tag_and_attribute_prefix():
    m = n8.classify_image(["a photo of a seagull"], ["a380 which has four engines"], 2, KEYS, IS_OLD)
    assert m["top1_kind"] == "other" and m["attr_correct"] == 1
    m = n8.classify_image([], ["has four engines"], 2, KEYS, IS_OLD)
    assert m["top1_kind"] == "no_tag" and m["attr_correct"] == 0 and m["correct_top3"] == 0


def test_summary_top1_parts_sum_to_100():
    rows = [
        {"is_old": 1, "covered_B": 1, "correct_top3": 1, "top1_kind": "correct", "attr_correct": 1},
        {"is_old": 0, "covered_B": 1, "correct_top3": 0, "top1_kind": "wrong_old", "attr_correct": 0},
        {"is_old": 0, "covered_B": 0, "correct_top3": 0, "top1_kind": "other", "attr_correct": 0},
        {"is_old": 1, "covered_B": 0, "correct_top3": 0, "top1_kind": "no_tag", "attr_correct": 0},
    ]
    s = {r["group"]: r for r in n8.summarize(rows, "aircraft", "B")}
    a = s["All"]
    assert a["top1_correct"] + a["top1_wrong"] + a["top1_other"] + a["no_tag"] == 100
    assert a["correct_top3"] == 25 and a["correct_top3_covered"] == 50
    assert s["New"]["top1_wrong_old"] == 50


# --------------------------------------------------------------------------- end-to-end on a mini dataset
import csv  # noqa: E402
import pickle  # noqa: E402

import pytest  # noqa: E402

# 4 Aircraft classes: labels 0,1 Old; 2,3 New. Class 3 has no tag in lexicon B (covered_B=0).
NAMES = [("0", "A380", "a380", "1"), ("1", "DC-3", "dc 3", "1"), ("2", "Tornado", "tornado", "0"), ("3", "Yak-42", "yak 42", "0")]
SPLIT = [("i0.jpg", 0, 1, 1), ("i1.jpg", 1, 0, 1), ("i2.jpg", 2, 0, 0), ("i3.jpg", 3, 0, 0)]
RTG = [
    {"image_id": "i0.jpg", "tags": ["a photo of an Airbus A380"], "attributes": ["a380 which has four engines"]},
    # own name only at rank 4 -> must NOT count as correct_top3
    {"image_id": "i1.jpg", "tags": ["a photo of a cat", "a photo of a dog", "a photo of a car", "a photo of a Douglas DC-3"],
     "attributes": []},
    {"image_id": "i2.jpg", "tags": ["a photo of a Tornado aircraft"], "attributes": []},
    {"image_id": "i3.jpg", "tags": ["a photo of a Tornado"], "attributes": []},  # names another New class
]


def make_repo(tmp_path, rtg_c=None, split=SPLIT):
    for sub in ("names", "splits", "coverage", "rtg"):
        (tmp_path / sub).mkdir()
    with (tmp_path / "names/aircraft.csv").open("w", newline="") as f:
        csv.writer(f).writerows([("label", "class_name", "name_norm", "is_old"), *NAMES])
    with (tmp_path / "splits/aircraft.csv").open("w", newline="") as f:
        csv.writer(f).writerows([("image_id", "path", "label", "is_labeled", "is_old"),
                                 *[(i, f"x/{i}", l, lab, o) for i, l, lab, o in split]])
    with (tmp_path / "coverage/aircraft.csv").open("w", newline="") as f:
        csv.writer(f).writerows([("label", "in_tag"), (0, 1), (1, 1), (2, 1), (3, 0)])
    for ver, data in (("B", RTG), ("C", rtg_c if rtg_c is not None else RTG)):
        with (tmp_path / f"rtg/aircraft_{ver}.pkl").open("wb") as f:
            pickle.dump(data, f)
    return tmp_path


def test_evaluate_end_to_end_and_top3_limit(tmp_path):
    root = make_repo(tmp_path)
    rows = {r["image_id"]: r for r in n8.evaluate("aircraft", "B", root / "rtg", root)}
    assert rows["i0.jpg"]["correct_top3"] == 1 and rows["i0.jpg"]["attr_correct"] == 1  # alias "airbus a380"
    assert rows["i1.jpg"]["correct_top3"] == 0 and rows["i1.jpg"]["top1_kind"] == "other"  # rank 4 ignored
    assert rows["i2.jpg"]["top1_kind"] == "correct"  # "tornado aircraft" suffix alias
    assert rows["i3.jpg"]["top1_kind"] == "wrong_new" and rows["i3.jpg"]["top1_label"] == 2


def test_covered_rate_is_nan_when_no_image_of_group_is_covered(tmp_path):
    root = make_repo(tmp_path, split=[("i3.jpg", 3, 0, 0)])
    rtg = [RTG[3]]
    with (root / "rtg/aircraft_B.pkl").open("wb") as f:
        pickle.dump(rtg, f)
    new = {r["group"]: r for r in n8.summarize(n8.evaluate("aircraft", "B", root / "rtg", root), "aircraft", "B")}["New"]
    assert new["correct_top3_covered"] != new["correct_top3_covered"]  # nan, not ZeroDivisionError
    assert "| – |" in n8.to_markdown([new])


def test_rtg_order_must_match_split(tmp_path):
    root = make_repo(tmp_path, rtg_c=list(reversed(RTG)))
    with pytest.raises(ValueError, match="image order"):
        n8.evaluate("aircraft", "C", root / "rtg", root)


def test_is_old_must_agree_between_names_and_splits(tmp_path):
    bad = [("i0.jpg", 0, 1, 0), *SPLIT[1:]]  # label 0 is Old in names/, New in splits/
    root = make_repo(tmp_path, split=bad)
    with pytest.raises(ValueError, match="is_old"):
        n8.evaluate("aircraft", "B", root / "rtg", root)


def test_main_writes_outputs_and_warns_on_new_name_in_c(tmp_path, capsys):
    root = make_repo(tmp_path)  # C == B here, so New image i2 still retrieves its own name
    out = tmp_path / "out"
    n8.main(["--root", str(root), "--rtg-dir", str(root / "rtg"), "--out", str(out), "--datasets", "aircraft"])
    assert {p.name for p in out.iterdir()} == {"aircraft_B_per_image.csv", "aircraft_C_per_image.csv",
                                               "n8_summary.csv", "n8_summary.md"}
    with (out / "n8_summary.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 6  # 2 versions x All/Old/New
    assert "CẢNH BÁO aircraft/C" in capsys.readouterr().err
