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


def test_prefix_leak_columns_use_own_tokens_and_uniqueness(tmp_path):
    rtg = [dict(RTG[0], tags=["a photo of a Spitfire mk ix"]), dict(RTG[1], tags=["a photo of a DC"]),
           dict(RTG[2], tags=["a photo of a Tornado"]), dict(RTG[3], tags=["a photo of a Yak"])]
    root = make_repo(tmp_path, rtg_c=rtg)
    rows = {r["image_id"]: r for r in n8.evaluate("aircraft", "C", root / "rtg", root)}
    assert rows["i1.jpg"]["prefix_top3"] == 1 and rows["i1.jpg"]["prefix_unique_top3"] == 1  # "dc" < "dc 3"
    assert rows["i3.jpg"]["prefix_unique_top3"] == 1  # "yak" < "yak 42"
    assert rows["i2.jpg"]["prefix_top3"] == 0  # exact name is not a prefix leak
    assert rows["i0.jpg"]["prefix_top3"] == 0  # longer name is not a prefix


def test_lexicon_status_flags_stale_rtg(tmp_path):
    (tmp_path / "lexicon/aircraft/C").mkdir(parents=True)
    (tmp_path / "lexicon/aircraft/C/tags.csv").write_text("a photo of a cat\n", encoding="utf-8")
    (tmp_path / "lexicon/aircraft/C/attributes.csv").write_text("has wings\n", encoding="utf-8")
    import hashlib, json
    sha = lambda p: hashlib.sha256((tmp_path / p).read_bytes()).hexdigest()  # noqa: E731
    meta = {"tags_sha256": sha("lexicon/aircraft/C/tags.csv"), "attributes_sha256": sha("lexicon/aircraft/C/attributes.csv")}
    (tmp_path / "aircraft_C.json").write_text(json.dumps(meta), encoding="utf-8")
    assert n8.lexicon_status("aircraft", "C", tmp_path, tmp_path) == "ok"
    (tmp_path / "lexicon/aircraft/C/tags.csv").write_text("a photo of a dog\n", encoding="utf-8")
    assert n8.lexicon_status("aircraft", "C", tmp_path, tmp_path) == "STALE"
    assert n8.lexicon_status("cub", "C", tmp_path, tmp_path) == "no-meta"
    assert "STALE" in n8.to_markdown([dict(n8.summarize([], "aircraft", "C", "STALE")[0], images=1)])


def test_attribute_prefix_leak(tmp_path):
    rtg = [dict(x, attributes=[]) for x in RTG]
    rtg[1]["attributes"] = ["dc which has two engines", "has wings"]
    root = make_repo(tmp_path, rtg_c=rtg)
    rows = {r["image_id"]: r for r in n8.evaluate("aircraft", "C", root / "rtg", root)}
    assert rows["i1.jpg"]["attr_prefix_top2"] == 1 and rows["i1.jpg"]["attr_prefix_unique_top2"] == 1
    assert rows["i0.jpg"]["attr_prefix_top2"] == 0


# --------------------------------------------------------------------------- coarse-name leak (review round 4)
def _index(tmp_path, rows, ds):
    path = tmp_path / f"{ds}.csv"
    path.write_text("label,class_name,name_norm,is_old\n" + "".join(f"{r}\n" for r in rows), encoding="utf-8")
    return n8.build_prefix_index(path, ds)


# Old classes first (required by names/ format). Labels: see comments.
CARS = ["0,Honda Accord Coupe 2012,honda accord coupe 2012,1", "1,Audi S4 Sedan 2012,audi s4 sedan 2012,1",
        "2,Aston Martin V8 Vantage Coupe 2012,aston martin v8 vantage coupe 2012,1",
        "3,Ford GT Coupe 2006,ford gt coupe 2006,0", "4,Acura Integra Type R 2001,acura integra type r 2001,0",
        "5,Chevrolet Impala Sedan 2007,chevrolet impala sedan 2007,0",
        "6,GMC Canyon Extended Cab 2012,gmc canyon extended cab 2012,0", "7,GMC Savana Van 2012,gmc savana van 2012,0",
        "8,Audi S4 Sedan 2007,audi s4 sedan 2007,0"]
AIR = ["0,747-100,747-100,1", "1,747-200,747-200,1", "2,Spitfire,spitfire,1",
       "3,747-300,747-300,0", "4,747-400,747-400,0", "5,Yak-42,yak-42,0"]
CUB = ["0,118.House_Sparrow,house sparrow,1", "1,092.Nighthawk,nighthawk,1", "2,116.Field_Sparrow,field sparrow,0"]


def test_tokens_strip_prompt_and_punctuation():
    assert n8.tokens("a photo of an F/A-18 Hornet") == ("f", "a", "18", "hornet")
    assert n8.tokens("  A Photo Of A Ford E-Series  ") == ("ford", "e", "series")
    assert n8.tokens("") == ()


def test_manufacturers_come_from_data(tmp_path):
    _, _, cars = _index(tmp_path, CARS, "scars")
    assert {("ford",), ("acura",), ("honda",), ("gmc",), ("aston", "martin")} <= cars
    _, _, air = _index(tmp_path, AIR, "aircraft")
    assert ("boeing",) in air  # from approved alias "boeing 747 400"
    _, _, cub = _index(tmp_path, CUB, "cub")
    assert cub == set()  # no manufacturer is ever dropped for birds


def test_boeing_747_is_old_new_collision(tmp_path):
    index, exact, makers = _index(tmp_path, AIR, "aircraft")
    labels = n8.prefix_labels("a photo of a Boeing 747", index, exact, makers)
    assert labels == {0, 1, 3, 4}  # direct: alias of New 747-400; after "boeing": Old 747-100/200 too
    assert labels & {0, 1} and labels != {4}  # contains Old -> not New-only, not unique


def test_manic_gt_and_lightning_gt_do_not_match_ford_gt(tmp_path):
    index, exact, makers = _index(tmp_path, CARS, "scars")
    assert n8.prefix_labels("a photo of a Manic gt", index, exact, makers) == set()
    assert n8.prefix_labels("a photo of a Lightning gt", index, exact, makers) == set()
    assert n8.prefix_labels("a photo of a Ford gt", index, exact, makers) == {3}  # the real coarse name still counts


def test_manufacturer_drop_is_real_maker_only(tmp_path):
    index, exact, makers = _index(tmp_path, CARS, "scars")
    assert n8.prefix_labels("a photo of a Honda integra", index, exact, makers) == {4}  # honda is a Cars make
    assert n8.prefix_labels("a photo of a Yenko integra", index, exact, makers) == set()  # yenko is not
    assert n8.prefix_labels("a photo of a Aston martin v8", index, exact, makers) == {2}  # 2-token make kept whole


def test_one_token_prefixes_are_detected(tmp_path):
    index, exact, makers = _index(tmp_path, CARS, "scars")
    assert n8.prefix_labels("a photo of a impala", index, exact, makers) == {5}
    assert n8.prefix_labels("a photo of a canyon", index, exact, makers) == {6}


def test_exact_and_qualified_names_are_not_prefix_leaks(tmp_path):
    index, exact, makers = _index(tmp_path, AIR, "aircraft")
    assert n8.prefix_labels("a photo of a Spitfire", index, exact, makers) == set()
    assert n8.prefix_labels("a photo of a Spitfire aircraft", index, exact, makers) == set()
    assert n8.prefix_labels("a photo of a Triumph Spitfire", index, exact, makers) == set()
    assert n8.prefix_labels("a photo of a Yak", index, exact, makers) == {5}
    assert n8.prefix_labels("a photo of a aircraft", index, exact, makers) == set()
    cindex, cexact, cmakers = _index(tmp_path, CUB, "cub")
    assert n8.prefix_labels("a photo of a Lesser nighthawk", cindex, cexact, cmakers) == set()


def test_attr_prefix_parsing():
    assert n8.attr_prefix("yak 42 which has three engines") == "yak 42"
    assert n8.attr_prefix("Yak 42 WHICH has three engines") == "Yak 42"
    assert n8.attr_prefix("has three engines") is None
    assert n8.attr_prefix("is a tang, which connects the blade") == "is a tang,"


def _cars_repo(tmp_path, records):
    """Mini Cars repo; records = [(label, is_old, tags, attributes)]."""
    for sub in ("names", "splits", "coverage", "rtg"):
        (tmp_path / sub).mkdir(exist_ok=True)
    (tmp_path / "names/scars.csv").write_text("label,class_name,name_norm,is_old\n" + "\n".join(CARS) + "\n", encoding="utf-8")
    olds = {int(r.split(",")[0]): r.split(",")[-1] for r in CARS}
    (tmp_path / "splits/scars.csv").write_text(
        "image_id,path,label,is_labeled,is_old\n" + "".join(f"i{k},x,{lab},0,{olds[lab]}\n" for k, (lab, *_r) in enumerate(records)),
        encoding="utf-8")
    (tmp_path / "coverage/scars.csv").write_text("label,in_tag\n" + "".join(f"{k},1\n" for k in olds), encoding="utf-8")
    with (tmp_path / "rtg/scars_C.pkl").open("wb") as f:
        pickle.dump([{"image_id": f"i{k}", "tags": t, "attributes": a} for k, (_l, t, a) in enumerate(records)], f)
    return {r["image_id"]: r for r in n8.evaluate("scars", "C", tmp_path / "rtg", tmp_path)}


def test_columns_one_token_vs_two_tokens_new_only_and_unique(tmp_path):
    rows = _cars_repo(tmp_path, [
        (7, ["a photo of a Gmc savana"], []),             # >=2 token, New-only, unique
        (6, ["a photo of a canyon"], []),                  # 1 token, New-only, unique
        (8, ["a photo of a Audi s4"], []),                 # matches Old+New -> leak but not New-only/unique
        (5, ["a photo of a cat"], ["impala which has a long hood"]),  # attribute prefix leak
        (1, ["a photo of a Audi s4"], []),                 # Old image: New-only is always 0
    ])
    savana, canyon, s4_new, impala, s4_old = (rows[f"i{k}"] for k in range(5))
    assert (savana["prefix_ge2_top3"], savana["prefix_1tok_top3"], savana["prefix_newonly_top3"], savana["prefix_unique_top3"]) == (1, 0, 1, 1)
    assert (canyon["prefix_ge2_top3"], canyon["prefix_1tok_top3"], canyon["prefix_newonly_top3"]) == (0, 1, 1)
    assert (s4_new["prefix_ge2_top3"], s4_new["prefix_newonly_top3"], s4_new["prefix_unique_top3"]) == (1, 0, 0)
    assert (impala["attr_prefix_top2"], impala["attr_prefix_newonly_top2"], impala["attr_prefix_unique_top2"]) == (1, 1, 1)
    assert impala["prefix_top3"] == 0
    assert s4_old["prefix_ge2_top3"] == 1 and s4_old["prefix_newonly_top3"] == 0


def test_new_only_counts_prefix_shared_by_two_new_classes(tmp_path):
    rows = _cars_repo(tmp_path, [(6, ["a photo of a Gmc"], []), (7, ["a photo of a Gmc"], [])])
    assert rows["i0"]["prefix_newonly_top3"] == 1 and rows["i0"]["prefix_unique_top3"] == 0  # GMC: 2 New classes only


def test_house_and_field_land_in_one_token_review_group(tmp_path):
    index, exact, makers = _index(tmp_path, CUB, "cub")
    assert n8.prefix_labels("a photo of a house", index, exact, makers) == {0}
    assert n8.prefix_labels("a photo of a field", index, exact, makers) == {2}
    # the token count, not the match, decides the column: these never reach the >=2-token column
    assert len(n8.tokens("a photo of a house")) == 1


def test_prefix_columns_only_look_at_top3_tags_and_top2_attributes(tmp_path):
    rows = _cars_repo(tmp_path, [(7, ["a photo of a cat", "a photo of a dog", "a photo of a car", "a photo of a Gmc savana"],
                                  ["has wheels", "has doors", "savana which is a van"])])
    assert rows["i0"]["prefix_top3"] == 0 and rows["i0"]["attr_prefix_top2"] == 0


def test_lexicon_status_for_b_path_and_markdown_marks_each_stale_row(tmp_path):
    lex = tmp_path / "third_party/textgcd/Lexicon"
    lex.mkdir(parents=True)
    (lex / "Lexicon_tags.csv").write_text("a photo of a cat\n", encoding="utf-8")
    (lex / "Lexicon_attributes.csv").write_text("has wings\n", encoding="utf-8")
    import hashlib
    import json
    meta = {k: hashlib.sha256((lex / f).read_bytes()).hexdigest()
            for k, f in (("tags_sha256", "Lexicon_tags.csv"), ("attributes_sha256", "Lexicon_attributes.csv"))}
    (tmp_path / "cub_B.json").write_text(json.dumps(meta), encoding="utf-8")
    assert n8.lexicon_status("cub", "B", tmp_path, tmp_path) == "ok"
    rows = [dict(n8.summarize([], "cub", "B", "ok")[0], images=1),
            dict(n8.summarize([], "cub", "C", "no-meta")[0], images=1)]
    md = n8.to_markdown(rows)
    assert "STALE" in md and "CUB C" in md
    assert md.count("⚠️") == 3  # status line + the no-meta row in each of the two tables


def test_main_writes_prefix_hits_file(tmp_path):
    rtg = [dict(x) for x in RTG]
    rtg[3] = dict(RTG[3], tags=["a photo of a Yak"])
    root = make_repo(tmp_path, rtg_c=rtg)
    out = tmp_path / "out"
    n8.main(["--root", str(root), "--rtg-dir", str(root / "rtg"), "--out", str(out), "--datasets", "aircraft"])
    with (out / "n8_prefix_hits.csv").open(encoding="utf-8") as f:
        hits = list(csv.DictReader(f))
    assert any(h["version"] == "C" and h["text"] == "a photo of a Yak" and h["n_tokens"] == "1" for h in hits)
