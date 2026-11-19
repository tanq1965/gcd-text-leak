from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import n3_lexicon_alias as lexicon


def write_csv(path: Path, rows: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_cub_aliases_are_exact_and_dataset_scoped():
    assert lexicon.aliases("105.Whip_poor_Will", "cub") == {
        "whip poor will",
        "eastern whip poor will",
        "mexican whip poor will",
    }
    assert lexicon.aliases("091.Mockingbird", "cub") == {
        "mockingbird",
        "northern mockingbird",
    }
    assert lexicon.aliases("022.Chuck_will_Widow", "cub") == {"chuck will widow"}
    assert lexicon.aliases("Whip-poor-will", "scars") == {"whip poor will"}
    assert lexicon.aliases("737-200", "aircraft") == {"737 200", "737 200 aircraft"}


def test_tag_and_attribute_parsers():
    assert lexicon.tag_name("a photo of a Eastern Whip poor will") == "eastern whip poor will"
    assert lexicon.tag_name("a photo of an Northern Mockingbird") == "northern mockingbird"
    assert lexicon.attr_parts("Northern Mockingbird which has long tail") == (
        "northern mockingbird",
        "has long tail",
    )
    assert lexicon.attr_parts("has long tail") == (None, None)


def test_load_names_rejects_noncontiguous_labels(tmp_path):
    path = tmp_path / "names.csv"
    write_csv(
        path,
        [
            ["label", "class_name", "name_norm", "is_old"],
            ["0", "Old Bird", "old bird", "1"],
            ["2", "New Bird", "new bird", "0"],
        ],
    )
    with pytest.raises(ValueError, match="contiguous"):
        lexicon.load_names(path)


def test_cub_aliases_are_applied_and_audited_in_c_and_d(tmp_path, monkeypatch):
    names = tmp_path / "names.csv"
    tags = tmp_path / "tags.csv"
    attrs = tmp_path / "attrs.csv"
    write_csv(
        names,
        [
            ["label", "class_name", "name_norm", "is_old"],
            ["0", "000.Old_Bird", "old bird", "1"],
            ["1", "105.Whip_poor_Will", "whip poor will", "0"],
            ["2", "091.Mockingbird", "mockingbird", "0"],
            ["3", "022.Chuck_will_Widow", "chuck will widow", "0"],
        ],
    )
    write_csv(
        tags,
        [
            ["a photo of an Old Bird"],
            ["a photo of a Eastern Whip poor will"],
            ["a photo of a Mexican Whip poor will"],
            ["a photo of a Northern Mockingbird"],
            ["a photo of a Chuck will Widow"],
        ],
    )
    write_csv(
        attrs,
        [
            ["old bird which has brown plumage"],
            ["eastern whip poor will which has nocturnal habits"],
            ["mexican whip poor will which has dark plumage"],
            ["northern mockingbird which has long tail"],
            ["chuck will widow which has a distinct call"],
        ],
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "n3_lexicon_alias.py",
            "--ds",
            "cub",
            "--names",
            str(names),
            "--tags",
            str(tags),
            "--attrs",
            str(attrs),
            "--out",
            str(tmp_path / "out"),
        ],
    )
    lexicon.main()

    output = tmp_path / "out" / "lexicon" / "cub"
    c_tags = lexicon.read_one_column(output / "C" / "tags.csv")
    assert c_tags == ["a photo of an Old Bird"]

    c_attrs = lexicon.read_one_column(output / "C" / "attributes.csv")
    assert c_attrs == [
        "old bird which has brown plumage",
        "has nocturnal habits",
        "has dark plumage",
        "has long tail",
        "has a distinct call",
    ]

    c_removed = read_csv(output / "C" / "removed.csv")
    alias_removed = [row for row in c_removed if row["rule"] == "alias"]
    assert {row["reason"] for row in alias_removed} == {
        "AOU 2010 split",
        "only common North American mockingbird",
    }
    assert len(alias_removed) == 3

    c_modified = read_csv(output / "C" / "modified.csv")
    alias_modified = [row for row in c_modified if row["rule"] == "alias"]
    assert len(alias_modified) == 3

    d_tags_path = output / "D" / "tags.csv"
    assert d_tags_path.read_bytes() == b""
    d_attrs = lexicon.read_one_column(output / "D" / "attributes.csv")
    assert d_attrs == [
        "has brown plumage",
        "has nocturnal habits",
        "has dark plumage",
        "has long tail",
        "has a distinct call",
    ]

    audit = json.loads((tmp_path / "out" / "coverage" / "cub_audit.json").read_text())
    assert audit["classes"] == 4
    assert audit["old"] == 1 and audit["new"] == 3
    assert len(audit["approved_aliases"]) == 2
    assert "022.Chuck_will_Widow" not in {
        row["class_name"] for row in audit["approved_aliases"]
    }


def test_alias_collision_is_rejected(tmp_path, monkeypatch):
    names = tmp_path / "names.csv"
    tags = tmp_path / "tags.csv"
    attrs = tmp_path / "attrs.csv"
    write_csv(
        names,
        [
            ["label", "class_name", "name_norm", "is_old"],
            ["0", "Mockingbird", "mockingbird", "1"],
            ["1", "Northern_Mockingbird", "northern mockingbird", "0"],
        ],
    )
    write_csv(tags, [["a photo of a Northern Mockingbird"]])
    write_csv(attrs, [["northern mockingbird which has a long tail"]])

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "n3_lexicon_alias.py",
            "--ds",
            "cub",
            "--names",
            str(names),
            "--tags",
            str(tags),
            "--attrs",
            str(attrs),
            "--out",
            str(tmp_path / "out"),
        ],
    )
    with pytest.raises(ValueError, match="Ambiguous"):
        lexicon.main()

@pytest.mark.parametrize(
    "class_name, expected_alias",
    [
        ("A320", "airbus a320 family"),
        ("DC-8", "douglas dc 8"),
        ("ERJ 145", "embraer erj 145 family"),
        ("MD-80", "mcdonnell douglas md 80"),
        ("747-400", "boeing 747 400"),
        ("C-130", "lockheed c 130 hercules"),
        ("DC-6", "douglas dc 6"),
        ("Challenger 600", "bombardier challenger 600"),
        ("Metroliner", "fairchild c 26 metroliner"),
    ],
)
def test_aircraft_reviewed_aliases(class_name, expected_alias):
    assert expected_alias in lexicon.aliases(
        class_name, "aircraft"
    )


def test_unrelated_names_are_not_aircraft_aliases():
    assert "north american b 45 tornado" not in lexicon.aliases(
        "Tornado", "aircraft"
    )

def test_cub_new_aliases_do_not_remove_unrelated_names(
    tmp_path, monkeypatch
):
    names = tmp_path / "names.csv"
    tags = tmp_path / "tags.csv"
    attrs = tmp_path / "attrs.csv"

    write_csv(names, [
        ["label", "class_name", "name_norm", "is_old"],
        ["0", "000.Old_Bird", "old bird", "1"],
        ["1", "101.White_Pelican", "white pelican", "0"],
        ["2", "017.Cardinal", "cardinal", "0"],
    ])

    write_csv(tags, [
        ["a photo of a American White Pelican"],
        ["a photo of a Northern Cardinal"],
        ["a photo of a Blue cardinal flower"],
    ])

    write_csv(attrs, [
        ["american white pelican which has white feathers"],
        ["northern cardinal which has red feathers"],
        ["cardinal directions which indicate north"],
    ])

    monkeypatch.setattr(
        sys, "argv", [
            "n3_lexicon_alias.py",
            "--ds", "cub",
            "--names", str(names),
            "--tags", str(tags),
            "--attrs", str(attrs),
            "--out", str(tmp_path / "out"),
        ]
    )

    lexicon.main()

    output = tmp_path / "out" / "lexicon" / "cub" / "C"

    c_tags = lexicon.read_one_column(
        output / "tags.csv"
    )
    c_attrs = lexicon.read_one_column(
        output / "attributes.csv"
    )

    assert c_tags == [
        "a photo of a Blue cardinal flower"
    ]

    assert c_attrs == [
        "has white feathers",
        "has red feathers",
        "cardinal directions which indicate north",
    ]

    removed = read_csv(output / "removed.csv")
    modified = read_csv(output / "modified.csv")

    assert len(removed) == 2
    assert len(modified) == 2
    assert all(row["rule"] == "alias" for row in removed)

def test_reviewed_residual_names_are_removed(
    tmp_path, monkeypatch
):
    names = tmp_path / "names.csv"
    tags = tmp_path / "tags.csv"
    attrs = tmp_path / "attrs.csv"

    write_csv(names, [
        ["label", "class_name", "name_norm", "is_old"],
        ["0", "Cessna 172", "cessna 172", "1"],
        ["1", "ERJ 135", "erj 135", "0"],
        ["2", "Tornado", "tornado", "0"],
    ])

    write_csv(tags, [
        ["a photo of a Cessna 172"],
        ["a photo of an ERJ 135"],
        ["a photo of a North American B-45 Tornado"],
    ])

    write_csv(attrs, [
        ["cessna 172 which has four seats"],
        ["cessna 172 which resembles the Cessna 172"],
        ["erj 135 which has ERJ 135 markings"],
        ["erj 135 which has swept wings"],
    ])

    # Test the same rules with smaller source row numbers.
    monkeypatch.setitem(
        lexicon.RESIDUAL_NAME_ROWS,
        "aircraft",
        {
            "C": {3: "erj 135"},
            "D": {
                2: "cessna 172",
                3: "erj 135",
            },
        },
    )

    monkeypatch.setattr(
        sys, "argv", [
            "n3_lexicon_alias.py",
            "--ds", "aircraft",
            "--names", str(names),
            "--tags", str(tags),
            "--attrs", str(attrs),
            "--out", str(tmp_path / "out"),
        ]
    )

    lexicon.main()

    output = tmp_path / "out" / "lexicon" / "aircraft"

    c_tags = lexicon.read_one_column(
        output / "C" / "tags.csv"
    )
    assert "a photo of a North American B-45 Tornado" in c_tags
    assert "a photo of an ERJ 135" not in c_tags

    c_attrs = lexicon.read_one_column(
        output / "C" / "attributes.csv"
    )
    assert len(c_attrs) == 3
    assert "has swept wings" in c_attrs

    d_attrs = lexicon.read_one_column(
        output / "D" / "attributes.csv"
    )
    assert d_attrs == [
        "has four seats",
        "has swept wings",
    ]

    c_removed = read_csv(output / "C" / "removed.csv")
    d_removed = read_csv(output / "D" / "removed.csv")

    c_residual = [
        row for row in c_removed
        if row["rule"] == "residual_name"
    ]
    d_residual = [
        row for row in d_removed
        if row["rule"] == "residual_name"
    ]

    assert {row["line"] for row in c_residual} == {"3"}
    assert {row["line"] for row in d_residual} == {"2", "3"}

    assert all(
        row["file"] == "Lexicon_attributes.csv"
        for row in c_residual + d_residual
    )

    audit = json.loads(
        (tmp_path / "out" / "coverage" / "aircraft_audit.json")
        .read_text(encoding="utf-8")
    )

    assert audit["C_removed_attributes"] == 1
    assert audit["D_removed_attributes"] == 2
    assert audit["D_removed_tags"] == 3

# Independent expectations from n3_cub_alias_list.txt (not derived from code tables).
@pytest.mark.parametrize(
    "base, expected",
    [
        ("anna hummingbird", "annas hummingbird"),
        ("bewick wren", "bewicks wren"),
        ("bewick wren", "bewick s wren"),
        ("brandt cormorant", "brandts cormorant"),
        ("brewer blackbird", "brewers blackbird"),
        ("brewer blackbird", "brewer's blackbird"),
        ("brewer sparrow", "brewers sparrow"),
        ("clark nutcracker", "clarks nutcracker"),
        ("heermann gull", "heermanns gull"),
        ("henslow sparrow", "henslows sparrow"),
        ("forsters tern", "forster s tern"),
        ("artic tern", "arctic tern"),
        ("le conte sparrow", "le contes sparrow"),
        ("lincoln sparrow", "lincolns sparrow"),
        ("scott oriole", "scotts oriole"),
        ("wilson warbler", "wilsons warbler"),
        ("harris sparrow", "harriss sparrow"),
    ],
)
def test_cub_reviewed_spelling_aliases(base, expected):
    assert expected in lexicon.aliases(base, "cub")
    assert expected not in lexicon.aliases(base, "aircraft")
    assert lexicon.ALIAS_REASONS[expected] == "possessive/typo spelling"


def run_small_d_fixture(tmp_path, monkeypatch, attributes, tags=None):
    names = tmp_path / "names.csv"
    tags_path = tmp_path / "tags.csv"
    attrs = tmp_path / "attrs.csv"
    out = tmp_path / "out"

    write_csv(names, [
        ["label", "class_name", "name_norm", "is_old"],
        ["0", "Old Class", "old class", "1"],
        ["1", "New Class", "new class", "0"],
    ])

    if tags is None:
        tags = ["a photo of an Old Class", "a photo of a New Class"]
    write_csv(tags_path, [[tag] for tag in tags])
    write_csv(attrs, [[text] for text in attributes])

    monkeypatch.setattr(sys, "argv", [
        "n3_lexicon_alias.py",
        "--ds", "scars",
        "--names", str(names),
        "--tags", str(tags_path),
        "--attrs", str(attrs),
        "--out", str(out),
        "--d-strip-all",
    ])

    lexicon.main()
    return out


def test_d_strip_all_removes_unrelated_prefixes(
    tmp_path, monkeypatch
):
    out = run_small_d_fixture(
        tmp_path,
        monkeypatch,
        [
            "alder flycatcher which has green tail",
            "another bird which has blue feathers",
        ],
    )

    d = out / "lexicon" / "scars" / "D"

    assert lexicon.read_one_column(d / "attributes.csv") == [
        "has green tail",
        "has blue feathers",
    ]

    assert (d / "tags.csv").read_bytes() == b""

    audit = json.loads(
        (out / "coverage" / "scars_audit.json")
        .read_text(encoding="utf-8")
    )

    assert audit["D_scope"] == "all_attribute_prefixes"
    assert audit["D_modified_attributes"] == 2


def test_d_deduplicates_after_strip(
    tmp_path, monkeypatch
):
    out = run_small_d_fixture(
        tmp_path,
        monkeypatch,
        [
            "alder flycatcher which has long tail",
            "cardinal which has long tail",
            "finch which has spotted wings",
        ],
    )

    d = out / "lexicon" / "scars" / "D"

    assert lexicon.read_one_column(d / "attributes.csv") == [
        "has long tail",
        "has spotted wings",
    ]

    removed = read_csv(d / "removed.csv")
    modified = read_csv(d / "modified.csv")

    assert len(removed) == 1
    assert removed[0]["line"] == "2"
    assert removed[0]["rule"] == "D_deduplicate_after_strip"
    assert {row["line"] for row in modified} == {"1", "3"}

    audit = json.loads(
        (out / "coverage" / "scars_audit.json")
        .read_text(encoding="utf-8")
    )

    assert audit["D_deduplicated_attributes"] == 1
    assert audit["D_removed_attributes"] == 1
    assert audit["D_modified_attributes"] == 2

def test_cub_spelling_aliases_filter_new_but_keep_old(tmp_path, monkeypatch):
    names = tmp_path / "names.csv"
    tags = tmp_path / "tags.csv"
    attrs = tmp_path / "attrs.csv"
    out = tmp_path / "out"
    write_csv(names, [
        ["label", "class_name", "name_norm", "is_old"],
        ["0", "122.Harris_Sparrow", "harris sparrow", "1"],
        ["1", "146.Forsters_Tern", "forsters tern", "0"],
        ["2", "141.Artic_Tern", "artic tern", "0"],
    ])
    write_csv(tags, [
        ["a photo of a Harriss Sparrow"],
        ["a photo of a Forster s Tern"],
        ["a photo of a Arctic Tern"],
        ["a photo of a Blue cardinal flower"],
    ])
    write_csv(attrs, [
        ["harriss sparrow which has a long tail"],
        ["forster s tern which has a forked tail"],
        ["arctic tern which has pale wings"],
    ])
    monkeypatch.setattr(sys, "argv", [
        "n3_lexicon_alias.py", "--ds", "cub",
        "--names", str(names), "--tags", str(tags),
        "--attrs", str(attrs), "--out", str(out),
        "--d-strip-all",
    ])
    lexicon.main()
    c = out / "lexicon" / "cub" / "C"
    assert lexicon.read_one_column(c / "tags.csv") == [
        "a photo of a Harriss Sparrow", "a photo of a Blue cardinal flower",
    ]
    assert lexicon.read_one_column(c / "attributes.csv") == [
        "harriss sparrow which has a long tail",
        "has a forked tail", "has pale wings",
    ]
    audit = json.loads((out / "coverage" / "cub_audit.json").read_text())
    assert audit["C_removed_tags"] == 2
    assert audit["C_modified_attributes"] == 2
    assert audit["D_scope"] == "all_attribute_prefixes"


def test_d_strip_all_handles_multiline_and_empty_tags(tmp_path, monkeypatch):
    out = run_small_d_fixture(
        tmp_path, monkeypatch,
        ["blue bird which has bright\nfeathers", "has dark spots"],
        tags=[],
    )
    d = out / "lexicon" / "scars" / "D"
    assert lexicon.read_one_column(d / "attributes.csv") == [
        "has bright\nfeathers", "has dark spots",
    ]
    assert (d / "tags.csv").read_bytes() == b""
    c = out / "lexicon" / "scars" / "C"
    assert (c / "tags.csv").read_bytes() == b""
