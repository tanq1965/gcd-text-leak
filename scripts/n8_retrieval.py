#!/usr/bin/env python3
"""N8: name-retrieval and name-confusion rates of RTG output (RQ1, RQ4).

Run from the repository root after rtg/{ds}_{B,C}.pkl exist:
    python scripts/n8_retrieval.py                      # B and C, 3 datasets
    python scripts/n8_retrieval.py --versions B C D     # D has no tags, only attribute columns are meaningful

Metrics (per image, then averaged over All/Old/New images of the split):
- correct_top3   : the image's own class name (exact name or N3-approved alias) is one of its top-3 tags.
- top1_correct   : the top-1 tag is the own class name.
- top1_wrong     : "nhận nhầm tên" - the top-1 tag is the name of ANOTHER class of the same dataset.
                   Split further into top1_wrong_old / top1_wrong_new by the class it names.
- top1_other     : the top-1 tag is not a class name of this dataset (other species, generic tag...).
- no_tag         : RTG kept no tag (all scores under threshold, or version D).
  top1_correct + top1_wrong + top1_other + no_tag = 100%.
- attr_correct   : the "X which" prefix of one of the top-2 attributes is the own class name.
- covered_B      : the own class name is a tag in the ORIGINAL lexicon B (coverage/{ds}.csv, in_tag);
  correct_top3_covered = correct_top3 restricted to those images (separates "name available" from
  "name retrieved").

- Coarse-name (token-prefix) leak, own tokenizer (lowercase, non-alphanumerics -> space), independent of
  the N3 filter. A text (tag, or the X of "X which" for attributes) points at the classes whose name/alias
  it is a strict token prefix of. A leading MANUFACTURER may be dropped on either side, so "gmc canyon ..."
  is also matched by "canyon" and "boeing 747" also matches "747 100". Only manufacturers are dropped
  (Cars: first token of class names + multi-token makes; Aircraft: manufacturer part of approved aliases;
  CUB: none): "Manic GT" is NOT matched to "Ford GT". Labels are the union over both forms, so
  "boeing 747" points at 747-100/200 (Old) and 747-300/400 (New).
  Per image, over the top-3 tags:
    prefix_ge2_top3     own class matched by a tag of >= 2 tokens
    prefix_1tok_top3    own class matched by a 1-token tag ("canyon", "impala", but also "house"):
                        manual review group, see n8_prefix_hits.csv
    prefix_newonly_top3 own class matched and every matched class is New (Tan's N3 rule; 0 for Old images)
    prefix_unique_top3  own class is the only matched class (stricter than New-only)
  attr_prefix_top2 / attr_prefix_newonly_top2 / attr_prefix_unique_top2: same on the top-2 attributes.
  Not counted: exact names, and "extra token + exact name" ("lesser nighthawk", "triumph spitfire"),
  i.e. the names N3 deliberately did not merge (all Old classes).
  The >=2-token threshold of N3 is a filter guard; N8 still measures 1-token prefixes (separate column).
- Staleness: every row checks the sha256 of the lexicon recorded in rtg/{ds}_{ver}.json against the
  current lexicon file (B: third_party/textgcd/Lexicon, C/D: lexicon/{ds}/{ver}). A mismatch is flagged
  at the top of n8_summary.md and on the row; do not cite those rows.

Name matching reuses scripts/n3_lexicon_alias.py (normalize, aliases, tag_name, attr_parts), so N8 counts
exactly what N3 counts as a class name.

Outputs (in --out, default retrieval/):
- {ds}_{ver}_per_image.csv : one row per image (input for N9 error analysis),
- n8_summary.csv           : one row per dataset x version x group,
- n8_summary.md            : RQ1 table + coarse-name leak table for the report,
- n8_prefix_hits.csv       : every coarse-name match that hit the own class (for manual review).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import n3_lexicon_alias as n3  # noqa: E402

DATASETS = ("cub", "scars", "aircraft")
DS_TITLE = {"cub": "CUB", "scars": "Stanford Cars", "aircraft": "FGVC-Aircraft"}
GROUPS = ("All", "Old", "New")
LEXICON = {"B": ("third_party/textgcd/Lexicon/Lexicon_tags.csv", "third_party/textgcd/Lexicon/Lexicon_attributes.csv"),
           "C": ("lexicon/{ds}/C/tags.csv", "lexicon/{ds}/C/attributes.csv"),
           "D": ("lexicon/{ds}/D/tags.csv", "lexicon/{ds}/D/attributes.csv")}
PCT_COLS = ("correct_top3", "correct_top3_covered", "top1_correct", "top1_wrong", "top1_wrong_old",
            "top1_wrong_new", "top1_other", "no_tag", "attr_correct")


def tokens(text: str) -> tuple[str, ...]:
    """Own tokenizer for the prefix-leak columns (independent of N3's normalize)."""
    text = re.sub(r"^\s*a photo of\s+(?:a|an)\s+", "", text, flags=re.I)
    return tuple(re.sub(r"[^a-z0-9]+", " ", text.casefold()).split())


CARS_MULTI_TOKEN_MAKES = {("am", "general"), ("aston", "martin"), ("land", "rover"), ("mercedes", "benz"),
                          ("rolls", "royce")}


def manufacturers(ds: str, classes: list[dict]) -> set[tuple[str, ...]]:
    """Manufacturer prefixes that may be dropped, taken from the data (no hand list except Cars multi-token makes)."""
    makers: set[tuple[str, ...]] = set()
    if ds == "scars":
        for row in classes:
            full = tokens(row["name_norm"])
            two = full[:2]
            makers.add(two if two in CARS_MULTI_TOKEN_MAKES else full[:1])
    elif ds == "aircraft":
        for row in classes:
            base = tokens(row["name_norm"])
            for alias in n3.aliases(row["name_norm"], ds):
                full = tokens(alias)
                if base and base[0] in full[1:]:
                    lead = full[: full.index(base[0])]
                    if lead and not any(ch.isdigit() for t in lead for ch in t):
                        makers.add(lead)
    return makers


def drop_maker(key: tuple[str, ...], makers: set) -> tuple[str, ...] | None:
    for size in (2, 1):
        if len(key) > size and key[:size] in makers:
            return key[size:]
    return None


def build_prefix_index(names_csv: Path, ds: str) -> tuple[dict, set, set]:
    """strict token prefix -> labels (name/alias as is, or without its manufacturer); exact names; makers."""
    classes = n3.load_names(names_csv)
    makers = manufacturers(ds, classes)
    index: dict[tuple, set[int]] = {}
    exact: set[tuple] = set()
    for row in classes:
        for name in n3.aliases(row["name_norm"], ds):
            full = tokens(name)
            exact.add(full)
            for seq in filter(None, (full, drop_maker(full, makers))):
                for size in range(1, len(seq)):
                    index.setdefault(seq[:size], set()).add(row["label"])
    return index, exact, makers


def prefix_labels(text: str, index: dict, exact: set, makers: set = frozenset()) -> set[int]:
    """Labels whose name/alias the text is a strict token prefix of (as is, or without a leading manufacturer)."""
    key = tokens(text)
    if not key or key in exact:
        return set()
    labels = set(index.get(key, set()))
    rest = drop_maker(key, makers)
    if rest and rest not in exact:
        labels |= index.get(rest, set())
    return labels


def attr_prefix(sentence: str) -> str | None:
    match = re.match(r"^\s*(.+?)\s+which\s+", sentence, re.I)
    return match.group(1) if match else None


def lexicon_status(ds: str, ver: str, rtg_dir: Path, root: Path) -> str:
    """'ok' if rtg/{ds}_{ver}.json was built from the current lexicon files, else 'STALE'/'no-meta'."""
    meta_path = rtg_dir / f"{ds}_{ver}.json"
    if not meta_path.is_file():
        return "no-meta"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    for key, rel in zip(("tags_sha256", "attributes_sha256"), LEXICON[ver]):
        current = hashlib.sha256((root / rel.format(ds=ds)).read_bytes()).hexdigest()
        if meta.get(key) != current:
            return "STALE"
    return "ok"


def build_name_index(names_csv: Path, ds: str) -> tuple[dict[str, int], dict[int, bool]]:
    """Map every normalized class name/alias -> label, and label -> is_old."""
    classes = n3.load_names(names_csv)
    key_to_label: dict[str, int] = {}
    for row in classes:
        for key in n3.aliases(row["name_norm"], ds):
            if key in key_to_label and key_to_label[key] != row["label"]:
                raise ValueError(f"{ds}: alias {key!r} maps to two classes")
            key_to_label[key] = row["label"]
    return key_to_label, {row["label"]: row["is_old"] for row in classes}


def load_split(split_csv: Path) -> list[dict]:
    with split_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["label"] = int(row["label"])
        row["is_old"] = row["is_old"] in ("1", "True", "true")
    return rows


def load_covered(coverage_csv: Path) -> dict[int, bool]:
    with coverage_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        return {int(r["label"]): r["in_tag"] == "1" for r in csv.DictReader(handle)}


def classify_image(tags: list[str], attrs: list[str], label: int,
                   key_to_label: dict[str, int], is_old: dict[int, bool]) -> dict:
    """Per-image name metrics for one RTG record."""
    tag_labels = [key_to_label.get(n3.tag_name(t)) for t in tags]
    attr_labels = [key_to_label.get(n3.attr_parts(a)[0] or "") for a in attrs]
    top1 = tag_labels[0] if tag_labels else None
    if not tags:
        top1_kind = "no_tag"
    elif top1 == label:
        top1_kind = "correct"
    elif top1 is not None:
        top1_kind = "wrong_old" if is_old[top1] else "wrong_new"
    else:
        top1_kind = "other"
    return {
        "correct_top3": int(label in tag_labels[:3]),
        "top1_kind": top1_kind,
        "top1_label": "" if top1 is None else top1,
        "attr_correct": int(label in attr_labels[:2]),
    }


def evaluate(ds: str, ver: str, rtg_dir: Path, root: Path = ROOT) -> list[dict]:
    """Return per-image rows for dataset ds, RTG version ver."""
    key_to_label, is_old = build_name_index(root / "names" / f"{ds}.csv", ds)
    split = load_split(root / "splits" / f"{ds}.csv")
    covered = load_covered(root / "coverage" / f"{ds}.csv")
    with (rtg_dir / f"{ds}_{ver}.pkl").open("rb") as handle:
        rtg = pickle.load(handle)

    if [r["image_id"] for r in rtg] != [s["image_id"] for s in split]:
        raise ValueError(f"rtg/{ds}_{ver}.pkl: image order differs from splits/{ds}.csv")
    for s in split:
        if is_old[s["label"]] != s["is_old"]:
            raise ValueError(f"{ds}: is_old of label {s['label']} differs between names/ and splits/")

    prefix_index, exact, makers = build_prefix_index(root / "names" / f"{ds}.csv", ds)
    rows = []
    for s, r in zip(split, rtg):
        own = s["label"]
        m = classify_image(r["tags"], r["attributes"], own, key_to_label, is_old)
        tag_hits = [(t, len(tokens(t)), prefix_labels(t, prefix_index, exact, makers)) for t in r["tags"][:3]]
        attr_hits = [(p, len(tokens(p)), prefix_labels(p, prefix_index, exact, makers))
                     for a in r["attributes"][:2] if (p := attr_prefix(a))]
        new_only = lambda labels: own in labels and not any(is_old[x] for x in labels)  # noqa: E731
        rows.append({
            "image_id": s["image_id"], "label": own, "is_old": int(s["is_old"]),
            "is_labeled": s["is_labeled"], "covered_B": int(covered[own]), **m,
            "top1_tag": r["tags"][0] if r["tags"] else "",
            "prefix_top3": int(any(own in h for _, _, h in tag_hits)),
            "prefix_ge2_top3": int(any(own in h and n >= 2 for _, n, h in tag_hits)),
            "prefix_1tok_top3": int(any(own in h and n == 1 for _, n, h in tag_hits)),
            "prefix_newonly_top3": int(any(new_only(h) for _, _, h in tag_hits)),
            "prefix_unique_top3": int(any(h == {own} for _, _, h in tag_hits)),
            "attr_prefix_top2": int(any(own in h for _, _, h in attr_hits)),
            "attr_prefix_newonly_top2": int(any(new_only(h) for _, _, h in attr_hits)),
            "attr_prefix_unique_top2": int(any(h == {own} for _, _, h in attr_hits)),
            "_hits": [(src, text, n, sorted(h)) for src, hits in (("tag", tag_hits), ("attribute", attr_hits))
                      for text, n, h in hits if own in h],
        })
    return rows
    return rows


def summarize(rows: list[dict], ds: str, ver: str, lexicon: str = "ok") -> list[dict]:
    out = []
    for group in GROUPS:
        g = [r for r in rows if group == "All" or bool(r["is_old"]) == (group == "Old")]
        n = len(g)
        cov = [r for r in g if r["covered_B"]]
        pct = lambda k: round(100 * k / n, 2) if n else float("nan")  # noqa: E731
        kinds = [r["top1_kind"] for r in g]
        out.append({
            "dataset": ds, "version": ver, "group": group, "images": n,
            "covered_B_images": len(cov),
            "correct_top3": pct(sum(r["correct_top3"] for r in g)),
            "correct_top3_covered": round(100 * sum(r["correct_top3"] for r in cov) / len(cov), 2)
            if cov else float("nan"),
            "top1_correct": pct(kinds.count("correct")),
            "top1_wrong": pct(kinds.count("wrong_old") + kinds.count("wrong_new")),
            "top1_wrong_old": pct(kinds.count("wrong_old")),
            "top1_wrong_new": pct(kinds.count("wrong_new")),
            "top1_other": pct(kinds.count("other")),
            "no_tag": pct(kinds.count("no_tag")),
            "attr_correct": pct(sum(r["attr_correct"] for r in g)),
            "prefix_top3": pct(sum(r.get("prefix_top3", 0) for r in g)),
            "prefix_ge2_top3": pct(sum(r.get("prefix_ge2_top3", 0) for r in g)),
            "prefix_1tok_top3": pct(sum(r.get("prefix_1tok_top3", 0) for r in g)),
            "prefix_newonly_top3": pct(sum(r.get("prefix_newonly_top3", 0) for r in g)),
            "prefix_unique_top3": pct(sum(r.get("prefix_unique_top3", 0) for r in g)),
            "attr_prefix_top2": pct(sum(r.get("attr_prefix_top2", 0) for r in g)),
            "attr_prefix_newonly_top2": pct(sum(r.get("attr_prefix_newonly_top2", 0) for r in g)),
            "attr_prefix_unique_top2": pct(sum(r.get("attr_prefix_unique_top2", 0) for r in g)),
            "lexicon": lexicon,
        })
    return out


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        w = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def to_markdown(summary: list[dict]) -> str:
    stale = sorted({f"{DS_TITLE[r['dataset']]} {r['version']}" for r in summary if r.get("lexicon", "ok") != "ok"})
    status = (f"> ⚠️ **STALE — không trích:** {', '.join(stale)} được tạo từ lexicon khác bản hiện tại "
              "(hash trong rtg/*.json không khớp). Chạy lại RTG rồi chạy lại N8." if stale else
              "> ✅ Mọi hàng dùng RTG khớp hash lexicon hiện tại (kiểm tra tự động khi chạy N8).")
    f = lambda v: "–" if v != v else f"{v:.1f}"  # noqa: E731  (nan -> –)
    g = lambda r, k: f(r.get(k, float("nan")))  # noqa: E731
    mark = lambda r: "" if r.get("lexicon", "ok") == "ok" else " ⚠️"  # noqa: E731
    lines = [
        "# N8 — Truy xuất đúng tên và nhận nhầm tên (RQ1)", "", status, "",
        "## Bảng 1 — Tên chính xác / alias", "",
        "Tỉ lệ % trên số ảnh của split (train, cả có nhãn và không nhãn). "
        "*Đúng top-3*: tên lớp đúng nằm trong top-3 tag. *Nhầm top-1*: tag đứng đầu là tên lớp khác "
        "trong cùng tập (tách theo lớp bị gọi tên là Old/New). *Đúng top-3 (có tên)*: chỉ tính ảnh của "
        "lớp có tên trong tag của lexicon B.", "",
        "| Tập | Bản | Nhóm | Ảnh | Đúng top-3 | Đúng top-3 (có tên) | Đúng top-1 | Nhầm top-1 "
        "| ↳ gọi tên Old | ↳ gọi tên New | Top-1 không phải tên lớp | Không tag | Thuộc tính đúng tên |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {DS_TITLE[r['dataset']]} | {r['version']} | {r['group']} | {r['images']} | "
            f"{f(r['correct_top3'])} | {f(r['correct_top3_covered'])} | {f(r['top1_correct'])} | "
            f"{f(r['top1_wrong'])} | {f(r['top1_wrong_old'])} | {f(r['top1_wrong_new'])} | "
            f"{f(r['top1_other'])} | {f(r['no_tag'])} | {f(r['attr_correct'])} |" + mark(r))
    lines += [
        "", "## Bảng 2 — Rò tên cấp thô (tiền tố token)", "",
        "Đo độc lập với bộ lọc N3 (bộ tách token riêng). Một tag (hoặc phần `X` của `X which …`) *rò tiền tố* "
        "nếu là tiền tố token ngắn hơn của tên/alias lớp đúng; chỉ bỏ **tên hãng** ở đầu (Cars: hãng xe; "
        "Aircraft: hãng trong alias đã duyệt; CUB: không bỏ), nên `canyon` < `gmc canyon …`, `boeing 747` khớp "
        "cả 747-100/200 (Old) lẫn 747-400 (New), còn `Manic GT` không khớp `Ford GT`. Không tính tên chính xác "
        "và dạng `từ thêm + tên chính xác` (`lesser nighthawk`). *≥2 token* / *1 token* tách theo số token của tag "
        "(nhóm 1 token gồm cả từ thông dụng như `house`, cần kiểm tay: `n8_prefix_hits.csv`). *Chỉ-New*: mọi lớp "
        "khớp đều New (luật N3 của Tân; ảnh Old luôn 0). *Chỉ lớp này*: chỉ khớp đúng lớp của ảnh (chặt hơn).", "",
        "| Tập | Bản | Nhóm | Tag ≥2 token | Tag 1 token (kiểm tay) | Tag chỉ-New | Tag chỉ lớp này "
        "| Thuộc tính | Thuộc tính chỉ-New | Thuộc tính chỉ lớp này |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in summary:
        lines.append(
            f"| {DS_TITLE[r['dataset']]} | {r['version']} | {r['group']} | {g(r, 'prefix_ge2_top3')} | "
            f"{g(r, 'prefix_1tok_top3')} | {g(r, 'prefix_newonly_top3')} | {g(r, 'prefix_unique_top3')} | "
            f"{g(r, 'attr_prefix_top2')} | {g(r, 'attr_prefix_newonly_top2')} | {g(r, 'attr_prefix_unique_top2')} |"
            + mark(r))
    return "\n".join(lines) + "\n"


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rtg-dir", type=Path, default=ROOT / "rtg")
    p.add_argument("--root", type=Path, default=ROOT, help="repo root holding names/, splits/, coverage/")
    p.add_argument("--out", type=Path, default=ROOT / "retrieval")
    p.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=DATASETS)
    p.add_argument("--versions", nargs="+", default=["B", "C"], choices=["B", "C", "D"])
    args = p.parse_args(argv)

    summary, hits = [], {}
    for ds in args.datasets:
        for ver in args.versions:
            rows = evaluate(ds, ver, args.rtg_dir, args.root)
            status = lexicon_status(ds, ver, args.rtg_dir, args.root)
            if status != "ok":
                print(f"CẢNH BÁO {ds}/{ver}: rtg {status} so với lexicon hiện tại -> không trích hàng này",
                      file=sys.stderr)
            names = {c["label"]: c["class_name"] for c in n3.load_names(args.root / "names" / f"{ds}.csv")}
            for row in rows:
                for src, text, n, labels in row.pop("_hits"):
                    key = (ds, ver, src, text.strip(), n, " | ".join(names[x] for x in labels), names[row["label"]])
                    hits[key] = hits.get(key, 0) + 1
            write_csv(args.out / f"{ds}_{ver}_per_image.csv", rows)
            summary += summarize(rows, ds, ver, status)
            if ver == "C":  # risk check in the plan: C must never retrieve a New class name
                new_hits = sum(r["correct_top3"] for r in rows if not r["is_old"])
                if new_hits:
                    print(f"CẢNH BÁO {ds}/C: {new_hits} ảnh New vẫn truy xuất được tên lớp đúng "
                          f"-> lọc tên còn sót, báo N3/N6", file=sys.stderr)
    write_csv(args.out / "n8_summary.csv", summary)
    hit_rows = [dict(zip(("dataset", "version", "source", "text", "n_tokens", "matched_classes", "own_class"), k),
                     images=v) for k, v in sorted(hits.items(), key=lambda kv: (kv[0][:2], -kv[1]))]
    if hit_rows:
        write_csv(args.out / "n8_prefix_hits.csv", hit_rows)
    (args.out / "n8_summary.md").write_text(to_markdown(summary), encoding="utf-8", newline="\n")
    print(to_markdown(summary))
    print(f"Đã ghi {args.out}/n8_summary.csv, n8_summary.md, n8_prefix_hits.csv và *_per_image.csv")


if __name__ == "__main__":
    main()
