"""N2/N6: Retrieval-based Text Generation (RTG) for B/C/D lexicons, in split order.

Retrieval logic follows TextGCD (textgcd@1175472, retrieval_based_text_generation.py):
- auxiliary model CLIP ViT-H-14 (laion2B) loaded with open_clip,
- lexicon sentences encoded with ``open_clip.tokenize`` and L2-normalised,
- score = ``logit_scale * image_feat @ text_feat.T`` (NOT plain cosine). Note: open_clip stores
  ``logit_scale`` in LOG space (CLIP applies ``.exp()`` itself); TextGCD multiplies by the raw
  parameter, ~ln(100) = 4.605 for pretrained CLIP, so threshold 0.2 means cosine >= ~0.043.
  We keep this exactly as TextGCD does and record the value in the output ``.json``,
- keep top-``num_tags`` tags / top-``num_attributes`` attributes whose score >= ``contrastive_th``.

Differences from the original script (logic of the retrieval itself is unchanged):
- Images are read from ``splits/{ds}.csv`` (+ ``--data-root``) instead of TextGCD's dataset
  classes, so the output covers exactly the split image_ids and Aircraft works without a
  TextGCD loader (TextGCD has none). ``image_id`` is the image basename, like TextGCD.
- Lexicon paths, cache paths and fp16 are command-line options.
- The embedding cache is keyed by the lexicon file's sha256 and the model name (in the file
  name and in a ``.json`` sidecar), so a cache of lexicon B can never be reused for C or D.
- An empty ``tags.csv`` (lexicon D) is allowed: tag retrieval is skipped and every image gets
  ``tags == []``.
- The image is encoded once per batch and shared by tag and attribute retrieval.

Output ``rtg/{ds}_{ver}.pkl``: list of ``{"image_id", "tags", "attributes"}`` in split order,
tags/attributes sorted by decreasing score. A ``.json`` sidecar records config and run time.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

DATASETS = ("cub", "scars", "aircraft")
VERSIONS = ("B", "C", "D")
DEFAULT_MODEL = "hf-hub:laion/CLIP-ViT-H-14-laion2B-s32B-b79K"
REPO_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_LEXICON = REPO_ROOT / "third_party" / "textgcd" / "Lexicon"


# --------------------------------------------------------------------------- lexicon
def load_lexicon(path: str | Path) -> list[str]:
    """Read a headerless one-column TextGCD lexicon CSV. An empty file gives []."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    for line, row in enumerate(rows, 1):
        if len(row) != 1:
            raise ValueError(f"{path}:{line}: expected 1 CSV column, got {len(row)}")
    return [row[0] for row in rows]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lexicon_paths(ver: str, ds: str, lexicon_root: str | Path = REPO_ROOT / "lexicon") -> tuple[Path, Path]:
    """B = TextGCD's original lexicon; C/D = lexicon/{ds}/{ver}/ produced by N3."""
    if ver == "B":
        return ORIGINAL_LEXICON / "Lexicon_tags.csv", ORIGINAL_LEXICON / "Lexicon_attributes.csv"
    base = Path(lexicon_root) / ds / ver
    return base / "tags.csv", base / "attributes.csv"


def cache_path(cache_dir: str | Path, lexicon_file: str | Path, sha: str, model_name: str) -> Path:
    model_slug = hashlib.sha256(model_name.encode()).hexdigest()[:8]
    stem = Path(lexicon_file).stem
    return Path(cache_dir) / f"{stem}_{sha[:12]}_{model_slug}.pt"


# --------------------------------------------------------------------------- encoding
def encode_texts(
    model: torch.nn.Module,
    tokenize: Callable[[Sequence[str]], torch.Tensor],
    texts: Sequence[str],
    device: torch.device,
    batch_size: int = 256,
) -> torch.Tensor:
    """Encode and L2-normalise sentences; returns a CPU tensor [N, D] (empty -> [0, 0])."""
    from tqdm import tqdm

    chunks = []
    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Encoding lexicon", leave=False):
            tokens = tokenize(list(texts[start:start + batch_size])).to(device)
            feats = model.encode_text(tokens)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            chunks.append(feats.cpu())
    if not chunks:
        return torch.zeros((0, 0))
    return torch.cat(chunks, dim=0)


def load_or_encode(
    model: torch.nn.Module,
    tokenize: Callable[[Sequence[str]], torch.Tensor],
    lexicon_file: str | Path,
    texts: Sequence[str],
    model_name: str,
    cache_dir: str | Path,
    device: torch.device,
) -> torch.Tensor:
    """Load cached lexicon embeddings if the sidecar matches (sha256, model, rows), else encode."""
    if not texts:
        return torch.zeros((0, 0))
    sha = file_sha256(lexicon_file)
    path = cache_path(cache_dir, lexicon_file, sha, model_name)
    meta_path = path.with_suffix(".json")
    expected = {"lexicon_sha256": sha, "model": model_name, "rows": len(texts)}
    if path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if {k: meta.get(k) for k in expected} != expected:
            raise ValueError(f"cache {path} does not match lexicon {lexicon_file}: {meta} vs {expected}")
        feats = torch.load(path, map_location="cpu")
        if feats.shape[0] != len(texts):
            raise ValueError(f"cache {path} has {feats.shape[0]} rows, lexicon has {len(texts)}")
        print(f"Loaded cached embeddings {path}")
        return feats
    feats = encode_texts(model, tokenize, texts, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(feats, path)
    meta_path.write_text(json.dumps({**expected, "lexicon_file": str(lexicon_file)}, indent=2), encoding="utf-8")
    print(f"Saved embeddings {path}")
    return feats


# --------------------------------------------------------------------------- retrieval
def retrieve(
    image_feats: torch.Tensor,
    text_feats: torch.Tensor,
    texts: Sequence[str],
    logit_scale: torch.Tensor | float,
    k: int,
    threshold: float,
) -> list[list[str]]:
    """TextGCD forward_tags/forward_attributes: top-k by logit_scale*cos, keep score >= threshold."""
    n = image_feats.shape[0]
    if len(texts) == 0 or k <= 0:
        return [[] for _ in range(n)]
    scores = logit_scale * image_feats @ text_feats.T
    top_scores, top_idx = scores.float().cpu().topk(k=min(k, len(texts)), dim=-1)
    out = []
    for row_scores, row_idx in zip(top_scores, top_idx):
        out.append([texts[int(i)] for s, i in zip(row_scores, row_idx) if s >= threshold])
    return out


class SplitImageDataset_RTG(Dataset):
    """Images listed in splits/{ds}.csv. Replaces CUB/Cars RTG loaders and adds Aircraft (N2)."""

    def __init__(self, split_csv: str | Path, data_root: str | Path, transform=None, limit: int = 0):
        df = pd.read_csv(split_csv, dtype={"image_id": str, "path": str})
        if df["image_id"].duplicated().any():
            raise ValueError(f"{split_csv}: duplicated image_id")
        bad = df[[Path(p).name != i for p, i in zip(df["path"], df["image_id"])]]
        if len(bad):
            raise ValueError(f"{split_csv}: image_id must equal basename(path), e.g. {bad.iloc[0].tolist()}")
        self.df = df.iloc[:limit] if limit else df
        self.root = Path(data_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        with Image.open(self.root / row["path"]) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "image_id": row["image_id"]}


# Name kept from the N2 task description; Aircraft has no TextGCD loader.
AircraftDataset_RTG = SplitImageDataset_RTG


def run_rtg(
    model: torch.nn.Module,
    loader: DataLoader,
    tags: Sequence[str],
    tag_feats: torch.Tensor,
    attributes: Sequence[str],
    attr_feats: torch.Tensor,
    device: torch.device,
    num_tags: int = 3,
    num_attributes: int = 2,
    threshold: float = 0.2,
) -> list[dict]:
    from tqdm import tqdm

    dtype = next(model.parameters()).dtype
    tag_feats = tag_feats.to(device, dtype) if len(tags) else tag_feats
    attr_feats = attr_feats.to(device, dtype) if len(attributes) else attr_feats
    results = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="RTG"):
            img = model.encode_image(batch["image"].to(device, dtype))
            img = img / img.norm(dim=-1, keepdim=True)
            batch_tags = retrieve(img, tag_feats, tags, model.logit_scale, num_tags, threshold)
            batch_attrs = retrieve(img, attr_feats, attributes, model.logit_scale, num_attributes, threshold)
            for image_id, t, a in zip(batch["image_id"], batch_tags, batch_attrs):
                results.append({"image_id": str(image_id), "tags": t, "attributes": a})
    return results


# --------------------------------------------------------------------------- CLI
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--version", required=True, choices=VERSIONS)
    p.add_argument("--data-root", required=True, help="dataset root; split paths are relative to it")
    p.add_argument("--split-csv", default=None, help="default: splits/{dataset}.csv")
    p.add_argument("--tags", default=None, help="override tags lexicon path")
    p.add_argument("--attributes", default=None, help="override attributes lexicon path")
    p.add_argument("--cache-dir", default="rtg/cache", help="lexicon embedding cache (keyed by sha256)")
    p.add_argument("--out-dir", default="rtg")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--fp16", action="store_true", help="run CLIP in half precision (CUDA only)")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--num-tags", type=int, default=3)
    p.add_argument("--num-attributes", type=int, default=2)
    p.add_argument("--threshold", type=float, default=0.2, help="on logit_scale * cosine, as TextGCD")
    p.add_argument("--limit", type=int, default=0, help="only the first N split rows (smoke test)")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    import open_clip

    args = parse_args(argv)
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.fp16 and device.type != "cuda":
        raise SystemExit("--fp16 needs CUDA")

    default_tags, default_attrs = lexicon_paths(args.version, args.dataset)
    tags_file = Path(args.tags or default_tags)
    attrs_file = Path(args.attributes or default_attrs)
    tags, attributes = load_lexicon(tags_file), load_lexicon(attrs_file)
    if not attributes:
        raise SystemExit(f"{attrs_file} is empty")
    if args.version != "D" and not tags:
        raise SystemExit(f"{tags_file} is empty but only version D may have no tags")
    print(f"{args.dataset}/{args.version}: {len(tags)} tags, {len(attributes)} attributes")

    t0 = time.time()
    model, _, preprocess = open_clip.create_model_and_transforms(args.model)
    model = model.to(device).eval()
    if args.fp16:
        model = model.half()
    t_model = time.time()

    tag_feats = load_or_encode(model, open_clip.tokenize, tags_file, tags, args.model, args.cache_dir, device)
    attr_feats = load_or_encode(model, open_clip.tokenize, attrs_file, attributes, args.model, args.cache_dir, device)
    t_lex = time.time()

    split_csv = args.split_csv or REPO_ROOT / "splits" / f"{args.dataset}.csv"
    dataset = SplitImageDataset_RTG(split_csv, args.data_root, preprocess, args.limit)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    results = run_rtg(model, loader, tags, tag_feats, attributes, attr_feats, device,
                      args.num_tags, args.num_attributes, args.threshold)
    t_end = time.time()

    expected_ids = dataset.df["image_id"].tolist()
    if [r["image_id"] for r in results] != expected_ids:
        raise RuntimeError("RTG output order does not match split order")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_limit{args.limit}" if args.limit else ""
    out_path = out_dir / f"{args.dataset}_{args.version}{suffix}.pkl"
    with out_path.open("wb") as handle:
        pickle.dump(results, handle)
    meta = {
        "dataset": args.dataset, "version": args.version, "images": len(results),
        "tags_file": str(tags_file), "tags_sha256": file_sha256(tags_file), "tags_rows": len(tags),
        "attributes_file": str(attrs_file), "attributes_sha256": file_sha256(attrs_file),
        "attributes_rows": len(attributes), "model": args.model, "fp16": args.fp16,
        "num_tags": args.num_tags, "num_attributes": args.num_attributes, "threshold": args.threshold,
        "logit_scale_raw": round(model.logit_scale.detach().float().item(), 4),
        "threshold_as_cosine": round(args.threshold / model.logit_scale.detach().float().item(), 4),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "seconds": {"load_model": round(t_model - t0, 1), "encode_lexicon": round(t_lex - t_model, 1),
                    "retrieve": round(t_end - t_lex, 1), "total": round(t_end - t0, 1)},
        "images_without_tags": sum(not r["tags"] for r in results),
        "images_without_attributes": sum(not r["attributes"] for r in results),
    }
    out_path.with_suffix(".json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
