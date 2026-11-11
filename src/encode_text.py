"""Build TextGCD prompts and encode them with frozen CLIP ViT-B/16."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np
import open_clip
import pandas as pd
import torch
from tqdm import tqdm

BASE_PROMPT = {
    "cub": "A photo of a bird",
    "scars": "A photo of a car",
    "aircraft": "A photo of an aircraft",
}
FEATURE_DIM = 512


def build_sentence(
    dataset_name: str,
    tags: Optional[Sequence[str]] = None,
    attributes: Optional[Sequence[str]] = None,
) -> str:
    """Construct one TextGCD sentence from tags and attribute sentences."""
    if dataset_name not in BASE_PROMPT:
        raise ValueError(f"unknown dataset {dataset_name!r}")
    clean_tags = [str(tag).strip() for tag in (tags or ()) if str(tag).strip()]
    sentence = BASE_PROMPT[dataset_name]
    if clean_tags:
        sentence += ", " + ", ".join(clean_tags)
    sentence += "."
    clean_attributes = []
    for attribute in attributes or ():
        text = str(attribute).strip()
        if text:
            clean_attributes.append(text if text.endswith(".") else text + ".")
    if clean_attributes:
        sentence += " " + " ".join(clean_attributes)
    return sentence


def encode_sentences(
    sentences: Sequence[str],
    batch_size: int = 128,
    device: Optional[str] = None,
) -> np.ndarray:
    """Return float32, L2-normalized CLIP text features with shape (N, 512)."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if not sentences:
        return np.empty((0, FEATURE_DIM), dtype=np.float32)
    target_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, _, _ = open_clip.create_model_and_transforms("ViT-B-16", pretrained="openai")
    tokenizer = open_clip.get_tokenizer("ViT-B-16")
    model = model.to(target_device)
    model.eval()

    encoded: list[np.ndarray] = []
    with torch.no_grad():
        for start in tqdm(range(0, len(sentences), batch_size), desc="Encoding text"):
            tokens = tokenizer(list(sentences[start : start + batch_size])).to(target_device)
            embeddings = model.encode_text(tokens)
            if embeddings.ndim != 2 or embeddings.shape[1] != FEATURE_DIM:
                raise ValueError(f"CLIP text features must have shape [N, {FEATURE_DIM}], got {tuple(embeddings.shape)}")
            norms = embeddings.norm(dim=-1, keepdim=True)
            if not torch.isfinite(norms).all() or (norms == 0).any():
                raise ValueError("CLIP text features contain a non-finite or zero-norm row")
            encoded.append((embeddings / norms).cpu().numpy().astype(np.float32, copy=False))
    return np.concatenate(encoded, axis=0)


def encode_rtg_pkl(
    dataset_name: str,
    split_csv_path: str | Path,
    rtg_pkl_path: str | Path,
    output_npy_path: str | Path,
    batch_size: int = 128,
) -> Path:
    """Encode RTG descriptions after reordering them to split CSV order."""
    df = pd.read_csv(split_csv_path, dtype={"image_id": str})
    if "image_id" not in df or df["image_id"].isna().any() or df["image_id"].duplicated().any():
        raise ValueError("split CSV must contain unique, non-null image_id values")
    with Path(rtg_pkl_path).open("rb") as handle:
        rtg_data = pickle.load(handle)
    if isinstance(rtg_data, Mapping):
        rtg_dict = {str(key): value for key, value in rtg_data.items()}
    elif isinstance(rtg_data, list):
        rtg_dict = {}
        for item in rtg_data:
            if not isinstance(item, Mapping) or "image_id" not in item:
                raise ValueError("RTG list entries must be mappings containing image_id")
            key = str(item["image_id"])
            if key in rtg_dict:
                raise ValueError(f"duplicate RTG image_id: {key}")
            rtg_dict[key] = item
    else:
        raise ValueError("RTG pickle must contain a mapping or list of mappings")

    sentences = []
    for image_id in df["image_id"].astype(str):
        item = rtg_dict.get(image_id, {})
        if not isinstance(item, Mapping):
            raise ValueError(f"RTG entry for {image_id} must be a mapping")
        sentences.append(build_sentence(dataset_name, item.get("tags", []), item.get("attributes", [])))
    features = encode_sentences(sentences, batch_size=batch_size)
    if features.shape != (len(df), FEATURE_DIM):
        raise ValueError(f"text feature shape mismatch: {features.shape}")
    output_path = Path(output_npy_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, features)
    print(f"Encoded {len(features)} sentences -> {output_path}")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Encode TextGCD descriptions")
    parser.add_argument("--dataset", required=True, choices=BASE_PROMPT)
    parser.add_argument("--split_csv", required=True)
    parser.add_argument("--rtg_pkl", required=True)
    parser.add_argument("--output_npy", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    encode_rtg_pkl(args.dataset, args.split_csv, args.rtg_pkl, args.output_npy, args.batch_size)
