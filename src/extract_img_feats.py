"""Extract frozen CLIP ViT-B/16 image features in split order."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

DATASETS = ("cub", "scars", "aircraft")
FEATURE_DIM = 512


def _set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SplitImageDataset(Dataset[tuple[torch.Tensor, str]]):
    def __init__(self, df: pd.DataFrame, transform: object, data_root: Optional[Path] = None) -> None:
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.data_root = data_root

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, str]:
        row = self.df.iloc[idx]
        image_path = Path(str(row["path"]))
        if self.data_root is not None and not image_path.is_absolute():
            image_path = self.data_root / image_path
        with Image.open(image_path) as image:
            image = image.convert("RGB")
        transformed = self.transform(image)
        return transformed, str(row["image_id"])


def _validate_split(df: pd.DataFrame, split_csv_path: Path) -> None:
    required = {"image_id", "path"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{split_csv_path}: missing columns {missing}")
    if df.empty:
        raise ValueError(f"{split_csv_path}: split is empty")
    if df["image_id"].isna().any() or df["image_id"].duplicated().any():
        raise ValueError(f"{split_csv_path}: image_id must be non-null and unique")


def extract_dataset_features(
    dataset_name: str,
    split_csv_path: str | Path,
    output_dir: str | Path = "feats",
    batch_size: int = 64,
    num_workers: int = 4,
    data_root: str | Path | None = None,
) -> tuple[Path, Path]:
    """Extract and save image features and IDs, both in split CSV row order."""
    _set_seed()
    if dataset_name not in DATASETS:
        raise ValueError(f"unknown dataset {dataset_name!r}; expected one of {DATASETS}")
    if batch_size < 1 or num_workers < 0:
        raise ValueError("batch_size must be positive and num_workers must be non-negative")

    split_path = Path(split_csv_path)
    df = pd.read_csv(split_path, dtype={"image_id": str, "path": str})
    _validate_split(df, split_path)
    root = Path(data_root) if data_root is not None else None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==> Processing {dataset_name} on device: {device}")

    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-16", pretrained="openai"
    )
    model = model.to(device)
    model.eval()
    dataset = SplitImageDataset(df, preprocess, root)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    features: list[np.ndarray] = []
    image_ids: list[str] = []
    with torch.no_grad():
        for images, ids in tqdm(loader, desc=f"Encoding {dataset_name} images"):
            embeddings = model.encode_image(images.to(device))
            if embeddings.ndim != 2 or embeddings.shape[1] != FEATURE_DIM:
                raise ValueError(f"CLIP image features must have shape [N, {FEATURE_DIM}], got {tuple(embeddings.shape)}")
            norms = embeddings.norm(dim=-1, keepdim=True)
            if not torch.isfinite(norms).all() or (norms == 0).any():
                raise ValueError("CLIP image features contain a non-finite or zero-norm row")
            embeddings = embeddings / norms
            features.append(embeddings.cpu().numpy().astype(np.float32, copy=False))
            image_ids.extend(str(image_id) for image_id in ids)

    all_features = np.concatenate(features, axis=0)
    split_ids = df["image_id"].astype(str).to_numpy()
    all_ids = np.asarray(image_ids, dtype=str)
    if all_features.shape != (len(df), FEATURE_DIM):
        raise ValueError(f"feature shape mismatch: {all_features.shape}, expected {(len(df), FEATURE_DIM)}")
    if not np.array_equal(all_ids, split_ids):
        raise ValueError("image_id order mismatch between extracted features and split")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    feature_path = output_path / f"{dataset_name}_img.npy"
    ids_path = output_path / f"{dataset_name}_img_ids.npy"
    np.save(feature_path, all_features)
    np.save(ids_path, all_ids)
    print(f"Saved {feature_path} {all_features.shape} and {ids_path}")
    return feature_path, ids_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract CLIP ViT-B/16 image features")
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--split_csv", default=None, help="default: splits/{dataset}.csv")
    parser.add_argument("--data-root", type=Path, default=None, help="root containing paths in the split CSV")
    parser.add_argument("--output_dir", type=Path, default=Path("feats"))
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    split_csv = args.split_csv or str(Path("splits") / f"{args.dataset}.csv")
    extract_dataset_features(
        args.dataset, split_csv, args.output_dir, args.batch_size, args.num_workers, args.data_root
    )
