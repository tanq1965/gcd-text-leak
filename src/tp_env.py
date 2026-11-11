"""Set up imports of the reused third-party code (SimGCD + GCD) without editing it.

TextGCD is never added to sys.path: its `data` and `config` modules clash with SimGCD's.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_VARS = (
    "cifar_10_root", "cifar_100_root", "cub_root", "aircraft_root", "car_root",
    "herbarium_dataroot", "imagenet_root", "osr_split_dir",
)


def add_third_party_paths(repo_root: Path = REPO_ROOT) -> None:
    """Put third_party/simgcd and third_party/gcd on sys.path; raise if a textgcd path is already there."""
    for p in sys.path:
        if "textgcd" in Path(p).parts:
            raise RuntimeError(f"third_party/textgcd must never be on sys.path (found {p!r})")
    for name in ("simgcd", "gcd"):
        path = str(repo_root / "third_party" / name)
        if path not in sys.path:
            sys.path.insert(0, path)


def inject_config(roots: dict[str, str]) -> None:
    """Replace the module `config` (imported at the top of every SimGCD loader) with our own values.

    Input: dict with every name in CONFIG_VARS. Must be called before `import data.*`;
    cached `data.*` modules are dropped so they re-import with the new roots.
    """
    missing = set(CONFIG_VARS) - set(roots)
    if missing:
        raise ValueError(f"missing config variables: {sorted(missing)}")
    module = types.ModuleType("config")
    for key, value in roots.items():
        setattr(module, key, value)
    for name in [m for m in sys.modules if m == "data" or m.startswith("data.")]:
        del sys.modules[name]
    sys.modules["config"] = module
