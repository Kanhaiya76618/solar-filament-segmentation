"""Configuration loading: a YAML file plus ``key.sub=value`` overrides."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def load_config(path: str | Path | None = None, overrides: list[str] | tuple = ()) -> dict:
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    for item in overrides:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        *parents, leaf = key.strip().split(".")
        node = cfg
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def repo_path(p: str | Path) -> Path:
    """Resolves a config path: absolute paths stay, relative ones are taken from the repo root."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


def resolve_device(name: str = "auto") -> str:
    import torch

    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def fold_tag(fold: int) -> str:
    """Weights trained without fold ``k`` are tagged ``fold{k}``; weights trained on all data ``all``."""
    return "all" if fold < 0 else f"fold{fold}"
