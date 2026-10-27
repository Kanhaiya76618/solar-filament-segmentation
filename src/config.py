"""Configuration loading: a YAML file plus ``key.sub=value`` overrides."""
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"


def _merge(base: dict, update: dict) -> dict:
    out = dict(base)
    for key, value in update.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def load_config(path: str | Path | None = None, overrides: list[str] | tuple = ()) -> dict:
    """Loads a YAML config. A file with ``inherit: other.yaml`` only lists what differs from it."""
    path = Path(path or DEFAULT_CONFIG)
    cfg = yaml.safe_load(path.read_text())
    if "inherit" in cfg:
        cfg = _merge(load_config(path.parent / cfg.pop("inherit")), cfg)
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
