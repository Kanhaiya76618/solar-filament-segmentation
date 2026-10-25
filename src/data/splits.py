"""
Leakage-safe cross-validation folds.

All annotator records of one physical image stay together, and observations closer in time than
``gap_days`` (any site) are chained into one group, because the same filaments are visible in
them. Groups are then assigned greedily to the fold with the fewest annotator records, which is
the unit the PQ metric counts.
"""
from datetime import datetime

import numpy as np
import pandas as pd


def observation_time(image_id: str) -> datetime:
    """Capture time from an id such as ``20150125172714Mh`` (YYYYMMDDHHMMSS + site)."""
    return datetime.strptime(image_id[:14], "%Y%m%d%H%M%S")


def make_folds(gt: pd.DataFrame, n_folds: int = 5, gap_days: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """One row per physical image: image_id, n_records, group, fold."""
    images = (gt.groupby("image_id")["annotator_image"].nunique().rename("n_records").reset_index())
    images["time"] = images["image_id"].map(observation_time)
    images = images.sort_values("time").reset_index(drop=True)
    gaps = images["time"].diff().dt.total_seconds().fillna(np.inf) / 86400.0
    images["group"] = np.cumsum(gaps.to_numpy() > gap_days) - 1

    weights = images.groupby("group")["n_records"].sum()
    rng = np.random.default_rng(seed)
    order = rng.permutation(weights.index.to_numpy())
    order = sorted(order, key=lambda g: -weights[g])  # stable: random among equal sizes
    load = np.zeros(n_folds)
    fold_of = {}
    for g in order:
        k = int(np.argmin(load))
        fold_of[g] = k
        load[k] += weights[g]
    images["fold"] = images["group"].map(fold_of)
    return images[["image_id", "n_records", "group", "fold"]]
