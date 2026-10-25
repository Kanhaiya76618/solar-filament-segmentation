"""Figures for the report: the organizers' diagnostic plots and prediction overlays."""
from collections import Counter

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.eval.pq import PQResult


def plot_diagnostics(result: PQResult, title: str = "") -> plt.Figure:
    """IoU and Dice histograms over overlapping pairs and the 1:n / n:1 bars, as in the self-evaluation notebook."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    for ax, scores, name in [(axes[0], result.pair_ious, "IoU"), (axes[1], result.pair_dices, "Dice")]:
        counts, edges = np.histogram(scores, bins=np.linspace(0, 1, 51))
        ax.bar((edges[:-1] + edges[1:]) / 2, counts / max(counts.sum(), 1), width=0.02,
               color="navy", alpha=0.7, edgecolor="black")
        if len(scores):
            ax.axvline(scores.mean(), color="crimson", linestyle="--", label=f"Mean = {scores.mean():.3f}")
            ax.legend(fontsize=8)
        ax.set(xlabel=name, ylabel="Probability", title=f"{name} distribution")

    gt, pred = Counter(result.gt_match_counts), Counter(result.pred_match_counts)
    ax = axes[2]
    for degree, count in gt.items():
        ax.bar(degree + 0.5, count, color="crimson" if degree == 0 else "forestgreen", edgecolor="black", alpha=0.7)
    for degree, count in pred.items():
        ax.bar(-(degree + 0.5), count, color="darkorange" if degree == 0 else "steelblue", edgecolor="black", alpha=0.7)
    ticks = sorted([d + 0.5 for d in gt] + [-(d + 0.5) for d in pred])
    ax.set_xticks(ticks, [f"1:{int(t - 0.5)}" if t > 0 else f"{int(-t - 0.5)}:1" for t in ticks], rotation=45)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(xlabel="Matching case", ylabel="Frequency", title="One-to-many (right) / many-to-one (left)")
    fig.suptitle(f"{title}  {result.summary()}", fontsize=9)
    fig.tight_layout()
    return fig


def overlay(image: np.ndarray, masks: list[np.ndarray], gt_masks: list[np.ndarray] | None = None) -> np.ndarray:
    """RGB image with each predicted filament outlined in its own colour (ground truth in white)."""
    rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    rng = np.random.default_rng(0)
    for m in gt_masks or []:
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, contours, -1, (255, 255, 255), 1)
    for m in masks:
        colour = tuple(int(c) for c in rng.integers(80, 256, 3))
        contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, contours, -1, colour, 2)
    return rgb
