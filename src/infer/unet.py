"""Full-disk U-Net probability maps: overlapping native tiles, Hann blending, 8-fold flip/rotation TTA."""
import numpy as np
import torch

from src.data.preprocess import Disk, model_input, radius_map


def _starts(lo: int, hi: int, tile: int, stride: int, size: int) -> list[int]:
    """Tile origins covering [lo, hi) inside an axis of length ``size`` (tile <= size)."""
    lo, hi = max(0, lo), min(size, hi)
    if hi - lo <= tile:
        return [int(np.clip(lo - (tile - (hi - lo)) // 2, 0, size - tile))]
    return sorted(set(list(range(lo, hi - tile, stride)) + [hi - tile]))


def _views(tta: bool) -> list[tuple[int, bool]]:
    """(rot90 count, horizontal flip) pairs: the 8 symmetries of the square, or the identity."""
    return [(k, f) for k in range(4) for f in (False, True)] if tta else [(0, False)]


def _apply(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    x = torch.rot90(x, k, dims=(2, 3))
    return torch.flip(x, dims=(3,)) if flip else x


def _invert(x: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    x = torch.flip(x, dims=(3,)) if flip else x
    return torch.rot90(x, -k, dims=(2, 3))


@torch.no_grad()
def predict_prob(models: list[torch.nn.Module], corr: np.ndarray, disk: Disk, device: str, tile: int = 1024,
                 overlap: int = 256, tta: bool = True, batch: int = 4) -> np.ndarray:
    """Mean sigmoid probability over models, tiles and TTA views; 0 outside the solar disk."""
    h, w = corr.shape
    tile = min(tile, h, w)
    window = np.outer(np.hanning(tile), np.hanning(tile)).astype(np.float32) + 1e-3
    margin = 8
    ys = _starts(int(disk.cy - disk.r) - margin, int(disk.cy + disk.r) + margin, tile, tile - overlap, h)
    xs = _starts(int(disk.cx - disk.r) - margin, int(disk.cx + disk.r) + margin, tile, tile - overlap, w)
    acc = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)
    amp = device.startswith("cuda")
    symmetries = _views(tta)

    for y0 in ys:
        for x0 in xs:
            x = torch.from_numpy(model_input(corr[y0:y0 + tile, x0:x0 + tile], disk, y0, x0))[None].to(device)
            views = [_apply(x, k, f) for k, f in symmetries]
            prob = torch.zeros((1, 1, tile, tile), device=device)
            for model in models:
                for i in range(0, len(views), batch):
                    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                        out = torch.sigmoid(model(torch.cat(views[i:i + batch])).float())
                    for j, (k, f) in enumerate(symmetries[i:i + batch]):
                        prob += _invert(out[j:j + 1], k, f)
            prob = prob[0, 0].cpu().numpy() / (len(views) * len(models))
            acc[y0:y0 + tile, x0:x0 + tile] += prob * window
            weight[y0:y0 + tile, x0:x0 + tile] += window

    out = np.divide(acc, weight, out=np.zeros_like(acc), where=weight > 0)
    out[radius_map(disk, (h, w)) >= 1.0] = 0
    return out
