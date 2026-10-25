"""U-Net (segmentation_models_pytorch) construction and compact fp16 checkpoints."""
from pathlib import Path

import segmentation_models_pytorch as smp
import torch


def build_unet(encoder: str = "resnet34", encoder_weights: str | None = "imagenet",
               in_channels: int = 2) -> torch.nn.Module:
    return smp.Unet(encoder_name=encoder, encoder_weights=encoder_weights,
                    in_channels=in_channels, classes=1)


def save_unet(model: torch.nn.Module, path: Path, encoder: str, in_channels: int = 2) -> None:
    """Saves an fp16 state dict (~49 MB for ResNet34, under GitHub's 100 MB file limit)."""
    model = getattr(model, "module", model)  # unwrap DataParallel
    state = {k: v.half() if v.is_floating_point() else v for k, v in model.state_dict().items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"encoder": encoder, "in_channels": in_channels, "state_dict": state}, path)


def load_unet(path: Path, device: str) -> torch.nn.Module:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = build_unet(ckpt["encoder"], None, ckpt["in_channels"])
    model.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in ckpt["state_dict"].items()})
    return model.to(device).eval()
