"""YOLO11-seg training (Ultralytics). The detectors decide which pixels form one filament."""
import shutil
from pathlib import Path

from src.config import fold_tag, repo_path, resolve_device
from src.data.cache import export_yolo


def yolo_device(cfg: dict) -> str | int:
    device = resolve_device(cfg["device"])
    return 0 if device.startswith("cuda") else device


def yolo_weights_path(cfg: dict, name: str, fold: int) -> Path:
    return repo_path(cfg["weights_dir"]) / f"{name}_{fold_tag(fold)}.pt"


def train_yolo(cfg: dict, fold: int) -> list[Path]:
    """Trains every detector in ``yolo.models`` without fold ``fold``; keeps the last epoch."""
    from ultralytics import YOLO

    cache = Path(cfg["data"]["cache_dir"])
    data_yaml = export_yolo(cache, fold)
    saved = []
    for m in cfg["yolo"]["models"]:
        model = YOLO(m["base"])
        model.train(data=str(data_yaml), imgsz=m["imgsz"], epochs=m["epochs"], batch=m["batch"],
                    device=yolo_device(cfg), workers=cfg["yolo"]["workers"], seed=cfg["seed"],
                    project=str(cache / "yolo_runs"), name=f"{m['name']}_{fold_tag(fold)}", exist_ok=True,
                    single_cls=True, **cfg["yolo"]["train_args"])
        last = Path(model.trainer.save_dir) / "weights" / "last.pt"
        dest = yolo_weights_path(cfg, m["name"], fold)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(last, dest)
        print(f"[yolo] saved {dest}")
        saved.append(dest)
    return saved
