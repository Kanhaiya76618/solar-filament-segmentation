"""Locates the competition data, locally or under /kaggle/input."""
import os
from dataclasses import dataclass
from pathlib import Path

DATASET_DIR = "MAGFiLO_1.0_Kaggle_2026"
CANDIDATES = [
    f"/kaggle/input/competitions/filament-segmentation-2026/{DATASET_DIR}",
    f"/kaggle/input/filament-segmentation-2026/{DATASET_DIR}",
    f"data/{DATASET_DIR}",
]
IMAGE_SUFFIXES = {".jpeg", ".jpg", ".png"}


@dataclass(frozen=True)
class DataPaths:
    root: Path

    @property
    def train_images(self) -> Path:
        return self.root / "train" / "train_images"

    @property
    def test_images(self) -> Path:
        return self.root / "test" / "test_images"

    @property
    def train_json(self) -> Path:
        default = self.root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
        if default.exists():
            return default
        found = sorted((self.root / "train").glob("*.json"))
        if not found:
            raise FileNotFoundError(f"no annotation JSON in {self.root / 'train'}")
        return found[0]


def find_data(root: str | Path | None = None) -> DataPaths:
    """Returns the first existing dataset folder: ``root``, $SOLAR_DATA_ROOT, then known locations."""
    options = [root, os.environ.get("SOLAR_DATA_ROOT"), *CANDIDATES]
    options += sorted(str(p) for p in Path("/kaggle/input").glob(f"**/{DATASET_DIR}")) \
        if Path("/kaggle/input").is_dir() else []
    for option in options:
        if option and ((Path(option) / "train").is_dir() or (Path(option) / "test").is_dir()):
            return DataPaths(Path(option))
    raise FileNotFoundError(f"{DATASET_DIR} not found; set data.root or $SOLAR_DATA_ROOT. Tried: {options}")


def list_images(folder: Path) -> dict[str, Path]:
    """Maps image id (file stem, e.g. 20150125172714Mh) to its path."""
    return {p.stem: p for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES}
