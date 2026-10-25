"""
Command-line entry point. Run from the repository root:

    python -m src.cli prepare
    python -m src.cli train-unet --fold 0
    python -m src.cli train-yolo --fold 0
    python -m src.cli validate   --fold 0
    python -m src.cli predict    --folds 0 --out submission.csv

``--fold k`` trains without fold k (k = -1 trains on all data). ``--set key=value`` overrides
any config entry, e.g. ``--set unet.epochs=5``.
"""
import argparse

from src.config import load_config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.cli", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["prepare", "train-unet", "train-yolo", "validate", "predict"])
    parser.add_argument("--config", default=None, help="YAML config (default: configs/default.yaml)")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override")
    parser.add_argument("--fold", type=int, default=0, help="held-out fold for training/validation (-1 = none)")
    parser.add_argument("--folds", type=int, nargs="+", default=[0], help="model folds to ensemble in predict")
    parser.add_argument("--out", default="submission.csv", help="submission path for predict")
    args = parser.parse_args(argv)
    cfg = load_config(args.config, args.set)

    if args.command == "prepare":
        from src.pipeline import prepare
        prepare(cfg)
    elif args.command == "train-unet":
        from src.train.unet import train_unet
        train_unet(cfg, args.fold)
    elif args.command == "train-yolo":
        from src.train.yolo import train_yolo
        train_yolo(cfg, args.fold)
    elif args.command == "validate":
        from src.pipeline import validate
        validate(cfg, args.fold)
    else:
        from src.pipeline import predict
        predict(cfg, args.folds, args.out)


if __name__ == "__main__":
    main()
