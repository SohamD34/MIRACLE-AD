"""Command-line interface for supervised MIRACLE-AD experiments."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import logging
import math
import os
from pathlib import Path
import random
import sys

import numpy as np
import torch
import yaml

from ..data import make_dataloaders
from ..models import available_backbones, available_networks
from ..split_registry import Language, available_datasets, summarize_records
from ..training import TrainHandler


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a raw-speech MIL model for Alzheimer's detection"
    )
    parser.add_argument("--config", help="YAML file whose keys match CLI option names")
    parser.add_argument("--datasets", nargs="+", choices=available_datasets())
    parser.add_argument("--data-root", default=os.environ.get("MIRACLE_AD_DATA_ROOT"))
    parser.add_argument("--split-dir", default=None)

    parser.add_argument("--backbone", choices=available_backbones(), default="gamma_gm_cnn")
    parser.add_argument("--network", choices=available_networks(), default="abmil")
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--chunk-duration", type=float, default=10.0)
    parser.add_argument("--overlap-factor", type=float, default=0.5)
    parser.add_argument("--combine-mci-ad", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--language-aware", action=argparse.BooleanOptionalAction, default=False)

    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-2)
    parser.add_argument("--final-learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--continue-training", action="store_true")

    parser.add_argument(
        "--loss", choices=["auto", "ce", "weighted_ce", "focal", "combined"], default="auto"
    )
    parser.add_argument("--class-weights", nargs="+", default=["auto"])
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--optimizer", choices=["adam", "adamw", "sgd"], default="adamw")
    parser.add_argument(
        "--scheduler", choices=["plateau", "onecycle", "cosine", "step"], default="cosine"
    )
    parser.add_argument("--language-loss-weight", type=float, default=0.5)

    parser.add_argument("--ssl-mode", choices=["none", "freeze", "finetune"], default="none")
    parser.add_argument("--pretrained-backbone", default=None)
    parser.add_argument("--ssl-root", default="outputs/ssl")
    parser.add_argument("--wav2vec2-model", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--cache-dir", default=None)

    parser.add_argument("--output-root", default="outputs/supervised")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--save-attention-maps", action="store_true")
    parser.add_argument("--skip-file-check", action="store_true")
    return parser


def _parse_args(argv=None):
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--config")
    known, _ = probe.parse_known_args(argv)
    parser = _build_parser()
    if known.config:
        with Path(known.config).open("r", encoding="utf-8") as stream:
            defaults = yaml.safe_load(stream) or {}
        if not isinstance(defaults, dict):
            parser.error("The configuration file must contain a YAML mapping")
        valid = {action.dest for action in parser._actions}
        unknown = sorted(set(defaults) - valid)
        if unknown:
            parser.error(f"Unknown configuration keys: {', '.join(unknown)}")
        parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    if not args.datasets:
        parser.error("--datasets is required (or set datasets in --config)")
    if not args.data_root:
        parser.error("--data-root is required (or set MIRACLE_AD_DATA_ROOT)")
    if args.language_aware and args.loss not in {"auto", "combined"}:
        parser.error("--language-aware requires --loss combined (or --loss auto)")
    if not args.language_aware and args.loss == "combined":
        parser.error("--loss combined requires --language-aware")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size != 1:
        parser.error("--batch-size must be 1 for variable-length recording bags")
    if args.accumulation_steps < 1:
        parser.error("--accumulation-steps must be at least 1")
    if args.patience < 1:
        parser.error("--patience must be at least 1")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative")
    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")
    if not 0 <= args.final_learning_rate <= args.learning_rate:
        parser.error(
            "--final-learning-rate must be non-negative and no greater than "
            "--learning-rate"
        )
    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative")
    return parser, args


def _parse_class_weights(values):
    if values == ["auto"] or values == "auto":
        return "auto"
    try:
        return [float(value) for value in values]
    except (TypeError, ValueError) as exc:
        raise ValueError("--class-weights must be 'auto' or a list of numbers") from exc


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_output_dir(args) -> Path:
    if args.output_dir:
        return Path(args.output_dir)
    dataset_string = "_".join(args.datasets)
    run_name = args.run_name or (
        f"{'2class' if args.combine_mci_ad else '3class'}"
        f"_{'language-aware' if args.language_aware else 'standard'}"
        f"_{args.ssl_mode}_seed{args.seed}"
    )
    return Path(args.output_root, args.network, args.backbone, dataset_string, run_name)


def _resolve_ssl_checkpoint(args) -> tuple[Path | None, bool]:
    if args.ssl_mode == "none":
        return None, False
    checkpoint = Path(args.pretrained_backbone) if args.pretrained_backbone else Path(
        args.ssl_root, "_".join(args.datasets), args.backbone, "_best.pt"
    )
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"SSL mode '{args.ssl_mode}' requested, but no backbone exists at {checkpoint}"
        )
    return checkpoint, args.ssl_mode == "freeze"


def _write_run_metadata(output_dir: Path, args, loss_name: str, partitions):
    config = vars(args).copy()
    config.update(
        {
            "resolved_loss": loss_name,
            "output_dir": str(output_dir.resolve()),
            "created_at": datetime.now().astimezone().isoformat(),
        }
    )
    output_dir.joinpath("run_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n", encoding="utf-8"
    )
    split_manifest = {
        name: {
            "summary": summarize_records(records),
            "records": [
                {
                    "source_path": record.source_path,
                    "resolved_path": str(record.path),
                    "label": record.label,
                    "language": record.language,
                    "dataset": record.dataset,
                }
                for record in records
            ],
        }
        for name, records in partitions.items()
    }
    output_dir.joinpath("resolved_splits.json").write_text(
        json.dumps(split_manifest, indent=2) + "\n", encoding="utf-8"
    )


def main(argv=None):
    parser, args = _parse_args(argv)
    _set_seed(args.seed)
    output_dir = _resolve_output_dir(args)
    checkpoint_files = [output_dir / "last_model.pth", output_dir / "best_model.pth"]
    if args.continue_training and not any(path.is_file() for path in checkpoint_files):
        parser.error(f"No supervised checkpoint found to resume in {output_dir}")
    if not args.continue_training and any(
        path.exists()
        for path in [output_dir / "run_config.json", *checkpoint_files]
    ):
        parser.error(
            f"Output directory already contains a run: {output_dir}. "
            "Choose --run-name/--output-dir or use --continue-training."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "train.log", encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )
    device_name = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    device = torch.device(device_name)
    loss_name = args.loss
    if loss_name == "auto":
        loss_name = "combined" if args.language_aware else "weighted_ce"

    if args.continue_training:
        # The full supervised checkpoint already contains the initialized
        # backbone. Preserve freeze mode without requiring the original SSL
        # artifact to remain at the same path.
        checkpoint = None
        freeze_backbone = args.ssl_mode == "freeze"
    else:
        checkpoint, freeze_backbone = _resolve_ssl_checkpoint(args)

    train_loader, validation_loader, test_loader, partitions = make_dataloaders(
        datasets=args.datasets,
        data_root=args.data_root,
        split_dir=args.split_dir,
        seed=args.seed,
        combine_mci_ad=args.combine_mci_ad,
        sample_rate=args.sample_rate,
        chunk_duration=args.chunk_duration,
        overlap_factor=args.overlap_factor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        verify_files=not args.skip_file_check,
    )
    _write_run_metadata(output_dir, args, loss_name, partitions)

    logging.info(
        "Run: datasets=%s backbone=%s network=%s device=%s",
        ",".join(args.datasets),
        args.backbone,
        args.network,
        device,
    )
    logging.info(
        "Recordings: train=%d validation=%d test=%d",
        len(train_loader.dataset),
        len(validation_loader.dataset),
        len(test_loader.dataset),
    )

    scheduler_kwargs = {}
    if args.scheduler == "onecycle":
        scheduler_kwargs = {
            "max_lr": args.learning_rate * 10,
            "steps_per_epoch": math.ceil(
                len(train_loader) / args.accumulation_steps
            ),
            "epochs": args.epochs,
        }
    elif args.scheduler == "cosine":
        scheduler_kwargs = {
            "T_max": args.epochs,
            "eta_min": args.final_learning_rate,
        }

    class_weights = _parse_class_weights(args.class_weights)
    num_classes = 2 if args.combine_mci_ad else 3
    if class_weights != "auto" and len(class_weights) != num_classes:
        parser.error(
            f"--class-weights requires {num_classes} values for this class configuration"
        )

    trainer = TrainHandler(
        backbone_name=args.backbone,
        network_name=args.network,
        num_classes=num_classes,
        combine_mci_ad=args.combine_mci_ad,
        lang_aware=args.language_aware,
        train_loader=train_loader,
        val_loader=validation_loader,
        test_loader=test_loader,
        device=device,
        num_language_classes=len(Language),
        sample_rate=args.sample_rate,
        wav2vec2_model=args.wav2vec2_model,
        cache_dir=args.cache_dir,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        scheduler_kwargs=scheduler_kwargs,
        criterion_name=loss_name,
        focal_alpha=class_weights,
        focal_gamma=args.focal_gamma,
        lambda_lang=args.language_loss_weight,
        checkpoint_dir=str(output_dir),
        accumulation_steps=args.accumulation_steps,
        continue_training=args.continue_training,
        save_attention_maps=args.save_attention_maps,
        pretrained_backbone_path=str(checkpoint) if checkpoint else None,
        freeze_backbone=freeze_backbone,
    )
    trainer.train(num_epochs=args.epochs, patience=args.patience)
    logging.info("Run complete: %s", output_dir.resolve())


if __name__ == "__main__":
    main()
