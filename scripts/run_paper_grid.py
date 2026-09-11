"""Generate or execute the primary MIRACLE-AD architecture grid."""

from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess
import sys

import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/paper_grid.yaml")
    parser.add_argument("--group", default="multilingual")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-root", default="outputs/paper-grid")
    parser.add_argument("--classes", choices=["2", "3", "both"], default="both")
    parser.add_argument(
        "--language-awareness", choices=["standard", "aware", "both"], default="both"
    )
    parser.add_argument("--ssl-mode", choices=["none", "freeze", "finetune"], default="none")
    parser.add_argument("--execute", action="store_true", help="Run commands instead of printing them")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    try:
        datasets = config["dataset_groups"][args.group]
    except KeyError as exc:
        groups = ", ".join(sorted(config.get("dataset_groups", {})))
        parser.error(f"Unknown group '{args.group}'. Available: {groups}")

    class_modes = [False, True] if args.classes == "both" else [args.classes == "2"]
    awareness_modes = (
        [False, True]
        if args.language_awareness == "both"
        else [args.language_awareness == "aware"]
    )
    training = config["training"]
    commands = []
    for backbone in config["grid"]["backbones"]:
        for network in config["grid"]["networks"]:
            for combine_mci_ad in class_modes:
                for language_aware in awareness_modes:
                    command = [
                        sys.executable,
                        "-m",
                        "miracle_ad.cli.train",
                        "--datasets",
                        *datasets,
                        "--data-root",
                        args.data_root,
                        "--output-root",
                        args.output_root,
                        "--backbone",
                        backbone,
                        "--network",
                        network,
                        "--sample-rate",
                        str(config["sample_rate"]),
                        "--chunk-duration",
                        str(config["chunk_duration"]),
                        "--overlap-factor",
                        str(config["overlap_factor"]),
                        "--seed",
                        str(config["seed"]),
                        "--epochs",
                        str(training["epochs"]),
                        "--batch-size",
                        str(training["batch_size"]),
                        "--accumulation-steps",
                        str(training["accumulation_steps"]),
                        "--learning-rate",
                        str(training["learning_rate"]),
                        "--final-learning-rate",
                        str(training["final_learning_rate"]),
                        "--weight-decay",
                        str(training["weight_decay"]),
                        "--patience",
                        str(training["patience"]),
                        "--optimizer",
                        training["optimizer"],
                        "--scheduler",
                        training["scheduler"],
                        "--ssl-mode",
                        args.ssl_mode,
                        "--language-aware" if language_aware else "--no-language-aware",
                    ]
                    if combine_mci_ad:
                        command.append("--combine-mci-ad")
                    commands.append(command)

    for command in commands:
        print(shlex.join(command))
        if args.execute:
            subprocess.run(command, check=True)
    print(f"\n{'Executed' if args.execute else 'Prepared'} {len(commands)} runs.")


if __name__ == "__main__":
    main()
