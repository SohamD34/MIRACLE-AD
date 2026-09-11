"""Inspect the fixed split artifacts without loading audio."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os

from ..split_registry import (
    available_datasets,
    build_partitions,
    load_split,
    missing_files,
    summarize_records,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Inspect MIRACLE-AD dataset splits")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--data-root", default=os.environ.get("MIRACLE_AD_DATA_ROOT"))
    parser.add_argument("--check-files", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)

    datasets = args.datasets or available_datasets(args.split_dir)
    source = {}
    for dataset in datasets:
        payload = load_split(dataset, args.split_dir)
        source[dataset] = {
            partition: {
                "total": len(entries),
                "labels": dict(sorted(Counter(entries.values()).items())),
            }
            for partition, entries in payload.items()
        }

    report = {"source_artifacts": source}
    status = 0
    if args.data_root:
        partitions = build_partitions(
            datasets,
            args.data_root,
            split_dir=args.split_dir,
        )
        report["resolved_partitions"] = {
            name: summarize_records(records) for name, records in partitions.items()
        }
        if args.check_files:
            absent = missing_files(
                record for records in partitions.values() for record in records
            )
            report["missing_files"] = [str(path) for path in absent]
            status = 1 if absent else 0
    elif args.check_files:
        parser.error("--check-files requires --data-root or MIRACLE_AD_DATA_ROOT")

    if args.as_json:
        print(json.dumps(report, indent=2))
    else:
        for dataset, summary in source.items():
            train = summary["train"]
            validation = summary["val"]
            test = summary["test"]
            print(
                f"{dataset:14} train={train['total']:4} "
                f"val={validation['total']:3} test={test['total']:4}"
            )
        if "resolved_partitions" in report:
            print("\nResolved partitions:")
            for name, summary in report["resolved_partitions"].items():
                print(f"  {name:10} {summary['total']:4} {summary['labels']}")
        if args.check_files:
            print(f"\nMissing files: {len(report['missing_files'])}")
    raise SystemExit(status)


if __name__ == "__main__":
    main()
