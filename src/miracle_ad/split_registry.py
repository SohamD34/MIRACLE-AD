"""Read and validate the fixed dataset splits shipped with MIRACLE-AD."""

from __future__ import annotations

from collections import Counter
import csv
from dataclasses import dataclass
from enum import IntEnum
from functools import lru_cache
from importlib import resources
import json
from pathlib import Path, PurePosixPath
import re
from typing import Iterable, Mapping


LABEL_NAMES = {0: "HC", 1: "AD", 2: "MCI"}
PARTITION_NAMES = ("train", "val", "test")


class Language(IntEnum):
    """Contiguous language IDs used by the four-class auxiliary head."""

    ENGLISH = 0
    CHINESE = 1
    SPANISH = 2
    GREEK = 3


DATASET_LANGUAGES: Mapping[str, Language] = {
    "elu": Language.ENGLISH,
    "pitt": Language.ENGLISH,
    "taukadial_e": Language.ENGLISH,
    "vas": Language.ENGLISH,
    "ncmmsc": Language.CHINESE,
    "mchou": Language.CHINESE,
    "taukadial_c": Language.CHINESE,
    "ivanova": Language.SPANISH,
    "gds3": Language.GREEK,
    "gpilot": Language.GREEK,
    # The unsplit TAUKADIAL artifact is retained for provenance. Prefer the
    # language-specific artifacts for experiments.
    "taukadial": Language.ENGLISH,
}


@dataclass(frozen=True)
class SampleRecord:
    """One recording in a fixed split."""

    path: Path
    label: int
    language: int
    dataset: str
    source_path: str


def split_resources():
    """Return the package resource directory containing the split artifacts."""

    return resources.files("miracle_ad.splits")


def available_datasets(split_dir: str | Path | None = None) -> list[str]:
    """List dataset names that have a bundled or user-supplied split file."""

    root = Path(split_dir) if split_dir else split_resources()
    names = []
    for item in root.iterdir():
        if item.name.endswith("_split.json"):
            names.append(item.name.removesuffix("_split.json"))
    return sorted(names)


def _open_split(dataset: str, split_dir: str | Path | None = None):
    filename = f"{dataset}_split.json"
    item = Path(split_dir, filename) if split_dir else split_resources().joinpath(filename)
    if not item.is_file():
        choices = ", ".join(available_datasets(split_dir))
        raise FileNotFoundError(f"No split artifact for '{dataset}'. Available: {choices}")
    return item.open("r", encoding="utf-8")


def load_split(dataset: str, split_dir: str | Path | None = None) -> dict:
    """Load a split JSON and verify its explicit three-part schema."""

    if dataset not in DATASET_LANGUAGES:
        raise ValueError(f"No language mapping registered for dataset '{dataset}'")
    with _open_split(dataset, split_dir) as stream:
        payload = json.load(stream)
    if set(payload) != set(PARTITION_NAMES):
        raise ValueError(
            f"{dataset}: expected JSON keys {list(PARTITION_NAMES)}, got {sorted(payload)}"
        )
    for partition, entries in payload.items():
        if not isinstance(entries, dict):
            raise TypeError(f"{dataset}:{partition} must map paths to integer labels")
        invalid = {path: label for path, label in entries.items() if label not in LABEL_NAMES}
        if invalid:
            raise ValueError(f"{dataset}:{partition} contains invalid labels: {invalid}")
    for index, first in enumerate(PARTITION_NAMES):
        for second in PARTITION_NAMES[index + 1 :]:
            overlap = set(payload[first]) & set(payload[second])
            if overlap:
                raise ValueError(
                    f"{dataset}: {len(overlap)} recordings occur in both "
                    f"'{first}' and '{second}'"
                )
    return payload


def relative_audio_path(source_path: str) -> PurePosixPath:
    """Convert a legacy split path into a path relative to the dataset root."""

    normalized = source_path.replace("\\", "/")
    marker = "/Datasets/"
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1]
    else:
        for prefix in ("../Datasets/", "Datasets/"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                break
    parts = [part for part in PurePosixPath(normalized).parts if part not in {".", "..", "/"}]
    if not parts:
        raise ValueError(f"Invalid empty audio path in split: {source_path!r}")
    return PurePosixPath(*parts)


def _records(
    dataset: str,
    entries: Mapping[str, int],
    data_root: str | Path,
    combine_mci_ad: bool,
) -> list[SampleRecord]:
    root = Path(data_root).expanduser()
    records = []
    for source_path, raw_label in entries.items():
        relative = relative_audio_path(source_path)
        label = 1 if combine_mci_ad and raw_label == 2 else int(raw_label)
        language = int(_record_language(dataset, source_path))
        records.append(
            SampleRecord(
                path=root.joinpath(*relative.parts),
                label=label,
                language=language,
                dataset=dataset,
                source_path=source_path,
            )
        )
    return records


@lru_cache(maxsize=1)
def _taukadial_languages() -> dict[str, Language]:
    resource = split_resources().joinpath("taukadial_language_map.csv")
    with resource.open("r", encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        mapping = {}
        for row in rows:
            value = row["language"].strip().lower()
            if value not in {"english", "chinese"}:
                raise ValueError(f"Unknown TAUKADIAL language: {value!r}")
            mapping[row["participant_id"].strip().zfill(3)] = (
                Language.ENGLISH if value == "english" else Language.CHINESE
            )
    return mapping


def _record_language(dataset: str, source_path: str) -> Language:
    if dataset != "taukadial":
        return DATASET_LANGUAGES[dataset]
    match = re.search(r"taukdial-(\d+)-", source_path, flags=re.IGNORECASE)
    if not match:
        raise ValueError(f"Cannot identify TAUKADIAL participant in {source_path!r}")
    participant = match.group(1).zfill(3)
    try:
        return _taukadial_languages()[participant]
    except KeyError as exc:
        raise ValueError(f"No language mapping for TAUKADIAL participant {participant}") from exc


def build_partitions(
    datasets: str | Iterable[str],
    data_root: str | Path,
    *,
    split_dir: str | Path | None = None,
    combine_mci_ad: bool = False,
) -> dict[str, list[SampleRecord]]:
    """Build train/validation/test recording lists from committed membership."""

    dataset_names = [datasets] if isinstance(datasets, str) else list(datasets)
    if not dataset_names:
        raise ValueError("At least one dataset is required")

    partitions: dict[str, list[SampleRecord]] = {
        name: [] for name in PARTITION_NAMES
    }
    for dataset in dataset_names:
        payload = load_split(dataset, split_dir)
        for partition in PARTITION_NAMES:
            partitions[partition].extend(
                _records(
                    dataset,
                    payload[partition],
                    data_root,
                    combine_mci_ad,
                )
            )
    return partitions


def summarize_records(records: Iterable[SampleRecord]) -> dict:
    """Return compact counts by dataset, language, and label."""

    records = list(records)
    return {
        "total": len(records),
        "datasets": dict(sorted(Counter(item.dataset for item in records).items())),
        "languages": dict(
            sorted(Counter(Language(item.language).name for item in records).items())
        ),
        "labels": {
            LABEL_NAMES[label]: count
            for label, count in sorted(Counter(item.label for item in records).items())
        },
    }


def missing_files(records: Iterable[SampleRecord]) -> list[Path]:
    """List split entries that do not resolve beneath the selected data root."""

    return [record.path for record in records if not record.path.is_file()]
