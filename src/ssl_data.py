"""Contrastive-learning data pipeline for backbone pretraining."""

from __future__ import annotations

from collections import defaultdict
import math
from pathlib import Path
import random
from typing import Iterable, Sequence

from audiomentations import AddGaussianNoise, Compose, Gain, PitchShift, TimeStretch
import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .split_registry import SampleRecord, build_partitions, missing_files


SSL_AUGMENTATIONS = Compose(
    [
        AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.010, p=0.8),
        TimeStretch(min_rate=0.95, max_rate=1.05, p=0.5, leave_length_unchanged=True),
        PitchShift(min_semitones=-1.0, max_semitones=1.0, p=0.5),
        Gain(min_gain_db=-3.0, max_gain_db=3.0, p=0.5),
    ]
)


def _duration(path: Path) -> float:
    try:
        return float(librosa.get_duration(path=path))
    except TypeError:  # librosa < 0.10 compatibility
        return float(librosa.get_duration(filename=path))


def cap_records_per_language(
    records: Sequence[SampleRecord], max_hours: float | None, seed: int
) -> list[SampleRecord]:
    """Deterministically cap SSL source duration independently per language."""

    if max_hours is None:
        return list(records)
    if max_hours <= 0:
        raise ValueError("max_hours must be positive or None")

    limit = max_hours * 3600.0
    grouped: dict[int, list[SampleRecord]] = defaultdict(list)
    for record in records:
        grouped[record.language].append(record)

    selected: list[SampleRecord] = []
    for language, group in sorted(grouped.items()):
        shuffled = sorted(group, key=lambda item: item.source_path.replace("\\", "/"))
        random.Random(f"{seed}:ssl:{language}").shuffle(shuffled)
        elapsed = 0.0
        for record in shuffled:
            duration = _duration(record.path)
            if duration <= 0:
                continue
            if elapsed + duration <= limit:
                selected.append(record)
                elapsed += duration
    return selected


class SSLChunkDataset(Dataset):
    """Return two independently augmented views of each fixed-length chunk."""

    def __init__(
        self,
        records: Sequence[SampleRecord],
        *,
        sample_rate: int = 24000,
        chunk_duration: float = 10.0,
        overlap_factor: float = 0.5,
        augment_both_probability: float = 1.0,
        iterations_multiplier: float = 1.0,
    ):
        if not 0.0 <= overlap_factor < 1.0:
            raise ValueError("overlap_factor must be in [0, 1)")
        if not 0.0 <= augment_both_probability <= 1.0:
            raise ValueError("augment_both_probability must be in [0, 1]")
        if iterations_multiplier <= 0:
            raise ValueError("iterations_multiplier must be positive")

        self.records = list(records)
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap_factor = overlap_factor
        self.augment_both_probability = augment_both_probability
        self.iterations_multiplier = iterations_multiplier
        self.chunk_index: list[tuple[SampleRecord, float]] = []

        stride = chunk_duration * (1.0 - overlap_factor)
        for record in self.records:
            duration = _duration(record.path)
            count = max(1, math.ceil((duration - chunk_duration) / stride) + 1)
            for index in range(count):
                start = min(index * stride, max(0.0, duration - chunk_duration))
                self.chunk_index.append((record, start))

    def __len__(self) -> int:
        return int(math.ceil(len(self.chunk_index) * self.iterations_multiplier))

    def _load_chunk(self, record: SampleRecord, start: float) -> np.ndarray:
        waveform, _ = librosa.load(
            record.path,
            sr=self.sample_rate,
            mono=True,
            offset=start,
            duration=self.chunk_duration,
        )
        target = int(round(self.sample_rate * self.chunk_duration))
        waveform = np.asarray(waveform, dtype=np.float32)
        if waveform.size < target:
            waveform = np.pad(waveform, (0, target - waveform.size))
        return waveform[:target]

    def __getitem__(self, index: int):
        if not self.chunk_index:
            raise IndexError("SSL dataset is empty")
        record, start = self.chunk_index[index % len(self.chunk_index)]
        clean = self._load_chunk(record, start)
        view_one = SSL_AUGMENTATIONS(samples=clean.copy(), sample_rate=self.sample_rate)
        if random.random() < self.augment_both_probability:
            view_two = SSL_AUGMENTATIONS(samples=clean.copy(), sample_rate=self.sample_rate)
        else:
            view_two = clean.copy()
        return (
            torch.from_numpy(np.asarray(view_one, dtype=np.float32)).unsqueeze(0),
            torch.from_numpy(np.asarray(view_two, dtype=np.float32)).unsqueeze(0),
        )


def make_ssl_dataloaders(
    datasets: str | Iterable[str],
    data_root: str | Path,
    *,
    split_dir: str | Path | None = None,
    seed: int = 42,
    sample_rate: int = 24000,
    chunk_duration: float = 10.0,
    overlap_factor: float = 0.5,
    batch_size: int = 8,
    num_workers: int = 0,
    augment_both_probability: float = 1.0,
    iterations_multiplier: float = 1.0,
    max_train_hours_per_language: float | None = 4.25,
    max_validation_hours_per_language: float | None = 1.25,
) -> tuple[DataLoader, DataLoader]:
    """Build leakage-safe SSL loaders from committed train/val partitions."""

    if batch_size < 2:
        raise ValueError("NT-Xent training requires batch_size >= 2")
    partitions = build_partitions(
        datasets,
        data_root,
        split_dir=split_dir,
    )
    absent = missing_files(partitions["train"] + partitions["val"])
    if absent:
        preview = "\n".join(f"  - {path}" for path in absent[:8])
        raise FileNotFoundError(
            f"{len(absent)} SSL source recordings are missing beneath {data_root}:\n{preview}"
        )

    train_records = cap_records_per_language(
        partitions["train"], max_train_hours_per_language, seed
    )
    validation_records = cap_records_per_language(
        partitions["val"], max_validation_hours_per_language, seed
    )
    train_dataset = SSLChunkDataset(
        train_records,
        sample_rate=sample_rate,
        chunk_duration=chunk_duration,
        overlap_factor=overlap_factor,
        augment_both_probability=augment_both_probability,
        iterations_multiplier=iterations_multiplier,
    )
    validation_dataset = SSLChunkDataset(
        validation_records,
        sample_rate=sample_rate,
        chunk_duration=chunk_duration,
        overlap_factor=overlap_factor,
        augment_both_probability=augment_both_probability,
    )
    if len(train_dataset) < batch_size or len(validation_dataset) < batch_size:
        raise ValueError(
            "SSL train and validation datasets must each contain at least one full batch. "
            "Select more data or reduce --batch-size."
        )
    generator = torch.Generator().manual_seed(seed)
    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    return (
        DataLoader(
            train_dataset,
            shuffle=True,
            drop_last=True,
            generator=generator,
            **common,
        ),
        DataLoader(validation_dataset, shuffle=False, drop_last=True, **common),
    )
