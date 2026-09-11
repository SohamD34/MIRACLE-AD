"""Raw-audio loading and MIL bag construction."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Sequence

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .split_registry import SampleRecord, build_partitions, missing_files


def chunk_waveform(
    waveform: np.ndarray,
    sample_rate: int = 24000,
    chunk_duration: float = 10.0,
    overlap_factor: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn one waveform into a variable-length bag of fixed-size chunks."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if chunk_duration <= 0:
        raise ValueError("chunk_duration must be positive")
    if not 0.0 <= overlap_factor < 1.0:
        raise ValueError("overlap_factor must be in [0, 1)")

    waveform = np.asarray(waveform, dtype=np.float32).reshape(-1)
    target_length = int(round(sample_rate * chunk_duration))
    if waveform.size == 0:
        waveform = np.zeros(target_length, dtype=np.float32)

    total_duration = waveform.size / sample_rate
    stride = chunk_duration * (1.0 - overlap_factor)
    count = max(1, math.ceil((total_duration - chunk_duration) / stride) + 1)

    chunks: list[np.ndarray] = []
    ranges: list[tuple[float, float]] = []
    for index in range(count):
        start_time = index * stride
        end_time = start_time + chunk_duration
        if end_time > total_duration:
            start_time = max(0.0, total_duration - chunk_duration)
            end_time = total_duration

        start = int(round(start_time * sample_rate))
        stop = min(start + target_length, waveform.size)
        chunk = waveform[start:stop]
        if chunk.size < target_length:
            chunk = np.pad(chunk, (0, target_length - chunk.size))
        chunks.append(chunk.astype(np.float32, copy=False))
        ranges.append((start_time, end_time))

    return np.stack(chunks), np.asarray(ranges, dtype=np.float32)


def load_audio_bag(
    path: str | Path,
    *,
    sample_rate: int = 24000,
    chunk_duration: float = 10.0,
    overlap_factor: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Load, mono-mix, resample, and chunk one recording."""

    waveform, _ = librosa.load(Path(path), sr=sample_rate, mono=True)
    return chunk_waveform(waveform, sample_rate, chunk_duration, overlap_factor)


class AudioBagDataset(Dataset):
    """A recording-level dataset where each item is a bag of audio chunks."""

    def __init__(
        self,
        records: Sequence[SampleRecord],
        *,
        sample_rate: int = 24000,
        chunk_duration: float = 10.0,
        overlap_factor: float = 0.5,
    ):
        self.records = list(records)
        # ``samples`` keeps compatibility with the original training handler
        # while exposing metadata without loading every audio file.
        self.samples = self.records
        self.sample_rate = sample_rate
        self.chunk_duration = chunk_duration
        self.overlap_factor = overlap_factor

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        chunks, ranges = load_audio_bag(
            record.path,
            sample_rate=self.sample_rate,
            chunk_duration=self.chunk_duration,
            overlap_factor=self.overlap_factor,
        )
        return (
            torch.from_numpy(chunks),
            torch.tensor(record.label, dtype=torch.long),
            torch.tensor(record.language, dtype=torch.long),
            torch.from_numpy(ranges),
            record.path.stem,
            record.dataset,
        )


def make_dataloaders(
    datasets: str | Iterable[str],
    data_root: str | Path,
    *,
    split_dir: str | Path | None = None,
    seed: int = 42,
    combine_mci_ad: bool = False,
    sample_rate: int = 24000,
    chunk_duration: float = 10.0,
    overlap_factor: float = 0.5,
    batch_size: int = 1,
    num_workers: int = 0,
    verify_files: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader, dict[str, list[SampleRecord]]]:
    """Create deterministic train/validation/test loaders from fixed splits."""

    if batch_size != 1:
        raise ValueError(
            "Variable-length MIL bags currently require batch_size=1. "
            "Use gradient accumulation for a larger effective batch."
        )
    partitions = build_partitions(
        datasets,
        data_root,
        split_dir=split_dir,
        combine_mci_ad=combine_mci_ad,
    )
    if verify_files:
        absent = missing_files(record for values in partitions.values() for record in values)
        if absent:
            preview = "\n".join(f"  - {path}" for path in absent[:8])
            suffix = f"\n  ... and {len(absent) - 8} more" if len(absent) > 8 else ""
            raise FileNotFoundError(
                f"{len(absent)} split recordings were not found beneath {Path(data_root)}:\n"
                f"{preview}{suffix}\nUse --data-root to select the dataset directory."
            )

    loader_kwargs = {
        "batch_size": 1,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    generator = torch.Generator().manual_seed(seed)
    train_dataset = AudioBagDataset(
        partitions["train"],
        sample_rate=sample_rate,
        chunk_duration=chunk_duration,
        overlap_factor=overlap_factor,
    )
    validation_dataset = AudioBagDataset(
        partitions["val"],
        sample_rate=sample_rate,
        chunk_duration=chunk_duration,
        overlap_factor=overlap_factor,
    )
    test_dataset = AudioBagDataset(
        partitions["test"],
        sample_rate=sample_rate,
        chunk_duration=chunk_duration,
        overlap_factor=overlap_factor,
    )
    return (
        DataLoader(train_dataset, shuffle=True, generator=generator, **loader_kwargs),
        DataLoader(validation_dataset, shuffle=False, **loader_kwargs),
        DataLoader(test_dataset, shuffle=False, **loader_kwargs),
        partitions,
    )
