from argparse import Namespace
import json
from types import SimpleNamespace

import pytest
import torch

from miracle_ad.split_registry import SampleRecord
from miracle_ad.ssl_training import NTXentLoss, _write_run_metadata


def test_nt_xent_is_finite_and_differentiable():
    first = torch.randn(4, 16, requires_grad=True)
    second = torch.randn(4, 16, requires_grad=True)
    loss = NTXentLoss(temperature=0.07)(first, second)
    assert torch.isfinite(loss)
    loss.backward()
    assert first.grad is not None
    assert second.grad is not None


def test_nt_xent_rejects_a_batch_without_negatives():
    with pytest.raises(ValueError, match="at least two"):
        NTXentLoss()(torch.randn(1, 16), torch.randn(1, 16))


def test_ssl_run_metadata_records_exact_membership(tmp_path):
    record = SampleRecord(
        path=tmp_path / "audio.wav",
        label=1,
        language=2,
        dataset="ivanova",
        source_path="../Datasets/Spanish/Ivanova/audio.wav",
    )

    class MetadataDataset:
        records = [record]

        def __len__(self):
            return 3

    loader = SimpleNamespace(dataset=MetadataDataset())
    _write_run_metadata(tmp_path, Namespace(seed=42), loader, loader)

    config = json.loads((tmp_path / "run_config.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (tmp_path / "resolved_splits.json").read_text(encoding="utf-8")
    )
    assert config["train_chunks"] == 3
    assert manifest["train"]["recordings"][0]["source_path"] == record.source_path
    assert manifest["fixed_test_holdout_used"] is False
