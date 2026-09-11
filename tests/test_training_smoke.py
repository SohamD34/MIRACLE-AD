import json

import numpy as np
import soundfile as sf
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from miracle_ad.data import make_dataloaders
from miracle_ad.training import TrainHandler, get_optimizer, get_scheduler


class TinyMILModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = nn.Identity()
        self.classifier = nn.Linear(1, num_classes)

    def forward(self, inputs, return_attention=False):
        logits = self.classifier(inputs.mean(dim=(1, 2)).unsqueeze(-1))
        if return_attention:
            weights = torch.full(
                (inputs.shape[0], inputs.shape[1]),
                1.0 / inputs.shape[1],
                device=inputs.device,
            )
            return logits, weights
        return logits


class InMemoryBags(Dataset):
    def __init__(self):
        self.records = [
            type("Record", (), {"label": label, "language": 0, "dataset": "synthetic"})()
            for label in range(2)
        ]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return (
            torch.zeros(1, 200),
            torch.tensor(record.label),
            torch.tensor(record.language),
            torch.tensor([[0.0, 1.0]]),
            f"sample_{index}",
            record.dataset,
        )


def test_cosine_scheduler_uses_configured_final_learning_rate():
    model = nn.Linear(2, 2)
    optimizer = get_optimizer(model, "adamw", lr=3e-2)
    assert optimizer.defaults["betas"] == (0.9, 0.999)
    scheduler = get_scheduler(
        optimizer,
        "cosine",
        T_max=100,
        eta_min=3e-5,
    )
    assert scheduler.eta_min == 3e-5


def test_one_epoch_training_uses_distinct_test_holdout(tmp_path, monkeypatch):
    data_root = tmp_path / "data"
    split_dir = tmp_path / "splits"
    output_dir = tmp_path / "run"
    data_root.mkdir()
    split_dir.mkdir()

    train_entries = {}
    validation_entries = {}
    test_entries = {}
    sample_rate = 200
    for label in range(3):
        for index in range(2):
            relative = f"audio/train_{label}_{index}.wav"
            path = data_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(path, np.zeros(sample_rate, dtype=np.float32), sample_rate)
            destination = train_entries if index == 0 else validation_entries
            destination[f"../Datasets/{relative}"] = label
        relative = f"audio/test_{label}.wav"
        sf.write(data_root / relative, np.zeros(sample_rate, dtype=np.float32), sample_rate)
        test_entries[f"../Datasets/{relative}"] = label

    (split_dir / "ncmmsc_split.json").write_text(
        json.dumps(
            {
                "train": train_entries,
                "val": validation_entries,
                "test": test_entries,
            }
        ),
        encoding="utf-8",
    )
    train, validation, test, _ = make_dataloaders(
        "ncmmsc",
        data_root,
        split_dir=split_dir,
        sample_rate=sample_rate,
        chunk_duration=1.0,
        num_workers=0,
    )
    assert len(train.dataset) == len(validation.dataset) == len(test.dataset) == 3
    assert test.dataset is not validation.dataset

    train = DataLoader(InMemoryBags(), batch_size=1, shuffle=False)
    validation = DataLoader(InMemoryBags(), batch_size=1, shuffle=False)
    test = DataLoader(InMemoryBags(), batch_size=1, shuffle=False)

    monkeypatch.setattr(
        "miracle_ad.training.get_model",
        lambda **kwargs: TinyMILModel(kwargs["num_disease_classes"]),
    )

    trainer = TrainHandler(
        backbone_name="m3",
        network_name="abmil",
        num_classes=3,
        combine_mci_ad=False,
        lang_aware=False,
        train_loader=train,
        val_loader=validation,
        test_loader=test,
        device=torch.device("cpu"),
        sample_rate=sample_rate,
        lr=1e-4,
        scheduler_name="cosine",
        scheduler_kwargs={"T_max": 1},
        criterion_name="weighted_ce",
        focal_alpha="auto",
        checkpoint_dir=str(output_dir),
    )
    trainer.train(num_epochs=1, patience=1)
    assert (output_dir / "best_model.pth").is_file()
    assert (output_dir / "best_optimizer.pth").is_file()
    assert (output_dir / "best_scheduler.pth").is_file()
    assert (output_dir / "best_test_cm.png").is_file()
