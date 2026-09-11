"""
Self-Supervised Pretraining for Audio Backbones.

Trains backbones using contrastive learning (NT-Xent loss) to produce similar
embeddings for audio chunks and their augmented versions.

Usage:
    miracle-ad-ssl --backbone m5 --datasets pitt --epochs 100 --data-root DATA

The pretrained backbone can be loaded in the main training loop:
    model = get_model('m5', 'abmil', ...)
    model.backbone.load_state_dict(torch.load('pretrained_backbone.pt'))
"""

import argparse
from datetime import datetime
import json
import logging
import os
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .models import available_backbones, create_backbone
from .ssl_data import make_ssl_dataloaders


# =============================================================================
# Projection Head for Contrastive Learning
# =============================================================================

class ProjectionHead(nn.Module):
    """
    MLP projection head that maps backbone embeddings to contrastive space.

    Following SimCLR design: Linear -> ReLU -> Linear
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


# =============================================================================
# NT-Xent Loss (Normalized Temperature-scaled Cross Entropy)
# =============================================================================
class NTXentLoss(nn.Module):
    """
    NT-Xent loss for contrastive learning (SimCLR-style).

    For a batch of N pairs (2N total samples), computes contrastive loss
    where positive pairs are (x_i, x_i') and negatives are all other samples.

    Args:
        temperature: Temperature scaling parameter (default: 0.07)
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss.

        Args:
            z_i: Embeddings of original samples [N, D]
            z_j: Embeddings of augmented samples [N, D]

        Returns:
            Scalar loss value
        """
        if z_i.ndim != 2 or z_i.shape != z_j.shape:
            raise ValueError("NT-Xent inputs must have matching [batch, feature] shapes")
        batch_size = z_i.shape[0]
        if batch_size < 2:
            raise ValueError("NT-Xent loss requires at least two positive pairs")
        device = z_i.device

        # Normalize embeddings
        z_i = F.normalize(z_i, dim=1)
        z_j = F.normalize(z_j, dim=1)

        # Concatenate: [z_1, z_2, ..., z_N, z'_1, z'_2, ..., z'_N]
        representations = torch.cat([z_i, z_j], dim=0)  # [2N, D]

        # Compute similarity matrix
        similarity_matrix = torch.mm(representations, representations.t())  # [2N, 2N]

        # Mask out self-similarity (diagonal)
        self_mask = torch.eye(2 * batch_size, device=device).bool()
        similarity_matrix = similarity_matrix.masked_fill(self_mask, -1e9)

        # Scale by temperature
        similarity_matrix = similarity_matrix / self.temperature

        # Compute loss using cross-entropy formulation
        logits = similarity_matrix  # Already masked out self
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=device),
            torch.arange(0, batch_size, device=device)
        ])

        loss = F.cross_entropy(logits, labels)

        return loss



# =============================================================================
# SSL Pretrainer
# =============================================================================

class SSLPretrainer:
    """
    Self-Supervised Learning Pretrainer for audio backbones.

    Uses contrastive learning with NT-Xent loss to pretrain backbones
    on unlabeled audio data.
    """

    def __init__(
        self,
        backbone_type: str,
        sample_rate: int = 24000,
        projection_hidden: int = 256,
        projection_dim: int = 128,
        temperature: float = 0.07,
        learning_rate: float = 1e-3,
        min_learning_rate: float = 1e-7,
        weight_decay: float = 1e-5,
        checkpoint_dir: str = "./pretrained_backbones",
        wav2vec2_model: str = "facebook/wav2vec2-base-960h",
        cache_dir: str = None,
        scheduler_epochs: int = 50,
        device: str | torch.device | None = None,
    ):
        self.backbone_type = backbone_type
        self.sample_rate = sample_rate
        self.checkpoint_dir = checkpoint_dir
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Create backbone
        self.backbone = self._create_backbone(backbone_type, sample_rate, wav2vec2_model, cache_dir)
        self.backbone = self.backbone.to(self.device)

        # Create projection head
        self.projection_head = ProjectionHead(
            input_dim=self.backbone.feature_dim,
            hidden_dim=projection_hidden,
            output_dim=projection_dim
        ).to(self.device)

        # Loss function
        self.criterion = NTXentLoss(temperature=temperature)

        # Optimizer (for both backbone and projection head)
        self.optimizer = optim.AdamW(
            list(self.backbone.parameters()) + list(self.projection_head.parameters()),
            lr=learning_rate,
            betas=(0.9, 0.999),
            weight_decay=weight_decay
        )

        # Cosine annealing LR scheduler: Reach min LR at epoch 30
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=max(1, scheduler_epochs),
            eta_min=min_learning_rate,
        )

        # Create checkpoint directory
        os.makedirs(checkpoint_dir, exist_ok=True)

        # TensorBoard
        self.writer = SummaryWriter(log_dir=os.path.join(checkpoint_dir, "logs"))

        # Logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(checkpoint_dir, "pretrain.log")),
                logging.StreamHandler()
            ],
            force=True,
        )
        self.logger = logging.getLogger(__name__)

    def _create_backbone(self, backbone_type: str, sample_rate: int,
                         wav2vec2_model: str, cache_dir: str):
        """Create backbone instance based on type."""
        return create_backbone(
            backbone_type,
            sample_rate=sample_rate,
            wav2vec2_model=wav2vec2_model,
            cache_dir=cache_dir,
        )

    def _forward_pass(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through backbone and projection head."""
        # x: [B, 1, L]
        features = self.backbone(x)  # [B, feature_dim]
        projections = self.projection_head(features)  # [B, projection_dim]
        return projections

    def train_epoch(self, dataloader) -> float:
        """Train for one epoch."""
        self.backbone.train()
        self.projection_head.train()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc="Training")
        for batch_idx, (original, augmented) in enumerate(pbar):
            # Move to device - shape: [B, 1, L]
            original = original.to(self.device)
            augmented = augmented.to(self.device)

            # Forward pass
            z_original = self._forward_pass(original)
            z_augmented = self._forward_pass(augmented)

            # Compute loss
            loss = self.criterion(z_original, z_augmented)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        return total_loss / max(num_batches, 1)


    def validate_epoch(self, dataloader) -> float:
        """Validate for one epoch."""
        self.backbone.eval()
        self.projection_head.eval()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(dataloader, desc="Validation", leave=False)
        with torch.no_grad():
            for original, augmented in pbar:
                original = original.to(self.device).float()
                augmented = augmented.to(self.device).float()

                z_original = self._forward_pass(original)
                z_augmented = self._forward_pass(augmented)

                loss = self.criterion(z_original, z_augmented)
                total_loss += loss.item()
                num_batches += 1

                pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})

        return total_loss / max(num_batches, 1)

    def train(
        self,
        train_dataloader,
        val_dataloader,
        epochs: int = 50,
        save_every: int = 10,
        start_epoch: int = 1,
        best_val_loss: float = float('inf'),
    ):
        """
        Main training loop.

        Args:
            dataloader: SSL DataLoader
            epochs: Number of training epochs
            save_every: Save checkpoint every N epochs
        """
        self.logger.info(f"Starting SSL pretraining for {epochs} epochs")
        self.logger.info(f"Backbone: {self.backbone_type}, Feature dim: {self.backbone.feature_dim}")
        self.logger.info(f"Device: {self.device}")
        self.logger.info(f"Train size: {len(train_dataloader.dataset)} chunks")
        self.logger.info(f"Val size:   {len(val_dataloader.dataset)} chunks")

        # best_val_loss passed in argument, helpful if resuming

        for epoch in range(start_epoch, epochs + 1):
            train_loss = self.train_epoch(train_dataloader)
            val_loss = self.validate_epoch(val_dataloader)

            # Step the learning rate scheduler
            self.scheduler.step()
            current_lr = self.scheduler.get_last_lr()[0]

            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/val', val_loss, epoch)
            self.writer.add_scalar('LearningRate', current_lr, epoch)

            self.logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - LR: {current_lr:.2e}")
            print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f} - LR: {current_lr:.2e}")

            # Save best model based on Validation Loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_backbone("best")
                self.logger.info(f"New best model saved (val_loss: {best_val_loss:.4f})")
                print(f"New best model saved (val_loss: {best_val_loss:.4f})")

            # Periodic save
            if epoch % save_every == 0:
                self.save_backbone(f"epoch_{epoch}")

            # Save full checkpoint for resuming (overwrites previous)
            self.save_checkpoint(epoch, best_val_loss, "checkpoint_last.pth")

        # Save final model
        self.save_backbone("final")
        self.writer.close()

        self.logger.info(f"Training complete. Best Val Loss: {best_val_loss:.4f}")

    def save_backbone(self, suffix: str = ""):
        """
        Save only the backbone weights (compatible with AudioClassificationModel).

        The saved state_dict can be loaded directly into model.backbone:
            model.backbone.load_state_dict(torch.load('backbone.pt'))
        """
        filename = ""
        if suffix:
            filename += f"_{suffix}"
        filename += ".pt"

        save_path = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.backbone.state_dict(), save_path)
        self.logger.info(f"Backbone saved to {save_path}")

    def load_backbone(self, checkpoint_path: str):
        """Load backbone weights from checkpoint."""
        self.backbone.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.logger.info(f"Backbone loaded from {checkpoint_path}")

    def save_checkpoint(self, epoch: int, best_val_loss: float, filename: str = "checkpoint_last.pth"):
        """
        Save full training state for resuming.
        """
        save_path = os.path.join(self.checkpoint_dir, filename)
        state = {
            'epoch': epoch,
            'backbone_state_dict': self.backbone.state_dict(),
            'projection_head_state_dict': self.projection_head.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': best_val_loss
        }
        torch.save(state, save_path)
        self.logger.info(f"Full checkpoint saved to {save_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """
        Load full training state.
        Returns: (start_epoch, best_val_loss)
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.backbone.load_state_dict(checkpoint['backbone_state_dict'])
        self.projection_head.load_state_dict(checkpoint['projection_head_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))

        self.logger.info(f"Resumed from epoch {start_epoch-1} (next epoch: {start_epoch})")
        return start_epoch, best_val_loss


def _write_run_metadata(output_dir: str | Path, args, train_loader, validation_loader):
    """Record the exact configuration and recording membership used by SSL."""

    destination = Path(output_dir)
    config = vars(args).copy()
    config.update(
        {
            "output_dir": str(destination.resolve()),
            "created_at": datetime.now().astimezone().isoformat(),
            "train_chunks": len(train_loader.dataset),
            "val_chunks": len(validation_loader.dataset),
        }
    )
    destination.joinpath("run_config.json").write_text(
        json.dumps(config, indent=2, default=str) + "\n",
        encoding="utf-8",
    )

    def serialize(loader):
        return [
            {
                "source_path": record.source_path,
                "resolved_path": str(record.path),
                "label": record.label,
                "language": record.language,
                "dataset": record.dataset,
            }
            for record in loader.dataset.records
        ]

    destination.joinpath("resolved_splits.json").write_text(
        json.dumps(
            {
                "train": {
                    "recordings": serialize(train_loader),
                    "chunks": len(train_loader.dataset),
                },
                "val": {
                    "recordings": serialize(validation_loader),
                    "chunks": len(validation_loader.dataset),
                },
                "fixed_test_holdout_used": False,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


# =============================================================================
# Main Entry Point
# =============================================================================

def main(argv=None):
    parser = argparse.ArgumentParser(description="SimCLR-style SSL pretraining for MIRACLE-AD backbones")
    parser.add_argument("--backbone", default="gamma_gm_cnn", choices=available_backbones())
    parser.add_argument("--projection-hidden", type=int, default=256)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--datasets", "--dataset", dest="datasets", nargs="+", required=True)
    parser.add_argument("--data-root", default=os.environ.get("MIRACLE_AD_DATA_ROOT"))
    parser.add_argument("--split-dir", default=None)
    parser.add_argument("--chunk-duration", type=float, default=10.0)
    parser.add_argument("--chunk-overlap", type=float, default=0.5)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--final-lr", type=float, default=1e-7)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--iterations-multiplier", type=float, default=1.0)
    parser.add_argument("--augment-both-probability", type=float, default=1.0)
    parser.add_argument("--max-train-hours-per-language", type=float, default=4.25)
    parser.add_argument("--max-validation-hours-per-language", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--wav2vec2-model", default="facebook/wav2vec2-base-960h")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    args = parser.parse_args(argv)

    if not args.data_root:
        parser.error("--data-root is required (or set MIRACLE_AD_DATA_ROOT)")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    if args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.batch_size < 2:
        parser.error("--batch-size must be at least 2 for NT-Xent loss")
    if args.save_every < 1:
        parser.error("--save-every must be at least 1")
    if args.lr <= 0:
        parser.error("--lr must be positive")
    if not 0 <= args.final_lr <= args.lr:
        parser.error("--final-lr must be non-negative and no greater than --lr")
    if args.weight_decay < 0:
        parser.error("--weight-decay cannot be negative")
    if args.temperature <= 0:
        parser.error("--temperature must be positive")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset_string = "_".join(args.datasets)
    output_dir = args.output_dir or os.path.join(
        "outputs", "ssl", dataset_string, args.backbone
    )
    checkpoint_path = os.path.join(output_dir, "checkpoint_last.pth")
    if args.continue_training and not os.path.isfile(checkpoint_path):
        parser.error(f"No SSL checkpoint found to resume at {checkpoint_path}")
    if not args.continue_training and any(
        os.path.exists(os.path.join(output_dir, name))
        for name in (
            "checkpoint_last.pth",
            "_best.pt",
            "_final.pt",
            "run_config.json",
        )
    ):
        parser.error(
            f"Output directory already contains an SSL run: {output_dir}. "
            "Choose --output-dir or use --continue-training."
        )
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 80)
    print("MIRACLE-AD self-supervised backbone pretraining")
    print(f"Backbone: {args.backbone}")
    print(f"Datasets: {', '.join(args.datasets)}")
    print(f"Chunks: {args.chunk_duration:g}s, overlap={args.chunk_overlap:g}")
    print(f"Output: {output_dir}")
    print("=" * 80)

    train_loader, val_loader = make_ssl_dataloaders(
        datasets=args.datasets,
        data_root=args.data_root,
        split_dir=args.split_dir,
        seed=args.seed,
        chunk_duration=args.chunk_duration,
        overlap_factor=args.chunk_overlap,
        sample_rate=args.sample_rate,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment_both_probability=args.augment_both_probability,
        iterations_multiplier=args.iterations_multiplier,
        max_train_hours_per_language=(
            args.max_train_hours_per_language or None
        ),
        max_validation_hours_per_language=(
            args.max_validation_hours_per_language or None
        ),
    )
    if not train_loader.dataset:
        parser.error("The selected datasets produced no SSL chunks")
    _write_run_metadata(output_dir, args, train_loader, val_loader)

    pretrainer = SSLPretrainer(
        backbone_type=args.backbone,
        sample_rate=args.sample_rate,
        projection_hidden=args.projection_hidden,
        projection_dim=args.projection_dim,
        temperature=args.temperature,
        learning_rate=args.lr,
        min_learning_rate=args.final_lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=output_dir,
        wav2vec2_model=args.wav2vec2_model,
        cache_dir=args.cache_dir,
        scheduler_epochs=args.epochs,
        device=args.device,
    )

    start_epoch = 1
    best_val_loss = float("inf")
    if args.continue_training:
        start_epoch, best_val_loss = pretrainer.load_checkpoint(checkpoint_path)

    pretrainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=args.epochs,
        save_every=args.save_every,
        start_epoch=start_epoch,
        best_val_loss=best_val_loss,
    )
    print(f"SSL pretraining complete. Backbone weights: {output_dir}")


if __name__ == "__main__":
    main()
