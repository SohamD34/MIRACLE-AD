"""Supervised training, evaluation, checkpointing, and attention exports."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import os
import logging
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from .models import get_model
from .split_registry import Language

def get_optimizer(model, name="adamw", lr=1e-4, weight_decay=1e-5):
    name = name.lower()
    if name == "adam":   return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay,
        )
    if name == "sgd":    return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unsupported optimizer: {name}")

def get_scheduler(optimizer, name="plateau", **kwargs):
    name = name.lower()
    if name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get("T_max", 100),
            eta_min=kwargs.get("eta_min", 0.0),
        )
    if name == "onecycle":
        steps = kwargs.get("steps_per_epoch", 100) * kwargs.get("epochs", 100)
        return optim.lr_scheduler.OneCycleLR(optimizer, max_lr=kwargs.get("max_lr", 1e-3), total_steps=steps, pct_start=0.3)
    if name == "step":
        return optim.lr_scheduler.StepLR(optimizer, step_size=kwargs.get("step_size", 5), gamma=0.1)
    raise ValueError(f"Unsupported scheduler: {name}")


# =============================================================================
# Class Weight Computation
# =============================================================================
def compute_class_weights(data_loader, num_classes, lang_aware=False):
    """
    Compute class weights based on inverse frequency from the training set.

    This helps handle class imbalance by giving higher weight to minority classes.
    Weight for class i = total_samples / (num_classes * count_i)

    Args:
        data_loader: DataLoader to compute class distribution from
        num_classes: Number of classes
        lang_aware: If True, labels are at index 1 (disease labels),
                   otherwise batch structure is (inputs, labels, ...)

    Returns:
        List of class weights (length = num_classes)
    """
    class_counts = np.zeros(num_classes)
    records = getattr(data_loader.dataset, "records", None)
    if records is not None:
        labels = [record.label for record in records]
        for label in labels:
            if 0 <= label < num_classes:
                class_counts[int(label)] += 1
    else:
        for batch in data_loader:
            labels = batch[1]
            if isinstance(labels, torch.Tensor):
                labels = labels.numpy()
            for label in labels:
                if 0 <= label < num_classes:
                    class_counts[int(label)] += 1

    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)

    # Compute inverse frequency weights
    total_samples = class_counts.sum()
    weights = total_samples / (num_classes * class_counts)

    # Normalize so that weights sum to num_classes (optional, keeps scale similar)
    weights = weights / weights.sum() * num_classes

    logging.info(f"Class distribution: {dict(enumerate(class_counts.astype(int)))}")
    logging.info(f"Computed class weights: {dict(enumerate(np.round(weights, 4)))}")

    return weights.tolist()


def compute_language_weights(data_loader, num_classes):
    """Compute inverse-frequency weights for the auxiliary language task."""
    counts = np.zeros(num_classes)
    records = getattr(data_loader.dataset, "records", None)
    if records is None:
        for batch in data_loader:
            for label in batch[2].numpy().tolist():
                counts[int(label)] += 1
    else:
        for record in records:
            counts[int(record.language)] += 1
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (num_classes * counts)
    return (weights / weights.sum() * num_classes).tolist()


# =============================================================================
# Loss Functions
# =============================================================================
class WeightedCrossEntropyLoss(nn.Module):
    """
    Weighted Cross Entropy Loss for handling class imbalance.

    Args:
        weight: Class weights as a list or tensor. Higher weight = higher penalty for misclassification.
                If None, uses standard unweighted cross entropy.
    """
    def __init__(self, weight=None):
        super().__init__()
        if weight is not None:
            self.register_buffer("weight", torch.tensor(weight, dtype=torch.float))
        else:
            self.weight = None

    def forward(self, inputs, targets):
        return F.cross_entropy(inputs, targets, weight=self.weight, reduction='mean')


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance by down-weighting easy examples.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha: Class weights as a list or tensor. If None, no class weighting is applied.
        gamma: Focusing parameter (default=2.0). Higher gamma = more focus on hard examples.
               gamma=0 is equivalent to cross entropy.
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        if alpha is not None:
            self.register_buffer("alpha", torch.tensor(alpha, dtype=torch.float))
        else:
            self.alpha = None

    def forward(self, inputs, targets):
        # Compute standard cross entropy (element-wise)
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        # Get probability of the true class
        pt = torch.exp(-ce_loss)
        # Apply focal modulation: (1 - pt)^gamma
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        # Return mean loss
        return focal_loss.mean()


class CombinedCrossEntropyLoss(nn.Module):
    """
    Combined loss for language-aware training.
    Combines disease classification loss with language classification loss.

    Args:
        lambda_lang: Weight for language loss (default=0.5).
        disease_weight: Class weights for disease classification.
        lang_weight: Class weights for language classification.
    """
    def __init__(self, lambda_lang=0.5, disease_weight=None, lang_weight=None):
        super().__init__()
        self.disease_ce = WeightedCrossEntropyLoss(weight=disease_weight)
        self.lang_ce = WeightedCrossEntropyLoss(weight=lang_weight)
        self.lambda_lang = lambda_lang

    def forward(self, outputs, targets):
        disease_out, lang_out = outputs
        disease_tgt, lang_tgt = targets
        return self.disease_ce(disease_out, disease_tgt) + self.lambda_lang * self.lang_ce(lang_out, lang_tgt)



# =============================================================================
# FINAL TRAINHANDLER — PER-CLASS, FULL RESUME, CLEAN
# =============================================================================
class TrainHandler:
    def __init__(
        self,
        backbone_name: str,
        network_name: str,
        num_classes: int,
        combine_mci_ad: bool,
        lang_aware: bool,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: torch.device,
        test_loader: DataLoader = None,
        num_language_classes: int = len(Language),
        sample_rate: int = 24000,
        wav2vec2_model: str = "facebook/wav2vec2-base-960h",
        cache_dir: str = None,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        optimizer_name: str = "adamw",
        scheduler_name: str = "plateau",
        scheduler_kwargs: dict = None,
        criterion_name: str = "ce",
        focal_alpha = None,
        focal_gamma: float = 2.0,
        lambda_lang: float = 0.5,
        checkpoint_dir: str = "./logs/run",
        accumulation_steps: int = 1,
        continue_training: bool = False,
        save_attention_maps: bool = False,
        pretrained_backbone_path: str = None,
        freeze_backbone: bool = False,
    ):
        if accumulation_steps < 1:
            raise ValueError("accumulation_steps must be at least 1")

        self.save_attention_maps_flag = save_attention_maps
        self.freeze_backbone = freeze_backbone
        self.backbone_name = backbone_name
        self.network_name = network_name
        self.num_classes = num_classes
        self.combine_mci_ad = combine_mci_ad
        self.lang_aware = lang_aware
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.accumulation_steps = accumulation_steps
        self.checkpoint_dir = checkpoint_dir
        self.sample_rate = sample_rate
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=f"{checkpoint_dir}/logs")

        # Class names
        if num_classes == 2:
            self.class_names = ["HC", "AD-MCI"] if combine_mci_ad else ["HC", "AD"]
        elif num_classes == 3:
            if combine_mci_ad: raise ValueError("combine_mci_ad=True not allowed with 3 classes")
            self.class_names = ["HC", "AD", "MCI"]
        else:
            raise ValueError("num_classes must be 2 or 3")

        # Read split metadata directly. This avoids decoding every audio file
        # several times before the first training epoch.
        self.language_names = {e.value: e.name for e in Language}
        records = getattr(train_loader.dataset, "records", [])
        unique_langs = {record.language for record in records}
        if not unique_langs:
            for batch in train_loader:
                langs = batch[2]
                if isinstance(langs, torch.Tensor):
                    unique_langs.update(langs.numpy().tolist())
                else:
                    unique_langs.add(langs)
        self.unique_languages = sorted(unique_langs)
        self.is_multilingual = len(self.unique_languages) > 1
        if self.is_multilingual:
            lang_names = [self.language_names.get(l, f"Lang{l}") for l in self.unique_languages]
            logging.info(f"Multilingual training detected: {len(self.unique_languages)} languages ({', '.join(lang_names)})")

        unique_ds = {record.dataset for record in records}
        if not unique_ds:
            for batch in train_loader:
                if len(batch) >= 6:
                    ds = batch[5]
                    if isinstance(ds, (list, tuple)):
                        unique_ds.update(ds)
                    else:
                        unique_ds.add(ds)
        self.unique_datasets = sorted(list(unique_ds))
        if len(self.unique_datasets) > 0:
             logging.info(f"Datasets detected: {len(self.unique_datasets)} ({', '.join(self.unique_datasets)})")

        # Loss - Compute class weights automatically if requested
        if focal_alpha == "auto" and criterion_name in ["weighted_ce", "focal", "combined"]:
            logging.info("Computing class weights automatically from training set...")
            focal_alpha = compute_class_weights(train_loader, num_classes, lang_aware)
        language_weights = (
            compute_language_weights(train_loader, num_language_classes)
            if lang_aware
            else None
        )

        if criterion_name == "ce":
            self.criterion = nn.CrossEntropyLoss()
        elif criterion_name == "weighted_ce":
            self.criterion = WeightedCrossEntropyLoss(weight=focal_alpha)
        elif criterion_name == "focal":
            self.criterion = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
        elif criterion_name == "combined":
            if not lang_aware: raise ValueError("combined loss requires lang_aware=True")
            self.criterion = CombinedCrossEntropyLoss(
                lambda_lang,
                disease_weight=focal_alpha,
                lang_weight=language_weights,
            )
        else:
            raise ValueError("criterion must be 'ce', 'weighted_ce', 'focal', or 'combined'")
        self.criterion = self.criterion.to(device)

        # Model
        last_model_path = os.path.join(checkpoint_dir, "last_model.pth")
        best_model_path = os.path.join(checkpoint_dir, "best_model.pth")

        load_path = None
        self.resume_prefix = None

        if continue_training:
            if os.path.exists(last_model_path):
                load_path = last_model_path
                self.resume_prefix = "last"
                logging.info(f"Resuming from LAST checkpoint: {load_path}")
            elif os.path.exists(best_model_path):
                load_path = best_model_path
                self.resume_prefix = "best"
                logging.info(f"Resuming from BEST checkpoint (last not found): {load_path}")

        self.model = get_model(
            backbone_type=backbone_name,
            network_type=network_name,
            num_disease_classes=num_classes,
            num_language_classes=num_language_classes,
            lang_aware=lang_aware,
            checkpoint_path=load_path,
            sample_rate=sample_rate,
            wav2vec2_model=wav2vec2_model,
            cache_dir=cache_dir,
        ).to(device)

        # Load pretrained backbone weights if provided
        if pretrained_backbone_path is not None:
            if os.path.exists(pretrained_backbone_path):
                logging.info(f"Loading pretrained backbone from: {pretrained_backbone_path}")
                backbone_state = torch.load(pretrained_backbone_path, map_location=device)
                if isinstance(backbone_state, dict) and "backbone_state_dict" in backbone_state:
                    backbone_state = backbone_state["backbone_state_dict"]
                if isinstance(backbone_state, dict) and any(
                    key.startswith("backbone.") for key in backbone_state
                ):
                    backbone_state = {
                        key.removeprefix("backbone."): value
                        for key, value in backbone_state.items()
                        if key.startswith("backbone.")
                    }
                self.model.backbone.load_state_dict(backbone_state)
                logging.info("Pretrained backbone weights loaded successfully!")
            else:
                logging.warning(f"Pretrained backbone path not found: {pretrained_backbone_path}")

        # Freeze backbone if requested
        if freeze_backbone:
            logging.info("Freezing backbone weights - only classification network will be trained")
            for param in self.model.backbone.parameters():
                param.requires_grad = False
            # Log trainable parameters
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in self.model.parameters())
            logging.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.1f}%)")

        # Optimizer & Scheduler
        self.optimizer = get_optimizer(self.model, optimizer_name, lr, weight_decay)
        scheduler_kwargs = scheduler_kwargs or {}
        if scheduler_name == "onecycle":
            scheduler_kwargs.setdefault("steps_per_epoch", len(train_loader))
            scheduler_kwargs.setdefault("epochs", 100)
        self.scheduler = get_scheduler(self.optimizer, scheduler_name, **scheduler_kwargs)

        # History
        self.history = {
            'train_loss': [], 'val_loss': [],
            'train_acc': [], 'val_acc': [],
            'train_macro_p': [], 'val_macro_p': [],
            'train_macro_r': [], 'val_macro_r': [],
            'train_macro_f1': [], 'val_macro_f1': [],
            'train_per_p': [], 'val_per_p': [],
            'train_per_r': [], 'val_per_r': [],
            'train_per_f1': [], 'val_per_f1': [],
            'train_lang_metrics': [], 'val_lang_metrics': [],
        }
        self.best_val_loss = float('inf')
        self.best_val_f1 = -1.0
        self.best_epoch = 0
        self.patience_counter = 0

        # FULL RESUME
        if continue_training and load_path and os.path.exists(load_path):
            logging.info(f"Loading optimizer/scheduler/history for {self.resume_prefix} checkpoint...")

            # Construct paths based on prefix
            opt_path = os.path.join(checkpoint_dir, f"{self.resume_prefix}_optimizer.pth")
            sched_path = os.path.join(checkpoint_dir, f"{self.resume_prefix}_scheduler.pth")
            hist_path = os.path.join(checkpoint_dir, f"{self.resume_prefix}_history.pth")

            # Backward compatibility for legacy "best" runs where history was just "history.pth"
            if self.resume_prefix == "best" and not os.path.exists(hist_path):
                legacy_hist = os.path.join(checkpoint_dir, "history.pth")
                if os.path.exists(legacy_hist):
                    hist_path = legacy_hist

            if os.path.exists(opt_path):
                self.optimizer.load_state_dict(torch.load(opt_path, map_location=device))
            if os.path.exists(sched_path):
                self.scheduler.load_state_dict(torch.load(sched_path, map_location=device))

            if os.path.exists(hist_path):
                saved = torch.load(hist_path, map_location="cpu")
                for k, v in saved.items():
                    if k in self.history: self.history[k] = v
                self.best_val_loss = saved.get("best_val_loss", float('inf'))
                self.best_val_f1 = saved.get("best_val_f1", -1.0)
                self.best_epoch = saved.get("best_epoch", 0)
                self.patience_counter = saved.get("patience_counter", 0)
                logging.info(f"Resumed from epoch {len(self.history['train_loss'])} | Best val F1: {self.best_val_f1:.4f}")

                # Verify language metrics length matches history (handling legacy checkpoints)
                hist_len = len(self.history['train_loss'])
                if len(self.history['train_lang_metrics']) < hist_len:
                    self.history['train_lang_metrics'].extend([None] * (hist_len - len(self.history['train_lang_metrics'])))
                if len(self.history['val_lang_metrics']) < hist_len:
                    self.history['val_lang_metrics'].extend([None] * (hist_len - len(self.history['val_lang_metrics'])))

                self._replay_tensorboard_history()

    def _log_metrics_tensorboard(self, epoch, split, loss, metrics, lang_metrics=None):
        self.writer.add_scalar(f"Loss/{split}", loss, epoch)
        self.writer.add_scalar(f"Accuracy/{split}", metrics['acc'], epoch)
        self.writer.add_scalar(f"Macro_F1/{split}", metrics['macro_f1'], epoch)
        self.writer.add_scalar(f"Macro_Precision/{split}", metrics['macro_p'], epoch)
        self.writer.add_scalar(f"Macro_Recall/{split}", metrics['macro_r'], epoch)

        for i, cls in enumerate(self.class_names):
            self.writer.add_scalar(f"Class_F1/{cls}_{split}", metrics['per_f1'][i], epoch)
            self.writer.add_scalar(f"Class_Precision/{cls}_{split}", metrics['per_p'][i], epoch)
            self.writer.add_scalar(f"Class_Recall/{cls}_{split}", metrics['per_r'][i], epoch)

        # Log per-language metrics if multilingual
        if lang_metrics:
            for lang_name, lm in lang_metrics.items():
                self.writer.add_scalar(f"Language_Accuracy/{lang_name}_{split}", lm['acc'], epoch)
                self.writer.add_scalar(f"Language_F1/{lang_name}_{split}", lm['macro_f1'], epoch)
                self.writer.add_scalar(f"Language_Precision/{lang_name}_{split}", lm['macro_p'], epoch)
                self.writer.add_scalar(f"Language_Recall/{lang_name}_{split}", lm['macro_r'], epoch)

        # Log per-dataset metrics
        if metrics.get('dataset_metrics'):
            for ds_name, dm in metrics['dataset_metrics'].items():
                self.writer.add_scalar(f"Dataset_Accuracy/{ds_name}_{split}", dm['acc'], epoch)
                self.writer.add_scalar(f"Dataset_F1/{ds_name}_{split}", dm['macro_f1'], epoch)

    def _replay_tensorboard_history(self):
        logging.info("Replaying history to TensorBoard...")
        epochs = len(self.history['train_loss'])
        for i in range(epochs):
            epoch = i + 1
            # Reconstruct metrics for Train
            train_metrics = {
                'acc': self.history['train_acc'][i],
                'macro_p': self.history['train_macro_p'][i],
                'macro_r': self.history['train_macro_r'][i],
                'macro_f1': self.history['train_macro_f1'][i],
                'per_p': self.history['train_per_p'][i],
                'per_r': self.history['train_per_r'][i],
                'per_f1': self.history['train_per_f1'][i],
            }
            self._log_metrics_tensorboard(epoch, "train", self.history['train_loss'][i], train_metrics)

            # Reconstruct metrics for Val
            val_metrics = {
                'acc': self.history['val_acc'][i],
                'macro_p': self.history['val_macro_p'][i],
                'macro_r': self.history['val_macro_r'][i],
                'macro_f1': self.history['val_macro_f1'][i],
                'per_p': self.history['val_per_p'][i],
                'per_r': self.history['val_per_r'][i],
                'per_f1': self.history['val_per_f1'][i],
            }
            self._log_metrics_tensorboard(epoch, "val", self.history['val_loss'][i], val_metrics)

    def _compute_metrics(self, y_true, y_pred):
        labels = list(range(self.num_classes))
        acc = accuracy_score(y_true, y_pred)
        macro_p = precision_score(
            y_true, y_pred, labels=labels, average='macro', zero_division=0
        )
        macro_r = recall_score(
            y_true, y_pred, labels=labels, average='macro', zero_division=0
        )
        macro_f1 = f1_score(
            y_true, y_pred, labels=labels, average='macro', zero_division=0
        )
        per_p = precision_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        ).tolist()
        per_r = recall_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        ).tolist()
        per_f1 = f1_score(
            y_true, y_pred, labels=labels, average=None, zero_division=0
        ).tolist()
        return {'acc': acc, 'macro_p': macro_p, 'macro_r': macro_r, 'macro_f1': macro_f1,
                'per_p': per_p, 'per_r': per_r, 'per_f1': per_f1}

    def _compute_metrics_by_language(self, y_true, y_pred, langs):
        """Compute metrics for each language subset."""
        lang_metrics = {}
        for lang_id in self.unique_languages:
            # Filter samples for this language
            indices = [i for i, l in enumerate(langs) if l == lang_id]
            if len(indices) == 0:
                continue
            y_true_lang = [y_true[i] for i in indices]
            y_pred_lang = [y_pred[i] for i in indices]
            lang_name = self.language_names.get(lang_id, f"Lang{lang_id}")
            lang_metrics[lang_name] = {
                'n_samples': len(indices),
                **self._compute_metrics(y_true_lang, y_pred_lang)
            }
        return lang_metrics

    def _compute_metrics_by_dataset(self, y_true, y_pred, ds_names):
        """Compute metrics for each dataset subset."""
        ds_metrics = {}
        for ds in self.unique_datasets:
            # Filter samples for this dataset
            indices = [i for i, d in enumerate(ds_names) if d == ds]
            if len(indices) == 0:
                continue
            y_true_ds = [y_true[i] for i in indices]
            y_pred_ds = [y_pred[i] for i in indices]
            ds_metrics[ds] = {
                'n_samples': len(indices),
                **self._compute_metrics(y_true_ds, y_pred_ds)
            }
        return ds_metrics

    def _get_disease_logits(self, outputs):
        if self.lang_aware:
            disease_logits = outputs[0]
            if isinstance(disease_logits, tuple):
                disease_logits = disease_logits[0]
            return disease_logits
        if isinstance(outputs, tuple):
            return outputs[0]
        return outputs

    def _compute_batch_loss(self, inputs, disease_tgt, targets_for_loss):
        outputs = self.model(inputs)
        if isinstance(self.criterion, CombinedCrossEntropyLoss):
            loss = self.criterion(outputs, targets_for_loss)
        else:
            loss = self.criterion(self._get_disease_logits(outputs), disease_tgt)
        return loss, outputs

    def _run_epoch(self, loader, training=True):
        if len(loader) == 0:
            raise ValueError("Cannot run an epoch with an empty data loader")
        self.model.train() if training else self.model.eval()
        total_loss = 0.0
        all_true, all_pred, all_langs, all_ds_names = [], [], [], []

        pbar = tqdm(loader, desc="Train" if training else "Val", leave=False)

        for i, batch in enumerate(pbar):
            # Unpack batch: (inputs, label, lang, ranges, audio_name) or (inputs, label, lang, ranges, audio_name, ds_name)
            inputs = batch[0].to(self.device)
            targets = batch[1]
            lang_labels = batch[2]  # Always present now

            ds_names = []
            if len(batch) >= 6:
                d = batch[5]
                if isinstance(d, (list, tuple)):
                    ds_names = list(d)
                else:
                    ds_names = [d] * inputs.size(0) # Should be list from collate usually
            else:
                ds_names = ["unknown"] * inputs.size(0)

            if self.lang_aware:
                disease_tgt = targets.to(self.device)
                lang_tgt = lang_labels.to(self.device)
                targets_for_loss = (disease_tgt, lang_tgt)
            else:
                disease_tgt = targets.to(self.device)
                targets_for_loss = disease_tgt

            with torch.set_grad_enabled(training):
                loss, outputs = self._compute_batch_loss(
                    inputs, disease_tgt, targets_for_loss
                )

            batch_loss = loss.item()
            if training:
                (loss / self.accumulation_steps).backward()
                if (i + 1) % self.accumulation_steps == 0 or (i + 1) == len(loader):
                    self.optimizer.step()
                    if isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                        self.scheduler.step()
                    self.optimizer.zero_grad()

            total_loss += batch_loss

            # ---- live loss in tqdm ----
            avg_loss_so_far = total_loss / (i + 1)
            pbar.set_postfix({"loss": f"{avg_loss_so_far:.4f}"})

            pred = self._get_disease_logits(outputs).argmax(1)
            all_true.extend(disease_tgt.cpu().tolist())
            all_pred.extend(pred.cpu().tolist())
            all_langs.extend(lang_labels.cpu().tolist())
            all_ds_names.extend(ds_names)

        avg_loss = total_loss / len(loader)
        metrics = self._compute_metrics(all_true, all_pred)

        # Compute per-language metrics if multilingual
        lang_metrics = None
        if self.is_multilingual:
            lang_metrics = self._compute_metrics_by_language(all_true, all_pred, all_langs)

        # Compute per-dataset metrics
        ds_metrics = self._compute_metrics_by_dataset(all_true, all_pred, all_ds_names)
        metrics['dataset_metrics'] = ds_metrics # Attach to main metrics dict for easier logging plumbing if desired, or return separately

        return avg_loss, metrics, lang_metrics, (all_true, all_pred, all_langs, all_ds_names)


    def train(self, num_epochs=100, patience=12):
        start_epoch = len(self.history['train_loss']) + 1
        for epoch in range(start_epoch, num_epochs + 1):
            train_loss, train_metrics, train_lang_metrics, _ = self._run_epoch(self.train_loader, training=True)
            val_loss, val_metrics, val_lang_metrics, _ = self._run_epoch(self.val_loader, training=False)

            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            elif not isinstance(self.scheduler, optim.lr_scheduler.OneCycleLR):
                self.scheduler.step()


            # ---- Print overall metrics ----
            print(
                f"Train | Loss: {train_loss:.4f} | "
                f"Acc: {train_metrics['acc']:.4f} | "
                f"P: {train_metrics['macro_p']:.4f} | "
                f"R: {train_metrics['macro_r']:.4f} | "
                f"F1: {train_metrics['macro_f1']:.4f}"
            )
            print(
                f"Val   | Loss: {val_loss:.4f} | "
                f"Acc: {val_metrics['acc']:.4f} | "
                f"P: {val_metrics['macro_p']:.4f} | "
                f"R: {val_metrics['macro_r']:.4f} | "
                f"F1: {val_metrics['macro_f1']:.4f}"
            )

            # ---- Print per-language metrics if multilingual ----
            if self.is_multilingual and train_lang_metrics:
                print("--- Per-Language Metrics ---")
                for lang_name, lm in train_lang_metrics.items():
                    print(f"  Train {lang_name} (n={lm['n_samples']}): Acc={lm['acc']:.4f} F1={lm['macro_f1']:.4f}")
                for lang_name, lm in val_lang_metrics.items():
                    print(f"  Val   {lang_name} (n={lm['n_samples']}): Acc={lm['acc']:.4f} F1={lm['macro_f1']:.4f}")

            # ---- Print per-dataset metrics ----
            if train_metrics.get('dataset_metrics'):
                print("--- Per-Dataset Metrics ---")
                for ds_name, dm in train_metrics['dataset_metrics'].items():
                     print(f"  Train {ds_name} (n={dm['n_samples']}): Acc={dm['acc']:.4f} F1={dm['macro_f1']:.4f}")
                if val_metrics.get('dataset_metrics'):
                    for ds_name, dm in val_metrics['dataset_metrics'].items():
                         print(f"  Val   {ds_name} (n={dm['n_samples']}): Acc={dm['acc']:.4f} F1={dm['macro_f1']:.4f}")

            # Logging
            def log(prefix, loss, m):
                per_str = " | ".join([f"{n}: P{m['per_p'][i]:.3f} R{m['per_r'][i]:.3f} F1{m['per_f1'][i]:.3f}"
                                      for i, n in enumerate(self.class_names)])
                logging.info(f"{prefix} | Loss {loss:.4f} | Acc {m['acc']:.4f} | "
                             f"Macro P{m['macro_p']:.3f} R{m['macro_r']:.3f} F1{m['macro_f1']:.3f} | {per_str}")

            def log_lang(prefix, lang_metrics):
                if lang_metrics:
                    for lang_name, lm in lang_metrics.items():
                        logging.info(f"  {prefix} {lang_name} (n={lm['n_samples']}): "
                                     f"Acc={lm['acc']:.4f} P={lm['macro_p']:.4f} R={lm['macro_r']:.4f} F1={lm['macro_f1']:.4f}")

            logging.info(f"\n=== EPOCH {epoch} ===")
            log("TRAIN", train_loss, train_metrics)
            if self.is_multilingual:
                log_lang("TRAIN", train_lang_metrics)
            log("  VAL", val_loss, val_metrics)
            if self.is_multilingual:
                log_lang("  VAL", val_lang_metrics)

            if train_metrics.get('dataset_metrics'):
                 log_lang("TRAIN DS", train_metrics['dataset_metrics']) # Reuse log_lang format as it's identical structure
            if val_metrics.get('dataset_metrics'):
                 log_lang("  VAL DS", val_metrics['dataset_metrics'])

            # Save history
            for split, loss, m in [
                ("train", train_loss, train_metrics),
                ("val", val_loss, val_metrics),
            ]:
                self.history[f'{split}_loss'].append(loss)
                self.history[f'{split}_acc'].append(m['acc'])
                self.history[f'{split}_macro_p'].append(m['macro_p'])
                self.history[f'{split}_macro_r'].append(m['macro_r'])
                self.history[f'{split}_macro_f1'].append(m['macro_f1'])
                self.history[f'{split}_per_p'].append(m['per_p'])
                self.history[f'{split}_per_r'].append(m['per_r'])
                self.history[f'{split}_per_f1'].append(m['per_f1'])

            self.history['train_lang_metrics'].append(train_lang_metrics)
            self.history['val_lang_metrics'].append(val_lang_metrics)

            self._log_metrics_tensorboard(
                epoch, "train", train_loss, train_metrics, train_lang_metrics,
            )
            self._log_metrics_tensorboard(
                epoch, "val", val_loss, val_metrics, val_lang_metrics,
            )

            # Save best
            if val_metrics['macro_f1'] > self.best_val_f1:
                self.best_val_f1 = val_metrics['macro_f1']
                self.best_epoch = epoch
                self.patience_counter = 0
                self._save_checkpoint("best")
                logging.info(f"NEW BEST MODEL @ epoch {epoch} | Val F1: {self.best_val_f1:.4f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= patience:
                    logging.info(f"EARLY STOPPING @ epoch {epoch}")
                    self._save_checkpoint("last") # Save state before exiting
                    break

            # Save last checkpoint every epoch
            self._save_checkpoint("last")

        self.model.load_state_dict(
            torch.load(
                os.path.join(self.checkpoint_dir, "best_model.pth"),
                map_location=self.device,
            )
        )
        self.plot_detailed_history()
        self.save_all_confusion_matrices()
        self.save_metrics_csv()
        self.writer.close()
        logging.info("TRAINING COMPLETE — BEST MODEL LOADED")

        if self.save_attention_maps_flag:
            self.save_attention_maps()

    def _save_checkpoint(self, prefix="best"):
        torch.save(self.model.state_dict(), os.path.join(self.checkpoint_dir, f"{prefix}_model.pth"))
        torch.save(
            self.optimizer.state_dict(),
            os.path.join(self.checkpoint_dir, f"{prefix}_optimizer.pth"),
        )
        torch.save(
            self.scheduler.state_dict(),
            os.path.join(self.checkpoint_dir, f"{prefix}_scheduler.pth"),
        )

        torch.save({
            **self.history,
            "best_val_loss": self.best_val_loss,
            "best_val_f1": self.best_val_f1,
            "best_epoch": self.best_epoch,
            "patience_counter": self.patience_counter,
        }, os.path.join(self.checkpoint_dir, f"{prefix}_history.pth"))

    def save_attention_maps(self):
        logging.info("Generating attention maps for Validation Set...")
        self.model.eval()

        save_dir = os.path.join(self.checkpoint_dir, "attention_maps")
        os.makedirs(save_dir, exist_ok=True)

        # We need to know which class is positive (Disease)
        # class_names: ["HC", "AD"] or ["HC", "AD", "MCI"]
        # Typically index 1 is disease in binary, or 1 and 2 in ternary.

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.val_loader, desc="Attention Maps")):
                # Unpack batch
                # Expected: (chunks, label, language, ranges, audio_name, dataset)
                # We need to handle variable unpacking dynamically or checking len

                parts = batch
                inputs = parts[0].to(self.device)
                labels = parts[1]

                # Check for ranges and audio_name in batch
                # Format: (inputs, labels, lang, ranges, audio_name, ds_name)
                if len(parts) >= 6:
                    ranges = parts[3]
                    audio_names = parts[4]
                elif len(parts) == 5:
                    # (inputs, labels, lang, ranges, audio_name)
                    ranges = parts[3]
                    audio_names = parts[4]
                elif len(parts) == 4:
                    # Legacy: (inputs, labels, ranges, audio_name) (assuming no lang)
                    ranges = parts[2]
                    audio_names = parts[3]
                else:
                    logging.warning(f"Unexpected batch tuple length: {len(parts)}, skipping attention map generation.")
                    continue

                # Forward pass
                # inputs: [B, N, L] -> batch size 1 usually
                # ranges: [B, N, 2]

                # We need to check if model supports return_attention
                # Attention-capable pooling networks implement this flag.

                try:
                    output = self.model(inputs, return_attention=True)
                except TypeError:
                    logging.warning("Model does not support return_attention, skipping.")
                    return

                # Parse output
                # (logits, attn_weights) or ((dis_logits, lang_logits), attn_weights) or just logits

                attn_weights = None
                logits = None
                if isinstance(output, tuple):
                    # Check if last element looks like attention weights
                    # It could be ((d, l), attn) or (d, attn)
                    potential_attn = output[-1]
                    if potential_attn is not None and isinstance(potential_attn, torch.Tensor):
                         if potential_attn.dim() >= 2: # [B, N] or [B, N, 1]
                             attn_weights = potential_attn
                    # Get logits for prediction
                    logits = self._get_disease_logits(output)
                else:
                    logits = output

                if attn_weights is None:
                    # Model returned None for attention (e.g. LSTM)
                    continue

                # Get predictions
                preds = logits.argmax(dim=1)

                # Process each sample in batch (usually 1)
                bs = inputs.size(0)
                for b in range(bs):
                    # Get weights and ranges for this sample
                    # weights: [N]
                    w = attn_weights[b].cpu().numpy()
                    if w.ndim > 1: w = w.squeeze()

                    # r: [N, 2]
                    r = ranges[b].numpy()

                    # Get audio name for this sample
                    audio_name = audio_names[b] if isinstance(audio_names, (list, tuple)) else audio_names

                    true_label_idx = labels[b].item()
                    pred_label_idx = preds[b].item()
                    true_class = self.class_names[true_label_idx] if true_label_idx < len(self.class_names) else str(true_label_idx)
                    pred_class = self.class_names[pred_label_idx] if pred_label_idx < len(self.class_names) else str(pred_label_idx)

                    # Plotting - 2 subplots: attention on top, waveform below
                    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8),
                                                    gridspec_kw={'height_ratios': [2, 1]})

                    max_time = r[-1, 1]
                    colors = ['#3498db', '#e74c3c'] # Blue and red for alternating chunks

                    # Top subplot: Attention weights
                    for k in range(len(w)):
                        start, end = r[k]
                        score = w[k]

                        # Draw attention bar
                        bar = ax1.bar(x=(start+end)/2, height=score, width=(end-start), align='center',
                                alpha=0.6, color=colors[k%2], edgecolor='black', linewidth=0.5)

                        # Add value label on top of bar
                        ax1.text((start+end)/2, score + 0.01, f'{score:.3f}',
                                ha='center', va='bottom', fontsize=8, rotation=45)

                    ax1.set_xlim(0, max_time)
                    ax1.set_ylim(bottom=0, top=max(w) * 1.2)  # Add headroom for labels
                    ax1.set_ylabel("Attention Weight")
                    ax1.set_title(f"{audio_name} | True: {true_class} | Pred: {pred_class} | Attention Map")
                    ax1.grid(axis='y', alpha=0.3)

                    # Bottom subplot: Audio waveform
                    # Reconstruct waveform from chunks
                    audio_input = inputs[b].cpu().numpy()  # [N, L]
                    n_chunks, chunk_len = audio_input.shape

                    # Create time axis based on ranges
                    # Plot each chunk at its correct position
                    for k in range(n_chunks):
                        start_t, end_t = r[k]
                        chunk_audio = audio_input[k]
                        t_chunk = np.linspace(start_t, end_t, len(chunk_audio))
                        ax2.plot(t_chunk, chunk_audio, color=colors[k%2], alpha=0.7, linewidth=0.5)

                    ax2.set_xlim(0, max_time)
                    ax2.set_xlabel("Time (s)")
                    ax2.set_ylabel("Amplitude")
                    ax2.set_title("Audio Waveform")
                    ax2.grid(axis='both', alpha=0.3)

                    # Save with audio name, true and predicted labels
                    fname = f"{audio_name}_true_{true_class}_pred_{pred_class}.png"
                    plt.tight_layout()
                    plt.savefig(os.path.join(save_dir, fname), dpi=150)
                    plt.close()

                    # Also log top attention chunks with audio name
                    top_k = min(3, len(w))
                    top_indices = np.argsort(w)[-top_k:][::-1]
                    logging.info(f"{audio_name} (True: {true_class}, Pred: {pred_class}): Top {top_k} attention chunks: " +
                                 ", ".join([f"Chunk{idx}({r[idx,0]:.1f}-{r[idx,1]:.1f}s)={w[idx]:.3f}"
                                           for idx in top_indices]))
    def save_all_confusion_matrices(self):
        self.model.eval()
        loaders = [(self.train_loader, "train"), (self.val_loader, "val")]
        if self.test_loader and self.test_loader is not self.val_loader:
             loaders.append((self.test_loader, "test"))

        for loader, name in loaders:
            y_true, y_pred, y_langs, y_ds = [], [], [], []
            with torch.no_grad():
                for batch in loader:
                    x = batch[0].to(self.device)
                    t = batch[1]
                    lang = batch[2]  # Always present now

                    if len(batch) >= 6:
                        ds = batch[5]
                        if not isinstance(ds, (list, tuple)): ds = [ds] * x.size(0)
                    else:
                        ds = ["unknown"] * x.size(0)

                    out = self.model(x)
                    pred = self._get_disease_logits(out).argmax(1)
                    true = t.to(self.device)
                    y_true.extend(true.cpu().tolist())
                    y_pred.extend(pred.cpu().tolist())
                    y_langs.extend(lang.cpu().tolist())
                    y_ds.extend(ds)

            # Overall confusion matrix
            cm = confusion_matrix(y_true, y_pred, labels=range(self.num_classes))
            plt.figure(figsize=(7,6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=self.class_names, yticklabels=self.class_names)
            plt.title(f"Best Model — {name.upper()} Confusion Matrix (Overall)")
            plt.xlabel("Predicted"); plt.ylabel("True")
            plt.savefig(os.path.join(self.checkpoint_dir, f"best_{name}_cm.png"), dpi=200)
            plt.close()

            # Per-language confusion matrices if multilingual
            if self.is_multilingual:
                for lang_id in self.unique_languages:
                    lang_name = self.language_names.get(lang_id, f"Lang{lang_id}")
                    # Filter samples for this language
                    indices = [i for i, l in enumerate(y_langs) if l == lang_id]
                    if len(indices) < 2:
                        continue
                    y_true_lang = [y_true[i] for i in indices]
                    y_pred_lang = [y_pred[i] for i in indices]

                    cm_lang = confusion_matrix(
                        y_true_lang, y_pred_lang, labels=range(self.num_classes)
                    )
                    plt.figure(figsize=(7,6))
                    sns.heatmap(cm_lang, annot=True, fmt='d', cmap='Greens',
                               xticklabels=self.class_names, yticklabels=self.class_names)
                    plt.title(f"Best Model — {name.upper()} Confusion Matrix ({lang_name})")
                    plt.xlabel("Predicted"); plt.ylabel("True")
                    plt.savefig(os.path.join(self.checkpoint_dir, f"best_{name}_cm_{lang_name}.png"), dpi=200)
                    plt.close()

            # Per-dataset confusion matrices
            for ds_id in self.unique_datasets:
                # Filter samples for this dataset
                indices = [i for i, d in enumerate(y_ds) if d == ds_id]
                if len(indices) < 2:
                    continue
                y_true_ds = [y_true[i] for i in indices]
                y_pred_ds = [y_pred[i] for i in indices]

                cm_ds = confusion_matrix(
                    y_true_ds, y_pred_ds, labels=range(self.num_classes)
                )
                plt.figure(figsize=(7,6))
                sns.heatmap(cm_ds, annot=True, fmt='d', cmap='Oranges',
                           xticklabels=self.class_names, yticklabels=self.class_names)
                plt.title(f"Best Model — {name.upper()} CM ({ds_id})")
                plt.xlabel("Predicted"); plt.ylabel("True")
                plt.savefig(os.path.join(self.checkpoint_dir, f"best_{name}_cm_{ds_id}.png"), dpi=200)
                plt.close()

    def plot_detailed_history(self):
        epochs = range(1, len(self.history['train_loss']) + 1)

        # 1. Loss and Accuracy
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        # Loss
        ax[0].plot(epochs, self.history['train_loss'], label='Train')
        ax[0].plot(epochs, self.history['val_loss'], label='Val')
        ax[0].set_title('Loss')
        ax[0].set_xlabel('Epoch')
        ax[0].set_ylabel('Loss')
        ax[0].legend()
        # Accuracy
        ax[1].plot(epochs, self.history['train_acc'], label='Train')
        ax[1].plot(epochs, self.history['val_acc'], label='Val')
        ax[1].set_title('Accuracy')
        ax[1].set_xlabel('Epoch')
        ax[1].set_ylabel('Accuracy')
        ax[1].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.checkpoint_dir, "history_loss_acc.png"), dpi=200)
        plt.close()

        # 2. Global Metrics (F1, P, R)
        fig, ax = plt.subplots(1, 3, figsize=(21, 6))
        metrics = ['macro_f1', 'macro_p', 'macro_r']
        titles = ['Macro F1', 'Macro Precision', 'Macro Recall']
        for i, (m, t) in enumerate(zip(metrics, titles)):
            ax[i].plot(epochs, self.history[f'train_{m}'], label='Train')
            ax[i].plot(epochs, self.history[f'val_{m}'], label='Val')
            ax[i].set_title(t)
            ax[i].set_xlabel('Epoch')
            ax[i].set_ylabel(t)
            ax[i].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.checkpoint_dir, "history_global_metrics.png"), dpi=200)
        plt.close()

        # 3. Classwise Metrics (3 separate plots: F1, P, R)
        # We create one figure per metric type, with subplots for each class
        for metric_name, nice_name in [('per_f1', 'F1'), ('per_p', 'Precision'), ('per_r', 'Recall')]:
            fig, ax = plt.subplots(1, self.num_classes, figsize=(7 * self.num_classes, 6))
            # Ensure ax is indexable if num_classes is 1 (though logic enforces >=2)
            if self.num_classes == 1: ax = [ax]

            for i, cls in enumerate(self.class_names):
                train_data = [ep[i] for ep in self.history[f'train_{metric_name}']]
                val_data = [ep[i] for ep in self.history[f'val_{metric_name}']]

                curr_ax = ax[i]
                curr_ax.plot(epochs, train_data, label='Train')
                curr_ax.plot(epochs, val_data, '--', label='Val')
                curr_ax.set_title(f'{nice_name} — {cls}')
                curr_ax.set_xlabel('Epoch')
                curr_ax.set_ylabel(nice_name)
                curr_ax.legend()

            plt.tight_layout()
            plt.savefig(os.path.join(self.checkpoint_dir, f"history_classwise_{nice_name.lower()}.png"), dpi=200)
            plt.close()

    def save_metrics_csv(self):
        import csv
        csv_path = os.path.join(self.checkpoint_dir, "metrics_history.csv")
        epochs = range(1, len(self.history['train_loss']) + 1)
        # Header
        header = ["epoch", "train_loss", "val_loss", "train_acc", "val_acc", "train_macro_p", "val_macro_p", "train_macro_r", "val_macro_r", "train_macro_f1", "val_macro_f1"]
        for cls in self.class_names:
            header += [f"train_f1_{cls}", f"val_f1_{cls}"]

        # Add language columns if multilingual
        if self.is_multilingual:
             for lang_id in self.unique_languages:
                 lang_name = self.language_names.get(lang_id, f"Lang{lang_id}")
                 header += [
                     f"train_{lang_name}_acc", f"train_{lang_name}_p", f"train_{lang_name}_r", f"train_{lang_name}_f1",
                     f"val_{lang_name}_acc", f"val_{lang_name}_p", f"val_{lang_name}_r", f"val_{lang_name}_f1"
                 ]

        with open(csv_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            for i in range(len(epochs)):
                row = [epochs[i], self.history['train_loss'][i], self.history['val_loss'][i], self.history['train_acc'][i], self.history['val_acc'][i], self.history['train_macro_p'][i], self.history['val_macro_p'][i], self.history['train_macro_r'][i], self.history['val_macro_r'][i], self.history['train_macro_f1'][i], self.history['val_macro_f1'][i]]
                for c in range(len(self.class_names)):
                    row += [self.history['train_per_f1'][i][c], self.history['val_per_f1'][i][c]]

                # Add language data
                if self.is_multilingual:
                    train_lm = self.history['train_lang_metrics'][i]
                    val_lm = self.history['val_lang_metrics'][i]
                    for lang_id in self.unique_languages:
                        lang_name = self.language_names.get(lang_id, f"Lang{lang_id}")
                        # Train
                        if train_lm and lang_name in train_lm:
                             m = train_lm[lang_name]
                             row += [m['acc'], m['macro_p'], m['macro_r'], m['macro_f1']]
                        else:
                             row += ["", "", "", ""]
                        # Val
                        if val_lm and lang_name in val_lm:
                             m = val_lm[lang_name]
                             row += [m['acc'], m['macro_p'], m['macro_r'], m['macro_f1']]
                        else:
                             row += ["", "", "", ""]

                writer.writerow(row)
