# MIRACLE-AD

**Multilingual Interpretable Raw-Speech Acoustic Learning for Alzheimer's Detection**

MIRACLE-AD is a raw-waveform multiple-instance learning (MIL) pipeline for
recording-level Alzheimer's disease detection. A recording is split into
overlapping chunks, an acoustic backbone embeds every chunk, and a temporal or
attention-based pooling network produces the recording prediction. The optional
language head supports multilingual training, while attention pooling exposes
the chunks that most influenced a prediction.

This repository is organized as an installable Python package. Training code is
under `src/miracle_ad/`; generated checkpoints, figures, logs, and licensed audio
are intentionally kept out of Git.

## Paper configuration

The primary paper grid uses:

- 24 kHz mono audio
- 10-second chunks with 5-second stride (50% overlap)
- `gamma_gm_cnn` and `gamma_erb_cnn` backbones
- `unilstm`, `bilstm`, `abmil`, and `gated_abmil` pooling
- inverse-frequency weighted cross-entropy
- optional four-language auxiliary classification with weight 0.5

The corresponding defaults and language groups are recorded in
[`configs/paper_grid.yaml`](configs/paper_grid.yaml).

### Hyperparameter details

| Parameter | SSL pretraining | Normal training |
|---|---:|---:|
| Epochs | 50 | 100 |
| Patience | — | 15 |
| Optimizer | AdamW (Loshchilov & Hutter) | AdamW (Loshchilov & Hutter) |
| $\beta_1$, $\beta_2$ | 0.9, 0.999 | 0.9, 0.999 |
| Scheduler | Cosine annealing | Cosine annealing |
| Initial LR, final LR | 1e-3, 1e-7 | 3e-2, 3e-5 |
| Batch size | 8 | 1 |
| Gradient accumulation | — | 4 |
| Hardware used | NVIDIA GTX 1650 | NVIDIA T4 |
| VRAM | 4 GB | 16 GB |

The hardware rows record the experimental setup; they are not runtime
requirements. AdamW uses its standard beta values shown above.

## Installation

Python 3.10 or newer is required.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e .
```

Install the optional Hugging Face dependency only when using the Wav2Vec2
backbone:

```bash
python -m pip install -e ".[wav2vec]"
```

For development and tests:

```bash
python -m pip install -e ".[dev]"
pytest
```

## Dataset setup

The audio corpora have their own licenses and are not redistributed. Arrange
them beneath one dataset root using the relative paths already recorded in the
split JSON files. For example:

```text
Datasets/
├── Chinese/NCMMSC/...
├── English/Pitt/...
├── English/VAS/...
├── Greek/Dem@Care/...
├── Mandarin/Chou/...
└── Spanish/Ivanova/...
```

Set the root once or pass it to every command:

```bash
export MIRACLE_AD_DATA_ROOT=/path/to/Datasets
# PowerShell: $env:MIRACLE_AD_DATA_ROOT = "D:\path\to\Datasets"
```

Check split integrity and path resolution before a long run:

```bash
miracle-ad-splits --check-files
```

The committed artifacts are in
[`src/miracle_ad/splits`](src/miracle_ad/splits). See
[`docs/DATASETS.md`](docs/DATASETS.md) for labels, partition semantics, and the
expected corpus layout.

Each JSON contains fixed `train`, `val`, and `test` membership. Validation is
never redrawn when a run starts.

## Supervised training

Run one paper-style configuration:

```bash
miracle-ad-train \
  --datasets ncmmsc taukadial_c mchou \
  --backbone gamma_gm_cnn \
  --network abmil \
  --combine-mci-ad \
  --no-language-aware
```

For multilingual language-aware training:

```bash
miracle-ad-train --config configs/example_multilingual.yaml
```

CLI values override values loaded from `--config`. Every run writes:

- `run_config.json` with the resolved hyperparameters
- `resolved_splits.json` with every exact recording assignment
- best/last model state, optimizer state, and scheduler state
- metric histories, confusion matrices, TensorBoard logs, and optional attention maps

Outputs default to `outputs/supervised/` and are ignored by Git.
See [`docs/CLI.md`](docs/CLI.md) for every supervised, SSL, inference, split,
and grid-runner argument.

## Self-supervised backbone pretraining

SimCLR-style pretraining uses NT-Xent loss and two augmented views of each raw
audio chunk. It draws only from the supervised training/validation pool; the
fixed test holdout is never used.

```bash
miracle-ad-ssl \
  --datasets ncmmsc taukadial_c mchou \
  --backbone gamma_gm_cnn \
  --data-root /path/to/Datasets \
  --epochs 50
```

The best backbone is saved as `_best.pt`. Use it in supervised training either
by auto-discovery or with an explicit path:

```bash
miracle-ad-train \
  --datasets ncmmsc taukadial_c mchou \
  --backbone gamma_gm_cnn \
  --network gated_abmil \
  --ssl-mode finetune
```

`--ssl-mode freeze` keeps the pretrained backbone fixed; `finetune` updates it
end to end. SSL runs also save their resolved configuration and exact recording
membership. More detail is in [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md).

## Inference

The inference command reads model settings from the checkpoint's neighboring
`run_config.json` and returns disease probabilities plus attention weights when
the selected pooling network supports them.

```bash
miracle-ad-predict \
  --input sample.wav \
  --checkpoint outputs/supervised/.../best_model.pth \
  --output-json prediction.json
```

## Retained experiment options

The two Gammatone backbones are the paper's primary models, but all backbone
options from the research workspace remain available:

`gamma_gm_cnn`, `gamma_erb_cnn`, `kurdish_cnn`, `soundnet8`, `soundnet5`,
`m3`, `m5`, `m11`, `m18`, `raw_audio_cnn`, `wavenet`, `raw_audio_lstm`, and
`wav2vec2`.

Pooling choices are `unilstm`, `bilstm`, `abmil`, `gated_abmil`, and the
experimental `transformer_abmil`.

The exact migration scope—including the generated files, notebooks, and
machine-specific comparison scripts intentionally left behind—is documented in
[`docs/MIGRATION.md`](docs/MIGRATION.md).

Generate the complete paper architecture grid without launching it:

```bash
python scripts/run_paper_grid.py \
  --group multilingual \
  --data-root /path/to/Datasets
```

Add `--execute` to run the generated commands.

## Repository layout

```text
.
├── configs/                  # Reproducible experiment presets
├── docs/                     # Dataset, CLI, experiment, and migration guidance
├── scripts/                  # Grid orchestration utilities
├── src/miracle_ad/
│   ├── cli/                  # Train, inference, and split inspection CLIs
│   ├── splits/               # Versioned JSON split artifacts
│   ├── data.py               # Raw-audio MIL bags
│   ├── models.py             # All retained backbones and pooling networks
│   ├── split_registry.py     # Portable split resolution and validation
│   ├── ssl_data.py           # Contrastive audio pairs
│   ├── ssl_training.py       # SSL trainer
│   └── training.py           # Supervised trainer and evaluation
└── tests/                    # Data, split, model, and SSL smoke tests
```

## Responsible use

This is research software, not a clinical diagnostic device. Predictions and
attention weights must not be interpreted as medical conclusions without
independent clinical validation.

## License

MIT. See [`LICENSE`](LICENSE).
